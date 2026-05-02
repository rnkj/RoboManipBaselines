import os
import sys
import warnings

import numpy as np
import torch
import torch.utils.data
from stable_pretraining.backbone.utils import vit_hf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robo_manip_baselines.common import CachedDataset, TrainBase

from .LeWmDataset import LeWmDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/le-wm"))
from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder, SIGReg  # noqa: E402

_VIT_SIZE_CHOICES = ("tiny", "small", "base", "large")


class TrainLeWm(TrainBase):
    DatasetClass = LeWmDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=False)
        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(lr=5e-5)

        parser.add_argument(
            "--camera_name",
            type=str,
            default="front",
            help=(
                "single camera name to feed into the ViT encoder "
                "(overrides --camera_names)"
            ),
        )
        parser.add_argument(
            "--history_size",
            type=int,
            default=3,
            help="number of context frames fed to the ARPredictor",
        )
        parser.add_argument(
            "--num_preds",
            type=int,
            default=1,
            help="number of prediction offset frames",
        )
        parser.add_argument("--img_size", type=int, default=224)
        parser.add_argument("--patch_size", type=int, default=14)
        parser.add_argument(
            "--encoder_scale",
            type=str,
            default="tiny",
            choices=list(_VIT_SIZE_CHOICES),
        )
        parser.add_argument("--embed_dim", type=int, default=192)
        parser.add_argument("--pred_depth", type=int, default=6)
        parser.add_argument("--pred_heads", type=int, default=16)
        parser.add_argument("--pred_mlp_dim", type=int, default=2048)
        parser.add_argument("--pred_dim_head", type=int, default=64)
        parser.add_argument("--pred_dropout", type=float, default=0.1)
        parser.add_argument("--pred_emb_dropout", type=float, default=0.0)
        parser.add_argument("--sigreg_weight", type=float, default=0.09)
        parser.add_argument("--sigreg_knots", type=int, default=17)
        parser.add_argument("--sigreg_num_proj", type=int, default=1024)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--grad_clip", type=float, default=1.0)
        parser.add_argument(
            "--warmup_ratio",
            type=float,
            default=0.2,
            help=(
                "Fraction of total epochs used for linear LR warmup "
                "(then cosine annealing). Set 0 to disable scheduler."
            ),
        )
        parser.add_argument(
            "--warmup_start_factor",
            type=float,
            default=0.01,
            help="Initial LR scale at the start of warmup (relative to --lr).",
        )
        parser.add_argument(
            "--use_bf16",
            action="store_true",
            default=False,
            help=(
                "Enable bf16 mixed-precision training via torch.autocast "
                "(matches official Lightning precision='bf16')."
            ),
        )
        parser.add_argument(
            "--pretrained",
            action="store_true",
            default=False,
            help=(
                "Load pretrained ViT weights from HuggingFace "
                "(google/vit-{scale}-patch{patch_size}-{img_size}). "
                "Requires patch_size=16 for standard Google ViT checkpoints."
            ),
        )
        parser.add_argument(
            "--freeze_encoder",
            action="store_true",
            default=False,
            help="Freeze encoder parameters; only predictor/projector/action_encoder are trained.",
        )

    def setup_model_meta_info(self):
        self.args.camera_names = [self.args.camera_name]
        super().setup_model_meta_info()

        num_steps = self.args.history_size + self.args.num_preds
        self.model_meta_info["data"].update(
            {
                "history_size": self.args.history_size,
                "num_preds": self.args.num_preds,
                "num_steps": num_steps,
                "img_size": self.args.img_size,
            }
        )

    def setup_dataset(self):
        if self.args.enable_rmb_cache and self.args.use_cached_dataset:
            raise ValueError(
                f"[{self.__class__.__name__}] Both 'enable_rmb_cache' and "
                "'use_cached_dataset' options cannot be True at the same time."
            )

        if self.args.val_ratio is not None:
            warnings.warn(
                f"[{self.__class__.__name__}] --val_ratio is ignored; "
                "LeWm uses (1 - train_ratio) for window-level random_split.",
                stacklevel=2,
            )

        self.set_data_stats()

        full_dataset = self.DatasetClass(
            self.all_filenames, self.model_meta_info, self.args.enable_rmb_cache
        )
        if self.args.use_cached_dataset:
            full_dataset = CachedDataset(full_dataset)
        self.full_dataset = full_dataset

        n = len(full_dataset)
        ratio = float(np.clip(self.args.train_ratio, 0.0, 1.0))
        n_train = max(int(ratio * n), 1)
        n_val = max(n - n_train, 1)
        n_train = n - n_val

        generator = torch.Generator().manual_seed(int(self.args.seed))
        train_set, val_set = torch.utils.data.random_split(
            full_dataset, lengths=[n_train, n_val], generator=generator
        )

        loader_kwargs = {
            "batch_size": self.args.batch_size,
            "pin_memory": True,
            "num_workers": self.args.num_workers,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        self.train_dataloader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        self.val_dataloader = DataLoader(val_set, shuffle=False, **loader_kwargs)

        self.writer = SummaryWriter(self.args.checkpoint_dir)
        self.print_dataset_info()

    def print_dataset_info(self):
        train_set = self.train_dataloader.dataset
        val_set = self.val_dataloader.dataset
        head = 8
        print(
            f"[{self.__class__.__name__}] Load dataset from {self.args.dataset_dir}\n"
            f"  - source files: {len(self.all_filenames)}\n"
            f"  - train windows: {len(train_set)}, val windows: {len(val_set)} "
            f"(total windows: {len(self.full_dataset)})\n"
            f"  - train_ratio: {self.args.train_ratio}, seed: {self.args.seed}\n"
            f"  - train indices[:{head}]: {list(train_set.indices[:head])}\n"
            f"  - val indices[:{head}]:   {list(val_set.indices[:head])}"
        )

    def setup_policy(self):
        predictor_kwargs = {
            "depth": self.args.pred_depth,
            "heads": self.args.pred_heads,
            "mlp_dim": self.args.pred_mlp_dim,
            "dim_head": self.args.pred_dim_head,
            "dropout": self.args.pred_dropout,
            "emb_dropout": self.args.pred_emb_dropout,
        }
        sigreg_kwargs = {
            "knots": self.args.sigreg_knots,
            "num_proj": self.args.sigreg_num_proj,
        }

        # Save reconstruction args to meta info before instantiation
        self.model_meta_info["policy"]["args"] = {
            "encoder_scale": self.args.encoder_scale,
            "patch_size": self.args.patch_size,
            "img_size": self.args.img_size,
            "embed_dim": self.args.embed_dim,
            "history_size": self.args.history_size,
            "num_preds": self.args.num_preds,
            "predictor": predictor_kwargs,
            "sigreg": {"weight": self.args.sigreg_weight, "kwargs": sigreg_kwargs},
            "warmup_ratio": self.args.warmup_ratio,
            "warmup_start_factor": self.args.warmup_start_factor,
            "use_bf16": self.args.use_bf16,
        }

        encoder = vit_hf(
            self.args.encoder_scale,
            patch_size=self.args.patch_size,
            image_size=self.args.img_size,
            pretrained=self.args.pretrained,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = self.args.embed_dim
        action_dim = len(self.model_meta_info["action"]["example"])
        effective_act_dim = self.args.skip * action_dim

        predictor = ARPredictor(
            num_frames=self.args.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **predictor_kwargs,
        )
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        projector = MLP(
            input_dim=hidden_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )
        pred_proj = MLP(
            input_dim=hidden_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )

        self.policy = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        ).cuda()
        self.sigreg = SIGReg(**sigreg_kwargs).cuda()
        self.sigreg_weight = self.args.sigreg_weight

        if self.args.freeze_encoder:
            for param in self.policy.encoder.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        if self.args.warmup_ratio > 0:
            total_epochs = max(self.args.num_epochs, 1)
            warmup_epochs = max(int(self.args.warmup_ratio * total_epochs), 1)
            cosine_epochs = max(total_epochs - warmup_epochs, 1)
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=self.args.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs)
            self.lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            self.lr_scheduler = None

        self.print_policy_info()
        print(
            f"  - encoder: ViT-{self.args.encoder_scale} "
            f"(hidden_dim={hidden_dim}, patch={self.args.patch_size}, "
            f"img={self.args.img_size}, "
            f"pretrained={self.args.pretrained}, freeze={self.args.freeze_encoder})"
        )
        print(
            f"  - wm: history_size={self.args.history_size}, "
            f"num_preds={self.args.num_preds}, "
            f"skip={self.args.skip}, "
            f"effective_act_dim={effective_act_dim}, embed_dim={embed_dim}"
        )

        # count the number of parameters
        total_all = sum(p.numel() for p in self.policy.parameters())
        trainable_all = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        print(
            f"  - params: total={total_all:,} ({total_all / 1e6:.2f}M), "
            f"trainable={trainable_all:,} ({trainable_all / 1e6:.2f}M)"
        )
        #for name, module in [
        #    ("encoder", self.policy.encoder),
        #    ("predictor", self.policy.predictor),
        #    ("action_encoder", self.policy.action_encoder),
        #    ("projector", self.policy.projector),
        #    ("pred_proj", self.policy.pred_proj),
        #]:
        #    total_m = sum(p.numel() for p in module.parameters())
        #    trainable_m = sum(p.numel() for p in module.parameters() if p.requires_grad)
        #    print(f"  - {name}: total={total_m:,}, trainable={trainable_m:,}")

    def _forward_batch(self, batch):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.policy.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_len = self.args.history_size
        n_preds = self.args.num_preds
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.policy.predict(ctx_emb, ctx_act)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }

    def _forward_with_amp(self, batch):
        if self.args.use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward_batch(batch)
        return self._forward_batch(batch)

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            self.policy.train()
            batch_result_list = []
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                result = self._forward_with_amp(batch)
                result["loss"].backward()
                if self.args.grad_clip is not None and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.policy.parameters() if p.requires_grad],
                        self.args.grad_clip,
                    )
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for batch in self.val_dataloader:
                    result = self._forward_with_amp(batch)
                    batch_result_list.append(self.detach_batch_result(result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)
                self.update_best_ckpt(epoch_summary)

            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        self.save_current_ckpt("last")
        self.save_best_ckpt()
