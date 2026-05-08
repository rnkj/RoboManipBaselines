import os
import sys
from collections import deque

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    RolloutBase,
    denormalize_data,
    find_rmb_files,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/le-wm"))
from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder  # noqa: E402
from stable_pretraining.backbone.utils import vit_hf  # noqa: E402

from .LeWmDataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


class RolloutLeWm(RolloutBase):
    """Closed-loop rollout for LeWm via CEM planning on top of JEPA.get_cost.

    LeWm is a world model (no direct action head), so we run Cross-Entropy
    Method planning over action sequences, scoring each candidate by the
    goal-embedding MSE returned from JEPA.get_cost. A first-in-first-out
    action buffer decouples planning cadence (once every receding_horizon *
    skip env steps) from the per-step env.step() consumption.

    The CEM loop is written to mirror stable_worldmodel.solver.CEMSolver so
    that a parity test is possible (see tests/TestLeWmCem.py).
    """

    def set_additional_args(self, parser):
        # `args.skip` is the rollout step interval (RolloutBase semantics);
        # forced to 1 because LeWm plans raw actions and pops one per env step.
        # The training-time bundle width (raw frames per LeWm step token) is
        # read separately from `model_meta_info["data"]["skip"]` in setup_policy.
        parser.set_defaults(skip=1)

        # Planning hyper-parameters.
        parser.add_argument(
            "--horizon",
            type=int,
            default=5,
            help="CEM planning horizon T in bundled-action steps (T >= history_size)",
        )
        parser.add_argument(
            "--receding_horizon",
            type=int,
            default=5,
            help="number of bundled-action steps executed before replanning",
        )
        parser.add_argument("--num_samples", type=int, default=300)
        parser.add_argument("--n_cem_iters", type=int, default=10)
        parser.add_argument("--topk", type=int, default=30)
        parser.add_argument("--var_scale", type=float, default=1.0)
        parser.add_argument(
            "--warm_start",
            action="store_true",
            help="carry (mean, std) across plans (shifted by receding_horizon)",
        )

        # Goal specification.
        parser.add_argument(
            "--goal_dataset_dir",
            type=str,
            required=True,
            help="RmbData directory or single .rmb/.hdf5 file providing the goal frame",
        )
        parser.add_argument("--goal_episode_idx", type=int, default=0)
        parser.add_argument(
            "--goal_step_idx",
            type=int,
            default=-1,
            help="frame index within the goal episode (negative allowed)",
        )

        parser.add_argument(
            "--action_clip",
            action="store_true",
            help="clip raw actions to env.action_space.low/high before executing",
        )

    def setup_policy(self):
        meta = self.model_meta_info
        data_meta = meta["data"]
        policy_args = meta["policy"]["args"]

        # `self.skip` is the training-time action bundle width (raw frames per
        # LeWm step token), which is unified with the RmbData decimation stride
        # under `data_meta["skip"]` (see LeWmDataset).
        # `self.args.skip` is the rollout step interval forced to 1 in
        # set_additional_args.
        self.skip = data_meta["skip"]
        self.history_size = data_meta["history_size"]
        self.num_preds = data_meta["num_preds"]
        self.img_size = data_meta["img_size"]
        self.effective_act_dim = self.skip * self.action_dim

        if self.args.horizon < self.history_size:
            raise ValueError(
                f"--horizon ({self.args.horizon}) must be >= history_size "
                f"({self.history_size}); JEPA.rollout splits action_sequence into "
                f"[H, T-H] and requires T >= H."
            )
        if not (1 <= self.args.receding_horizon <= self.args.horizon):
            raise ValueError(
                f"--receding_horizon must be in [1, horizon={self.args.horizon}], "
                f"got {self.args.receding_horizon}"
            )
        if not (1 <= self.args.topk <= self.args.num_samples):
            raise ValueError(
                f"--topk ({self.args.topk}) must be in [1, num_samples="
                f"{self.args.num_samples}]"
            )

        # Mirror TrainLeWm.setup_policy so the state_dict loads cleanly.
        encoder = vit_hf(
            policy_args["encoder_scale"],
            patch_size=policy_args["patch_size"],
            image_size=policy_args["img_size"],
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = policy_args["embed_dim"]

        predictor = ARPredictor(
            num_frames=policy_args["history_size"],
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **policy_args["predictor"],
        )
        action_encoder = Embedder(input_dim=self.effective_act_dim, emb_dim=embed_dim)
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
        )

        # image_transforms must exist before _load_goal; setup_variables runs later.
        self._build_image_transforms()

        self.load_ckpt()

        self.print_policy_info()
        print(
            f"  - encoder: ViT-{policy_args['encoder_scale']} "
            f"(hidden_dim={hidden_dim}, patch={policy_args['patch_size']}, "
            f"img={policy_args['img_size']})"
        )
        print(
            f"  - wm: history_size={self.history_size}, num_preds={self.num_preds}, "
            f"skip={self.skip}, effective_act_dim={self.effective_act_dim}, "
            f"embed_dim={embed_dim}"
        )
        print(
            f"  - planner: horizon={self.args.horizon}, "
            f"receding_horizon={self.args.receding_horizon}, "
            f"num_samples={self.args.num_samples}, n_cem_iters={self.args.n_cem_iters}, "
            f"topk={self.args.topk}, var_scale={self.args.var_scale}, "
            f"warm_start={self.args.warm_start}"
        )

        self._load_goal()

    def setup_variables(self):
        # Override base: base would install a float32-only transform, but
        # JEPA expects ImageNet-normalized 224x224 input.
        self._build_image_transforms()

    def reset_variables(self):
        super().reset_variables()
        self.past_pixels = deque(maxlen=self.history_size)
        self.action_plan_buf = []
        self.cem_mean = None
        self.cem_std = None
        # list of [best_cost, mean_cost] per planning call for diagnostics
        self.cem_cost_log = []
        self._cost_log_path = os.path.join(
            os.path.dirname(self.args.checkpoint),
            f"cem_cost_log_ep{self.data_manager.episode_idx:0>3}.npy",
        )

    def setup_plot(self, fig_ax=None):
        if fig_ax is None:
            fig_ax = plt.subplots(
                1,
                3,
                figsize=(13.5, 4.5),
                dpi=60,
                squeeze=False,
                constrained_layout=True,
            )
        super().setup_plot(fig_ax)

    def _build_image_transforms(self):
        self.image_transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((self.img_size, self.img_size), antialias=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _preprocess_rgb(self, rgb_hwc_u8):
        img = np.moveaxis(rgb_hwc_u8, -1, 0)
        img = torch.from_numpy(np.ascontiguousarray(img)).to(dtype=torch.uint8)
        return self.image_transforms(img)

    def _push_pixels(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        if len(self.past_pixels) == 0:
            for _ in range(self.history_size):
                self.past_pixels.append(img_tensor)
        else:
            self.past_pixels.append(img_tensor)

    def _load_goal(self):
        rmb_paths = find_rmb_files(self.args.goal_dataset_dir)
        if self.args.goal_episode_idx >= len(rmb_paths):
            raise ValueError(
                f"--goal_episode_idx {self.args.goal_episode_idx} out of range; "
                f"found {len(rmb_paths)} episodes in {self.args.goal_dataset_dir}"
            )
        goal_path = rmb_paths[self.args.goal_episode_idx]
        camera_name = self.camera_names[0]
        with RmbData(goal_path) as rmb_data:
            rgb_key = DataKey.get_rgb_image_key(camera_name)
            rgb = np.array(rmb_data[rgb_key][self.args.goal_step_idx])

        self.goal_rgb_raw = rgb
        goal_tensor = self._preprocess_rgb(rgb).to(self.device)
        # Shape: (B=1, T_goal=1, 3, H, W). The S dim is added on the fly in _plan_cem.
        self.goal_pixels = goal_tensor.unsqueeze(0).unsqueeze(0)

        print(
            f"[{self.__class__.__name__}] Loaded goal from {goal_path} "
            f"(episode_idx={self.args.goal_episode_idx}, "
            f"step_idx={self.args.goal_step_idx})"
        )

    def infer_policy(self):
        img = self._preprocess_rgb(self.info["rgb_images"][self.camera_names[0]])
        self._push_pixels(img)

        if len(self.action_plan_buf) == 0:
            self.action_plan_buf.extend(self._plan_cem())
            np.save(self._cost_log_path, np.array(self.cem_cost_log))

        self.policy_action = self.action_plan_buf.pop(0)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def _plan_cem(self, generator=None):
        """Run CEM for one planning call and return raw actions to execute.

        Mirrors stable_worldmodel.solver.CEMSolver.solve so that a parity
        test is possible. CEM optimizes the full horizon (all T bundled
        steps); the "history" embedding for the predictor comes from
        past_pixels via JEPA.encode, not from fixed past actions.
        """
        T = self.args.horizon
        S = self.args.num_samples
        K = self.args.topk
        A = self.effective_act_dim
        device = self.device

        if self.args.warm_start and self.cem_mean is not None:
            shift = min(self.args.receding_horizon, T)
            mean = torch.roll(self.cem_mean, shifts=-shift, dims=0)
            mean[-shift:] = 0.0
            std = self.cem_std.clone()
        else:
            mean = torch.zeros(T, A, device=device)
            std = torch.full((T, A), float(self.args.var_scale), device=device)

        pixels_hist = torch.stack(list(self.past_pixels), dim=0)
        pixels_hist = pixels_hist.unsqueeze(0).unsqueeze(0)  # (1, 1, H, 3, h, w)
        goal_expanded = self.goal_pixels.unsqueeze(1)  # (1, 1, 1, 3, h, w)
        # Dummy action so that JEPA.get_cost's `goal.pop("action")` does not
        # raise KeyError; the value is unused (rollout overwrites info["action"]).
        dummy_action = torch.zeros(1, 1, 1, A, device=device)

        for _ in range(self.args.n_cem_iters):
            noise = torch.randn(S, T, A, device=device, generator=generator)
            samples = mean + std * noise
            samples[0] = mean  # parity with CEMSolver: force first sample = mean

            info = {
                "pixels": pixels_hist,
                "goal": goal_expanded,
                "action": dummy_action,
            }
            cost = self.policy.get_cost(info, samples.unsqueeze(0))[0]  # (S,)

            top_idx = torch.topk(cost, K, largest=False).indices
            elite = samples[top_idx]
            mean = elite.mean(0)
            std = elite.std(0)

        if self.args.warm_start:
            self.cem_mean = mean.detach().clone()
            self.cem_std = std.detach().clone()

        best_cost = float(cost.min().item())
        mean_cost = float(cost.mean().item())
        self.cem_cost_log.append([best_cost, mean_cost])
        print(
            f"[CEM plan #{len(self.cem_cost_log):>3d}]"
            f"  best={best_cost:.6f}  mean={mean_cost:.6f}",
            flush=True,
        )

        best = mean.detach().cpu().numpy()  # (T, A)
        raw_list = []
        for k in range(self.args.receding_horizon):
            bundled = best[k].reshape(self.skip, self.action_dim)
            for f in range(self.skip):
                raw = denormalize_data(bundled[f], self.model_meta_info["action"])
                if self.args.action_clip:
                    raw = np.clip(
                        raw,
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )
                raw_list.append(raw.astype(np.float64))
        return raw_list

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        camera_name = self.camera_names[0]
        self.ax[0, 0].imshow(self.info["rgb_images"][camera_name])
        self.ax[0, 0].set_title(f"current ({camera_name})", fontsize=18)

        self.ax[0, 1].imshow(self.goal_rgb_raw)
        self.ax[0, 1].set_title("goal", fontsize=18)

        self.plot_action(self.ax[0, 2])

        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
