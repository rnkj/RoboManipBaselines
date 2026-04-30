# LeWm (LeWorldModel) policy

## Installation

```bash
pip install -e ".[le-wm]"
```

## Model Training

```bash
cd robo_manip_baselines
python ./bin/Train.py LeWm \
    --dataset_dir ./dataset/MujocoUR5eToolbox_Dataset100 \
    --camera_name front \
    --skip 1 \
    --history_size 3 \
    --num_preds 1 \
    --encoder_scale tiny \
    --img_size 224 \
    --patch_size 14 \
    --embed_dim 192 \
    --pred_depth 4 \
    --pred_heads 8 \
    --pred_mlp_dim 1024 \
    --pred_dim_head 64 \
    --pred_dropout 0.1 \
    --batch_size 64 \
    --num_epochs 1000 \
    --lr 3e-4 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --sigreg_weight 0.09 \
    --use_cached_dataset
```

Note: Due to the computational time,
- turning on `--use_cached_dataset` is highly recommended, and
- RmbData-SingleHDF5 (`.hdf5`) format is better.

## Policy rollout

```bash
cd robo_manip_baselines
python ./bin/Rollout.py LeWm MujocoUR5eToolbox \
    --checkpoint ./checkpoint/LeWm/<run_dir>/policy_last.ckpt \
    --goal_dataset_dir ./dataset/MujocoUR5eToolbox_Dataset100 \
    --goal_episode_idx 0 \
    --goal_step_idx -1 \
    --horizon 5 \
    --receding_horizon 1 \
    --num_samples 300 \
    --n_cem_iters 3 \
    --topk 30 \
    --var_scale 0.5 \
    --warm_start \
    --max_duration 30 \
    --action_clip
```

Note: at training time, `--skip` controls the stride at which states/images
are decimated from the raw dataset (RmbData decimation stride), while
`frameskip` is the number of raw action frames bundled into one LeWm step
token (stored separately in `model_meta_info["data"]["frameskip"]`). At
rollout time, `--skip` (in the RolloutBase sense of an env-step interval) is
always forced to 1 because LeWm feeds one raw action per env step; the
training-time bundle width is restored from
`model_meta_info["data"]["frameskip"]`.

## Limitations

- **Single camera**: `JEPA.encode` expects `batch["pixels"]` as `(B, T, C, H, W)`;
  multi-camera fusion is outside the scope of this implementation.
- **Proprioception is unused**: normalized state from `model_meta_info["state"]`
  is placed in `batch["observation"]`, but the upstream `lejepa_forward` only
  reads `pixels` and `action`. The Rollout side likewise ignores proprio, and
  the goal is image-only.
- **Goal/world consistency**: the goal is taken directly from a single frame
  of the specified dataset, so `--world_idx` (initial robot pose) and
  `--goal_episode_idx` must correspond to the same world configuration to be
  meaningful. For initial testing, align them explicitly, e.g.
  `--world_idx 0 --goal_episode_idx 0`.

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@article{leworldmodel2026,
  author    = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  title     = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  year      = {2026},
}
```
