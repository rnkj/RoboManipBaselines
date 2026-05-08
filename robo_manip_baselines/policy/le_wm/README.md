# LeWm (LeWorldModel) policy

## Installation

```bash
pip install -e ".[le-wm]"
```

## Model Training

```bash
cd robo_manip_baselines
python ./bin/Train.py LeWm \
    --dataset_dir ./dataset/MujocoXarm7Pusht_Dataset100 \
    --camera_name front \
    --num_epoch 10000 \
    --use_bf16 \
    --num_workers 4 \
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
    --receding_horizon 5 \
    --num_samples 300 \
    --n_cem_iters 30 \
    --topk 30 \
    --var_scale 0.5 \
    --max_duration 30
```

Note: two stride parameters are applied in order:

1. `--skip` (RoboManipBaselines convention): raw-frame decimation stride.
   For example, `--skip 3` turns 30 FPS data into a pseudo 10 FPS timeline.
2. `--frameskip` (upstream le-wm semantics, default 5): number of consecutive
   post-skip action frames bundled into one world-model token. The action
   encoder input dimension is therefore `frameskip * action_dim`.

Both values are persisted in `model_meta_info["data"]` (`"skip"` and
`"frameskip"`). At rollout time, `--skip` defaults to the training-time
decimation (RolloutBase's standard fallback) and `--frameskip` is always
restored from the checkpoint; the planner consumes one bundled action per
`args.skip` env steps.

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
