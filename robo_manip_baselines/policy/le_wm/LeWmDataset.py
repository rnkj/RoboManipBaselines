import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    normalize_data,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeWmDataset(DatasetBase):
    """Dataset to train LeWm (LeWorldModel) policy.

    Produces num_steps-windowed samples compatible with le-wm's lejepa_forward.
    Two stride parameters are applied in order:
      1. `skip` (RoboManipBaselines convention): raw-frame decimation stride.
         e.g. `--skip 3` turns 30 FPS data into a pseudo 10 FPS timeline.
      2. `frameskip` (upstream le-wm semantics): number of consecutive
         (post-skip) action frames bundled into one world-model token.
    Output shapes:
      - pixels: (num_steps, 3, img_size, img_size) float32, ImageNet-normalized
      - action: (num_steps, frameskip * action_dim) float32, normalized
      - observation: (num_steps, state_dim) float32, normalized
        (currently not consumed by JEPA.encode; carried through for future extension)
    """

    def setup_image_transforms(self):
        img_size = self.model_meta_info["data"]["img_size"]
        self.image_transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((img_size, img_size), antialias=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        frameskip = self.model_meta_info["data"]["frameskip"]
        num_steps = self.model_meta_info["data"]["num_steps"]
        span_thinned = num_steps * frameskip

        self.chunk_info_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len_thinned = rmb_data[DataKey.TIME][::skip].shape[0]
            if episode_len_thinned < span_thinned:
                continue
            for start_time_idx in range(0, episode_len_thinned - span_thinned + 1):
                self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        frameskip = self.model_meta_info["data"]["frameskip"]
        num_steps = self.model_meta_info["data"]["num_steps"]
        span_thinned = num_steps * frameskip
        camera_name = self.model_meta_info["image"]["camera_names"][0]
        episode_idx, start = self.chunk_info_list[chunk_idx]
        end = start + span_thinned

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            # Load state on the thinned timeline, sampled at `frameskip` stride
            # (num_steps, state_dim).
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros((num_steps, 0), dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[
                            start:end:frameskip
                        ]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action on the thinned timeline; keep every (post-skip) frame
            # in the window: (span_thinned, action_dim) -> (num_steps, frameskip, action_dim)
            action_window = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[start:end]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )
            action_dim = action_window.shape[-1]
            action = action_window.reshape(num_steps, frameskip, action_dim)

            # Load images: decimate raw frames by `skip`, then take one per
            # macro-step at `frameskip` stride. (num_steps, H, W, 3) uint8.
            images = rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][
                start:end:frameskip
            ]

        # Normalize state/action (action mean/std broadcasts over the frameskip axis)
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])
        action = action.reshape(num_steps, frameskip * action_dim)

        # Reorder image axes: (num_steps, H, W, 3) -> (num_steps, 3, H, W)
        images = np.moveaxis(images, -1, -3)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images.copy(), dtype=torch.uint8)

        # Apply ImageNet transforms (uint8 -> float32 [0,1] -> resize -> ImageNet normalize)
        images_tensor = self.image_transforms(images_tensor)

        # State/action augmentation (image augmentation is intentionally skipped here)
        state_tensor, action_tensor, _ = self.augment_data(
            state_tensor, action_tensor, None
        )

        return {
            "pixels": images_tensor,
            "action": action_tensor,
            "observation": state_tensor,
        }
