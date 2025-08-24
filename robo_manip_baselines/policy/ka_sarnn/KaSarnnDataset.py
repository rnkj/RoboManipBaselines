import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    crop_and_resize,
    get_skipped_data_seq,
    normalize_data,
)


class KaSarnnDataset(DatasetBase):
    """Dataset to train knowledge adapter SARNN policy."""

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        skip = self.model_meta_info["data"]["skip"]
        max_episode_len = self.model_meta_info["data"]["max_episode_len"]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]

            # Load state
            state = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)
                    for key in self.model_meta_info["state"]["keys"]
                ],
                axis=1,
            )

            # Load images
            image_list = [
                rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip]
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]

            # Load worker and robot action indices
            worker_action_indices = get_skipped_data_seq(
                rmb_data[DataKey.WORKER_ACTION_CLASSES][:], DataKey.WORKER_ACTION_CLASSES, skip
            )
            robot_action_indices = get_skipped_data_seq(
                rmb_data[DataKey.ROBOT_ACTION_CLASSES][:], DataKey.ROBOT_ACTION_CLASSES, skip
            )

        # Crop and resize images
        image_crop_size_list = self.model_meta_info["data"]["image_crop_size_list"]
        image_size_list = self.model_meta_info["data"]["image_size_list"]
        image_list = [
            crop_and_resize(*image_and_sizes)
            for image_and_sizes in zip(
                image_list, image_crop_size_list, image_size_list
            )
        ]

        # Add padding
        state = self.pad_last_element(state)
        image_list = [self.pad_last_element(image) for image in image_list]
        worker_action_indices = self.pad_last_element(worker_action_indices)
        robot_action_indices = self.pad_last_element(robot_action_indices)

        # Setup mask
        mask = np.concatenate(
            [np.ones(episode_len), np.zeros(max_episode_len - episode_len)]
        )

        # Pre-convert data
        state, image_list = self.pre_convert_data(state, image_list)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        image_tensor_list = [
            torch.tensor(image, dtype=torch.uint8) for image in image_list
        ]
        action_indices_tensor_list = [
            torch.tensor(action_indices, dtype=torch.long)
            for action_indices in [worker_action_indices, robot_action_indices]
        ]
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        # Since these data are used for both input and output when the policy reconstructs the data,
        # no data augmentation is performed here. Only data conversion is applied.
        image_tensor_list = [
            self.image_transforms(image_tensor) for image_tensor in image_tensor_list
        ]

        # Sort in the order of policy inputs and outputs
        return (
            state_tensor,  # (max_episode_len, state_dim)
            image_tensor_list,  # (num_images, max_episode_len, width, height, 3)
            action_indices_tensor_list,  # (2, max_episode_length)
            mask_tensor,  # (max_episode_len)
        )

    def setup_image_transforms(self):
        """Setup image transforms."""
        self.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def pre_convert_data(self, state, image_list):
        """Pre-convert data. Arguments must be numpy arrays (not torch tensors)."""
        state = normalize_data(state, self.model_meta_info["state"])
        image_list = [np.moveaxis(image, -1, -3) for image in image_list]

        return state, image_list

    def pad_last_element(self, data_seq):
        """Add padding with the last element. Arguments must be numpy array (not torch tensor)."""
        current_len = data_seq.shape[0]
        max_episode_len = self.model_meta_info["data"]["max_episode_len"]

        if current_len == max_episode_len:
            return data_seq

        return np.concatenate(
            [data_seq, np.repeat(data_seq[-1:], max_episode_len - current_len, axis=0)],
            axis=0,
        )
