import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class ViveInputDevice(InputDeviceBase):
    """HTC Vive Controller for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        device_params,
        pos_scale=1.0,
        gripper_scale=5.0,
        vive_to_eef_frame_rotation=None,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.name = device_params["name"]
        self.serial_number = device_params["serial_number"]
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        if vive_to_eef_frame_rotation is None:
            self.vive_to_eef_frame_rotation = np.eye(3)
        else:
            self.vive_to_eef_frame_rotation = np.array(
                vive_to_eef_frame_rotation, dtype=np.float64
            )
        assert self.vive_to_eef_frame_rotation.shape == (3, 3)

    def connect(self):
        self.enabled_teleop = False
        self.prev_is_enable_teleop_pressed = False
        self.vive_se3_at_enable = None
        self.eef_se3_at_enable = None
        self.has_announced_ready = False

        if self.connected:
            return

        import openvr

        self.vr_system = openvr.init(openvr.VRApplication_Other)
        self.openvr = openvr
        self.connected = True

    def close(self):
        if self.connected:
            self.openvr.shutdown()
            self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self._read_vive()

        if self.state is None:
            self.prev_is_enable_teleop_pressed = False
            self.has_announced_ready = False
            return

        # Use the trigger axis (axes[2]) to enable teleop
        is_enable_teleop_pressed = self.state["axes"][2] == 1.0
        if is_enable_teleop_pressed and not self.prev_is_enable_teleop_pressed:
            self.enabled_teleop = True
            self.vive_se3_at_enable = self.state["se3"].copy()
            self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
            print(
                f"[{self.__class__.__name__}] Teleoperation enabled for Vive '{self.name}'."
            )
        self.prev_is_enable_teleop_pressed = is_enable_teleop_pressed

    def _read_vive(self):
        openvr = self.openvr
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        state = None

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            pose = poses[i]

            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                continue

            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class != openvr.TrackedDeviceClass_Controller:
                continue

            sn = self.vr_system.getStringTrackedDeviceProperty(
                i, openvr.Prop_SerialNumber_String
            )
            if sn != self.serial_number:
                continue

            mat = np.asarray(pose.mDeviceToAbsoluteTracking.m, dtype=np.float64)
            se3 = pin.SE3(mat[:, :3], mat[:, 3])

            _, vive_state = self.vr_system.getControllerState(i)
            axes = np.array(
                [
                    vive_state.rAxis[0].x,
                    vive_state.rAxis[0].y,
                    vive_state.rAxis[1].x,
                ]
            )
            buttons = {
                "application_menu": bool(
                    vive_state.ulButtonPressed >> openvr.k_EButton_ApplicationMenu & 1
                ),
                "trackpad_pressed": bool(
                    vive_state.ulButtonPressed >> openvr.k_EButton_SteamVR_Touchpad & 1
                ),
            }

            state = {
                "se3": se3,
                "axes": axes,
                "buttons": buttons,
            }

            break

        self.state = state
        if (state is not None) and (not self.has_announced_ready):
            print(
                f"[{self.__class__.__name__}] Vive '{self.name}' is ready. Fully pull the trigger to start teleoperation."
            )
            self.has_announced_ready = True

    def is_ready(self):
        return self.state is not None

    def set_command_data(self):
        if (not self.enabled_teleop) or (self.state is None):
            return

        # Set arm command
        delta_vive_se3 = self.vive_se3_at_enable.inverse() * self.state["se3"]
        adjusted_delta_vive_se3 = pin.SE3(
            self.vive_to_eef_frame_rotation
            @ delta_vive_se3.rotation
            @ self.vive_to_eef_frame_rotation.T,
            self.vive_to_eef_frame_rotation @ delta_vive_se3.translation,
        )
        scaled_delta_vive_se3 = pin.SE3(
            adjusted_delta_vive_se3.rotation,
            self.pos_scale * adjusted_delta_vive_se3.translation,
        )
        target_se3 = self.eef_se3_at_enable * scaled_delta_vive_se3

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        buttons = self.state["buttons"]
        if buttons["application_menu"] and not buttons["trackpad_pressed"]:
            gripper_joint_pos -= self.gripper_scale
        elif buttons["trackpad_pressed"] and not buttons["application_menu"]:
            gripper_joint_pos += self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
