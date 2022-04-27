"""An implementation of a healthy muscosceletal model."""

from math import sqrt, atan2, pi, sin, cos

from gym import spaces
from opensim import (
    CoordinateActuator,
    Vec3,
)
import numpy as np

from src.envs.stable_pd import QTripplet, StablePD
from src.data import DataRow, TrainingData

from .base_env import BaseOpenSimEnv


PELVIS_VEL_RUNNING_AVG_FACTOR = 0.5
TARGET_PELVIS_VEL = 0.9

JOINT_NAME_TRANSLATION = dict(
    ankle_angle_r="ankle_angle_r",
    hip_adduction_l="hip_adduction_l",
    hip_adduction_r="hip_adduction_r",
    hip_flexion_l="hip_flexion_l",
    hip_flexion_r="hip_flexion_r",
    hip_rotation_l="hip_rotation_l",
    hip_rotation_r="hip_rotation_r",
    knee_angle_r="knee_angle_r",
    pelvis_list="pelvis_list",
    pelvis_rotation="pelvis_rotation",
    pelvis_tilt="pelvis_tilt",
    pelvis_tx="pelvis_tx",
    pelvis_ty="pelvis_ty",
    pelvis_tz="pelvis_tz",
    pros_ankle_angle="knee_angle_l",
    pros_knee_angle="ankle_angle_l",
    lumbar_extension="lumbar_extension",
)


def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return Vec3(x, y, z)


def soft_clip_penalty_to_reward(penalty, softness, reward_max):
    """
    Given a positive penalty, this will compute `reward_max` - `penalty`. With a
    rounded point at `reward_max` = `penalty`, according to `softness`.
    """
    if softness < 0.001:
        return np.maximum(0.0, reward_max - penalty)

    hardness = 4 / softness
    scale = reward_max / np.log1p(np.exp(hardness))
    return scale * np.log1p(np.exp(hardness - penalty / scale))


class ProsthesisOpenSimEnv(BaseOpenSimEnv):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Box(-10, 10, shape=(124,))
        self.action_space = spaces.Box(-1, 1, shape=(17,))

        data_path = kwargs.pop("data_path")
        data_start_time = kwargs.pop("data_start_time", 0)
        data_tempo = kwargs.pop("data_tempo", 1)
        self.data = TrainingData(
            data_path, tempo=data_tempo, start_time=data_start_time
        )

        super().__init__(*args, **kwargs)

        self.record_next()

        self.stable_pd = StablePD(delta_time=self.delta_time)

        # Used in get_reward
        self.pelvis_vel_avg = 0
        self.last_pelvis_pos = None

    def action_to_control_value(self, action, actuator):
        actuator = CoordinateActuator.safeDownCast(actuator)
        if actuator is None:
            return action / 2 + 0.5

        coord = actuator.getCoordinate()
        coord_name = coord.getName()

        coord_name = JOINT_NAME_TRANSLATION[coord_name]

        pos = coord.getValue(self.state)
        vel = coord.getSpeedValue(self.state)
        acc = coord.getAccelerationValue(self.state)
        coord = QTripplet(pos, vel, acc)

        # This is cached in the set_control_values_from_action method we overwrite.
        row = self.cached_data_row
        setpoint = getattr(row, coord_name) + action
        control = self.stable_pd.control(coord, setpoint)

        return max(-1.0, min(1.0, control))

    def set_control_values_from_action(self, controller, action):
        self.cached_data_row = self.data.get_row(self.current_step * self.delta_time)
        super().set_control_values_from_action(controller, action)
        del self.cached_data_row

    def get_observation(self):
        """Compute the observation vector.

        Observation Components:
            - Pelvis position in global frame
            - Pelvis orientation in global frame
            - Pelvis linear velocity in local frame
            - Pelvis angular velocity in local frame. Instead of using xyz
              euler, I decided to meassure local linear velocity at points 0.3
              unit away. The meassured velocity does not contain the velocity of
              the pelvis itself. In total I use two points, the forward point
              meassuring left and up, while the up point meassures left as well.
            - For each joint:
                - Joint position in local frame
                - Joint velocity in local frame
                - Joint orientation in parent frame (ie. angle(s) of the joint)
                - Joint angular velocity in parent frame
            - For each end effector:
                - Position in local space
        """
        self.model.realizeVelocity(self.state)

        body_set = self.model.get_BodySet()

        pelvis = body_set.get("pelvis")
        pelvis_transform = pelvis.getTransformInGround(self.state)

        # --- Compute pelvis position ---
        pelvis_pos_glob = pelvis_transform.p()
        pelvis_obs_pos = [pelvis_pos_glob[i] for i in range(3)]

        # --- Compute orientation representation ---
        # First transform the forward vector to global space
        pelvis_forward_glob = pelvis_transform.xformFrameVecToBase(Vec3(1, 0, 0))

        # then compute angle to (1, 0, 0) and normalize from -pi to pi
        plevis_y_orient = atan2(pelvis_forward_glob[2], pelvis_forward_glob[0])
        if plevis_y_orient > pi:
            plevis_y_orient -= 2 * pi
        elif plevis_y_orient <= -pi:
            plevis_y_orient += 2 * pi

        # Next we transform the up vector to global space and rotate it so it points forward.
        # This only works, becuase the model should always face upwards.
        pelvis_up_glob = pelvis_transform.xformFrameVecToBase(Vec3(0, 1, 0))

        # rotate around y axis
        pyo_cos = cos(plevis_y_orient)
        pyo_sin = sin(plevis_y_orient)
        pug_x = pelvis_up_glob[0] * pyo_cos - pelvis_up_glob[2] * pyo_sin
        pug_y = pelvis_up_glob[1]
        pug_z = pelvis_up_glob[0] * pyo_sin + pelvis_up_glob[2] * pyo_cos

        plevis_x_orient = atan2(pug_z, pug_y)
        if plevis_x_orient > pi:
            plevis_x_orient -= 2 * pi
        elif plevis_x_orient <= -pi:
            plevis_x_orient += 2 * pi

        plevis_z_orient = atan2(pug_x, pug_y)
        if plevis_z_orient > pi:
            plevis_z_orient -= 2 * pi
        elif plevis_z_orient <= -pi:
            plevis_z_orient += 2 * pi

        pelvis_obs_orient = [plevis_x_orient, plevis_y_orient, plevis_z_orient]

        # --- Compute linear velocity representation ---
        pelvis_lin_vel_glob = pelvis.getLinearVelocityInGround(self.state)
        pelvis_lin_vel_loca = pelvis_transform.xformBaseVecToFrame(pelvis_lin_vel_glob)

        pelvis_obs_lin_vel = [pelvis_lin_vel_loca[i] for i in range(3)]

        # --- Compute angular velocity representation ---
        pelvis_ang_vel_glob = pelvis.getAngularVelocityInGround(self.state)
        pelvis_ang_vel_loca = pelvis_transform.xformBaseVecToFrame(pelvis_ang_vel_glob)

        # Compute how local angular velocity affects these two points, to extract more usefull angular velocity. (avl = angular velocity local)
        pelvis_avl_forward = cross(pelvis_ang_vel_loca, Vec3(0.3, 0, 0))
        pelvis_avl_up = cross(pelvis_ang_vel_loca, Vec3(0, 0.3, 0))

        # Local rotation around the [x, y, z] axis. See the function docstring for details
        pelvis_obs_ang_vel = [
            pelvis_avl_up[2],
            pelvis_avl_forward[2],
            pelvis_avl_forward[1],
        ]
        # print(pelvis_obs_ang_vel)

        # Add joint coordinates to observation
        signifficant_coords = [
            "ankle_angle_r",
            "hip_adduction_l",
            "hip_adduction_r",
            "hip_flexion_l",
            "hip_flexion_r",
            "hip_rotation_l",
            "hip_rotation_r",
            "knee_angle_r",
            "pros_ankle_angle",
            "pros_knee_angle",
        ]

        coordinate_obs = []
        coordinate_set = self.model.getCoordinateSet()
        for coord in signifficant_coords:
            coord = coordinate_set.get(coord)
            coordinate_obs.append(coord.getValue(self.state))
            coordinate_obs.append(coord.getSpeedValue(self.state))

        # Add body part locations to observation
        signifficant_bodys = [
            "femur_r",
            "tibia_r",
            "calcn_r",
            "toes_r",
            "transfemur",
            "osseo_pylon",
            "tibia_pylon",
            "foot",
            "torso",
            "head",
        ]
        body_obs = []
        for body in signifficant_bodys:
            body = body_set.get(body)
            pos_global = body.getTransformInGround(self.state).p()
            vel_global = body.getLinearVelocityInGround(self.state)
            pos_local = pelvis_transform.shiftBaseStationToFrame(pos_global)
            vel_local = pelvis_transform.xformBaseVecToFrame(vel_global)
            body_obs += [pos_local[i] for i in range(3)]
            body_obs += [vel_local[i] for i in range(3)]

        time = self.current_step * self.delta_time
        # Scale the observations, this ensures that they stay in the given range.
        data_obs = self.data.get_row(time).as_array()

        # print(pelvis_obs_pos, pelvis_obs_orient, pelvis_obs_lin_vel, pelvis_obs_ang_vel, actuator_obs, body_obs)

        # return pelvis_obs_pos + pelvis_obs_orient + pelvis_obs_lin_vel + pelvis_obs_ang_vel + actuator_obs + body_obs

        observation = np.array(
            pelvis_obs_pos  # 0 + 3
            + pelvis_obs_orient  # 3 + 3
            + pelvis_obs_lin_vel  # 6 + 3
            + pelvis_obs_ang_vel  # 9 + 3
            + coordinate_obs  # 12 + 20
            + body_obs  # 32 + 60
            + data_obs  # 92 + 32 + 124
        )

        # Check observations
        assert (
            observation.shape == self.observation_space.shape
        ), "Invalid observation space: " + str(observation.shape)
        illegal = np.logical_not(np.isfinite(observation))
        if np.any(illegal):
            print(
                "Illegal value(s) in observation:",
                np.where(illegal),
                observation[illegal],
            )
            observation[illegal] = 0.0

        return observation

    def _get_joint_imitation_reward(self, imi_data: DataRow) -> float:
        imi_pos_penalty = 0
        imi_vel_penalty = 0

        ignore_coords = frozenset(["pelvis_tx", "pelvis_ty", "pelvis_tz"])

        for coord in self.model.getCoordinateSet():
            name = JOINT_NAME_TRANSLATION[coord.getName()]

            if coord.getLocked(self.state) or name in ignore_coords:
                continue

            pos = getattr(imi_data, name)
            vel = getattr(imi_data, "d_" + name)

            imi_pos_penalty += (coord.getValue(self.state) - pos) ** 2 / 10
            imi_vel_penalty += (coord.getSpeedValue(self.state) - vel) ** 2 / 10

        imi_pos_reward = soft_clip_penalty_to_reward(imi_pos_penalty, 1, 1)
        imi_vel_reward = soft_clip_penalty_to_reward(imi_vel_penalty, 1, 1)

        self.step_info["reward_imi_pos"] = imi_pos_reward
        self.step_info["reward_imi_vel"] = imi_vel_reward

        return imi_pos_reward + imi_vel_reward

    def get_reward(self):
        """Returns the reward at the current state."""

        time = self.current_step * self.delta_time
        imi_data = self.data.get_row(time)

        reward = 0

        # Imitation reward, for the joints
        reward += self._get_joint_imitation_reward(imi_data)

        # Positional reward (pelvis ie. root frame)
        body_set = self.model.get_BodySet()
        pelvis = body_set.get("pelvis")
        pelvis_transform = pelvis.getTransformInGround(self.state)
        pelvis_pos = pelvis_transform.p()

        pos_penalty = (
            sqrt(
                (pelvis_pos[0] - imi_data.pelvis_tx) ** 2
                + (pelvis_pos[2] - imi_data.pelvis_tz) ** 2
            )
            * 10
        )
        pos_reward = soft_clip_penalty_to_reward(pos_penalty, 1, 1)

        reward += pos_reward
        self.step_info["reward_pos"] = pos_reward

        # Velocity reward (pelvis ie. root_frame)
        if self.last_pelvis_pos is None:
            self.last_pelvis_pos = Vec3(pelvis_pos)

        pelvis_vel_x = (pelvis_pos[0] - self.last_pelvis_pos[0]) / self.delta_time
        pelvis_vel_z = (pelvis_pos[2] - self.last_pelvis_pos[2]) / self.delta_time
        pelvis_vel = sqrt(pelvis_vel_x ** 2 + pelvis_vel_z ** 2)

        self.last_pelvis_pos = Vec3(pelvis_pos)

        self.pelvis_vel_avg *= 1 - PELVIS_VEL_RUNNING_AVG_FACTOR
        self.pelvis_vel_avg += pelvis_vel * PELVIS_VEL_RUNNING_AVG_FACTOR

        vel_penalty = np.abs(self.pelvis_vel_avg - TARGET_PELVIS_VEL) * 2
        vel_reward = soft_clip_penalty_to_reward(vel_penalty, 1, 1)

        reward += vel_reward
        self.step_info["reward_vel"] = vel_reward

        if pelvis_pos[1] < 0.8:
            reward = reward / (1 + (pelvis_pos[1] * 10 - 8) ** 2)

        return reward

    def is_done(self):
        """Returns true, if this episode is over."""
        body_set = self.model.get_BodySet()

        pelvis = body_set.get("pelvis")
        pelvis_transform = pelvis.getTransformInGround(self.state)
        pelvis_position = pelvis_transform.p()
        return pelvis_position[1] < 0.6

    def set_pose(self, pose: DataRow):
        """Called during reset, set the pose of the model from a DataRow."""
        for coordinate in self.model.getCoordinateSet():
            if coordinate.get_locked():
                continue
            name = JOINT_NAME_TRANSLATION[coordinate.getName()]

            coordinate.setValue(self.state, getattr(pose, name))
            coordinate.setSpeedValue(self.state, getattr(pose, "d_" + name))

    def get_reset_pose(self):
        """Implement this to reset to a specific pose"""
        return self.data.get_row(self.current_step * self.delta_time)

    def state_as_row(self):
        pose = DataRow()
        for coord in self.model.getCoordinateSet():
            name = JOINT_NAME_TRANSLATION[coord.getName()]
            setattr(pose, name, coord.getValue(self.state))
            setattr(pose, "d_" + name, coord.getSpeedValue(self.state))
        return pose

    def reset(self):
        self.last_pelvis_pos = None
        return super().reset()

    def get_info(self):
        """return additional info for every step."""

        body_set = self.model.get_BodySet()
        pelvis = body_set.get("pelvis")
        pelvis_transform = pelvis.getTransformInGround(self.state)
        ppos = pelvis_transform.p()

        self.step_info["dist_from_origin"] = sqrt(ppos[0] ** 2 + ppos[2] ** 2)

        # This will simply return the step_info dictionary
        return super().get_info()
