"""Position task base configuration."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from . import mdp


def _position_error_obs_scale(value_range: tuple[float, float]) -> float:
    return 1.0 / max(abs(value_range[0]), abs(value_range[1]), 0.25)


def make_position_env_cfg(
    *,
    robot_base_env_cfg: ManagerBasedRlEnvCfg,
    command_pos_x_range_m: tuple[float, float],
    command_pos_y_range_m: tuple[float, float],
    command_pos_z_range_m: tuple[float, float],
    command_roll_range_rad: tuple[float, float],
    command_pitch_range_rad: tuple[float, float],
    command_yaw_range_rad: tuple[float, float],
    command_resampling_time_s: tuple[float, float],
    reset_pos_x_range_m: tuple[float, float],
    reset_pos_y_range_m: tuple[float, float],
    reset_pos_z_range_m: tuple[float, float],
    reset_roll_range_rad: tuple[float, float],
    reset_pitch_range_rad: tuple[float, float],
    reset_yaw_range_rad: tuple[float, float],
    reset_lin_vel_x_range_m_s: tuple[float, float],
    reset_lin_vel_y_range_m_s: tuple[float, float],
    reset_lin_vel_z_range_m_s: tuple[float, float],
    reset_ang_vel_x_range_rad_s: tuple[float, float],
    reset_ang_vel_y_range_rad_s: tuple[float, float],
    reset_ang_vel_z_range_rad_s: tuple[float, float],
) -> ManagerBasedRlEnvCfg:
    cfg = robot_base_env_cfg

    position_error_scale = (
        _position_error_obs_scale(command_pos_x_range_m),
        _position_error_obs_scale(command_pos_y_range_m),
        _position_error_obs_scale(command_pos_z_range_m),
    )
    orientation_error_scale = (1.0 / math.pi, 1.0 / math.pi, 1.0 / math.pi)

    cfg.commands = {
        "pose": mdp.UniformPoseCommandCfg(
            entity_name="robot",
            resampling_time_range=command_resampling_time_s,
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=command_pos_x_range_m,
                pos_y=command_pos_y_range_m,
                pos_z=command_pos_z_range_m,
                roll=command_roll_range_rad,
                pitch=command_pitch_range_rad,
                yaw=command_yaw_range_rad,
            ),
        )
    }

    actor_terms = {
        "position_error_b": ObservationTermCfg(
            func=mdp.body_position_error_b,
            params={"command_name": "pose"},
            scale=position_error_scale,
        ),
        "orientation_error": ObservationTermCfg(
            func=mdp.orientation_error,
            params={"command_name": "pose"},
            scale=orientation_error_scale,
        ),
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "applied_body_wrench": ObservationTermCfg(
            func=mdp.applied_body_wrench,
            params={"action_name": "body_wrench"},
        ),
    }

    critic_terms = {
        **actor_terms,
        "current_position_rel": ObservationTermCfg(func=mdp.current_position_rel),
        "current_orientation_quat": ObservationTermCfg(
            func=mdp.current_orientation_quat,
        ),
        "desired_position_rel": ObservationTermCfg(
            func=mdp.desired_position_rel,
            params={"command_name": "pose"},
        ),
        "desired_orientation_quat": ObservationTermCfg(
            func=mdp.desired_orientation_quat,
            params={"command_name": "pose"},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.events["reset_root_state_uniform"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": reset_pos_x_range_m,
                "y": reset_pos_y_range_m,
                "z": reset_pos_z_range_m,
                "roll": reset_roll_range_rad,
                "pitch": reset_pitch_range_rad,
                "yaw": reset_yaw_range_rad,
            },
            "velocity_range": {
                "x": reset_lin_vel_x_range_m_s,
                "y": reset_lin_vel_y_range_m_s,
                "z": reset_lin_vel_z_range_m_s,
                "roll": reset_ang_vel_x_range_rad_s,
                "pitch": reset_ang_vel_y_range_rad_s,
                "yaw": reset_ang_vel_z_range_rad_s,
            },
        },
    )

    cfg.rewards = {
        "track_position": RewardTermCfg(
            func=mdp.track_position,
            weight=4.0,
            params={"command_name": "pose", "std": 0.60},
        ),
        "track_orientation": RewardTermCfg(
            func=mdp.track_orientation,
            weight=1.5,
            params={"command_name": "pose", "std": 0.75},
        ),
        "linear_velocity_l2": RewardTermCfg(
            func=mdp.linear_velocity_l2,
            weight=-5.0e-2,
        ),
        "angular_velocity_l2": RewardTermCfg(
            func=mdp.angular_velocity_l2,
            weight=-5.0e-2,
        ),
    }

    return cfg
