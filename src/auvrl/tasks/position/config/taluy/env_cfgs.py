"""Taluy position environment configurations."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from auvrl.envs.taluy_env_cfg import EventMode, make_taluy_base_env_cfg
from auvrl.tasks.position import mdp
from auvrl.tasks.position.position_env_cfg import make_position_env_cfg


def _current_obs_scale(
    current_speed_range_m_s: tuple[float, float],
    vertical_current_range_m_s: tuple[float, float],
) -> tuple[float, float, float]:
    horizontal_extent = max(
        abs(current_speed_range_m_s[0]),
        abs(current_speed_range_m_s[1]),
        0.25,
    )
    vertical_extent = max(
        abs(vertical_current_range_m_s[0]),
        abs(vertical_current_range_m_s[1]),
        0.25,
    )
    return (
        1.0 / horizontal_extent,
        1.0 / horizontal_extent,
        1.0 / vertical_extent,
    )


def _default_position_curriculum_stages() -> list[mdp.PositionCurriculumStage]:
    return [
        {
            "step": 0,
            "command_position_scale": 0.40,
            "command_orientation_scale": 0.35,
            "reset_pose_scale": 0.35,
            "reset_velocity_scale": 0.35,
        },
        {
            "step": 4_000,
            "command_position_scale": 0.60,
            "command_orientation_scale": 0.55,
            "reset_pose_scale": 0.55,
            "reset_velocity_scale": 0.55,
        },
        {
            "step": 8_000,
            "command_position_scale": 0.80,
            "command_orientation_scale": 0.75,
            "reset_pose_scale": 0.75,
            "reset_velocity_scale": 0.75,
        },
        {
            "step": 12_000,
            "command_position_scale": 1.00,
            "command_orientation_scale": 1.00,
            "reset_pose_scale": 1.00,
            "reset_velocity_scale": 1.00,
        },
    ]


def make_taluy_position_env_cfg(
    *,
    num_envs: int = 1,
    episode_length_s: float = 30.0,
    command_resampling_time_s: tuple[float, float] = (8.0, 14.0),
    command_pos_x_range_m: tuple[float, float] = (-0.75, 0.75),
    command_pos_y_range_m: tuple[float, float] = (-0.75, 0.75),
    command_pos_z_range_m: tuple[float, float] = (-0.50, 0.50),
    command_roll_range_rad: tuple[float, float] = (0.0, 0.0),
    command_pitch_range_rad: tuple[float, float] = (0.0, 0.0),
    command_yaw_range_rad: tuple[float, float] = (-2.0, 2.0),
    reset_pos_x_range_m: tuple[float, float] = (-0.30, 0.30),
    reset_pos_y_range_m: tuple[float, float] = (-0.30, 0.30),
    reset_pos_z_range_m: tuple[float, float] = (-0.20, 0.20),
    reset_roll_range_rad: tuple[float, float] = (-0.10, 0.10),
    reset_pitch_range_rad: tuple[float, float] = (-0.10, 0.10),
    reset_yaw_range_rad: tuple[float, float] = (-0.40, 0.40),
    reset_lin_vel_x_range_m_s: tuple[float, float] = (-0.10, 0.10),
    reset_lin_vel_y_range_m_s: tuple[float, float] = (-0.10, 0.10),
    reset_lin_vel_z_range_m_s: tuple[float, float] = (-0.10, 0.10),
    reset_ang_vel_x_range_rad_s: tuple[float, float] = (-0.20, 0.20),
    reset_ang_vel_y_range_rad_s: tuple[float, float] = (-0.20, 0.20),
    reset_ang_vel_z_range_rad_s: tuple[float, float] = (-0.20, 0.20),
    curriculum_enabled: bool = True,
    curriculum_stages: list[mdp.PositionCurriculumStage] | None = None,
    thruster_voltage_event_mode: EventMode = "disabled",
    thruster_voltage_range_v: tuple[float, float] = (16.0, 16.0),
    current_event_mode: EventMode = "disabled",
    current_speed_range_m_s: tuple[float, float] = (0.0, 0.0),
    current_yaw_range_rad: tuple[float, float] = (0.0, 0.0),
    vertical_current_range_m_s: tuple[float, float] = (0.0, 0.0),
) -> ManagerBasedRlEnvCfg:
    robot_base_env_cfg = make_taluy_base_env_cfg(
        action_space="body_wrench",
        thruster_voltage_event_mode=thruster_voltage_event_mode,
        thruster_voltage_range_v=thruster_voltage_range_v,
        current_event_mode=current_event_mode,
        current_speed_range_m_s=current_speed_range_m_s,
        current_yaw_range_rad=current_yaw_range_rad,
        vertical_current_range_m_s=vertical_current_range_m_s,
    )

    cfg = make_position_env_cfg(
        robot_base_env_cfg=robot_base_env_cfg,
        command_pos_x_range_m=command_pos_x_range_m,
        command_pos_y_range_m=command_pos_y_range_m,
        command_pos_z_range_m=command_pos_z_range_m,
        command_roll_range_rad=command_roll_range_rad,
        command_pitch_range_rad=command_pitch_range_rad,
        command_yaw_range_rad=command_yaw_range_rad,
        command_resampling_time_s=command_resampling_time_s,
        reset_pos_x_range_m=reset_pos_x_range_m,
        reset_pos_y_range_m=reset_pos_y_range_m,
        reset_pos_z_range_m=reset_pos_z_range_m,
        reset_roll_range_rad=reset_roll_range_rad,
        reset_pitch_range_rad=reset_pitch_range_rad,
        reset_yaw_range_rad=reset_yaw_range_rad,
        reset_lin_vel_x_range_m_s=reset_lin_vel_x_range_m_s,
        reset_lin_vel_y_range_m_s=reset_lin_vel_y_range_m_s,
        reset_lin_vel_z_range_m_s=reset_lin_vel_z_range_m_s,
        reset_ang_vel_x_range_rad_s=reset_ang_vel_x_range_rad_s,
        reset_ang_vel_y_range_rad_s=reset_ang_vel_y_range_rad_s,
        reset_ang_vel_z_range_rad_s=reset_ang_vel_z_range_rad_s,
    )

    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_length_s

    if curriculum_enabled:
        if curriculum_stages is None:
            curriculum_stages = _default_position_curriculum_stages()
        if len(curriculum_stages) == 0:
            raise ValueError("curriculum_stages must not be empty when enabled.")
        pose_command_cfg = cfg.commands["pose"]
        pose_command_cfg.base_ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=tuple(pose_command_cfg.ranges.pos_x),
            pos_y=tuple(pose_command_cfg.ranges.pos_y),
            pos_z=tuple(pose_command_cfg.ranges.pos_z),
            roll=tuple(pose_command_cfg.ranges.roll),
            pitch=tuple(pose_command_cfg.ranges.pitch),
            yaw=tuple(pose_command_cfg.ranges.yaw),
        )
        reset_cfg = cfg.events["reset_root_state_uniform"]
        reset_cfg.base_pose_range = {
            key: tuple(value) for key, value in reset_cfg.params["pose_range"].items()
        }
        reset_cfg.base_velocity_range = {
            key: tuple(value)
            for key, value in reset_cfg.params["velocity_range"].items()
        }
        cfg.curriculum = {
            "command_and_reset_pose_ranges": CurriculumTermCfg(
                func=mdp.command_and_reset_pose_ranges,
                params={
                    "command_name": "pose",
                    "reset_event_name": "reset_root_state_uniform",
                    "stages": curriculum_stages,
                },
            )
        }

    if current_event_mode != "disabled":
        cfg.observations["critic"].terms["current_velocity_b"] = ObservationTermCfg(
            func=mdp.current_velocity_b,
            scale=_current_obs_scale(
                current_speed_range_m_s,
                vertical_current_range_m_s,
            ),
        )

    if thruster_voltage_event_mode != "disabled":
        cfg.observations["critic"].terms["thruster_voltage_offset"] = (
            ObservationTermCfg(
                func=mdp.thruster_voltage_offset,
            )
        )

    cfg.rewards["action_l2"] = RewardTermCfg(
        func=mdp.body_wrench_action_l2,
        weight=-4.0e-3,
        params={"action_name": "body_wrench"},
    )
    cfg.rewards["settle_linear_velocity"] = RewardTermCfg(
        func=mdp.settle_linear_velocity_near_goal,
        weight=-1.5e-1,
        params={
            "command_name": "pose",
            "position_std": 0.25,
            "orientation_std": 0.30,
        },
    )
    cfg.rewards["settle_angular_velocity"] = RewardTermCfg(
        func=mdp.settle_angular_velocity_near_goal,
        weight=-2.5e-1,
        params={
            "command_name": "pose",
            "position_std": 0.25,
            "orientation_std": 0.30,
        },
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.body_wrench_action_rate_l2,
        weight=-2.0e-2,
        params={"action_name": "body_wrench"},
    )
    cfg.rewards["settle_action_rate"] = RewardTermCfg(
        func=mdp.settle_action_rate_near_goal,
        weight=-1.5e-1,
        params={
            "command_name": "pose",
            "position_std": 0.25,
            "orientation_std": 0.30,
            "action_name": "body_wrench",
        },
    )
    cfg.rewards["saturation"] = RewardTermCfg(
        func=mdp.body_wrench_saturation_penalty,
        weight=-1.0e-1,
        params={"action_name": "body_wrench"},
    )

    return cfg
