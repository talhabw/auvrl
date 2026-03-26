from __future__ import annotations

from typing import TypedDict, cast

import torch

from auvrl.tasks.position.mdp.pose_command import UniformPoseCommandCfg


class PositionCurriculumStage(TypedDict, total=False):
    step: int
    command_position_scale: float
    command_orientation_scale: float
    reset_pose_scale: float
    reset_velocity_scale: float


def _scaled_range(value: tuple[float, float], scale: float) -> tuple[float, float]:
    return (float(value[0]) * scale, float(value[1]) * scale)


def command_and_reset_pose_ranges(
    env,
    env_ids: torch.Tensor | slice | None,
    command_name: str,
    reset_event_name: str,
    stages: list[PositionCurriculumStage],
) -> dict[str, torch.Tensor]:
    del env_ids

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    command_cfg = cast(UniformPoseCommandCfg, command_term.cfg)
    command_ranges = command_cfg.ranges

    reset_event_cfg = env.event_manager.get_term_cfg(reset_event_name)
    reset_pose_range = reset_event_cfg.params["pose_range"]
    reset_velocity_range = reset_event_cfg.params["velocity_range"]

    base_command_ranges = getattr(command_cfg, "base_ranges")
    base_reset_pose_range = getattr(reset_event_cfg, "base_pose_range")
    base_reset_velocity_range = getattr(reset_event_cfg, "base_velocity_range")

    stage_index = 0
    active_stage = stages[0]
    for idx, stage in enumerate(stages):
        if env.common_step_counter >= stage["step"]:
            stage_index = idx
            active_stage = stage

    command_position_scale = float(active_stage.get("command_position_scale", 1.0))
    command_orientation_scale = float(
        active_stage.get("command_orientation_scale", 1.0)
    )
    reset_pose_scale = float(active_stage.get("reset_pose_scale", 1.0))
    reset_velocity_scale = float(active_stage.get("reset_velocity_scale", 1.0))

    command_ranges.pos_x = _scaled_range(
        base_command_ranges.pos_x,
        command_position_scale,
    )
    command_ranges.pos_y = _scaled_range(
        base_command_ranges.pos_y,
        command_position_scale,
    )
    command_ranges.pos_z = _scaled_range(
        base_command_ranges.pos_z,
        command_position_scale,
    )
    command_ranges.roll = _scaled_range(
        base_command_ranges.roll,
        command_orientation_scale,
    )
    command_ranges.pitch = _scaled_range(
        base_command_ranges.pitch,
        command_orientation_scale,
    )
    command_ranges.yaw = _scaled_range(
        base_command_ranges.yaw,
        command_orientation_scale,
    )

    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        reset_pose_range[key] = _scaled_range(
            base_reset_pose_range[key], reset_pose_scale
        )
        reset_velocity_range[key] = _scaled_range(
            base_reset_velocity_range[key],
            reset_velocity_scale,
        )

    return {
        "stage_index": torch.tensor(stage_index, dtype=torch.float32),
        "command_position_scale": torch.tensor(command_position_scale),
        "command_orientation_scale": torch.tensor(command_orientation_scale),
        "reset_pose_scale": torch.tensor(reset_pose_scale),
        "reset_velocity_scale": torch.tensor(reset_velocity_scale),
        "command_pos_x_min": torch.tensor(command_ranges.pos_x[0]),
        "command_pos_x_max": torch.tensor(command_ranges.pos_x[1]),
        "command_pos_y_min": torch.tensor(command_ranges.pos_y[0]),
        "command_pos_y_max": torch.tensor(command_ranges.pos_y[1]),
        "command_pos_z_min": torch.tensor(command_ranges.pos_z[0]),
        "command_pos_z_max": torch.tensor(command_ranges.pos_z[1]),
        "command_yaw_min": torch.tensor(command_ranges.yaw[0]),
        "command_yaw_max": torch.tensor(command_ranges.yaw[1]),
        "reset_pos_x_min": torch.tensor(reset_pose_range["x"][0]),
        "reset_pos_x_max": torch.tensor(reset_pose_range["x"][1]),
        "reset_pos_z_min": torch.tensor(reset_pose_range["z"][0]),
        "reset_pos_z_max": torch.tensor(reset_pose_range["z"][1]),
        "reset_yaw_min": torch.tensor(reset_pose_range["yaw"][0]),
        "reset_yaw_max": torch.tensor(reset_pose_range["yaw"][1]),
    }


__all__ = ["PositionCurriculumStage", "command_and_reset_pose_ranges"]
