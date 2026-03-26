from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_box_minus,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
)


def _validate_range(name: str, value: tuple[float, float]) -> None:
    if float(value[1]) < float(value[0]):
        raise ValueError(f"{name} must satisfy lower <= upper, got {value}.")


class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env) -> None:
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)

        self.metrics["error_position"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_orientation"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command

    def _update_metrics(self) -> None:
        max_command_time = max(
            float(self.cfg.resampling_time_range[1]),
            self._env.step_dt,
        )
        max_command_steps = max(max_command_time / max(self._env.step_dt, 1.0e-6), 1.0)

        current_pos_rel = self.robot.data.root_link_pos_w - self._env.scene.env_origins
        current_quat = quat_unique(self.robot.data.root_link_quat_w)

        position_error = self.pose_command[:, :3] - current_pos_rel
        orientation_error = quat_box_minus(
            quat_unique(self.pose_command[:, 3:7]),
            current_quat,
        )

        self.metrics["error_position"] += (
            torch.linalg.norm(
                position_error,
                dim=1,
            )
            / max_command_steps
        )
        self.metrics["error_orientation"] += (
            torch.linalg.norm(
                orientation_error,
                dim=1,
            )
            / max_command_steps
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        sample = torch.empty(len(env_ids), device=self.device)
        ranges = self.cfg.ranges
        default_root_state = self.robot.data.default_root_state[env_ids]

        self.pose_command[env_ids, 0] = default_root_state[:, 0] + sample.uniform_(
            *ranges.pos_x
        )
        self.pose_command[env_ids, 1] = default_root_state[:, 1] + sample.uniform_(
            *ranges.pos_y
        )
        self.pose_command[env_ids, 2] = default_root_state[:, 2] + sample.uniform_(
            *ranges.pos_z
        )

        roll = sample.uniform_(*ranges.roll).clone()
        pitch = sample.uniform_(*ranges.pitch).clone()
        yaw = sample.uniform_(*ranges.yaw).clone()
        delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
        desired_quat = quat_mul(default_root_state[:, 3:7], delta_quat)
        self.pose_command[env_ids, 3:7] = quat_unique(desired_quat)

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer) -> None:
        del visualizer


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    entity_name: str

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    ranges: Ranges
    debug_vis: bool = False

    def build(self, env) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)

    def __post_init__(self) -> None:
        _validate_range("resampling_time_range", self.resampling_time_range)
        _validate_range("ranges.pos_x", self.ranges.pos_x)
        _validate_range("ranges.pos_y", self.ranges.pos_y)
        _validate_range("ranges.pos_z", self.ranges.pos_z)
        _validate_range("ranges.roll", self.ranges.roll)
        _validate_range("ranges.pitch", self.ranges.pitch)
        _validate_range("ranges.yaw", self.ranges.yaw)
