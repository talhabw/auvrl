from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from auvrl.actuator.body_wrench_action import BodyWrenchAction
from auvrl.actuator.thruster_actuator import THRUSTER_LOCAL_AXIS
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_box_minus,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _validate_range(name: str, value: tuple[float, float]) -> None:
    if float(value[1]) < float(value[0]):
        raise ValueError(f"{name} must satisfy lower <= upper, got {value}.")


class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)

        self.metrics["error_position"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_orientation"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["error_position_xy"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["error_position_z"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["error_yaw"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["linear_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_l2"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_rate_l2"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["saturation_fraction"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_get_env_idx: Callable[[], int] | None = None
        self._ghost_model: mujoco.MjModel | None = None

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
        desired_roll, desired_pitch, desired_yaw = euler_xyz_from_quat(
            quat_unique(self.pose_command[:, 3:7]),
        )
        current_roll, current_pitch, current_yaw = euler_xyz_from_quat(current_quat)
        del desired_roll, desired_pitch, current_roll, current_pitch
        yaw_error = torch.atan2(
            torch.sin(desired_yaw - current_yaw),
            torch.cos(desired_yaw - current_yaw),
        )
        lin_vel_b = self.robot.data.root_link_lin_vel_b
        ang_vel_b = self.robot.data.root_link_ang_vel_b

        try:
            wrench_term = self._env.action_manager.get_term("body_wrench")
        except KeyError:
            wrench_term = None
        if isinstance(wrench_term, BodyWrenchAction):
            action_l2 = torch.mean(torch.square(wrench_term.raw_action), dim=1)
            action_rate_l2 = torch.mean(
                torch.square(
                    self._env.action_manager.action
                    - self._env.action_manager.prev_action
                ),
                dim=1,
            )
            saturation_fraction = wrench_term.step_saturation_fraction
        else:
            zeros = torch.zeros(self.num_envs, device=self.device)
            action_l2 = zeros
            action_rate_l2 = zeros
            saturation_fraction = zeros

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
        self.metrics["error_position_xy"] += (
            torch.linalg.norm(position_error[:, :2], dim=1) / max_command_steps
        )
        self.metrics["error_position_z"] += (
            torch.abs(position_error[:, 2]) / max_command_steps
        )
        self.metrics["error_yaw"] += torch.abs(yaw_error) / max_command_steps
        self.metrics["linear_speed"] += (
            torch.linalg.norm(lin_vel_b, dim=1) / max_command_steps
        )
        self.metrics["angular_speed"] += (
            torch.linalg.norm(ang_vel_b, dim=1) / max_command_steps
        )
        self.metrics["action_l2"] += action_l2 / max_command_steps
        self.metrics["action_rate_l2"] += action_rate_l2 / max_command_steps
        self.metrics["saturation_fraction"] += saturation_fraction / max_command_steps

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

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
    ) -> None:
        from viser import Icon

        ranges = self.cfg.ranges
        axes: list[tuple[str, tuple[float, float], float, float]] = [
            ("pos_x (m)", ranges.pos_x, 0.01, 5.0),
            ("pos_y (m)", ranges.pos_y, 0.01, 5.0),
            ("pos_z (m)", ranges.pos_z, 0.01, 5.0),
            ("roll (rad)", ranges.roll, 0.01, 3.15),
            ("pitch (rad)", ranges.pitch, 0.01, 3.15),
            ("yaw (rad)", ranges.yaw, 0.05, 6.3),
        ]
        sliders: list[viser.GuiSliderHandle] = []

        with server.gui.add_folder(name.replace("_", " ").title()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for label, axis_range, step, max_limit in axes:
                max_mag = max(
                    abs(float(axis_range[0])), abs(float(axis_range[1])), step
                )
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_mag,
                    step=step,
                    min=step,
                    max=max_limit,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_mag,
                    max=max_mag,
                    step=step,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _(_ev, _slider=slider, _max_input=max_input) -> None:
                    _slider.min = -_max_input.value
                    _slider.max = _max_input.value

                sliders.append(slider)

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_) -> None:
                for s in sliders:
                    s.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            env_idx = self._joystick_get_env_idx()
            default_state = self.robot.data.default_root_state[env_idx]
            # Position: offset from default
            for i in range(3):
                self.pose_command[env_idx, i] = (
                    default_state[i] + self._joystick_sliders[i].value
                )
            # Orientation: euler offset applied to default quaternion
            roll = float(self._joystick_sliders[3].value)
            pitch = float(self._joystick_sliders[4].value)
            yaw = float(self._joystick_sliders[5].value)
            delta_quat = quat_from_euler_xyz(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device),
                torch.tensor([yaw], device=self.device),
            )
            desired_quat = quat_mul(default_state[3:7].unsqueeze(0), delta_quat)
            self.pose_command[env_idx, 3:7] = quat_unique(desired_quat).squeeze(0)
            self.time_left[env_idx] = 1.0e9

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        commands = self.command.cpu().numpy()
        env_origins_ws = self._env.scene.env_origins.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
        desired_mat_ws = (
            matrix_from_quat(quat_unique(self.pose_command[:, 3:7])).cpu().numpy()
        )

        viz = self.cfg.viz

        # Thruster data (same pattern as velocity command)
        try:
            wrench_term = self._env.action_manager.get_term("body_wrench")
        except KeyError:
            wrench_term = None
        thruster_site_pos_ws: np.ndarray | None = None
        thruster_force_axis_ws: np.ndarray | None = None
        thruster_targets: np.ndarray | None = None
        if isinstance(wrench_term, BodyWrenchAction):
            site_ids = wrench_term.site_ids
            num_sites = int(site_ids.numel())
            thruster_site_pos_ws = (
                self.robot.data.data.site_xpos[:, site_ids].detach().cpu().numpy()
            )
            thruster_site_mat_ws = (
                self.robot.data.data.site_xmat[:, site_ids]
                .reshape(self.num_envs, num_sites, 3, 3)
                .detach()
                .cpu()
                .numpy()
            )
            thruster_force_axis_ws = thruster_site_mat_ws @ np.asarray(
                THRUSTER_LOCAL_AXIS, dtype=float
            )
            thruster_targets = wrench_term.thruster_targets.detach().cpu().numpy()

        for env_idx in env_indices:
            base_pos_w = base_pos_ws[env_idx]
            base_mat_w = base_mat_ws[env_idx]
            command = commands[env_idx]
            desired_pos = command[:3] + env_origins_ws[env_idx]
            desired_mat_w = desired_mat_ws[env_idx]

            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_contype[:] = 0
                self._ghost_model.geom_conaffinity[:] = 0

            # Ghost mesh at desired pose
            qpos = self.robot.data.data.qpos[env_idx].detach().cpu().numpy().copy()
            qpos[:3] = desired_pos
            qpos[3:7] = command[3:7]
            visualizer.add_ghost_mesh(
                qpos=qpos,
                model=self._ghost_model,
                alpha=viz.ghost_alpha,
            )

            # Coordinate frame at desired pose
            visualizer.add_frame(
                position=desired_pos,
                rotation_matrix=desired_mat_w,
                scale=viz.frame_scale,
                axis_radius=viz.frame_axis_radius,
                alpha=0.7,
            )

            # Coordinate frame at current pose (dimmer)
            visualizer.add_frame(
                position=base_pos_w,
                rotation_matrix=base_mat_w,
                scale=viz.frame_scale * 0.7,
                axis_radius=viz.frame_axis_radius * 0.7,
                alpha=0.35,
            )

            # Position error arrow: current -> desired
            pos_error = desired_pos - base_pos_w
            error_norm = float(np.linalg.norm(pos_error))
            if error_norm > viz.error_arrow_zero_threshold:
                visualizer.add_arrow(
                    start=base_pos_w,
                    end=desired_pos,
                    color=viz.error_arrow_color,
                    width=viz.error_arrow_width,
                )
            else:
                visualizer.add_sphere(
                    center=base_pos_w,
                    radius=viz.converged_marker_radius,
                    color=(0.1, 0.9, 0.3, 0.7),
                )

            # Thruster force arrows
            if (
                thruster_site_pos_ws is None
                or thruster_force_axis_ws is None
                or thruster_targets is None
            ):
                continue

            for thruster_idx, force_n in enumerate(thruster_targets[env_idx]):
                force_n = float(force_n)
                if abs(force_n) < viz.thruster_force_zero_threshold_n:
                    continue
                start = thruster_site_pos_ws[env_idx, thruster_idx]
                end = start + (
                    thruster_force_axis_ws[env_idx, thruster_idx]
                    * (force_n * viz.thruster_force_scale)
                )
                color = (
                    (1.0, 0.65, 0.1, 0.85) if force_n >= 0.0 else (1.0, 0.35, 0.6, 0.85)
                )
                visualizer.add_arrow(
                    start, end, color=color, width=viz.thruster_force_width
                )


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

    @dataclass
    class VizCfg:
        ghost_alpha: float = 0.3
        frame_scale: float = 0.25
        frame_axis_radius: float = 0.008
        error_arrow_color: tuple[float, float, float, float] = (0.95, 0.4, 0.1, 0.9)
        error_arrow_width: float = 0.015
        error_arrow_zero_threshold: float = 0.02
        converged_marker_radius: float = 0.025
        thruster_force_scale: float = 0.015
        thruster_force_width: float = 0.012
        thruster_force_zero_threshold_n: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)

    def __post_init__(self) -> None:
        _validate_range("resampling_time_range", self.resampling_time_range)
        _validate_range("ranges.pos_x", self.ranges.pos_x)
        _validate_range("ranges.pos_y", self.ranges.pos_y)
        _validate_range("ranges.pos_z", self.ranges.pos_z)
        _validate_range("ranges.roll", self.ranges.roll)
        _validate_range("ranges.pitch", self.ranges.pitch)
        _validate_range("ranges.yaw", self.ranges.yaw)
