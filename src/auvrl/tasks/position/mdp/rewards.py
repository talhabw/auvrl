from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_box_minus, quat_unique

from auvrl.actuator.body_wrench_action import BodyWrenchAction

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def _pose_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    if command.ndim != 2 or command.shape[1] != 7:
        raise ValueError(
            f"Expected pose command '{command_name}' to have shape (N, 7), got {tuple(command.shape)}."
        )
    return command


def track_position(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    robot: Entity = env.scene[entity_name]
    command = _pose_command(env, command_name)
    current_pos_rel = robot.data.root_link_pos_w - env.scene.env_origins
    error = command[:, :3] - current_pos_rel
    return torch.exp(-torch.sum(torch.square(error), dim=1) / float(std * std))


def track_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    robot: Entity = env.scene[entity_name]
    command = _pose_command(env, command_name)
    error = quat_box_minus(
        quat_unique(command[:, 3:7]),
        quat_unique(robot.data.root_link_quat_w),
    )
    return torch.exp(-torch.sum(torch.square(error), dim=1) / float(std * std))


def linear_velocity_l2(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    return torch.mean(torch.square(robot.data.root_link_lin_vel_b), dim=1)


def angular_velocity_l2(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    return torch.mean(torch.square(robot.data.root_link_ang_vel_b), dim=1)


def body_wrench_action_l2(
    env: ManagerBasedRlEnv,
    action_name: str,
    wrench_limits: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")

    if wrench_limits is None:
        return torch.mean(torch.square(action_term.raw_action), dim=1)

    limits = torch.tensor(wrench_limits, dtype=torch.float, device=env.device).view(
        1, 6
    )
    if torch.any(limits <= 0.0):
        raise ValueError("wrench_limits must all be positive.")

    normalized_action = action_term.desired_wrench_b / limits
    return torch.mean(torch.square(normalized_action), dim=1)


def body_wrench_action_rate_l2(
    env: ManagerBasedRlEnv,
    wrench_limits: tuple[float, float, float, float, float, float] | None = None,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")

    if wrench_limits is None:
        delta = env.action_manager.action - env.action_manager.prev_action
        return torch.mean(torch.square(delta), dim=1)

    limits = torch.tensor(wrench_limits, dtype=torch.float, device=env.device).view(
        1, 6
    )
    if torch.any(limits <= 0.0):
        raise ValueError("wrench_limits must all be positive.")

    current_wrench = action_term.action_to_wrench(env.action_manager.action)
    prev_wrench = action_term.action_to_wrench(env.action_manager.prev_action)
    delta = (current_wrench - prev_wrench) / limits
    return torch.mean(torch.square(delta), dim=1)


def body_wrench_saturation_penalty(
    env: ManagerBasedRlEnv,
    action_name: str,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")
    return action_term.step_saturation_fraction
