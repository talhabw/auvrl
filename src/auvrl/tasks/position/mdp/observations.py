from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_box_minus, quat_unique

from auvrl.actuator.body_wrench_action import BodyWrenchAction
from auvrl.actuator.thruster_actuator import ThrusterActuator
from auvrl.sim.underwater_hydro_action import UnderwaterHydroAction

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


def body_position_error_b(
    env: ManagerBasedRlEnv,
    command_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    command = _pose_command(env, command_name)
    current_pos_rel = robot.data.root_link_pos_w - env.scene.env_origins
    error_rel = command[:, :3] - current_pos_rel
    return quat_apply_inverse(robot.data.root_link_quat_w, error_rel)


def orientation_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    command = _pose_command(env, command_name)
    current_quat = quat_unique(robot.data.root_link_quat_w)
    desired_quat = quat_unique(command[:, 3:7])
    return quat_box_minus(desired_quat, current_quat)


def current_position_rel(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_pos_w - env.scene.env_origins


def current_orientation_quat(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    robot: Entity = env.scene[entity_name]
    return quat_unique(robot.data.root_link_quat_w)


def desired_position_rel(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _pose_command(env, command_name)[:, :3]


def desired_orientation_quat(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return quat_unique(_pose_command(env, command_name)[:, 3:7])


def current_velocity_b(
    env: ManagerBasedRlEnv,
    action_name: str = "hydro",
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, UnderwaterHydroAction):
        raise ValueError(f"Action term '{action_name}' is not UnderwaterHydroAction.")
    return action_term.current_velocity_b


def thruster_voltage_offset(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    nominal_voltage_v: float = 16.0,
    scale_v: float = 2.0,
) -> torch.Tensor:
    if scale_v <= 0.0:
        raise ValueError(f"scale_v must be positive, got {scale_v}.")
    entity = env.scene[entity_name]
    thruster_actuators = [
        actuator
        for actuator in entity.actuators
        if isinstance(actuator, ThrusterActuator)
    ]
    if len(thruster_actuators) != 1:
        raise ValueError(
            f"Expected exactly one ThrusterActuator on '{entity_name}', got {len(thruster_actuators)}."
        )
    thruster_actuator = thruster_actuators[0]
    mean_voltage = thruster_actuator.supply_voltage.mean(dim=1, keepdim=True)
    return (mean_voltage - float(nominal_voltage_v)) / float(scale_v)


def applied_body_wrench(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
    normalize: bool = True,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")
    wrench = action_term.applied_wrench_origin_b
    if not normalize:
        return wrench
    return wrench / action_term.wrench_limit
