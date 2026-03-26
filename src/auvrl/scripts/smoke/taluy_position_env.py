"""Smoke checks for the Taluy position-task scaffold."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_apply_inverse, quat_box_minus, quat_unique
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. Ensure mjlab is available "
        "(local ../mjlab or installed dependency)."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    UniformPoseCommand,
    make_taluy_position_env_cfg,
)


EXPECTED_ACTOR_OBS_DIM = 21
EXPECTED_CRITIC_OBS_DIM = 35
EXPECTED_REWARD_TERMS = {
    "track_position",
    "track_orientation",
    "linear_velocity_l2",
    "angular_velocity_l2",
    "action_l2",
    "action_rate_l2",
    "saturation",
}
COMMAND_POS_X_RANGE_M = (0.25, 0.25)
COMMAND_POS_Y_RANGE_M = (-0.20, -0.20)
COMMAND_POS_Z_RANGE_M = (0.15, 0.15)
POSITION_ERROR_SCALE = torch.tensor(
    [
        1.0 / max(abs(COMMAND_POS_X_RANGE_M[0]), abs(COMMAND_POS_X_RANGE_M[1]), 0.25),
        1.0 / max(abs(COMMAND_POS_Y_RANGE_M[0]), abs(COMMAND_POS_Y_RANGE_M[1]), 0.25),
        1.0 / max(abs(COMMAND_POS_Z_RANGE_M[0]), abs(COMMAND_POS_Z_RANGE_M[1]), 0.25),
    ]
)
ORIENTATION_ERROR_SCALE = torch.tensor([1.0 / torch.pi, 1.0 / torch.pi, 1.0 / torch.pi])


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _make_env() -> ManagerBasedRlEnv:
    cfg = make_taluy_position_env_cfg(
        num_envs=2,
        command_resampling_time_s=(0.04, 0.04),
        command_pos_x_range_m=COMMAND_POS_X_RANGE_M,
        command_pos_y_range_m=COMMAND_POS_Y_RANGE_M,
        command_pos_z_range_m=COMMAND_POS_Z_RANGE_M,
        command_roll_range_rad=(0.0, 0.0),
        command_pitch_range_rad=(0.0, 0.0),
        command_yaw_range_rad=(0.35, 0.35),
        curriculum_enabled=False,
        reset_pos_x_range_m=(0.0, 0.0),
        reset_pos_y_range_m=(0.0, 0.0),
        reset_pos_z_range_m=(0.0, 0.0),
        reset_roll_range_rad=(0.0, 0.0),
        reset_pitch_range_rad=(0.0, 0.0),
        reset_yaw_range_rad=(0.0, 0.0),
        reset_lin_vel_x_range_m_s=(0.0, 0.0),
        reset_lin_vel_y_range_m_s=(0.0, 0.0),
        reset_lin_vel_z_range_m_s=(0.0, 0.0),
        reset_ang_vel_x_range_rad_s=(0.0, 0.0),
        reset_ang_vel_y_range_rad_s=(0.0, 0.0),
        reset_ang_vel_z_range_rad_s=(0.0, 0.0),
    )
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _check_command_and_observations(env: ManagerBasedRlEnv) -> None:
    obs, _ = env.reset()

    if set(obs.keys()) != {"actor", "critic"}:
        raise AssertionError(f"Unexpected observation groups: {sorted(obs.keys())}")

    actor_obs = obs["actor"]
    critic_obs = obs["critic"]
    if not isinstance(actor_obs, torch.Tensor) or not isinstance(
        critic_obs, torch.Tensor
    ):
        raise AssertionError("Actor/critic observations should be tensors.")
    if actor_obs.shape != (env.num_envs, EXPECTED_ACTOR_OBS_DIM):
        raise AssertionError(
            f"Actor observation has unexpected shape. Observed {tuple(actor_obs.shape)}"
        )
    if critic_obs.shape != (env.num_envs, EXPECTED_CRITIC_OBS_DIM):
        raise AssertionError(
            "Critic observation has unexpected shape. "
            f"Observed {tuple(critic_obs.shape)}"
        )

    command_term = env.command_manager.get_term("pose")
    if not isinstance(command_term, UniformPoseCommand):
        raise AssertionError("Expected pose term to be UniformPoseCommand")

    command = command_term.command
    if command.shape != (env.num_envs, 7):
        raise AssertionError(f"Unexpected command shape: {tuple(command.shape)}")

    robot = env.scene["robot"]
    current_pos_rel = robot.data.root_link_pos_w - env.scene.env_origins
    current_quat = robot.data.root_link_quat_w

    expected_position_error = quat_apply_inverse(
        current_quat,
        command[:, :3] - current_pos_rel,
    ) * POSITION_ERROR_SCALE.to(device=env.device).view(1, 3)
    observed_position_error = actor_obs[:, 0:3]
    if not torch.allclose(
        observed_position_error, expected_position_error, atol=1.0e-5
    ):
        raise AssertionError(
            "Actor position-error slice does not match expected value."
        )

    expected_orientation_error = quat_box_minus(
        quat_unique(command[:, 3:7]),
        quat_unique(current_quat),
    ) * ORIENTATION_ERROR_SCALE.to(device=env.device).view(1, 3)
    observed_orientation_error = actor_obs[:, 3:6]
    if not torch.allclose(
        observed_orientation_error,
        expected_orientation_error,
        atol=1.0e-5,
    ):
        raise AssertionError(
            "Actor orientation-error slice does not match expected value."
        )

    base_lin_vel = actor_obs[:, 6:9]
    base_ang_vel = actor_obs[:, 9:12]
    projected_gravity = actor_obs[:, 12:15]
    applied_wrench = actor_obs[:, 15:21]
    if not torch.allclose(base_lin_vel, torch.zeros_like(base_lin_vel)):
        raise AssertionError("Base linear velocity should start at zero after reset.")
    if not torch.allclose(base_ang_vel, torch.zeros_like(base_ang_vel)):
        raise AssertionError("Base angular velocity should start at zero after reset.")
    if not torch.allclose(applied_wrench, torch.zeros_like(applied_wrench)):
        raise AssertionError("Applied body wrench should start at zero after reset.")
    if not torch.isfinite(projected_gravity).all():
        raise AssertionError("Projected gravity should remain finite.")

    critic_current_pos = critic_obs[:, 21:24]
    critic_current_quat = critic_obs[:, 24:28]
    critic_desired_pos = critic_obs[:, 28:31]
    critic_desired_quat = critic_obs[:, 31:35]
    if not torch.allclose(critic_current_pos, current_pos_rel, atol=1.0e-5):
        raise AssertionError("Critic current-position slice is incorrect.")
    if not torch.allclose(
        critic_current_quat,
        quat_unique(current_quat),
        atol=1.0e-5,
    ):
        raise AssertionError("Critic current-orientation slice is incorrect.")
    if not torch.allclose(critic_desired_pos, command[:, :3], atol=1.0e-5):
        raise AssertionError("Critic desired-position slice is incorrect.")
    if not torch.allclose(
        critic_desired_quat,
        quat_unique(command[:, 3:7]),
        atol=1.0e-5,
    ):
        raise AssertionError("Critic desired-orientation slice is incorrect.")

    if int(command_term.command_counter.min().item()) != 1:
        raise AssertionError(
            "Command term should have sampled once during reset. "
            f"Observed counters {command_term.command_counter.tolist()}"
        )

    reward_terms = set(env.reward_manager.active_terms)
    if reward_terms != EXPECTED_REWARD_TERMS:
        raise AssertionError(f"Unexpected reward terms: {sorted(reward_terms)}")


def _check_resampling_and_step(env: ManagerBasedRlEnv) -> None:
    command_term = env.command_manager.get_term("pose")
    assert isinstance(command_term, UniformPoseCommand)

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    initial_counter = command_term.command_counter.clone()

    for _ in range(6):
        _, reward, _, _, _ = env.step(action)
        _assert_finite_env_state(env)
        if not torch.isfinite(reward).all():
            raise AssertionError(f"Encountered non-finite rewards: {reward.tolist()}")

    if not torch.all(command_term.command_counter >= initial_counter + 1):
        raise AssertionError(
            "Command did not resample after the configured interval. "
            f"Initial={initial_counter.tolist()} current={command_term.command_counter.tolist()}"
        )


def main() -> None:
    env = _make_env()
    try:
        _check_command_and_observations(env)
        _check_resampling_and_step(env)
    finally:
        env.close()

    print("Taluy MJLab position-task scaffold smoke passed.")


if __name__ == "__main__":
    main()
