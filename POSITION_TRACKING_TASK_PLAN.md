# Position Tracking Task Plan

## Goal

Add a new Taluy position-tracking RL task on top of the existing layered env/task
design, without changing or trying to finish the current velocity task first.

The first version should target station-keeping / waypoint-style control with the
existing `body_wrench` action space in a clean environment. Randomization stays
supported, but off by default.

## Recommended Task Definition

- Task name: `position`
- Vehicle entrypoint: `make_taluy_position_env_cfg()`
- Action space: existing `body_wrench`
- Command type: sampled pose setpoints, not trajectory tracking
- Initial orientation scope: full quaternion interface, but default command
  sampling should vary yaw and position first while keeping desired roll/pitch
  at zero. This keeps the task useful for AUV waypoint holding while staying
  6-DOF-ready.

Why this shape:

- Position control is a more natural AUV baseline than direct velocity tracking.
- The actor should see errors, not raw goals, so the policy stays goal-relative.
- Orientation should be represented as an error rotation vector, not Euler-angle
  deltas and not raw desired quaternion.

## Observation Design

### Actor observations

Use the following actor observation set as the default v1 design:

- Body-frame position error
  `e_p^b = R^T (p_d - p)` - 3
- Orientation error as a rotation vector - 3
- Body linear velocity `[u, v, w]` - 3
- Body angular velocity `[p, q, r]` - 3
- Gravity direction in body frame / projected gravity - 3
- Previous applied or commanded body wrench, normalized - 6

Total actor dim: `21`

Recommended implementation details:

- Compute position error in the body frame with `quat_apply_inverse`.
- Compute orientation error with quaternion-safe math such as
  `quat_box_minus(q_d, q)` and standardize quaternions with `quat_unique` where
  needed.
- Prefer normalized applied body wrench over raw policy output because it is
  closer to the physical command that reaches the vehicle.

### Critic observations

Start from all actor observations, then add:

- Current position relative to env origin - 3
- Current orientation quaternion - 4
- Desired position relative to env origin - 3
- Desired orientation quaternion - 4
- Optional privileged dynamics terms when enabled, such as body-frame water
  current and thruster voltage offset

Base critic dim without privileged randomization terms: `35`

Important note:

- Do not feed raw simulator-global position to the critic because MJLab multi-env
  grid placement leaks environment index through `env_origins`. Use env-origin-
  relative position instead.

### Keep / avoid

Keep:

- Position error, not raw actor position
- Orientation error, not desired Euler angles
- Body-frame position error
- Gravity-in-body-frame so the policy can distinguish attitudes cleanly

Avoid:

- Raw world position for actor observations
- Euler-angle differences
- Raw quaternion comparisons without sign handling

## Command Design

Create a new command term, for example `UniformPoseCommand`, under
`src/auvrl/tasks/position/mdp/`.

Recommended stored command contents:

- Desired position relative to env origin: 3
- Desired orientation quaternion: 4

Recommended v1 behavior:

- Sample target positions from configurable `x/y/z` ranges around each env
  origin.
- Sample target yaw from a configurable range.
- Keep desired roll/pitch ranges defaulted to `(0.0, 0.0)` in the first pass.
- Resample more slowly than the velocity task, for example on the order of
  `6-12 s`, so the robot has time to settle.

This gives a position+heading task immediately, while the command interface still
supports later expansion to fully varying 3D orientation.

## Reset Design

The position task should not always start from the exact nominal pose.

Recommended reset flow:

1. Keep `reset_scene_to_default` from the vehicle base env.
2. Add a second reset event using MJLab's `reset_root_state_uniform`.
3. Randomize initial root pose and velocity within modest ranges.

Suggested initial ranges:

- Position reset offsets: smaller than the command ranges
- Yaw reset: moderate randomization
- Roll/pitch reset: small randomization
- Linear/angular velocities: small ranges around zero

This should make the task learn recovery and station-keeping, not just origin
hovering.

## Reward Design

Recommended v1 reward terms:

- `track_position`: main reward from position error norm
- `track_orientation`: reward from orientation error norm
- `linear_velocity_l2` or `settle_linear_velocity`: keep motion low near the
  goal
- `angular_velocity_l2` or `settle_angular_velocity`: damp residual rotation
- `action_l2`: reuse current body-wrench action penalty
- `action_rate_l2`: reuse action smoothness penalty
- `saturation`: reuse thruster saturation penalty

Recommended shaping:

- Use smooth dense rewards first, for example Gaussian or similar kernels on the
  position-error norm and orientation-error norm.
- Weight position reward higher than orientation reward in v1.
- Keep velocity penalties secondary; they should stabilize the hover, not become
  the main task.

If needed later:

- Add a small near-goal bonus
- Gate some orientation reward on being reasonably close in position
- Add a simple curriculum by widening command/reset ranges over time

## Config / Code Layout

Follow the same layered pattern as the velocity task.

### New runtime files

- `src/auvrl/tasks/position/__init__.py`
- `src/auvrl/tasks/position/position_env_cfg.py`
- `src/auvrl/tasks/position/mdp/__init__.py`
- `src/auvrl/tasks/position/mdp/pose_command.py`
- `src/auvrl/tasks/position/mdp/observations.py`
- `src/auvrl/tasks/position/mdp/rewards.py`
- `src/auvrl/tasks/position/config/taluy/__init__.py`
- `src/auvrl/tasks/position/config/taluy/env_cfgs.py`
- `src/auvrl/tasks/position/config/taluy/rl_cfg.py`

### New scripts

- `src/auvrl/scripts/train/taluy_position.py`
- `src/auvrl/scripts/smoke/taluy_position_env.py`

### Package/docs updates

- `src/auvrl/__init__.py`
- `README.md`
- `STATUS.md`

## Reuse Strategy

To keep the current velocity task untouched, prefer minimal reuse rather than a
shared refactor on the first pass.

Recommended approach:

- Reuse the existing Taluy vehicle base env as-is.
- Reuse generic body-wrench penalties and privileged helpers where practical.
- Do not spend the first implementation pass cleaning up the velocity task.

If the position task works well, the common action/reward helpers can be lifted
into a shared module later.

## Validation Plan

The first implementation pass should include at least:

1. `uv run ruff check src/auvrl`
2. `uv run python -m compileall -q src/auvrl`
3. A deterministic smoke test for the new task
4. A short CPU training smoke run

Suggested smoke-test checks:

- Actor observation dim is `21`
- Critic observation dim matches the enabled privileged terms
- Position error observation matches the sampled target at reset
- Orientation error is zero when target and robot orientation match
- Command resampling updates the target pose after the configured interval
- Zero-action stepping keeps state and rewards finite

## Explicit v1 Non-Goals

Do not block the first version on:

- Finishing or cleaning up the current velocity task
- A rich viewer/demo workflow for the new task
- Full trajectory tracking with desired velocity/acceleration references
- Heavy domain randomization from the start

## Recommended Defaults To Implement Later

Unless we decide otherwise during implementation, the default implementation
should use:

- Body-frame position error for actor observations
- Orientation error rotation vector for actor observations
- Env-origin-relative position for critic observations
- Desired yaw sampling with desired roll/pitch fixed to zero in v1
- Clean environment defaults, with current/voltage randomization remaining
  optional
