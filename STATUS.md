# Project Status

This is the single internal state/status note for the repository.

## Current scope

- The project is centered on Taluy in MJLab.
- Taluy has both raw thruster control and 6D body-wrench control paths.
- The current RL focus is Taluy body-velocity tracking in a clean environment
  before leaning heavily on domain randomization.
- A Taluy body-wrench position-tracking scaffold now exists alongside velocity
  tracking so waypoint and station-keeping experiments can start without waiting
  on more velocity-task iteration.

## Current config workflow

- Env/task configuration is now intentionally layered.
- Vehicle base env config files live under `src/auvrl/envs/`.
- Task base env config files live under `src/auvrl/tasks/<task>/` and should
  only add task logic that stays invariant across vehicles.
- Final task+vehicle env creators live under
  `src/auvrl/tasks/<task>/config/<vehicle>/` and should hold the parts of that
  task that are specific to the selected vehicle.
- When adding a new task, the expected flow is:
  - start from the vehicle base env cfg
  - build a task base env cfg on top of it
  - build a vehicle-specific final env cfg in the task config folder
- Scripts and package exports should point at the final task+vehicle env
  creator, not the intermediate base builders, unless the base builder itself
  is what you are actively extending.

## Current structure

- Runtime code lives under `src/auvrl/`.
- Taluy assets live under `src/auvrl/asset_zoo/vehicles/taluy/`.
- Thruster assets live under `src/auvrl/asset_zoo/thrusters/`.
- Vehicle base env builders live under `src/auvrl/envs/`.
- Task base env builders live under `src/auvrl/tasks/<task>/`.
- Vehicle-specific task configs live under `src/auvrl/tasks/<task>/config/`.
- User-facing scripts live under `src/auvrl/scripts/`.

## Implemented pieces

- Taluy vehicle base env builder: `src/auvrl/envs/taluy_env_cfg.py`
- AUV YAML loader/model: `src/auvrl/config/auv_cfg.py`
- Thruster YAML loader/model: `src/auvrl/config/thruster_cfg.py`
- Thruster actuator: `src/auvrl/actuator/thruster_actuator.py`
- Thruster allocation helpers: `src/auvrl/actuator/thruster_allocation.py`
- Body-wrench action term: `src/auvrl/actuator/body_wrench_action.py`
- Hydro action and shared hydro math:
  - `src/auvrl/sim/underwater_hydro_action.py`
  - `src/auvrl/sim/hydrodynamics.py`
- Vehicle-invariant velocity task base:
  - `src/auvrl/tasks/velocity/velocity_env_cfg.py`
  - `src/auvrl/tasks/velocity/mdp/`
- Vehicle-invariant position task base:
  - `src/auvrl/tasks/position/position_env_cfg.py`
  - `src/auvrl/tasks/position/mdp/`
- Taluy-specific position final config:
  - `src/auvrl/tasks/position/config/taluy/env_cfgs.py`
  - `src/auvrl/tasks/position/config/taluy/rl_cfg.py`
- Taluy-specific velocity final config:
  - `src/auvrl/tasks/velocity/config/taluy/env_cfgs.py`
  - `src/auvrl/tasks/velocity/config/taluy/rl_cfg.py`

## Restructure notes

- The recent restructuring split the old single Taluy velocity env path into
  three layers:
  - vehicle base: `src/auvrl/envs/taluy_env_cfg.py`
  - task base: `src/auvrl/tasks/velocity/velocity_env_cfg.py`
  - task+vehicle final: `src/auvrl/tasks/velocity/config/taluy/env_cfgs.py`
- The same layering pattern now exists for Taluy position tracking:
  - vehicle base: `src/auvrl/envs/taluy_env_cfg.py`
  - task base: `src/auvrl/tasks/position/position_env_cfg.py`
  - task+vehicle final: `src/auvrl/tasks/position/config/taluy/env_cfgs.py`
- `make_velocity_env_cfg()` is the reusable task-level builder for 6-DOF body
  velocity tracking. It sets up commands, core observations, and generic
  velocity-tracking rewards, but leaves vehicle-specific tuning out.
- `make_position_env_cfg()` is the reusable task-level builder for 6-DOF pose
  tracking. It sets up pose commands, goal-relative observations, reset
  randomization, and generic position/orientation rewards.
- `make_taluy_velocity_env_cfg()` is now the concrete Taluy velocity entrypoint.
  It starts from `make_taluy_base_env_cfg()`, applies the velocity task base,
  then adds Taluy-only observation/reward terms and run-specific overrides.
- `make_taluy_position_env_cfg()` is the concrete Taluy position entrypoint.
  It starts from `make_taluy_base_env_cfg()`, applies the position task base,
  then adds Taluy-only observation/reward terms and run-specific overrides.
- PPO config for this task/vehicle pair now lives beside the final env builder
  under `src/auvrl/tasks/velocity/config/taluy/rl_cfg.py`.
- PPO config for Taluy position lives beside the final env builder under
  `src/auvrl/tasks/position/config/taluy/rl_cfg.py`.
- Public imports still expose the Taluy final env creator through
  `src/auvrl/tasks/velocity/__init__.py`, `src/auvrl/tasks/position/__init__.py`,
  and `src/auvrl/__init__.py`.

## Important behavioral notes

- Taluy uses a force-target thruster path with actuator lag and voltage-aware
  saturation.
- `com-neutral` is a demo/debug allocation mode, not the primary RL control path.
- Positive thrust follows the local site `-Z` axis convention.
- Taluy hydro/restoring parameters come from YAML, while rigid-body mass,
  inertia, and center of gravity come from the MuJoCo XML referenced by
  `model_xml_path`.

## Current validation baseline

- Core lint/build checks:
  - `uv run ruff check src/auvrl`
  - `uv run python -m compileall -q src/auvrl`
- Useful smoke tests:
  - `uv run python -m auvrl.scripts.smoke.taluy_dynamics_regression`
  - `uv run python -m auvrl.scripts.smoke.taluy_position_env`
  - `uv run python -m auvrl.scripts.smoke.taluy_velocity_env`
  - `uv run python -m auvrl.scripts.smoke.taluy_body_wrench`

Last known working direction:

- The layered vehicle base -> task base -> task+vehicle final config workflow
  is in place for Taluy velocity and Taluy position.
- Taluy env/task wiring is in place.
- Taluy velocity training runs in MJLab.
- Taluy position task scaffolding is now available for training/smoke checks.
- Clean-environment training is the current baseline path.

## Immediate priorities

- Establish a clean-environment Taluy position-tracking baseline.
- Continue improving Taluy velocity-tracking quality in the clean environment.
- Keep following the new layered config pattern when adding more tasks or more
  vehicles.
- Use demos and smoke tests to catch regressions in hydro/action/allocation
  wiring.
- Keep randomization secondary until the clean baseline is strong.
