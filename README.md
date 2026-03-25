# auvrl

`auvrl` is a Taluy-focused underwater RL project built on top of `mjlab`.

The repository contains the Taluy MJLab environment, hydrodynamics and thruster
models, task configuration builders, RL wiring, and a small set of scripts for
smoke testing, demos, diagnostics, and training.

## What this repo contains

- Taluy vehicle assets in `src/auvrl/asset_zoo/vehicles/taluy`
- Thruster model assets in `src/auvrl/asset_zoo/thrusters`
- Runtime code under `src/auvrl/actuator`, `src/auvrl/sim`, `src/auvrl/envs`, and `src/auvrl/tasks`
- Vehicle-specific task finals under `src/auvrl/tasks/<task>/config/<vehicle>`
- Executable scripts under `src/auvrl/scripts`

## Main entry points

- Taluy vehicle base env builder: `src/auvrl/envs/taluy_env_cfg.py`
- Vehicle-invariant velocity task base: `src/auvrl/tasks/velocity/velocity_env_cfg.py`
- Taluy velocity final env builder: `src/auvrl/tasks/velocity/config/taluy/env_cfgs.py`
- Taluy velocity PPO config: `src/auvrl/tasks/velocity/config/taluy/rl_cfg.py`
- Training script: `src/auvrl/scripts/train/taluy_velocity.py`
- Visual demo: `src/auvrl/scripts/demo/taluy_visual.py`

## Config layering workflow

- Each vehicle owns a base env config in `src/auvrl/envs/`; this file contains the vehicle-specific MJLab scene, assets, actuator wiring, hydro action, events, viewer, and other settings that are not task-specific.
- Each task owns a base env config in `src/auvrl/tasks/<task>/`; this file starts from a vehicle base env config and adds only task logic that should stay vehicle-invariant.
- Each task and vehicle pair owns a final env creator in `src/auvrl/tasks/<task>/config/<vehicle>/`; this layer starts from the task base env config and applies the task tuning, observations, rewards, and overrides that are specific to that vehicle.
- Scripts should normally import the final env creator for the concrete task and vehicle combination they run.

Current example:

- `make_taluy_base_env_cfg()` in `src/auvrl/envs/taluy_env_cfg.py`
- `make_velocity_env_cfg()` in `src/auvrl/tasks/velocity/velocity_env_cfg.py`
- `make_taluy_velocity_env_cfg()` in `src/auvrl/tasks/velocity/config/taluy/env_cfgs.py`

## Setup

```bash
uv sync
```

## Common commands

Smoke tests:

```bash
uv run python -m auvrl.scripts.smoke.thruster
uv run python -m auvrl.scripts.smoke.hydro_action
uv run python -m auvrl.scripts.smoke.env
uv run python -m auvrl.scripts.smoke.taluy_body_wrench
uv run python -m auvrl.scripts.smoke.taluy_dynamics_regression
uv run python -m auvrl.scripts.smoke.taluy_velocity_env
```

Training:

```bash
uv run python -m auvrl.scripts.train.taluy_velocity
```

Viewer / demo:

```bash
uv run python -m auvrl.scripts.demo.taluy_visual
uv run python -m auvrl.scripts.demo.taluy_velocity_play
```

Taluy velocity debug overlay legend (`Scene -> Body_velocity`):

- Purple arrow: commanded linear body velocity, anchored above the robot center.
- Blue arrow: measured linear body velocity, sharing the same anchor so it sits inside the command arrow.
- Orange / pink thruster arrows: commanded thruster force along each thruster's local `-Z` force axis (orange = positive thrust, pink = reverse thrust).
- Tiny thruster commands below about `1 N` are hidden to reduce idle jitter.

Diagnostics:

```bash
uv run python -m auvrl.scripts.diagnostics.taluy_velocity
```

## Important project conventions

- Taluy remains the primary vehicle in this repo; generic helpers exist only
  where they reduce duplication.
- MuJoCo fluid and gravity stay disabled in the Taluy XML; restoring and drag
  are applied through the hydro action.
- Positive thruster force acts along each thruster site's local `-Z` axis.

## Additional docs

- Internal project status: `STATUS.md`
- AUV YAML schema: `MJLAB_AUV_CONFIG_SCHEMA.md`
- Thruster YAML schema: `MJLAB_THRUSTER_CONFIG_SCHEMA.md`
- Thruster calibration guide: `THRUSTER_CALIBRATION.md`

## Python version

- `python >= 3.10`
