# MJLab AUV Config Schema

Canonical schema for AUV vehicle YAML files parsed by `AUVYamlModel` in
`src/auvrl/config/auv_cfg.py`.

Use the exact field names below. Aliases and the legacy top-level
`thruster_model` / `thruster_site_names` keys are not supported.

## Conventions

- `model_xml_path` is resolved relative to `src/auvrl/`.
- Names ending in `_b_` are expressed in the vehicle body frame.
- `current_world_m_s` is expressed in the world frame.
- Wrench vectors use `[Fx, Fy, Fz, Mx, My, Mz]` in `[N, N, N, N*m, N*m, N*m]`.
- Demo command vectors use the actuator command units defined by the referenced
  thruster model YAML.

## Top-level fields

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `model_xml_path` | string | path | MuJoCo XML used to load rigid-body mass, inertia, and center of gravity. |
| `body_name` | string | - | MuJoCo body that receives hydro and control forces. |
| `hydro` | mapping | - | Hydrodynamic and restoring-force parameters. |
| `thruster` | mapping | - | Thruster model selection and site bindings. |
| `body_wrench_limit` | 6-vector | mixed | Per-axis limit for normalized body-wrench control. |
| `demo` | mapping | - | Scripted Taluy demo commands and stabilization tweaks. |
| `viewer` | mapping | - | Default viewer camera settings. |

## `hydro`

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `linear_damping_matrix` | 6x6 matrix | mixed | Linear damping matrix applied to body-frame relative twist. |
| `quadratic_damping_matrix` | 6x6 matrix | mixed | Quadratic damping matrix applied to body-frame relative twist magnitude. |
| `added_mass_6x6` | 6x6 matrix | kg / kg*m / kg*m^2 | Added-mass matrix. Defaults to all zeros when omitted. |
| `current_world_m_s` | 3-vector | m/s | Ambient water current in the world frame. Defaults to zero current. |
| `current_body_m_s` | 3-vector or `null` | m/s | Optional body-fixed water current. When set, it is used directly instead of rotating `current_world_m_s` into the body frame. |
| `fluid_density_kg_m3` | float | kg/m^3 | Fluid density used for buoyancy and hydro terms. |
| `gravity_m_s2` | float | m/s^2 | Gravity magnitude used for restoring forces. |
| `buoyancy_n` | float or `null` | N | Net buoyant force magnitude. |
| `displaced_volume_m3` | float or `null` | m^3 | Displaced fluid volume used to derive buoyancy. |
| `center_of_buoyancy_b_m` | 3-vector | m | Center of buoyancy in the body frame. |
| `include_damping` | bool | - | Enables damping forces. |
| `include_restoring` | bool | - | Enables buoyancy and gravity restoring forces. |
| `include_added_mass` | bool | - | Enables added-mass forces. |
| `include_added_coriolis` | bool | - | Enables added-mass Coriolis and centripetal terms. |

At least one of `buoyancy_n` or `displaced_volume_m3` must be provided. If
both are present they must agree with:

```text
buoyancy_n = displaced_volume_m3 * fluid_density_kg_m3 * gravity_m_s2
```

Rigid-body mass, inertia, and center of gravity are not configured here; they
come from the MuJoCo XML referenced by `model_xml_path`.

## `thruster`

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `model` | string | - | Thruster model YAML basename without `.yaml`, for example `t200`. This must match the file name exactly and use the schema documented in `MJLAB_THRUSTER_CONFIG_SCHEMA.md`. |
| `site_names` | string list | - | Ordered MuJoCo site names for the vehicle thrusters. |

## `demo`

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `surge_command` | 8-vector | model-dependent | Scripted forward-motion thruster command. |
| `yaw_command` | 8-vector | model-dependent | Scripted yaw thruster command. |
| `heave_command` | 8-vector | model-dependent | Scripted vertical thruster command. |
| `coast_command` | 8-vector | model-dependent | Zero or near-zero command used during the coast phase. |

## `viewer`

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `distance` | float | m | Viewer camera distance from the tracked body. |
| `elevation` | float | deg | Viewer camera elevation angle. |
| `azimuth` | float | deg | Viewer camera azimuth angle. |
