# MJLab Thruster Config Schema

Canonical schema for thruster YAML files parsed by `ThrusterModelCfg` in
`src/auvrl/config/thruster_cfg.py`.

Use the exact field names below.

## Conventions

- All thrust values are in newtons.
- `supply_voltage` may be either a single scalar voltage or a tuple of per-thruster
  voltages.
- The force-to-PWM mapping uses the polynomial:

```text
pwm_us = a * force_kgf^2 + b * force_kgf * voltage + c * voltage^2
       + d * force_kgf + e * voltage + f
```

  where `force_kgf = force_n / newton_per_kgf`.

## Fields

| Field | Type | Units | Meaning |
| --- | --- | --- | --- |
| `command_limit` | float | N | Command clamp applied before thruster dynamics and calibration. |
| `tau_s` | float | s | First-order thruster time constant. |
| `force_deadzone_n` | float | N | Commands with smaller magnitude are treated as zero. |
| `min_thrust_n` | float | N | Minimum achievable thrust after dynamics and saturation. |
| `max_thrust_n` | float | N | Maximum achievable thrust after dynamics and saturation. |
| `supply_voltage` | float or tuple | V | Supply voltage used by the calibration model. |
| `pwm_min_us` | float | us | Minimum allowable PWM command. |
| `pwm_max_us` | float | us | Maximum allowable PWM command. |
| `pwm_neutral_us` | float | us | Neutral PWM that should produce zero thrust. |
| `force_to_pwm_coeffs_forward` | 6-vector | mixed | Forward-thrust polynomial coefficients `[a, b, c, d, e, f]`. |
| `force_to_pwm_coeffs_reverse` | 6-vector | mixed | Reverse-thrust polynomial coefficients `[a, b, c, d, e, f]`. |
| `newton_per_kgf` | float | N/kgf | Unit conversion between kilogram-force and newtons. |

## Validation rules

- `command_limit`, `tau_s`, `newton_per_kgf`, and all `supply_voltage` values must
  be positive.
- `force_deadzone_n` must be non-negative.
- `max_thrust_n` must be greater than or equal to `min_thrust_n`.
- `pwm_neutral_us` must lie within `[pwm_min_us, pwm_max_us]`.
