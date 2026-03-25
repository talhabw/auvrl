# Thruster Calibration Guide

This guide describes the thruster model currently used by `auvrl` and how to
calibrate it from hardware data.

The canonical YAML fields are documented in `MJLAB_THRUSTER_CONFIG_SCHEMA.md`
and implemented in `src/auvrl/config/thruster_cfg.py`.

## Current model

`auvrl` currently supports a force-target thruster model.

- The command sent to each thruster is a force target in newtons.
- A voltage-aware force-to-PWM polynomial models saturation effects.
- A first-order lag models thruster dynamics.
- The final thrust is clamped to configured min/max thrust limits.

Old `isaac_pwm`-style fields such as `command_model`, `pwm_deadzone`,
`pwm_to_omega_forward`, `pwm_to_omega_reverse`, `time_constant_s`, and
`rotor_constant` are not used by the current code.

## Per-step behavior

For each thruster and each control step:

1. Clamp the commanded force:

   ```text
   force_cmd = clip(command, -command_limit, command_limit)
   ```

2. Apply the deadzone in newtons:

   ```text
   if |force_cmd| < force_deadzone_n:
       force_cmd = 0
   ```

3. Choose forward or reverse polynomial coefficients based on the sign of
   `force_cmd`.

4. Convert force from newtons to kilogram-force:

   ```text
   force_kgf = force_cmd / newton_per_kgf
   ```

5. Compute the requested PWM using the configured 2D polynomial:

   ```text
   pwm_req = a * force_kgf^2 + b * force_kgf * voltage + c * voltage^2
           + d * force_kgf + e * voltage + f
   ```

6. Saturate PWM to the allowed range:

   ```text
   pwm_sat = clip(pwm_req, pwm_min_us, pwm_max_us)
   ```

7. Convert the saturated PWM back to the achievable force by solving the same
   polynomial for force. If the saturated PWM is exactly `pwm_neutral_us`, the
   achievable force is set to zero.

8. Clamp achievable force to `[min_thrust_n, max_thrust_n]`.

9. Apply first-order lag:

   ```text
   alpha = exp(-dt / tau_s)
   thrust_state[k+1] = alpha * thrust_state[k] + (1 - alpha) * achievable_force
   ```

10. Clamp the resulting state again to `[min_thrust_n, max_thrust_n]`.

This means the model is not a direct static force map. PWM saturation and lag
both affect the final thrust.

## YAML fields to calibrate

The main calibration fields are:

- `command_limit`
- `tau_s`
- `force_deadzone_n`
- `min_thrust_n`
- `max_thrust_n`
- `supply_voltage`
- `pwm_min_us`
- `pwm_max_us`
- `pwm_neutral_us`
- `force_to_pwm_coeffs_forward`
- `force_to_pwm_coeffs_reverse`
- `newton_per_kgf`

See `MJLAB_THRUSTER_CONFIG_SCHEMA.md` for meanings and validation rules.

## Recommended calibration workflow

Use a thrust stand, synchronized PWM logs, and measured supply voltage. If
possible, calibrate at several voltages spanning the operating range.

### 1. Determine PWM bounds and neutral

These usually come from the ESC and thruster interface:

- `pwm_min_us`
- `pwm_max_us`
- `pwm_neutral_us`

For a typical T200 setup this may be `1100 / 1900 / 1500`, but use the actual
values for your system.

### 2. Estimate force deadzone

Sweep PWM around neutral and measure thrust.

- Find the smallest forward PWM that produces repeatable non-zero thrust.
- Find the smallest reverse PWM that produces repeatable non-zero thrust.
- Convert both to force using the measured thrust stand output.
- Set `force_deadzone_n` to a conservative magnitude below which you want the
  model to return zero force.

This deadzone is in newtons, not normalized command units.

### 3. Fit the force-to-PWM polynomials

Collect steady-state samples over the expected operating envelope.

For each sample record:

- thrust in newtons
- PWM in microseconds
- supply voltage in volts
- thrust direction sign

Convert thrust to kilogram-force:

```text
force_kgf = force_n / newton_per_kgf
```

Then fit two separate least-squares models, one for forward thrust and one for
reverse thrust:

```text
pwm_us = a * force_kgf^2 + b * force_kgf * voltage + c * voltage^2
       + d * force_kgf + e * voltage + f
```

Store the fitted coefficients as:

- `force_to_pwm_coeffs_forward = [a, b, c, d, e, f]`
- `force_to_pwm_coeffs_reverse = [a, b, c, d, e, f]`

Notes:

- Fit forward and reverse separately; they are usually asymmetric.
- Use the measured voltage, not just nominal battery voltage.
- Check that the fitted surface is monotonic enough in the operating region so
  the inverse step remains well-behaved.

### 4. Set thrust limits

Set:

- `min_thrust_n`
- `max_thrust_n`

to the achievable force envelope you want after saturation.

These should reflect physical capability, not just what the policy is allowed to
request. The model clamps to these values after computing achievable force and
after applying lag.

### 5. Set command limit

`command_limit` is the input-force clamp before the calibration map is applied.

In many cases it is reasonable to set it equal to the physical thrust envelope,
but it does not have to be exactly equal. Treat it as the maximum force command
the higher-level controller or policy is allowed to ask for.

### 6. Fit the time constant

Run step tests in force command space and measure force response over time.

Fit:

```text
force(t) = force_ss + (force_0 - force_ss) * exp(-t / tau_s)
```

Then set `tau_s` from the fitted response.

If you do not fit directly, a practical approximation is:

```text
tau_s ~= rise_time_10_to_90 / 2.2
```

## Validation sequence

After fitting the parameters, validate in this order.

### Single-thruster checks

- commanded force sign matches measured force sign
- neutral command gives zero thrust
- deadzone behaves as expected
- saturation occurs near expected PWM limits
- step response roughly matches measured lag

### Voltage sensitivity checks

- the same force request at different voltages produces different requested PWM
- lower voltage reduces achievable force when PWM saturates earlier

### Vehicle-level checks

- one-thruster-at-a-time tests produce the expected body wrench direction
- combined Taluy maneuvers such as surge, yaw, and heave have the expected sign
- site orientation follows the convention that positive thrust acts along local
  `-Z`

## Example YAML template

```yaml
command_limit: 60.0
tau_s: 0.05

force_deadzone_n: 0.5

min_thrust_n: -60.0
max_thrust_n: 60.0

supply_voltage: 16.0
pwm_min_us: 1100.0
pwm_max_us: 1900.0
pwm_neutral_us: 1500.0
newton_per_kgf: 9.81

force_to_pwm_coeffs_forward:
  - a_f
  - b_f
  - c_f
  - d_f
  - e_f
  - f_f

force_to_pwm_coeffs_reverse:
  - a_r
  - b_r
  - c_r
  - d_r
  - e_r
  - f_r
```

## Common pitfalls

- Fitting PWM as a function of force in newtons instead of kilogram-force.
- Ignoring battery voltage and then expecting one polynomial to work everywhere.
- Using a polynomial fit that looks good numerically but is not invertible or is
  poorly behaved around neutral.
- Setting `command_limit` much larger than physical capability and then training
  through heavy saturation all the time.
- Confusing `force_deadzone_n` with a PWM deadzone.
- Getting the site axis sign wrong at the vehicle level.

## Quick checklist

- [ ] measured `pwm_min_us`, `pwm_max_us`, and `pwm_neutral_us`
- [ ] calibrated `force_deadzone_n`
- [ ] fitted `force_to_pwm_coeffs_forward`
- [ ] fitted `force_to_pwm_coeffs_reverse`
- [ ] fitted `tau_s`
- [ ] set realistic `min_thrust_n` and `max_thrust_n`
- [ ] set realistic `command_limit`
- [ ] validated sign conventions on the full vehicle
