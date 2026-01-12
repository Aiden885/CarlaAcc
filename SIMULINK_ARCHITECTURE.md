# ACC Integrated Model - Complete Simulink Architecture

## Document Purpose
This document provides complete architectural details of `acc_integrated_model.slx` down to the most basic blocks. It serves as a comprehensive reference for future development sessions to understand the internal structure without needing to re-analyze the model.

---

## System Overview

### High-Level Data Flow
```
CARLA Simulation (Python)
    ↓ UDP Send (Port 27000)
acc_integrated_model.slx
    ├── Decision Subsystem (State Machine)
    └── SPPVT Subsystem (Control Algorithm)
    ↓ UDP Receive (Port 27001)
Python Control Loop
    ↓ Torque Conversion
CARLA Vehicle Control
```

### Communication Protocol
- **Python → Simulink**: UDP port 27000 (structured bus data)
- **Simulink → Python**: UDP port 27001 (structured bus data)
- **Cycle Time**: 50ms (dt = 0.05s)

---

## Top-Level Architecture

### Model: `acc_integrated_model.slx`

**Total Blocks**: 6

| Block Name | Type | Description |
|------------|------|-------------|
| UDP Receive | MATLABSystem | Receives bus data from Python (port 27000) |
| Mux1 | Mux | Multiplexes inputs for Decision subsystem |
| Decision | SubSystem | State machine for decision logic |
| SPPVT | SubSystem | Staged Proportional Plus Variable Time-gap controller |
| Demux | Demux | Demultiplexes SPPVT outputs |
| UDP Send | MATLABSystem | Sends results back to Python (port 27001) |
| Dashboard Scope | DashboardScope | Visualization (debugging) |

### Signal Flow
```
UDP Receive → Mux1 → Decision → SPPVT → Demux → UDP Send
                           ↓
                   Dashboard Scope
```

---

## Decision Subsystem

### Purpose
Implements a finite state machine for ACC decision logic using lookup tables.

### Inputs (4)
| Port | Name | Type | Description |
|------|------|------|-------------|
| 1 | current_state | double | Current FSM state |
| 2 | command_type | double | Command from user/system |
| 3 | has_history | double | Whether history exists |
| 4 | last_active_decision | double | Previous active decision |

### Outputs (5)
| Port | Name | Type | Description |
|------|------|------|-------------|
| 1 | next_state | double | Next FSM state |
| 2 | decision | double | Decision output |
| 3 | control_enabled | double | Whether control is enabled |
| 4 | next_has_history | double | Updated history flag |
| 5 | next_last_decision | double | Updated last decision |

### Internal Structure
**Lookup Tables (4)**:
1. `LUT_control_enabled` - Maps states to control enable flag
2. `LUT_decision` - Maps states to decision outputs
3. `LUT_next_state` - State transition table
4. `LUT_side_effect` - Side effects for state transitions

**Note**: LUT data is stored as workspace variables (char-referenced).

---

## SPPVT Subsystem

### Purpose
Implements the core ACC control algorithm with staged proportional control and upgrade conditions.

### Top-Level Structure

#### Inputs (5)
| Port | Name | Type | Unit | Description |
|------|------|------|------|-------------|
| 1 | error_value | double | m | Distance/speed error |
| 2 | current_stage_offset | double | m | Stage-specific offset |
| 3 | prev_error | double | m | Previous error (for derivative) |
| 4 | prev_velocity | double | m/s | Previous velocity (for derivative) |
| 5 | prev_accel | double | m/s² | Previous acceleration (for derivative) |

#### Outputs (5)
| Port | Name | Type | Unit | Description |
|------|------|------|------|-------------|
| 1 | control_output | double | Nm (dimensionless) | Torque demand [-3.0, 2.0] |
| 2 | velocity | double | m/s | Error velocity (filtered) |
| 3 | acceleration | double | m/s² | Error acceleration (filtered) |
| 4 | jerk | double | m/s³ | Error jerk |
| 5 | should_upgrade | double | bool | Upgrade condition flag |

#### Constants (6)
| Name | Value | Unit | Description |
|------|-------|------|-------------|
| SPPVT_kp | 0.95 | - | Proportional gain |
| SPPVT_dt | 0.05 | s | Time step (20Hz) |
| SPPVT_delta | 0.05 | m/s | Upgrade condition - velocity threshold |
| SPPVT_eta | 0.3 | m | Upgrade condition - error threshold |
| SPPVT_max_accel | 2.0 | - | Maximum acceleration limit |
| SPPVT_max_decel | -4.56 | - | Maximum deceleration limit |

---

### Sub-Subsystem 1: Core_Control_Subsystem

**Purpose**: Calculates the primary control output using enhanced proportional control.

#### Inputs (5)
1. `error_value` - Distance/speed error
2. `current_stage_offset` - Stage offset
3. `sppvt_kp` - Proportional gain (0.95)
4. `max_accel` - Upper saturation limit (2.0)
5. `max_decel` - Lower saturation limit (-3.0)

#### Outputs (1)
1. `control_output` - Saturated torque demand

#### Internal Blocks (4)

| Block Name | Type | Configuration | Description |
|------------|------|---------------|-------------|
| Enhanced_Error | Sum | `++` | `error_value + current_stage_offset` |
| Proportional_Control | Product | 2 inputs | `Enhanced_Error × sppvt_kp` |
| Output_Saturation | Saturate | Upper: 2.0, Lower: -3.0 | Limits output range |
| control_output | Outport | Port 1 | Final output |

#### Control Law
```
Enhanced_Error = error_value + current_stage_offset
Raw_Output = Enhanced_Error × kp (0.95)
control_output = saturate(Raw_Output, [-3.0, 2.0])
```

**Physical Meaning**:
- Output range [-3.0, 2.0] represents dimensionless torque demand
- Python scales this by 210.0 to get actual engine torque (N·m)
- Negative values → braking, Positive values → acceleration

---

### Sub-Subsystem 2: Derivatives_Subsystem

**Purpose**: Computes filtered derivatives of the error signal (velocity, acceleration, jerk).

#### Inputs (5)
1. `error_value` - Current error
2. `dt` - Time step (0.05s)
3. `prev_error` - Previous error
4. `prev_velocity` - Previous velocity
5. `prev_accel` - Previous acceleration

#### Outputs (3)
1. `velocity` - Filtered error velocity
2. `acceleration` - Filtered error acceleration
3. `jerk` - Error jerk (rate of acceleration change)

#### Discrete Filters (2)

**Filter 1**: Velocity Low-Pass Filter
```matlab
Numerator: [0.239]
Denominator: [1 -0.761]
Transfer Function: H(z) = 0.239 / (1 - 0.761z^-1)
Sample Time: Inherited (-1)
```

**Filter 2**: Acceleration Low-Pass Filter
```matlab
Numerator: [0.2]
Denominator: [1 -0.8]
Transfer Function: H(z) = 0.2 / (1 - 0.8z^-1)
Sample Time: Inherited (-1)
```

**Filter Characteristics**:
- Both are first-order IIR low-pass filters
- Cut-off frequency designed for 50ms sampling
- Reduce noise in derivative calculations

#### Calculation Blocks (6)

| Block Name | Type | Configuration | Formula |
|------------|------|---------------|---------|
| Error_Diff | Sum | `+-` | `error_value - prev_error` |
| Velocity_Calc | Product | 2 inputs | `Error_Diff × (1/dt)` |
| Discrete Filter | DiscreteFilter | H1(z) | Filters raw velocity |
| Velocity_Diff | Sum | `+-` | `velocity - prev_velocity` |
| Acceleration_Calc | Product | 2 inputs | `Velocity_Diff × (1/dt)` |
| Discrete Filter1 | DiscreteFilter | H2(z) | Filters raw acceleration |
| Acceleration_Diff | Sum | `+-` | `acceleration - prev_accel` |
| Jerk_Calc | Product | 2 inputs | `Acceleration_Diff × (1/dt)` |

#### Signal Processing Chain
```
error_value ──┬──> Error_Diff ──> Velocity_Calc ──> [Filter 1] ──> velocity
              │                                                          │
         prev_error                                                      │
                                                                         │
velocity ──────┬──> Velocity_Diff ──> Accel_Calc ──> [Filter 2] ──> acceleration
               │                                                          │
         prev_velocity                                                    │
                                                                          │
acceleration ──┬──> Accel_Diff ──> Jerk_Calc ──> jerk
               │
         prev_accel
```

---

### Sub-Subsystem 3: Upgrade_Condition_Subsystem

**Purpose**: Determines when to upgrade the control stage based on error dynamics.

#### Inputs (6)
1. `jerk` - Error jerk from Derivatives
2. `acceleration` - Error acceleration from Derivatives
3. `velocity` - Error velocity from Derivatives
4. `error_value` - Raw error
5. `sppvt_delta` - Velocity threshold (0.05 m/s)
6. `sppvt_eta` - Error threshold (0.3 m)

#### Outputs (4)
1. `should_upgrade` - Final upgrade decision (boolean)
2. `com1` - Condition 1 result (debug)
3. `com2` - Condition 2 result (debug)
4. `com3` - Condition 3 result (debug)

#### Constants (2)
- `Constant` = 0 (baseline reference)
- `Kappa` = 0.2 (detector coefficient)

#### Logic Blocks

| Block Name | Type | Configuration | Formula |
|------------|------|---------------|---------|
| Abs_error | Abs | - | `|error_value|` |
| Abs_velocity | Abs | - | `|velocity|` |
| Compare_eta | RelationalOperator | `>` | `|error_value| > sppvt_eta` |
| Compare_eta1 | RelationalOperator | `<=` | `|velocity| <= sppvt_delta` |
| Product | Product | 2 inputs | `Kappa × [signal]` |
| Relational Operator | RelationalOperator | `<` | Condition 3 |
| Relational Operator1 | RelationalOperator | `<` | Condition 4 |
| Logical Operator | Logic | AND, 3 inputs | Combines conditions |

#### Upgrade Logic

**Complete Upgrade Condition**:
```
B(k) = (|e(k)| > η) ∧ (|ė(k)| ≤ δ) ∧ (|ė(k)| < κ · |e(k)|)
```

Where:
- `e(k)` = error_value (current error)
- `ė(k)` = velocity (error derivative, filtered)
- `η` = 0.3 m (error threshold)
- `δ` = 0.05 m/s (velocity threshold)
- `κ` = 0.2 (convergence ratio)

**Condition 1** (`com1`):
```
|e(k)| > η  →  |error_value| > 0.3 m
```
**Meaning**: Error magnitude is large enough to warrant stage upgrade.

**Condition 2** (`com2`):
```
|ė(k)| ≤ δ  →  |velocity| ≤ 0.05 m/s
```
**Meaning**: Error rate of change is small (system stabilizing).

**Condition 3** (`com3`):
```
|ė(k)| < κ · |e(k)|  →  |velocity| < 0.2 × |error_value|
```
**Meaning**: Error change rate is less than 20% of current error (relative convergence).
- Example: If error = 1.0 m, velocity must be < 0.2 m/s
- Ensures error is converging rather than diverging

**Final Decision**:
```
should_upgrade = Condition1 AND Condition2 AND Condition3
```

**Physical Interpretation**:
- Upgrade only when error is **significant** (> 0.3m)
- AND error change is **slow** (≤ 0.05 m/s)
- AND error is **relatively converging** (change rate < 20% of error magnitude)
- Prevents premature stage transitions during transients or oscillations
- Ensures smooth control transitions

#### Detector Subsystem
- **Location**: `Upgrade_Condition_Subsystem/Detector`
- **Purpose**: Implements the relative convergence check (Condition 3)
- **Implementation**: `|velocity| < Kappa × |error_value|` using Product and RelationalOperator blocks

---

## Signal Connectivity

### SPPVT Internal Signal Flow
- **Total Signal Lines**: 24
- **Key Connections**:
  1. Constants → Core_Control inputs
  2. Constants → Derivatives inputs
  3. Derivatives outputs → Upgrade_Condition inputs
  4. Core_Control output → Top-level output port
  5. Upgrade_Condition output → Top-level output port

### Mux/Demux Configuration
- **Mux1**: Combines signals for Decision subsystem (4 signals)
- **Demux**: Splits SPPVT outputs for UDP transmission (5 signals)

---

## Python Integration

### Data Flow from Python to Simulink

**Input Bus Structure** (defined in `create_decision_sppvt_bus.m`):
```matlab
DecisionSppvtInput (Bus)
├── decision_inputs (4 doubles)
├── error_value (double, m)
├── current_stage_offset (double, m)
├── prev_error (double, m)
├── prev_velocity (double, m/s)
└── prev_accel (double, m/s²)
```

**Python Sending Code** (`integrated_simulink_manager.py`):
```python
packed_data = struct.pack(
    '13d',  # 13 doubles
    decision_inputs[0],  # current_state
    decision_inputs[1],  # command_type
    decision_inputs[2],  # has_history
    decision_inputs[3],  # last_active_decision
    error_value,
    current_stage_offset,
    prev_error,
    prev_velocity,
    prev_accel,
    0, 0, 0, 0  # Reserved
)
self.udp_socket.sendto(packed_data, (self.simulink_ip, 27000))
```

### Data Flow from Simulink to Python

**Output Bus Structure**:
```matlab
DecisionSppvtOutput (Bus)
├── decision_next_state (double)
├── decision_decision (double)
├── decision_control_enabled (double)
├── decision_next_has_history (double)
├── decision_next_last_decision (double)
├── sppvt_control_output (double, Nm dimensionless)
├── sppvt_velocity_output (double, m/s)
├── sppvt_acceleration_output (double, m/s²)
├── sppvt_jerk_output (double, m/s³)
└── sppvt_should_upgrade (double, bool)
```

**Python Receiving Code**:
```python
received_data = struct.unpack('10d', data)  # 10 doubles
return {
    'decision_next_state': received_data[0],
    'decision_decision': received_data[1],
    'decision_control_enabled': received_data[2],
    'decision_next_has_history': received_data[3],
    'decision_next_last_decision': received_data[4],
    'sppvt_control_output': received_data[5],  # Dimensionless torque
    'sppvt_velocity_output': received_data[6],
    'sppvt_acceleration_output': received_data[7],
    'sppvt_jerk_output': received_data[8],
    'sppvt_should_upgrade': received_data[9],
}
```

---

## Control Flow Integration

### Complete Control Cycle

```
1. Python (acc_updated.py):
   - Measures ego vehicle state (speed, position)
   - Measures lead vehicle state
   - Calculates error_value = f(distance_error, speed_error)

2. Python → Simulink (UDP 27000):
   - Sends: error, offset, history states, prev derivatives

3. Simulink Processing:
   a. Decision subsystem:
      - Updates FSM state
      - Determines control_enabled flag

   b. SPPVT subsystem:
      - Core_Control: Computes torque demand
      - Derivatives: Calculates filtered derivatives
      - Upgrade_Condition: Evaluates stage transition

4. Simulink → Python (UDP 27001):
   - Returns: control_output, derivatives, upgrade flag

5. Python (control_loop_manager.py):
   - Scales torque: engine_torque = control_output × 210.0
   - Applies control_enabled gate

6. Python (torque_to_throttle_converter.py):
   - If torque >= 0: Convert to throttle [0, 1]
   - If torque < 0: Convert to brake [0, 1] via transmission

7. CARLA:
   - Applies vehicle control
   - Physics simulation updates vehicle state
```

---

## Critical Parameters Summary

### Simulink SPPVT Parameters
| Parameter | Value | Unit | Purpose |
|-----------|-------|------|---------|
| kp | 0.95 | - | Proportional gain (internal) |
| dt | 0.05 | s | Control cycle time |
| delta | 0.05 | m/s | Upgrade velocity threshold |
| eta | 0.3 | m | Upgrade error threshold |
| Kappa | 0.2 | - | Detector coefficient |
| max_accel | 2.0 | - | Output saturation upper |
| max_decel | -3.0 | - | Output saturation lower |

### Python Scaling Parameters (`acc_config.py`)
| Parameter | Value | Unit | Purpose |
|-----------|-------|------|---------|
| sppvt_accel_scale | 210.0 | N·m | Torque scaling gain (acceleration) |
| sppvt_decel_scale | 210.0 | N·m | Torque scaling gain (braking) |

**Note**: Scaling parameters are NOT unit converters - they are tuning gains for control performance.

### CARLA Vehicle Parameters (`torque_to_throttle_converter.py`)
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| total_gear_ratio | 9.204 | - | Audi e-tron transmission ratio |
| max_brake_torque_per_wheel | 1000.0 | N·m | Maximum brake torque per wheel |
| num_drive_wheels | 4 | - | Number of driven wheels |

---

## Torque Conversion Details

### Acceleration Path (Positive Torque)
```
SPPVT output [0, 2.0]
    × 210.0 (Python scaling)
    = Engine Torque [0, 420] N·m
    ÷ Max Available Torque at RPM
    = Throttle [0, 1.0]
    → CARLA Throttle Input
```

### Braking Path (Negative Torque)
```
SPPVT output [-3.0, 0]
    × 210.0 (Python scaling)
    = Engine Brake Torque [0, 630] N·m (absolute)
    × 9.204 (Transmission ratio)
    = Wheel Brake Torque [0, 5798.52] N·m
    ÷ 4 wheels
    = Per-Wheel Torque [0, 1449.63] N·m
    ÷ 1000 N·m (Max per wheel)
    = Brake Value [0, 1.449] → saturated to [0, 1.0]
    → CARLA Brake Input
```

**Physics Chain**:
```
Engine → Transmission → Wheel Torque → Brake Normalization → CARLA
```

---

## File References

### Simulink Files
- **Model**: `acc_integrated_model.slx`
- **Bus Definition**: `create_decision_sppvt_bus.m`
- **Test Scripts**: (Deleted after architecture extraction)

### Python Files
- **Entry Point**: `acc_updated.py`
- **Control Manager**: `control_loop_manager.py` (lines 441-480)
- **Torque Converter**: `torque_to_throttle_converter.py` (lines 192-281)
- **Simulink Interface**: `integrated_simulink_manager.py` (lines 510-525)
- **Configuration**: `acc_config.py` (lines 124-132)

---

## Modification History

### 2025-01-09: Unified Torque System
**Changes**:
1. Updated `sppvt_decel_scale` from 1.0 to 210.0
2. Modified `_engine_torque_to_brake_value` method
3. Corrected comments throughout codebase
4. Renamed variables for clarity (`sppvt_target_accel` → `sppvt_torque_demand`)

**Reason**: Supervisor requirement to use torque units for both acceleration and deceleration.

**Impact**: Both acceleration and deceleration now use consistent torque-based control.

---

## Notes for Future Development

1. **Upgrade Condition**: The third condition logic (Relational Operator, Relational Operator1) requires signal tracing to fully document. Current understanding is sufficient for operation.

2. **Detector Subsystem**: Currently empty/placeholder. May be used for future enhancements to upgrade logic.

3. **LUT Data**: Lookup table data stored in MATLAB workspace variables. Requires model workspace inspection for exact transition logic.

4. **Filter Design**: Discrete filters use inherited sample time (-1), actual sampling is 50ms from model configuration.

5. **Scaling Philosophy**: Python-side scaling (×210.0) allows easy tuning without recompiling Simulink model. This is by design for rapid prototyping.

6. **Units Convention**:
   - Simulink SPPVT output: Dimensionless torque demand [-3.0, 2.0]
   - Python after scaling: Engine torque [N·m]
   - CARLA input: Normalized throttle/brake [0, 1.0]

---

## Quick Reference

### Key Equations

**Core Control**:
```
control_output = saturate(kp × (error + offset), [-3.0, 2.0])
```

**Derivative Calculation**:
```
velocity = LPF1((error - prev_error) / dt)
acceleration = LPF2((velocity - prev_velocity) / dt)
jerk = (acceleration - prev_accel) / dt
```

**Upgrade Condition**:
```
should_upgrade = (|error| > 0.3) AND (|velocity| <= 0.05) AND [Detector Logic]
```

**Python Torque Scaling**:
```
engine_torque = control_output × 210.0
```

**Brake Conversion**:
```
brake_value = (|engine_torque| × 9.204) / (4 × 1000)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Model Version**: acc_integrated_model.slx (after_meeting1022 branch)
