# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CarlaAcc is an Adaptive Cruise Control (ACC) system built on the CARLA simulation platform. The project uses a hybrid Python-MATLAB/Simulink architecture to implement complete ACC functionality including perception, decision-making, control, and simulation validation.

## Key Architecture Components

- **System Integration**: `acc_updated.py` (945 lines) serves as the main system integrator, handling CARLA environment management, sensor integration, user interface, and the main control loop
- **Decision Module**: `acc_decision.py` implements a comprehensive ACC state machine with 7 states and 9 commands, supporting both adaptive cruise control and manual cruise control modes
- **Control Algorithms**: 
  - `sppvt_longitudinal_control.py` - SPPVT (Self-Paced Policy Value Target) longitudinal control with Python fallback when MATLAB unavailable
  - `three_mode_controller.py` - Three-phase control strategy (distance, time, speed control modes)
- **Perception**: Lane detection (`lane_detection.py`), radar clustering (`radar_cluster.py`), and Kalman filtering (`kalman_filter.py`)
- **MATLAB Integration**: Simulink models and Stateflow charts for control algorithms, with Python-MATLAB communication via `matlab_connect.py`

## Running the System

### Prerequisites
- CARLA simulator 0.9.14 running on localhost:2000
- MATLAB with Simulink (optional - system has Python fallbacks)
- Required Python packages: carla, opencv-python, numpy, pygame

### Main Execution
```bash
python acc_updated.py
```

The main system will:
1. Initialize CARLA environment with Town05 map
2. Set up ego vehicle and target vehicles
3. Configure sensors (camera, radar)
4. Start the main control loop with keyboard input handling

### Keyboard Controls
- **Space**: Engage ACC
- **C**: Switch to cruise-only mode  
- **Arrow Keys**: Adjust target speed and following distance
- **ESC**: Exit system
- **M**: Toggle manual control override

## MATLAB/Simulink Integration

### Creating Stateflow Models
```matlab
% Create the base model structure
create_acc_stateflow.m

% Configure state machine logic (replicates Python decision module)
configure_chart_logic.m  

% Fix signal connections if needed
fix_stateflow_connections.m
```

### Model Testing
```matlab
% Simple integration test
test_stateflow_simple.py

% Full integration test  
test_stateflow_integration.py
```

## Development Notes

### Code Architecture Principles
The system follows modular design with clear separation of concerns:
- **High cohesion modules**: Decision logic, control algorithms, perception modules are well-contained
- **System integration**: `acc_updated.py` reasonably handles integration responsibilities
- **Hybrid approach**: Python provides core functionality with MATLAB extensions for advanced control algorithms

### Key Module Interactions
```
acc_updated.py (System Integrator)
├── Depends on: lane_detection.py, kalman_filter.py, radar_cluster.py
├── Depends on: acc_decision.py (Complete decision module)  
├── Depends on: acc_planning_control.py
├── Depends on: three_mode_controller.py, sppvt_longitudinal_control.py
└── Depends on: display_manager.py

acc_decision.py (Decision Module)
└── Depends on: three_mode_controller.py

acc_planning_control.py (Needs refactoring - multiple responsibilities)
├── Depends on: sppvt_longitudinal_control.py  
└── Depends on: three_mode_controller.py
```

### Testing Approach
- Individual module testing via `test_*.py` files
- Integration testing through `cruisetest.py`
- CARLA environment validation via `verify_carla.py`
- Manual testing through main system execution

### Configuration Management
Key parameters are currently embedded in code. For modifications:
- ACC decision parameters: `acc_decision.py` ACCDecisionModule initialization
- Three-mode control parameters: `three_mode_controller.py` set_three_mode_parameters()
- SPPVT control parameters: `sppvt_longitudinal_control.py` class initialization
- CARLA environment settings: `acc_updated.py` init_carla() method

### Data Flow
```
Sensor Data → Perception Modules → Environmental State
                                  ↓
User Commands → Decision Module → Control Commands  
                                  ↓
Control Commands → Control Module → Vehicle Control
                                  ↓
Vehicle Control → CARLA Environment → Simulation Feedback
```

### Known Architecture Notes
- `acc_planning_control.py` handles multiple responsibilities and is a candidate for refactoring into separate modules
- The system supports both Python-only operation and Python+MATLAB hybrid modes
- Display management and user interface are integrated into the main system file but could be further modularized
- The decision module supports both immediate execution and historical state management for seamless mode transitions

## File Documentation Reference

See `PROJECT_STRUCTURE_ANALYSIS.md` for comprehensive analysis of all modules, coupling relationships, and architectural improvement recommendations.