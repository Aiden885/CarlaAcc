#!/usr/bin/env python3
"""
Performance comparison: simple vs complex external input
Test to identify the performance bottleneck
"""

import matlab.engine
import matlab
import time
import logging

def test_simple_simulation():
    """Test simple simulation without external input (like test_dataset_access.py)"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        print("=== Test 1: Simple simulation (no external input) ===")
        start_time = time.time()

        # Start MATLAB engine
        matlab_eng = matlab.engine.start_matlab()

        # Create bus definitions
        matlab_eng.eval("create_decision_sppvt_bus()", nargout=0)

        # Load model
        model_name = 'ACC_Decision_SPPVT_Integrated'
        matlab_eng.eval(f"load_system('{model_name}')", nargout=0)

        # Configure simulation
        dt = 0.05
        matlab_eng.eval(f"set_param('{model_name}', 'SimulationMode', 'normal')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'StopTime', '{dt}')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'SaveOutput', 'on')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'OutputSaveName', 'yout')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'SaveFormat', 'Dataset')", nargout=0)

        # Clear external input - use default model behavior
        matlab_eng.eval(f"set_param('{model_name}', 'ExternalInput', '')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'LoadExternalInput', 'off')", nargout=0)

        setup_time = time.time() - start_time
        print(f"Setup time: {setup_time*1000:.1f}ms")

        # Run simulation
        sim_start = time.time()
        matlab_eng.eval(f"simOut = sim('{model_name}');", nargout=0)
        sim_time = time.time() - sim_start

        # Extract output
        extract_start = time.time()
        final_value = float(matlab_eng.eval("simOut.yout{1}.Values.sppvt_control_output.Data(end)"))
        extract_time = time.time() - extract_start

        total_time = time.time() - start_time

        print(f"✅ Simple simulation results:")
        print(f"   Setup: {setup_time*1000:.1f}ms")
        print(f"   Simulation: {sim_time*1000:.1f}ms")
        print(f"   Extraction: {extract_time*1000:.1f}ms")
        print(f"   Total: {total_time*1000:.1f}ms")
        print(f"   SPPVT output: {final_value}")

        return matlab_eng, total_time

    except Exception as e:
        print(f"❌ Simple simulation failed: {e}")
        return None, None

def test_complex_simulation(matlab_eng):
    """Test complex simulation with external input (like realtime manager)"""
    try:
        print("\n=== Test 2: Complex simulation (with external input) ===")
        start_time = time.time()

        model_name = 'ACC_Decision_SPPVT_Integrated'
        dt = 0.05

        # Create complex external input data (like realtime manager)
        time_points = matlab.double([0.0, dt])
        control_error = 1.5
        ego_speed_ms = 13.89
        control_enabled = True
        control_mode_flag = True

        # Create input data dict with 14 timeseries objects
        input_data_dict = {
            'ego_speed_kmh': matlab_eng.timeseries(
                matlab.double([50.0, 50.0]), time_points, 'Name', 'ego_speed_kmh'
            ),
            'ego_speed_ms': matlab_eng.timeseries(
                matlab.double([ego_speed_ms, ego_speed_ms]), time_points, 'Name', 'ego_speed_ms'
            ),
            'control_error': matlab_eng.timeseries(
                matlab.double([control_error, control_error]), time_points, 'Name', 'control_error'
            ),
            'V_target_kmh': matlab_eng.timeseries(
                matlab.double([50.0, 50.0]), time_points, 'Name', 'V_target_kmh'
            ),
            'V_min_kmh': matlab_eng.timeseries(
                matlab.double([30.0, 30.0]), time_points, 'Name', 'V_min_kmh'
            ),
            'G2_s': matlab_eng.timeseries(
                matlab.double([2.0, 2.0]), time_points, 'Name', 'G2_s'
            ),
            'timestamp': matlab_eng.timeseries(
                time_points, time_points, 'Name', 'timestamp'
            ),
            'command_type': matlab_eng.timeseries(
                matlab.int32([1, 1]), time_points, 'Name', 'command_type'
            ),
            'control_mode_flag': matlab_eng.timeseries(
                matlab.int32([1, 1]), time_points, 'Name', 'control_mode_flag'
            ),
            'command_active': matlab_eng.timeseries(
                matlab.logical([control_enabled, control_enabled]), time_points, 'Name', 'command_active'
            ),
            'manual_throttle_active': matlab_eng.timeseries(
                matlab.logical([False, False]), time_points, 'Name', 'manual_throttle_active'
            ),
            'external_stage_offset': matlab_eng.timeseries(
                matlab.double([0.0, 0.0]), time_points, 'Name', 'external_stage_offset'
            ),
            'external_stage_manager_states': matlab_eng.timeseries(
                matlab.double([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), time_points, 'Name', 'external_stage_manager_states'
            ),
            'external_adapter_states': matlab_eng.timeseries(
                matlab.double([[0.0, 13.89, 0.1], [0.0, 13.89, 0.1]]), time_points, 'Name', 'external_adapter_states'
            )
        }

        data_creation_time = time.time() - start_time
        print(f"Data creation time: {data_creation_time*1000:.1f}ms")

        # Set time units
        units_start = time.time()
        matlab_eng.workspace['input_data_temp'] = input_data_dict
        matlab_eng.eval("""
        field_names = fieldnames(input_data_temp);
        for i = 1:length(field_names)
            input_data_temp.(field_names{i}).TimeInfo.Units = 'seconds';
        end
        """, nargout=0)
        input_data_dict = matlab_eng.workspace['input_data_temp']
        units_time = time.time() - units_start
        print(f"Time units setting: {units_time*1000:.1f}ms")

        # Put into workspace
        workspace_start = time.time()
        matlab_eng.workspace['input_data'] = input_data_dict
        workspace_time = time.time() - workspace_start
        print(f"Workspace assignment: {workspace_time*1000:.1f}ms")

        # Configure external input
        config_start = time.time()
        matlab_eng.eval(f"set_param('{model_name}', 'ExternalInput', 'input_data')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'LoadExternalInput', 'on')", nargout=0)
        config_time = time.time() - config_start
        print(f"Config time: {config_time*1000:.1f}ms")

        # Run simulation
        sim_start = time.time()
        matlab_eng.eval(f"simOut = sim('{model_name}');", nargout=0)
        sim_time = time.time() - sim_start

        # Extract output
        extract_start = time.time()
        final_value = float(matlab_eng.eval("simOut.yout{1}.Values.sppvt_control_output.Data(end)"))
        extract_time = time.time() - extract_start

        total_time = time.time() - start_time

        print(f"✅ Complex simulation results:")
        print(f"   Data creation: {data_creation_time*1000:.1f}ms")
        print(f"   Time units: {units_time*1000:.1f}ms")
        print(f"   Workspace: {workspace_time*1000:.1f}ms")
        print(f"   Config: {config_time*1000:.1f}ms")
        print(f"   Simulation: {sim_time*1000:.1f}ms")
        print(f"   Extraction: {extract_time*1000:.1f}ms")
        print(f"   Total: {total_time*1000:.1f}ms")
        print(f"   SPPVT output: {final_value}")

        return total_time

    except Exception as e:
        print(f"❌ Complex simulation failed: {e}")
        return None

if __name__ == "__main__":
    # Test simple simulation
    matlab_eng, simple_time = test_simple_simulation()

    if matlab_eng and simple_time:
        # Test complex simulation
        complex_time = test_complex_simulation(matlab_eng)

        if complex_time:
            print(f"\n=== Performance Comparison ===")
            print(f"Simple simulation: {simple_time*1000:.1f}ms")
            print(f"Complex simulation: {complex_time*1000:.1f}ms")
            print(f"Overhead ratio: {complex_time/simple_time:.1f}x")

            if complex_time > 5.0:  # > 5 seconds
                print("🚨 Complex simulation is extremely slow - external input is the bottleneck!")
            elif simple_time > 5.0:
                print("🚨 Even simple simulation is slow - Simulink model itself is the bottleneck!")
            else:
                print("✅ Both simulations are reasonable - bottleneck is elsewhere")