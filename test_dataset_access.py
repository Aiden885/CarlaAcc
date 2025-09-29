#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Dataset access functionality
Simplified test to debug the Dataset access issue
"""

import matlab.engine
import matlab
import time
import logging

def test_dataset_access():
    """Test Simulink Dataset access without the full interface"""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        # Start MATLAB engine
        logger.info("Starting MATLAB engine...")
        matlab_eng = matlab.engine.start_matlab()
        logger.info("MATLAB engine started successfully")

        # Create bus definitions
        logger.info("Creating bus definitions...")
        matlab_eng.eval("create_decision_sppvt_bus()", nargout=0)

        # Load model
        model_name = 'ACC_Decision_SPPVT_Integrated'
        logger.info(f"Loading model: {model_name}")
        matlab_eng.eval(f"load_system('{model_name}')", nargout=0)

        # Configure simulation
        dt = 0.05
        matlab_eng.eval(f"set_param('{model_name}', 'SimulationMode', 'normal')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'StopTime', '{dt}')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'SaveOutput', 'on')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'OutputSaveName', 'yout')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'SaveFormat', 'Dataset')", nargout=0)

        # Clear external input
        matlab_eng.eval(f"set_param('{model_name}', 'ExternalInput', '')", nargout=0)
        matlab_eng.eval(f"set_param('{model_name}', 'LoadExternalInput', 'off')", nargout=0)

        logger.info("Running simulation...")
        matlab_eng.eval(f"simOut = sim('{model_name}');", nargout=0)

        # Check if simOut exists
        if not matlab_eng.eval("exist('simOut', 'var')"):
            raise RuntimeError("Simulation failed to generate output")

        logger.info("Simulation completed, analyzing output...")

        # Debug Dataset structure
        matlab_eng.eval("yout = simOut.yout;", nargout=0)
        yout_class = matlab_eng.eval("class(yout)")
        logger.info(f"yout class: {yout_class}")

        is_dataset = matlab_eng.eval("isa(yout, 'Simulink.SimulationData.Dataset')")
        logger.info(f"Is Dataset: {is_dataset}")

        if is_dataset:
            num_elements = int(matlab_eng.eval("yout.numElements"))
            logger.info(f"Dataset has {num_elements} elements")

            if num_elements > 0:
                element_class = matlab_eng.eval("class(yout{1})")
                logger.info(f"Element 1 class: {element_class}")

                # Check if element has Values property
                has_values = matlab_eng.eval("isprop(yout{1}, 'Values')")
                logger.info(f"Element 1 has Values property: {has_values}")

                if has_values:
                    values_class = matlab_eng.eval("class(yout{1}.Values)")
                    logger.info(f"Values class: {values_class}")

                    # Check if Values is a struct
                    is_struct = matlab_eng.eval("isstruct(yout{1}.Values)")
                    logger.info(f"Values is struct: {is_struct}")

                    if is_struct:
                        # Get field names
                        field_names = matlab_eng.eval("fieldnames(yout{1}.Values)")
                        logger.info(f"Field names: {field_names}")

                        # Try to access specific fields
                        try:
                            has_sppvt_control = matlab_eng.eval("isfield(yout{1}.Values, 'sppvt_control_output')")
                            logger.info(f"Has sppvt_control_output field: {has_sppvt_control}")

                            if has_sppvt_control:
                                sppvt_control_class = matlab_eng.eval("class(yout{1}.Values.sppvt_control_output)")
                                logger.info(f"sppvt_control_output class: {sppvt_control_class}")

                                # Try to access the data
                                sppvt_data = matlab_eng.eval("yout{1}.Values.sppvt_control_output.Data")
                                logger.info(f"sppvt_control_output data shape: {matlab_eng.eval('size(yout{1}.Values.sppvt_control_output.Data)')}")
                                logger.info(f"sppvt_control_output data: {sppvt_data}")

                                # Get the final value
                                final_value = float(matlab_eng.eval("yout{1}.Values.sppvt_control_output.Data(end)"))
                                logger.info(f"SUCCESS: Final SPPVT output value: {final_value}")

                        except Exception as field_error:
                            logger.error(f"Field access error: {field_error}")

        logger.info("Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_access()
    exit(0 if success else 1)