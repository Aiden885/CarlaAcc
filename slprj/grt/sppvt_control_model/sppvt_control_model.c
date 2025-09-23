/*
 * Code generation for system model 'sppvt_control_model'
 *
 * Model                      : sppvt_control_model
 * Model version              : 1.2
 * Simulink Coder version : 24.2 (R2024b) 21-Jun-2024
 * C source code generated on : Fri Sep 19 09:50:33 2025
 *
 * Note that the functions contained in this file are part of a Simulink
 * model, and are not self-contained algorithms.
 */

#include "sppvt_control_model.h"
#include "rtwtypes.h"
#include "sppvt_control_model_private.h"
#include <math.h>

P_sppvt_control_model_T sppvt_control_model_P = {
  /* Expression: 2.0
   * Referenced by: '<S1>/Output_Saturation'
   */
  2.0,

  /* Expression: -3.0
   * Referenced by: '<S1>/Output_Saturation'
   */
  -3.0
};

/* Output and update for referenced model: 'sppvt_control_model' */
void sppvt_control_model(const real_T *rtu_error_value, const real_T *rtu_dt,
  const real_T *rtu_current_stage_offset, const real_T *rtu_sppvt_kp, const
  real_T *rtu_prev_error, const real_T *rtu_prev_velocity, const real_T
  *rtu_prev_accel, const real_T *rtu_sppvt_eta, real_T *rty_control_output,
  real_T *rty_velocity, real_T *rty_acceleration, real_T *rty_jerk, boolean_T
  *rty_should_upgrade)
{
  real_T rtb_Enhanced_Error;

  /* Sum: '<S1>/Enhanced_Error' */
  rtb_Enhanced_Error = *rtu_error_value + *rtu_current_stage_offset;

  /* Product: '<S1>/Proportional_Control' */
  rtb_Enhanced_Error *= *rtu_sppvt_kp;

  /* Saturate: '<S1>/Output_Saturation' */
  if (rtb_Enhanced_Error > sppvt_control_model_P.Output_Saturation_UpperSat) {
    *rty_control_output = sppvt_control_model_P.Output_Saturation_UpperSat;
  } else if (rtb_Enhanced_Error <
             sppvt_control_model_P.Output_Saturation_LowerSat) {
    *rty_control_output = sppvt_control_model_P.Output_Saturation_LowerSat;
  } else {
    *rty_control_output = rtb_Enhanced_Error;
  }

  /* End of Saturate: '<S1>/Output_Saturation' */

  /* Sum: '<S2>/Error_Diff' */
  rtb_Enhanced_Error = *rtu_error_value - *rtu_prev_error;

  /* Product: '<S2>/Velocity_Calc' */
  *rty_velocity = rtb_Enhanced_Error / *rtu_dt;

  /* Sum: '<S2>/Velocity_Diff' */
  rtb_Enhanced_Error = *rty_velocity - *rtu_prev_velocity;

  /* Product: '<S2>/Acceleration_Calc' */
  *rty_acceleration = rtb_Enhanced_Error / *rtu_dt;

  /* Sum: '<S2>/Acceleration_Diff' */
  rtb_Enhanced_Error = *rty_acceleration - *rtu_prev_accel;

  /* Product: '<S2>/Jerk_Calc' */
  *rty_jerk = rtb_Enhanced_Error / *rtu_dt;

  /* Abs: '<S3>/Abs_error' */
  rtb_Enhanced_Error = fabs(*rtu_error_value);

  /* RelationalOperator: '<S3>/Compare_eta' */
  *rty_should_upgrade = (rtb_Enhanced_Error > *rtu_sppvt_eta);
}

/* Model initialize function */
void sppvt_control_model_initialize(const char_T **rt_errorStatus,
  RT_MODEL_sppvt_control_model_T *const sppvt_control_model_M)
{
  /* Registration code */

  /* initialize error status */
  rtmSetErrorStatusPointer(sppvt_control_model_M, rt_errorStatus);
}
