/*
 * Code generation for system model 'sppvt_control_model'
 * For more details, see corresponding source file sppvt_control_model.c
 *
 */

#ifndef sppvt_control_model_h_
#define sppvt_control_model_h_
#ifndef sppvt_control_model_COMMON_INCLUDES_
#define sppvt_control_model_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "math.h"
#endif                                /* sppvt_control_model_COMMON_INCLUDES_ */

#include "sppvt_control_model_types.h"

/* Parameters (default storage) */
struct P_sppvt_control_model_T_ {
  real_T Output_Saturation_UpperSat;   /* Expression: 2.0
                                        * Referenced by: '<S1>/Output_Saturation'
                                        */
  real_T Output_Saturation_LowerSat;   /* Expression: -3.0
                                        * Referenced by: '<S1>/Output_Saturation'
                                        */
};

/* Real-time Model Data Structure */
struct tag_RTM_sppvt_control_model_T {
  const char_T **errorStatus;
};

typedef struct {
  RT_MODEL_sppvt_control_model_T rtm;
} MdlrefDW_sppvt_control_model_T;

/* Model reference registration function */
extern void sppvt_control_model_initialize(const char_T **rt_errorStatus,
  RT_MODEL_sppvt_control_model_T *const sppvt_control_model_M);
extern void sppvt_control_model(const real_T *rtu_error_value, const real_T
  *rtu_dt, const real_T *rtu_current_stage_offset, const real_T *rtu_sppvt_kp,
  const real_T *rtu_prev_error, const real_T *rtu_prev_velocity, const real_T
  *rtu_prev_accel, const real_T *rtu_sppvt_eta, real_T *rty_control_output,
  real_T *rty_velocity, real_T *rty_acceleration, real_T *rty_jerk, boolean_T
  *rty_should_upgrade);

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'sppvt_control_model'
 * '<S1>'   : 'sppvt_control_model/Core_Control_Subsystem'
 * '<S2>'   : 'sppvt_control_model/Derivatives_Subsystem'
 * '<S3>'   : 'sppvt_control_model/Upgrade_Condition_Subsystem'
 */
#endif                                 /* sppvt_control_model_h_ */
