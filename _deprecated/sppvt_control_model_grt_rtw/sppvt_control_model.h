/*
 * sppvt_control_model.h
 *
 * Code generation for model "sppvt_control_model".
 *
 * Model version              : 1.6
 * Simulink Coder version : 24.2 (R2024b) 21-Jun-2024
 * C source code generated on : Wed Sep 24 10:14:29 2025
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef sppvt_control_model_h_
#define sppvt_control_model_h_
#ifndef sppvt_control_model_COMMON_INCLUDES_
#define sppvt_control_model_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_logging.h"
#include "rt_nonfinite.h"
#include "math.h"
#endif                                /* sppvt_control_model_COMMON_INCLUDES_ */

#include "sppvt_control_model_types.h"
#include <float.h>
#include <string.h>
#include <stddef.h>

/* Macros for accessing real-time model data structure */
#ifndef rtmGetFinalTime
#define rtmGetFinalTime(rtm)           ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetRTWLogInfo
#define rtmGetRTWLogInfo(rtm)          ((rtm)->rtwLogInfo)
#endif

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   ((rtm)->Timing.taskTime0)
#endif

#ifndef rtmGetTFinal
#define rtmGetTFinal(rtm)              ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                (&(rtm)->Timing.taskTime0)
#endif

/* External inputs (root inport signals with default storage) */
typedef struct {
  real_T error_value;                  /* '<Root>/error_value' */
  real_T dt;                           /* '<Root>/dt' */
  real_T current_stage_offset;         /* '<Root>/current_stage_offset' */
  real_T sppvt_kp;                     /* '<Root>/sppvt_kp' */
  real_T max_accel;                    /* '<Root>/max_accel' */
  real_T max_decel;                    /* '<Root>/max_decel' */
  real_T prev_error;                   /* '<Root>/prev_error' */
  real_T prev_velocity;                /* '<Root>/prev_velocity' */
  real_T prev_accel;                   /* '<Root>/prev_accel' */
  real_T sppvt_delta;                  /* '<Root>/sppvt_delta' */
  real_T sppvt_eta;                    /* '<Root>/sppvt_eta' */
  real_T control_mode_flag;            /* '<Root>/control_mode_flag' */
} ExtU_sppvt_control_model_T;

/* External outputs (root outports fed by signals with default storage) */
typedef struct {
  real_T control_output;               /* '<Root>/control_output' */
  real_T velocity;                     /* '<Root>/velocity' */
  real_T acceleration;                 /* '<Root>/acceleration' */
  real_T jerk;                         /* '<Root>/jerk' */
  boolean_T should_upgrade;            /* '<Root>/should_upgrade' */
} ExtY_sppvt_control_model_T;

/* Parameters (default storage) */
struct P_sppvt_control_model_T_ {
  real_T Output_Saturation_UpperSat;   /* Expression: 2.0
                                        * Referenced by: '<S1>/Output_Saturation'
                                        */
  real_T Output_Saturation_LowerSat;   /* Expression: -3.0
                                        * Referenced by: '<S1>/Output_Saturation'
                                        */
  real_T Constant_Value;               /* Expression: 0
                                        * Referenced by: '<S3>/Constant'
                                        */
};

/* Real-time Model Data Structure */
struct tag_RTM_sppvt_control_model_T {
  const char_T *errorStatus;
  RTWLogInfo *rtwLogInfo;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    time_T taskTime0;
    uint32_T clockTick0;
    uint32_T clockTickH0;
    time_T stepSize0;
    time_T tFinal;
    boolean_T stopRequestedFlag;
  } Timing;
};

/* Block parameters (default storage) */
extern P_sppvt_control_model_T sppvt_control_model_P;

/* External inputs (root inport signals with default storage) */
extern ExtU_sppvt_control_model_T sppvt_control_model_U;

/* External outputs (root outports fed by signals with default storage) */
extern ExtY_sppvt_control_model_T sppvt_control_model_Y;

/* Model entry point functions */
extern void sppvt_control_model_initialize(void);
extern void sppvt_control_model_step(void);
extern void sppvt_control_model_terminate(void);

/* Real-time Model object */
extern RT_MODEL_sppvt_control_model_T *const sppvt_control_model_M;

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
