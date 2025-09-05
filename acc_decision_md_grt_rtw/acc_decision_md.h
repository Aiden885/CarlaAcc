/*
 * acc_decision_md.h
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "acc_decision_md".
 *
 * Model version              : 1.2
 * Simulink Coder version : 24.2 (R2024b) 21-Jun-2024
 * C source code generated on : Tue Sep  2 11:32:40 2025
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef acc_decision_md_h_
#define acc_decision_md_h_
#ifndef acc_decision_md_COMMON_INCLUDES_
#define acc_decision_md_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_logging.h"
#include "rt_nonfinite.h"
#include "math.h"
#endif                                 /* acc_decision_md_COMMON_INCLUDES_ */

#include "acc_decision_md_types.h"
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

/* Block states (default storage) for system '<Root>' */
typedef struct {
  real_T history_V3_kmh;               /* '<Root>/ACC_Decision_Chart' */
  real_T history_G1_m;                 /* '<Root>/ACC_Decision_Chart' */
  real_T history_G2_s;                 /* '<Root>/ACC_Decision_Chart' */
  real_T pending_adj;                  /* '<Root>/ACC_Decision_Chart' */
  uint8_T is_active_c3_acc_decision_md;/* '<Root>/ACC_Decision_Chart' */
  uint8_T is_during_;                  /* '<Root>/ACC_Decision_Chart' */
} DW_acc_decision_md_T;

/* External inputs (root inport signals with default storage) */
typedef struct {
  real_T command_input;                /* '<Root>/command_input' */
  real_T ego_speed_kmh;                /* '<Root>/ego_speed_kmh' */
  boolean_T has_target;                /* '<Root>/has_target' */
  real_T current_distance;             /* '<Root>/current_distance' */
  boolean_T reset_signal;              /* '<Root>/reset_signal' */
} ExtU_acc_decision_md_T;

/* External outputs (root outports fed by signals with default storage) */
typedef struct {
  real_T current_state;                /* '<Root>/current_state' */
  real_T control_mode;                 /* '<Root>/control_mode' */
  boolean_T acc_active;                /* '<Root>/acc_active' */
  boolean_T control_enabled;           /* '<Root>/control_enabled' */
  real_T V3_kmh;                       /* '<Root>/V3_kmh' */
  real_T G1_m;                         /* '<Root>/G1_m' */
  real_T G2_s;                         /* '<Root>/G2_s' */
  boolean_T has_history;               /* '<Root>/has_history' */
  real_T message_code;                 /* '<Root>/message_code' */
  real_T pending_distance_adj;         /* '<Root>/pending_distance_adj' */
  boolean_T is_in_control;             /* '<Root>/is_in_control' */
} ExtY_acc_decision_md_T;

/* Real-time Model Data Structure */
struct tag_RTM_acc_decision_md_T {
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

/* Block states (default storage) */
extern DW_acc_decision_md_T acc_decision_md_DW;

/* External inputs (root inport signals with default storage) */
extern ExtU_acc_decision_md_T acc_decision_md_U;

/* External outputs (root outports fed by signals with default storage) */
extern ExtY_acc_decision_md_T acc_decision_md_Y;

/* Model entry point functions */
extern void acc_decision_md_initialize(void);
extern void acc_decision_md_step(void);
extern void acc_decision_md_terminate(void);

/* Real-time Model object */
extern RT_MODEL_acc_decision_md_T *const acc_decision_md_M;

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
 * '<Root>' : 'acc_decision_md'
 * '<S1>'   : 'acc_decision_md/ACC_Decision_Chart'
 */
#endif                                 /* acc_decision_md_h_ */
