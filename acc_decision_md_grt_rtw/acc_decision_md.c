/*
 * acc_decision_md.c
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

#include "acc_decision_md.h"
#include "rtwtypes.h"
#include <math.h>
#include <string.h>
#include "acc_decision_md_private.h"

/* Named constants for Chart: '<Root>/ACC_Decision_Chart' */
#define IN_S1_ADAPTIVE_HISTORY_STANDBY ((uint8_T)2U)
#define IN_S2_ADAPTIVE_NO_HISTORY_STAND ((uint8_T)3U)
#define acc_decision_IN_NO_ACTIVE_CHILD ((uint8_T)0U)
#define acc_decision_m_IN_S0_IN_CONTROL ((uint8_T)1U)
#define acc_decision_md_DISTANCE_STEP  (1.0)
#define acc_decision_md_IN_S3_LOW_SPEED ((uint8_T)4U)
#define acc_decision_md_SPEED_STEP     (1.0)
#define acc_decision_md_V1_KMH         (0.0)

/* Block states (default storage) */
DW_acc_decision_md_T acc_decision_md_DW;

/* External inputs (root inport signals with default storage) */
ExtU_acc_decision_md_T acc_decision_md_U;

/* External outputs (root outports fed by signals with default storage) */
ExtY_acc_decision_md_T acc_decision_md_Y;

/* Real-time model */
static RT_MODEL_acc_decision_md_T acc_decision_md_M_;
RT_MODEL_acc_decision_md_T *const acc_decision_md_M = &acc_decision_md_M_;

/* Forward declaration for local functions */
static void acc_decisi_reset_all_parameters(void);
static void acc_decision_md_save_history(void);
static void acc_decision_md_restore_history(void);

/* Function for Chart: '<Root>/ACC_Decision_Chart' */
static void acc_decisi_reset_all_parameters(void)
{
  /* Outport: '<Root>/has_history' */
  acc_decision_md_Y.has_history = false;
  acc_decision_md_DW.history_V3_kmh = 50.0;
  acc_decision_md_DW.history_G1_m = 15.0;
  acc_decision_md_DW.history_G2_s = 2.0;
  acc_decision_md_DW.pending_adj = 0.0;

  /* Outport: '<Root>/V3_kmh' */
  acc_decision_md_Y.V3_kmh = 50.0;

  /* Outport: '<Root>/G1_m' */
  acc_decision_md_Y.G1_m = 15.0;

  /* SystemInitialize for Outport: '<Root>/G2_s' */
  acc_decision_md_Y.G2_s = 2.0;

  /* Outport: '<Root>/control_mode' */
  acc_decision_md_Y.control_mode = 0.0;

  /* Outport: '<Root>/message_code' */
  acc_decision_md_Y.message_code = 0.0;
}

/* Function for Chart: '<Root>/ACC_Decision_Chart' */
static void acc_decision_md_save_history(void)
{
  /* Outport: '<Root>/has_history' */
  acc_decision_md_Y.has_history = true;

  /* Outport: '<Root>/V3_kmh' */
  acc_decision_md_DW.history_V3_kmh = acc_decision_md_Y.V3_kmh;

  /* Outport: '<Root>/G1_m' */
  acc_decision_md_DW.history_G1_m = acc_decision_md_Y.G1_m;

  /* SystemInitialize for Outport: '<Root>/G2_s' */
  acc_decision_md_DW.history_G2_s = acc_decision_md_Y.G2_s;
}

/* Function for Chart: '<Root>/ACC_Decision_Chart' */
static void acc_decision_md_restore_history(void)
{
  /* Outport: '<Root>/has_history' incorporates:
   *  Outport: '<Root>/G1_m'
   *  Outport: '<Root>/G2_s'
   *  Outport: '<Root>/V3_kmh'
   */
  if (acc_decision_md_Y.has_history) {
    acc_decision_md_Y.V3_kmh = acc_decision_md_DW.history_V3_kmh;
    acc_decision_md_Y.G1_m = acc_decision_md_DW.history_G1_m;
    acc_decision_md_Y.G2_s = acc_decision_md_DW.history_G2_s;
    if (acc_decision_md_DW.pending_adj != 0.0) {
      acc_decision_md_Y.G1_m = fmax(5.0, acc_decision_md_Y.G1_m +
        acc_decision_md_DW.pending_adj);
      acc_decision_md_DW.pending_adj = 0.0;
    }
  }

  /* End of Outport: '<Root>/has_history' */
}

/* Model step function */
void acc_decision_md_step(void)
{
  boolean_T guard1;

  /* Chart: '<Root>/ACC_Decision_Chart' incorporates:
   *  Inport: '<Root>/command_input'
   *  Inport: '<Root>/ego_speed_kmh'
   *  Inport: '<Root>/reset_signal'
   *  Outport: '<Root>/G1_m'
   *  Outport: '<Root>/V3_kmh'
   *  Outport: '<Root>/has_history'
   */
  if (acc_decision_md_DW.is_active_c3_acc_decision_md == 0) {
    acc_decision_md_DW.is_active_c3_acc_decision_md = 1U;

    /* Outport: '<Root>/pending_distance_adj' */
    acc_decision_md_Y.pending_distance_adj = acc_decision_md_DW.pending_adj;
    if (acc_decision_md_U.reset_signal) {
      acc_decisi_reset_all_parameters();
    }

    if (acc_decision_md_Y.V3_kmh <= 0.0) {
      acc_decision_md_Y.V3_kmh = 50.0;
    }

    if (acc_decision_md_Y.G1_m <= 0.0) {
      acc_decision_md_Y.G1_m = 15.0;
    }

    acc_decision_md_DW.is_during_ = IN_S2_ADAPTIVE_NO_HISTORY_STAND;

    /* Outport: '<Root>/current_state' incorporates:
     *  Inport: '<Root>/reset_signal'
     *  Outport: '<Root>/G1_m'
     *  Outport: '<Root>/V3_kmh'
     */
    acc_decision_md_Y.current_state = 2.0;

    /* Outport: '<Root>/acc_active' */
    acc_decision_md_Y.acc_active = false;

    /* Outport: '<Root>/control_enabled' */
    acc_decision_md_Y.control_enabled = false;

    /* Outport: '<Root>/is_in_control' */
    acc_decision_md_Y.is_in_control = false;

    /* Outport: '<Root>/has_history' */
    acc_decision_md_Y.has_history = false;
  } else {
    /* Outport: '<Root>/pending_distance_adj' */
    acc_decision_md_Y.pending_distance_adj = acc_decision_md_DW.pending_adj;
    if (acc_decision_md_U.reset_signal) {
      acc_decisi_reset_all_parameters();
    }

    if (acc_decision_md_Y.V3_kmh <= 0.0) {
      acc_decision_md_Y.V3_kmh = 50.0;
    }

    if (acc_decision_md_Y.G1_m <= 0.0) {
      acc_decision_md_Y.G1_m = 15.0;
    }

    guard1 = false;
    switch (acc_decision_md_DW.is_during_) {
     case acc_decision_m_IN_S0_IN_CONTROL:
      /* Outport: '<Root>/current_state' */
      acc_decision_md_Y.current_state = 0.0;

      /* Outport: '<Root>/acc_active' */
      acc_decision_md_Y.acc_active = true;

      /* Outport: '<Root>/control_enabled' */
      acc_decision_md_Y.control_enabled = true;

      /* Outport: '<Root>/is_in_control' */
      acc_decision_md_Y.is_in_control = true;
      if (acc_decision_md_U.command_input == 5.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 8.0;
        acc_decision_md_save_history();

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 108.0;
        guard1 = true;
      } else if (acc_decision_md_U.command_input == 6.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 8.0;
        acc_decision_md_save_history();

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 109.0;
        guard1 = true;
      } else if (acc_decision_md_U.command_input == 0.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 1.0;
        acc_decision_md_Y.V3_kmh = fmax(31.0, acc_decision_md_Y.V3_kmh -
          acc_decision_md_SPEED_STEP);

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 101.0;
      } else if (acc_decision_md_U.command_input == 1.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 2.0;
        acc_decision_md_Y.V3_kmh = fmin(120.0, acc_decision_md_Y.V3_kmh +
          acc_decision_md_SPEED_STEP);

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 102.0;
      } else if (acc_decision_md_U.command_input == 2.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 3.0;
        acc_decision_md_Y.G1_m = fmax(5.0, acc_decision_md_Y.G1_m -
          acc_decision_md_DISTANCE_STEP);

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 103.0;
      } else if (acc_decision_md_U.command_input == 3.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 4.0;
        acc_decision_md_Y.G1_m = fmin(50.0, acc_decision_md_Y.G1_m +
          acc_decision_md_DISTANCE_STEP);

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 104.0;
      } else if (acc_decision_md_U.command_input == 4.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 7.0;

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 107.0;
      }
      break;

     case IN_S1_ADAPTIVE_HISTORY_STANDBY:
      /* Outport: '<Root>/current_state' */
      acc_decision_md_Y.current_state = 1.0;

      /* Outport: '<Root>/acc_active' */
      acc_decision_md_Y.acc_active = false;

      /* Outport: '<Root>/control_enabled' */
      acc_decision_md_Y.control_enabled = false;

      /* Outport: '<Root>/is_in_control' */
      acc_decision_md_Y.is_in_control = false;
      if (acc_decision_md_U.command_input == 1.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 6.0;
        acc_decision_md_restore_history();

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 206.0;
        acc_decision_md_DW.is_during_ = acc_decision_m_IN_S0_IN_CONTROL;

        /* Outport: '<Root>/current_state' */
        acc_decision_md_Y.current_state = 0.0;

        /* Outport: '<Root>/acc_active' */
        acc_decision_md_Y.acc_active = true;

        /* Outport: '<Root>/control_enabled' */
        acc_decision_md_Y.control_enabled = true;

        /* Outport: '<Root>/is_in_control' */
        acc_decision_md_Y.is_in_control = true;
      } else if (acc_decision_md_U.ego_speed_kmh < acc_decision_md_V1_KMH) {
        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 999.0;
        acc_decision_md_DW.is_during_ = acc_decision_md_IN_S3_LOW_SPEED;

        /* Outport: '<Root>/current_state' */
        acc_decision_md_Y.current_state = 3.0;
      } else if (acc_decision_md_U.command_input == 0.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 5.0;

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 205.0;
      } else if ((acc_decision_md_U.command_input >= 2.0) &&
                 (acc_decision_md_U.command_input <= 6.0)) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 8.0;

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = acc_decision_md_U.command_input + 200.0;
      }
      break;

     case IN_S2_ADAPTIVE_NO_HISTORY_STAND:
      /* Outport: '<Root>/current_state' */
      acc_decision_md_Y.current_state = 2.0;

      /* Outport: '<Root>/acc_active' */
      acc_decision_md_Y.acc_active = false;

      /* Outport: '<Root>/control_enabled' */
      acc_decision_md_Y.control_enabled = false;

      /* Outport: '<Root>/is_in_control' */
      acc_decision_md_Y.is_in_control = false;
      if (acc_decision_md_U.ego_speed_kmh < acc_decision_md_V1_KMH) {
        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 999.0;
        acc_decision_md_DW.is_during_ = acc_decision_md_IN_S3_LOW_SPEED;

        /* Outport: '<Root>/current_state' */
        acc_decision_md_Y.current_state = 3.0;
      } else if (acc_decision_md_U.command_input == 0.0) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 5.0;

        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 305.0;
      } else if ((acc_decision_md_U.command_input >= 1.0) &&
                 (acc_decision_md_U.command_input <= 6.0)) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 8.0;

        /* Outport: '<Root>/message_code' incorporates:
         *  Inport: '<Root>/command_input'
         */
        acc_decision_md_Y.message_code = acc_decision_md_U.command_input + 300.0;
      }
      break;

     default:
      /* Outport: '<Root>/current_state' */
      /* case IN_S3_LOW_SPEED: */
      acc_decision_md_Y.current_state = 3.0;

      /* Outport: '<Root>/acc_active' */
      acc_decision_md_Y.acc_active = false;

      /* Outport: '<Root>/control_enabled' */
      acc_decision_md_Y.control_enabled = false;

      /* Outport: '<Root>/is_in_control' */
      acc_decision_md_Y.is_in_control = false;
      if ((acc_decision_md_U.ego_speed_kmh >= acc_decision_md_V1_KMH) &&
          acc_decision_md_Y.has_history) {
        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 901.0;
        acc_decision_md_DW.is_during_ = IN_S1_ADAPTIVE_HISTORY_STANDBY;

        /* Outport: '<Root>/current_state' */
        acc_decision_md_Y.current_state = 1.0;
      } else if ((acc_decision_md_U.ego_speed_kmh >= acc_decision_md_V1_KMH) &&
                 (!acc_decision_md_Y.has_history)) {
        /* Outport: '<Root>/message_code' */
        acc_decision_md_Y.message_code = 902.0;
        acc_decision_md_DW.is_during_ = IN_S2_ADAPTIVE_NO_HISTORY_STAND;

        /* Outport: '<Root>/current_state' */
        acc_decision_md_Y.current_state = 2.0;
      } else if ((acc_decision_md_U.command_input >= 0.0) &&
                 (acc_decision_md_U.command_input <= 6.0) &&
                 (acc_decision_md_U.ego_speed_kmh < acc_decision_md_V1_KMH)) {
        /* Outport: '<Root>/control_mode' */
        acc_decision_md_Y.control_mode = 8.0;

        /* Outport: '<Root>/message_code' incorporates:
         *  Inport: '<Root>/command_input'
         */
        acc_decision_md_Y.message_code = acc_decision_md_U.command_input + 400.0;
      }
      break;
    }

    if (guard1) {
      acc_decision_md_DW.is_during_ = IN_S1_ADAPTIVE_HISTORY_STANDBY;

      /* Outport: '<Root>/current_state' */
      acc_decision_md_Y.current_state = 1.0;

      /* Outport: '<Root>/acc_active' */
      acc_decision_md_Y.acc_active = false;

      /* Outport: '<Root>/control_enabled' */
      acc_decision_md_Y.control_enabled = false;

      /* Outport: '<Root>/is_in_control' */
      acc_decision_md_Y.is_in_control = false;
    }
  }

  /* End of Chart: '<Root>/ACC_Decision_Chart' */

  /* Matfile logging */
  rt_UpdateTXYLogVars(acc_decision_md_M->rtwLogInfo,
                      (&acc_decision_md_M->Timing.taskTime0));

  /* signal main to stop simulation */
  {                                    /* Sample time: [0.01s, 0.0s] */
    if ((rtmGetTFinal(acc_decision_md_M)!=-1) &&
        !((rtmGetTFinal(acc_decision_md_M)-acc_decision_md_M->Timing.taskTime0) >
          acc_decision_md_M->Timing.taskTime0 * (DBL_EPSILON))) {
      rtmSetErrorStatus(acc_decision_md_M, "Simulation finished");
    }
  }

  /* Update absolute time for base rate */
  /* The "clockTick0" counts the number of times the code of this task has
   * been executed. The absolute time is the multiplication of "clockTick0"
   * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
   * overflow during the application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick0 and the high bits
   * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
   */
  if (!(++acc_decision_md_M->Timing.clockTick0)) {
    ++acc_decision_md_M->Timing.clockTickH0;
  }

  acc_decision_md_M->Timing.taskTime0 = acc_decision_md_M->Timing.clockTick0 *
    acc_decision_md_M->Timing.stepSize0 + acc_decision_md_M->Timing.clockTickH0 *
    acc_decision_md_M->Timing.stepSize0 * 4294967296.0;
}

/* Model initialize function */
void acc_decision_md_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)acc_decision_md_M, 0,
                sizeof(RT_MODEL_acc_decision_md_T));
  rtmSetTFinal(acc_decision_md_M, 0.1);
  acc_decision_md_M->Timing.stepSize0 = 0.01;

  /* Setup for data logging */
  {
    static RTWLogInfo rt_DataLoggingInfo;
    rt_DataLoggingInfo.loggingInterval = (NULL);
    acc_decision_md_M->rtwLogInfo = &rt_DataLoggingInfo;
  }

  /* Setup for data logging */
  {
    rtliSetLogXSignalInfo(acc_decision_md_M->rtwLogInfo, (NULL));
    rtliSetLogXSignalPtrs(acc_decision_md_M->rtwLogInfo, (NULL));
    rtliSetLogT(acc_decision_md_M->rtwLogInfo, "tout");
    rtliSetLogX(acc_decision_md_M->rtwLogInfo, "");
    rtliSetLogXFinal(acc_decision_md_M->rtwLogInfo, "");
    rtliSetLogVarNameModifier(acc_decision_md_M->rtwLogInfo, "rt_");
    rtliSetLogFormat(acc_decision_md_M->rtwLogInfo, 4);
    rtliSetLogMaxRows(acc_decision_md_M->rtwLogInfo, 0);
    rtliSetLogDecimation(acc_decision_md_M->rtwLogInfo, 1);
    rtliSetLogY(acc_decision_md_M->rtwLogInfo, "");
    rtliSetLogYSignalInfo(acc_decision_md_M->rtwLogInfo, (NULL));
    rtliSetLogYSignalPtrs(acc_decision_md_M->rtwLogInfo, (NULL));
  }

  /* states (dwork) */
  (void) memset((void *)&acc_decision_md_DW, 0,
                sizeof(DW_acc_decision_md_T));

  /* external inputs */
  (void)memset(&acc_decision_md_U, 0, sizeof(ExtU_acc_decision_md_T));

  /* external outputs */
  (void)memset(&acc_decision_md_Y, 0, sizeof(ExtY_acc_decision_md_T));

  /* Matfile logging */
  rt_StartDataLoggingWithStartTime(acc_decision_md_M->rtwLogInfo, 0.0,
    rtmGetTFinal(acc_decision_md_M), acc_decision_md_M->Timing.stepSize0,
    (&rtmGetErrorStatus(acc_decision_md_M)));

  /* SystemInitialize for Outport: '<Root>/current_state' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.current_state = 2.0;

  /* SystemInitialize for Outport: '<Root>/control_mode' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.control_mode = 0.0;

  /* SystemInitialize for Outport: '<Root>/acc_active' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.acc_active = false;

  /* SystemInitialize for Outport: '<Root>/control_enabled' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.control_enabled = false;

  /* SystemInitialize for Outport: '<Root>/V3_kmh' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.V3_kmh = 50.0;

  /* SystemInitialize for Outport: '<Root>/G1_m' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.G1_m = 15.0;

  /* SystemInitialize for Outport: '<Root>/G2_s' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.G2_s = 2.0;

  /* SystemInitialize for Outport: '<Root>/has_history' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.has_history = false;

  /* SystemInitialize for Outport: '<Root>/message_code' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.message_code = 0.0;

  /* SystemInitialize for Outport: '<Root>/pending_distance_adj' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.pending_distance_adj = 0.0;

  /* SystemInitialize for Outport: '<Root>/is_in_control' incorporates:
   *  Chart: '<Root>/ACC_Decision_Chart'
   */
  acc_decision_md_Y.is_in_control = false;

  /* SystemInitialize for Chart: '<Root>/ACC_Decision_Chart' */
  acc_decision_md_DW.history_V3_kmh = 50.0;
  acc_decision_md_DW.history_G1_m = 15.0;
  acc_decision_md_DW.history_G2_s = 2.0;
  acc_decision_md_DW.pending_adj = 0.0;
  acc_decision_md_DW.is_active_c3_acc_decision_md = 0U;
  acc_decision_md_DW.is_during_ = acc_decision_IN_NO_ACTIVE_CHILD;
}

/* Model terminate function */
void acc_decision_md_terminate(void)
{
  /* (no terminate code required) */
}
