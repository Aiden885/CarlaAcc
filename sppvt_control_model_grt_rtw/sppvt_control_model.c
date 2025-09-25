/*
 * sppvt_control_model.c
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

#include "sppvt_control_model.h"
#include <math.h>
#include "rtwtypes.h"
#include <string.h>
#include "sppvt_control_model_private.h"

/* External inputs (root inport signals with default storage) */
ExtU_sppvt_control_model_T sppvt_control_model_U;

/* External outputs (root outports fed by signals with default storage) */
ExtY_sppvt_control_model_T sppvt_control_model_Y;

/* Real-time model */
static RT_MODEL_sppvt_control_model_T sppvt_control_model_M_;
RT_MODEL_sppvt_control_model_T *const sppvt_control_model_M =
  &sppvt_control_model_M_;

/* Model step function */
void sppvt_control_model_step(void)
{
  real_T rtb_Abs_error;
  real_T rtb_Abs_velocity;

  /* Product: '<S1>/Proportional_Control' incorporates:
   *  Inport: '<Root>/current_stage_offset'
   *  Inport: '<Root>/error_value'
   *  Inport: '<Root>/sppvt_kp'
   *  Sum: '<S1>/Enhanced_Error'
   */
  rtb_Abs_velocity = (sppvt_control_model_U.error_value +
                      sppvt_control_model_U.current_stage_offset) *
    sppvt_control_model_U.sppvt_kp;

  /* Saturate: '<S1>/Output_Saturation' */
  if (rtb_Abs_velocity > sppvt_control_model_P.Output_Saturation_UpperSat) {
    /* Outport: '<Root>/control_output' */
    sppvt_control_model_Y.control_output =
      sppvt_control_model_P.Output_Saturation_UpperSat;
  } else if (rtb_Abs_velocity < sppvt_control_model_P.Output_Saturation_LowerSat)
  {
    /* Outport: '<Root>/control_output' */
    sppvt_control_model_Y.control_output =
      sppvt_control_model_P.Output_Saturation_LowerSat;
  } else {
    /* Outport: '<Root>/control_output' */
    sppvt_control_model_Y.control_output = rtb_Abs_velocity;
  }

  /* End of Saturate: '<S1>/Output_Saturation' */

  /* Product: '<S2>/Velocity_Calc' incorporates:
   *  Inport: '<Root>/dt'
   *  Inport: '<Root>/error_value'
   *  Inport: '<Root>/prev_error'
   *  Sum: '<S2>/Error_Diff'
   */
  rtb_Abs_velocity = (sppvt_control_model_U.error_value -
                      sppvt_control_model_U.prev_error) /
    sppvt_control_model_U.dt;

  /* Outport: '<Root>/velocity' */
  sppvt_control_model_Y.velocity = rtb_Abs_velocity;

  /* Product: '<S2>/Acceleration_Calc' incorporates:
   *  Inport: '<Root>/dt'
   *  Inport: '<Root>/prev_velocity'
   *  Sum: '<S2>/Velocity_Diff'
   */
  rtb_Abs_error = (rtb_Abs_velocity - sppvt_control_model_U.prev_velocity) /
    sppvt_control_model_U.dt;

  /* Outport: '<Root>/acceleration' */
  sppvt_control_model_Y.acceleration = rtb_Abs_error;

  /* Outport: '<Root>/jerk' incorporates:
   *  Inport: '<Root>/dt'
   *  Inport: '<Root>/prev_accel'
   *  Product: '<S2>/Jerk_Calc'
   *  Sum: '<S2>/Acceleration_Diff'
   */
  sppvt_control_model_Y.jerk = (rtb_Abs_error - sppvt_control_model_U.prev_accel)
    / sppvt_control_model_U.dt;

  /* Outport: '<Root>/should_upgrade' incorporates:
   *  Abs: '<S3>/Abs_error'
   *  Abs: '<S3>/Abs_velocity'
   *  Constant: '<S3>/Constant'
   *  Inport: '<Root>/error_value'
   *  Inport: '<Root>/sppvt_delta'
   *  Inport: '<Root>/sppvt_eta'
   *  Logic: '<S3>/Logical Operator'
   *  RelationalOperator: '<S3>/Compare_eta'
   *  RelationalOperator: '<S3>/Compare_eta1'
   *  RelationalOperator: '<S3>/Relational Operator'
   */
  sppvt_control_model_Y.should_upgrade = ((rtb_Abs_error <
    sppvt_control_model_P.Constant_Value) && (fabs(rtb_Abs_velocity) <=
    sppvt_control_model_U.sppvt_delta) && (fabs
    (sppvt_control_model_U.error_value) > sppvt_control_model_U.sppvt_eta));

  /* Matfile logging */
  rt_UpdateTXYLogVars(sppvt_control_model_M->rtwLogInfo,
                      (&sppvt_control_model_M->Timing.taskTime0));

  /* signal main to stop simulation */
  {                                    /* Sample time: [0.05s, 0.0s] */
    if ((rtmGetTFinal(sppvt_control_model_M)!=-1) &&
        !((rtmGetTFinal(sppvt_control_model_M)-
           sppvt_control_model_M->Timing.taskTime0) >
          sppvt_control_model_M->Timing.taskTime0 * (DBL_EPSILON))) {
      rtmSetErrorStatus(sppvt_control_model_M, "Simulation finished");
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
  if (!(++sppvt_control_model_M->Timing.clockTick0)) {
    ++sppvt_control_model_M->Timing.clockTickH0;
  }

  sppvt_control_model_M->Timing.taskTime0 =
    sppvt_control_model_M->Timing.clockTick0 *
    sppvt_control_model_M->Timing.stepSize0 +
    sppvt_control_model_M->Timing.clockTickH0 *
    sppvt_control_model_M->Timing.stepSize0 * 4294967296.0;
}

/* Model initialize function */
void sppvt_control_model_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)sppvt_control_model_M, 0,
                sizeof(RT_MODEL_sppvt_control_model_T));
  rtmSetTFinal(sppvt_control_model_M, 0.05);
  sppvt_control_model_M->Timing.stepSize0 = 0.05;

  /* Setup for data logging */
  {
    static RTWLogInfo rt_DataLoggingInfo;
    rt_DataLoggingInfo.loggingInterval = (NULL);
    sppvt_control_model_M->rtwLogInfo = &rt_DataLoggingInfo;
  }

  /* Setup for data logging */
  {
    rtliSetLogXSignalInfo(sppvt_control_model_M->rtwLogInfo, (NULL));
    rtliSetLogXSignalPtrs(sppvt_control_model_M->rtwLogInfo, (NULL));
    rtliSetLogT(sppvt_control_model_M->rtwLogInfo, "tout");
    rtliSetLogX(sppvt_control_model_M->rtwLogInfo, "");
    rtliSetLogXFinal(sppvt_control_model_M->rtwLogInfo, "");
    rtliSetLogVarNameModifier(sppvt_control_model_M->rtwLogInfo, "rt_");
    rtliSetLogFormat(sppvt_control_model_M->rtwLogInfo, 2);
    rtliSetLogMaxRows(sppvt_control_model_M->rtwLogInfo, 0);
    rtliSetLogDecimation(sppvt_control_model_M->rtwLogInfo, 1);

    /*
     * Set pointers to the data and signal info for each output
     */
    {
      static void * rt_LoggedOutputSignalPtrs[] = {
        &sppvt_control_model_Y.control_output,
        &sppvt_control_model_Y.velocity,
        &sppvt_control_model_Y.acceleration,
        &sppvt_control_model_Y.jerk,
        &sppvt_control_model_Y.should_upgrade
      };

      rtliSetLogYSignalPtrs(sppvt_control_model_M->rtwLogInfo,
                            ((LogSignalPtrsType)rt_LoggedOutputSignalPtrs));
    }

    {
      static int_T rt_LoggedOutputWidths[] = {
        1,
        1,
        1,
        1,
        1
      };

      static int_T rt_LoggedOutputNumDimensions[] = {
        1,
        1,
        1,
        1,
        1
      };

      static int_T rt_LoggedOutputDimensions[] = {
        1,
        1,
        1,
        1,
        1
      };

      static boolean_T rt_LoggedOutputIsVarDims[] = {
        0,
        0,
        0,
        0,
        0
      };

      static void* rt_LoggedCurrentSignalDimensions[] = {
        (NULL),
        (NULL),
        (NULL),
        (NULL),
        (NULL)
      };

      static int_T rt_LoggedCurrentSignalDimensionsSize[] = {
        4,
        4,
        4,
        4,
        4
      };

      static BuiltInDTypeId rt_LoggedOutputDataTypeIds[] = {
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_BOOLEAN
      };

      static int_T rt_LoggedOutputComplexSignals[] = {
        0,
        0,
        0,
        0,
        0
      };

      static RTWPreprocessingFcnPtr rt_LoggingPreprocessingFcnPtrs[] = {
        (NULL),
        (NULL),
        (NULL),
        (NULL),
        (NULL)
      };

      static const char_T *rt_LoggedOutputLabels[] = {
        "",
        "",
        "",
        "",
        "" };

      static const char_T *rt_LoggedOutputBlockNames[] = {
        "sppvt_control_model/control_output",
        "sppvt_control_model/velocity",
        "sppvt_control_model/acceleration",
        "sppvt_control_model/jerk",
        "sppvt_control_model/should_upgrade" };

      static RTWLogDataTypeConvert rt_RTWLogDataTypeConvert[] = {
        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_BOOLEAN, SS_BOOLEAN, 0, 0, 0, 1.0, 0, 0.0 }
      };

      static RTWLogSignalInfo rt_LoggedOutputSignalInfo[] = {
        {
          5,
          rt_LoggedOutputWidths,
          rt_LoggedOutputNumDimensions,
          rt_LoggedOutputDimensions,
          rt_LoggedOutputIsVarDims,
          rt_LoggedCurrentSignalDimensions,
          rt_LoggedCurrentSignalDimensionsSize,
          rt_LoggedOutputDataTypeIds,
          rt_LoggedOutputComplexSignals,
          (NULL),
          rt_LoggingPreprocessingFcnPtrs,

          { rt_LoggedOutputLabels },
          (NULL),
          (NULL),
          (NULL),

          { rt_LoggedOutputBlockNames },

          { (NULL) },
          (NULL),
          rt_RTWLogDataTypeConvert
        }
      };

      rtliSetLogYSignalInfo(sppvt_control_model_M->rtwLogInfo,
                            rt_LoggedOutputSignalInfo);

      /* set currSigDims field */
      rt_LoggedCurrentSignalDimensions[0] = &rt_LoggedOutputWidths[0];
      rt_LoggedCurrentSignalDimensions[1] = &rt_LoggedOutputWidths[1];
      rt_LoggedCurrentSignalDimensions[2] = &rt_LoggedOutputWidths[2];
      rt_LoggedCurrentSignalDimensions[3] = &rt_LoggedOutputWidths[3];
      rt_LoggedCurrentSignalDimensions[4] = &rt_LoggedOutputWidths[4];
    }

    rtliSetLogY(sppvt_control_model_M->rtwLogInfo, "yout");
  }

  /* external inputs */
  (void)memset(&sppvt_control_model_U, 0, sizeof(ExtU_sppvt_control_model_T));

  /* external outputs */
  (void)memset(&sppvt_control_model_Y, 0, sizeof(ExtY_sppvt_control_model_T));

  /* Matfile logging */
  rt_StartDataLoggingWithStartTime(sppvt_control_model_M->rtwLogInfo, 0.0,
    rtmGetTFinal(sppvt_control_model_M), sppvt_control_model_M->Timing.stepSize0,
    (&rtmGetErrorStatus(sppvt_control_model_M)));
}

/* Model terminate function */
void sppvt_control_model_terminate(void)
{
  /* (no terminate code required) */
}
