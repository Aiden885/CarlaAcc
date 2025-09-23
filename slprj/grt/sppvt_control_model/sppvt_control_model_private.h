/*
 * sppvt_control_model_private.h
 *
 * Code generation for model "sppvt_control_model".
 *
 * Model version              : 1.2
 * Simulink Coder version : 24.2 (R2024b) 21-Jun-2024
 * C source code generated on : Fri Sep 19 09:50:33 2025
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef sppvt_control_model_private_h_
#define sppvt_control_model_private_h_
#include "rtwtypes.h"
#include "multiword_types.h"
#include "sppvt_control_model.h"
#include "sppvt_control_model_types.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         (*((rtm)->errorStatus))
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    (*((rtm)->errorStatus) = (val))
#endif

#ifndef rtmGetErrorStatusPointer
#define rtmGetErrorStatusPointer(rtm)  (rtm)->errorStatus
#endif

#ifndef rtmSetErrorStatusPointer
#define rtmSetErrorStatusPointer(rtm, val) ((rtm)->errorStatus = (val))
#endif

extern P_sppvt_control_model_T sppvt_control_model_P;

#endif                                 /* sppvt_control_model_private_h_ */
