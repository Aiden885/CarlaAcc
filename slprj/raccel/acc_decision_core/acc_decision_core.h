#ifndef acc_decision_core_h_
#define acc_decision_core_h_
#ifndef acc_decision_core_COMMON_INCLUDES_
#define acc_decision_core_COMMON_INCLUDES_
#include <stdlib.h>
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sigstream_rtw.h"
#include "simtarget/slSimTgtSigstreamRTW.h"
#include "simtarget/slSimTgtSlioCoreRTW.h"
#include "simtarget/slSimTgtSlioClientsRTW.h"
#include "simtarget/slSimTgtSlioSdiRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "raccel.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "rt_logging_simtarget.h"
#include "rt_nonfinite.h"
#include "math.h"
#include "dt_info.h"
#include "ext_work.h"
#endif
#include "acc_decision_core_types.h"
#include <stddef.h>
#include "rtw_modelmap_simtarget.h"
#include "rt_defines.h"
#include <string.h>
#define MODEL_NAME acc_decision_core
#define NSAMPLE_TIMES (2) 
#define NINPUTS (4)       
#define NOUTPUTS (5)     
#define NBLOCKIO (0) 
#define NUM_ZC_EVENTS (0) 
#ifndef NCSTATES
#define NCSTATES (0)   
#elif NCSTATES != 0
#error Invalid specification of NCSTATES defined in compiler command
#endif
#ifndef rtmGetDataMapInfo
#define rtmGetDataMapInfo(rtm) (*rt_dataMapInfoPtr)
#endif
#ifndef rtmSetDataMapInfo
#define rtmSetDataMapInfo(rtm, val) (rt_dataMapInfoPtr = &val)
#endif
#ifndef IN_RACCEL_MAIN
#endif
typedef struct { struct { void * AQHandles ; } oqzsmkjjtm ; struct { void *
AQHandles ; } csohmthn3c ; struct { void * AQHandles ; } hevbco1h2d ; struct
{ void * AQHandles ; } py4u22s2hr ; struct { void * AQHandles ; } b2cw0c2ffb
; } DW ; typedef struct { real_T noyaiydphn ; real_T c5sdyktups ; real_T
dehz1famfm ; real_T ca3u04tmfz ; } ExtU ; typedef struct { real_T fvup4tazez
; real_T ecrtk30kvf ; real_T gf2wpcs10h ; real_T dsoc3sr5fb ; real_T
hbheu55okb ; } ExtY ; typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
DataMapInfo ; struct P_ { real_T command_bp [ 8 ] ; real_T
control_enabled_table [ 32 ] ; real_T decision_table [ 32 ] ; real_T
next_state_table [ 32 ] ; real_T side_effect_table [ 32 ] ; real_T state_bp [
4 ] ; real_T Const_Minus1_Value ; real_T Const_One_Value ; real_T
Const_One_Decision_Value ; real_T Const_Six_Value ; uint32_T
LUT_next_state_maxIndex [ 2 ] ; uint32_T LUT_decision_maxIndex [ 2 ] ;
uint32_T LUT_control_enabled_maxIndex [ 2 ] ; uint32_T
LUT_side_effect_maxIndex [ 2 ] ; } ; extern const char_T *
RT_MEMORY_ALLOCATION_ERROR ; extern DW rtDW ; extern ExtU rtU ; extern ExtY
rtY ; extern P rtP ; extern mxArray * mr_acc_decision_core_GetDWork ( ) ;
extern void mr_acc_decision_core_SetDWork ( const mxArray * ssDW ) ; extern
mxArray * mr_acc_decision_core_GetSimStateDisallowedBlocks ( ) ; extern const
rtwCAPI_ModelMappingStaticInfo * acc_decision_core_GetCAPIStaticMap ( void )
; extern SimStruct * const rtS ; extern DataMapInfo * rt_dataMapInfoPtr ;
extern rtwCAPI_ModelMappingInfo * rt_modelMapInfoPtr ; void MdlOutputs ( int_T
tid ) ; void MdlOutputsParameterSampleTime ( int_T tid ) ; void MdlUpdate ( int_T tid ) ; void MdlTerminate ( void ) ; void MdlInitializeSizes ( void ) ; void MdlInitializeSampleTimes ( void ) ; SimStruct * raccel_register_model ( ssExecutionInfo * executionInfo ) ;
#endif
