#ifndef sppvt_control_model_h_
#define sppvt_control_model_h_
#ifndef sppvt_control_model_COMMON_INCLUDES_
#define sppvt_control_model_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "math.h"
#endif
#include "sppvt_control_model_types.h"
#include "rtw_modelmap_simtarget.h"
#include "rt_nonfinite.h"
#include <string.h>
#include <stddef.h>
struct nvhpshikqfm_ { real_T P_0 ; real_T P_1 ; real_T P_2 ; } ; struct
hzh1clgcok { struct SimStruct_tag * _mdlRefSfcnS ; struct {
rtwCAPI_ModelMappingInfo mmi ; rtwCAPI_ModelMapLoggingInstanceInfo
mmiLogInstanceInfo ; sysRanDType * systemRan [ 2 ] ; int_T systemTid [ 2 ] ;
} DataMapInfo ; struct { int_T mdlref_GlobalTID [ 2 ] ; } Timing ; } ;
typedef struct { lla4rxpd2v rtm ; } honz4qowvac ; extern void moodhcvwbn ( SimStruct * _mdlRefSfcnS , int_T mdlref_TID0 , int_T mdlref_TID1 , lla4rxpd2v * const ninbu0alvp , void * sysRanPtr , int contextTid , rtwCAPI_ModelMappingInfo * rt_ParentMMI , const char_T * rt_ChildPath , int_T rt_ChildMMIIdx , int_T rt_CSTATEIdx ) ; extern void mr_sppvt_control_model_MdlInfoRegFcn ( SimStruct * mdlRefSfcnS , char_T * modelName , int_T * retVal ) ; extern mxArray * mr_sppvt_control_model_GetDWork ( const honz4qowvac * mdlrefDW ) ; extern void mr_sppvt_control_model_SetDWork ( honz4qowvac * mdlrefDW , const mxArray * ssDW ) ; extern void mr_sppvt_control_model_RegisterSimStateChecksum ( SimStruct * S ) ; extern mxArray * mr_sppvt_control_model_GetSimStateDisallowedBlocks ( ) ; extern const rtwCAPI_ModelMappingStaticInfo * sppvt_control_model_GetCAPIStaticMap ( void ) ; extern void sppvt_control_model ( const real_T * jlwnlbickz , const real_T * oa2l3hxdft , const real_T * p5d10symtt , const real_T * iij0tvraqk , const real_T * b153mqkknq , const real_T * pgs12lddo2 , const real_T * jypzdwp30j , const real_T * ipeapiysbn , const real_T * olayervcpn , real_T * aavict1um0 , real_T * adxopizfhg , real_T * di0zwxyhhe , real_T * kzejqtvb1e , boolean_T * cpewdcpadj ) ; extern void i24dvpsv4q ( lla4rxpd2v * const ninbu0alvp ) ;
#endif
