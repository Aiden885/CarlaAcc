#ifndef acc_decision_core_cap_host_h__
#define acc_decision_core_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
acc_decision_core_host_DataMapInfo_T ;
#ifdef __cplusplus
extern "C" {
#endif
void acc_decision_core_host_InitializeDataMapInfo ( acc_decision_core_host_DataMapInfo_T * dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
