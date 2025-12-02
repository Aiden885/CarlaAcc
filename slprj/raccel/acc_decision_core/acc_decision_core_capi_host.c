#include "acc_decision_core_capi_host.h"
static acc_decision_core_host_DataMapInfo_T root;
static int initialized = 0;
rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        acc_decision_core_host_InitializeDataMapInfo(&(root), "acc_decision_core");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
