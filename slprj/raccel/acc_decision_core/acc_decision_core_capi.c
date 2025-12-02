#include "rtw_capi.h"
#ifdef HOST_CAPI_BUILD
#include "acc_decision_core_capi_host.h"
#define sizeof(...) ((size_t)(0xFFFF))
#undef rt_offsetof
#define rt_offsetof(s,el) ((uint16_T)(0xFFFF))
#define TARGET_CONST
#define TARGET_STRING(s) (s)
#ifndef SS_UINT64
#define SS_UINT64 17
#endif
#ifndef SS_INT64
#define SS_INT64 18
#endif
#else
#include "builtin_typeid_types.h"
#include "acc_decision_core.h"
#include "acc_decision_core_capi.h"
#include "acc_decision_core_private.h"
#ifdef LIGHT_WEIGHT_CAPI
#define TARGET_CONST
#define TARGET_STRING(s)               ((NULL))
#else
#define TARGET_CONST                   const
#define TARGET_STRING(s)               (s)
#endif
#endif
static const rtwCAPI_Signals rtBlockSignals [ ] = { { 0 , 0 , ( NULL ) , ( NULL
) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_BlockParameters
rtBlockParameters [ ] = { { 0 , TARGET_STRING ( "acc_decision_core/Const_Minus1"
) , TARGET_STRING ( "Value" ) , 0 , 0 , 0 } , { 1 , TARGET_STRING ( "acc_decision_core/Const_One" ) , TARGET_STRING ( "Value" ) , 0 , 0 , 0 } , { 2 , TARGET_STRING ( "acc_decision_core/Const_One_Decision" ) , TARGET_STRING ( "Value" ) , 0 , 0 , 0 } , { 3 , TARGET_STRING ( "acc_decision_core/Const_Six" ) , TARGET_STRING ( "Value" ) , 0 , 0 , 0 } , { 4 , TARGET_STRING ( "acc_decision_core/LUT_control_enabled" ) , TARGET_STRING ( "maxIndex" ) , 1 , 1 , 0 } , { 5 , TARGET_STRING ( "acc_decision_core/LUT_decision" ) , TARGET_STRING ( "maxIndex" ) , 1 , 1 , 0 } , { 6 , TARGET_STRING ( "acc_decision_core/LUT_next_state" ) , TARGET_STRING ( "maxIndex" ) , 1 , 1 , 0 } , { 7 , TARGET_STRING ( "acc_decision_core/LUT_side_effect" ) , TARGET_STRING ( "maxIndex" ) , 1 , 1 , 0 } , { 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 } } ; static int_T rt_LoggedStateIdxList [ ] = { - 1 } ; static const rtwCAPI_Signals rtRootInputs [ ] = { { 8 , 0 , TARGET_STRING ( "acc_decision_core/current_state" ) , TARGET_STRING ( "" ) , 1 , 0 , 0 , 0 , 0 } , { 9 , 0 , TARGET_STRING ( "acc_decision_core/command_type" ) , TARGET_STRING ( "" ) , 2 , 0 , 0 , 0 , 0 } , { 10 , 0 , TARGET_STRING ( "acc_decision_core/has_history" ) , TARGET_STRING ( "" ) , 3 , 0 , 0 , 0 , 0 } , { 11 , 0 , TARGET_STRING ( "acc_decision_core/last_active_decision" ) , TARGET_STRING ( "" ) , 4 , 0 , 0 , 0 , 0 } , { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_Signals rtRootOutputs [ ] = { { 12 , 0 , TARGET_STRING ( "acc_decision_core/next_state" ) , TARGET_STRING ( "" ) , 1 , 0 , 0 , 0 , 0 } , { 13 , 0 , TARGET_STRING ( "acc_decision_core/decision" ) , TARGET_STRING ( "" ) , 2 , 0 , 0 , 0 , 0 } , { 14 , 0 , TARGET_STRING ( "acc_decision_core/control_enabled" ) , TARGET_STRING ( "" ) , 3 , 0 , 0 , 0 , 0 } , { 15 , 0 , TARGET_STRING ( "acc_decision_core/next_has_history" ) , TARGET_STRING ( "" ) , 4 , 0 , 0 , 0 , 0 } , { 16 , 0 , TARGET_STRING ( "acc_decision_core/next_last_decision" ) , TARGET_STRING ( "" ) , 5 , 0 , 0 , 0 , 0 } , { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_ModelParameters rtModelParameters [ ] = { { 17 , TARGET_STRING ( "command_bp" ) , 0 , 2 , 0 } , { 18 , TARGET_STRING ( "control_enabled_table" ) , 0 , 3 , 0 } , { 19 , TARGET_STRING ( "decision_table" ) , 0 , 3 , 0 } , { 20 , TARGET_STRING ( "next_state_table" ) , 0 , 3 , 0 } , { 21 , TARGET_STRING ( "side_effect_table" ) , 0 , 3 , 0 } , { 22 , TARGET_STRING ( "state_bp" ) , 0 , 4 , 0 } , { 0 , ( NULL ) , 0 , 0 , 0 } } ;
#ifndef HOST_CAPI_BUILD
static void * rtDataAddrMap [ ] = { & rtP . Const_Minus1_Value , & rtP .
Const_One_Value , & rtP . Const_One_Decision_Value , & rtP . Const_Six_Value
, & rtP . LUT_control_enabled_maxIndex [ 0 ] , & rtP . LUT_decision_maxIndex
[ 0 ] , & rtP . LUT_next_state_maxIndex [ 0 ] , & rtP .
LUT_side_effect_maxIndex [ 0 ] , & rtU . noyaiydphn , & rtU . c5sdyktups , &
rtU . dehz1famfm , & rtU . ca3u04tmfz , & rtY . fvup4tazez , & rtY .
ecrtk30kvf , & rtY . gf2wpcs10h , & rtY . dsoc3sr5fb , & rtY . hbheu55okb , &
rtP . command_bp [ 0 ] , & rtP . control_enabled_table [ 0 ] , & rtP .
decision_table [ 0 ] , & rtP . next_state_table [ 0 ] , & rtP .
side_effect_table [ 0 ] , & rtP . state_bp [ 0 ] , } ; static int32_T *
rtVarDimsAddrMap [ ] = { ( NULL ) } ;
#endif
static TARGET_CONST rtwCAPI_DataTypeMap rtDataTypeMap [ ] = { { "double" ,
"real_T" , 0 , 0 , sizeof ( real_T ) , ( uint8_T ) SS_DOUBLE , 0 , 0 , 0 } ,
{ "unsigned int" , "uint32_T" , 0 , 0 , sizeof ( uint32_T ) , ( uint8_T )
SS_UINT32 , 0 , 0 , 0 } } ;
#ifdef HOST_CAPI_BUILD
#undef sizeof
#endif
static TARGET_CONST rtwCAPI_ElementMap rtElementMap [ ] = { { ( NULL ) , 0 ,
0 , 0 , 0 } , } ; static const rtwCAPI_DimensionMap rtDimensionMap [ ] = { {
rtwCAPI_SCALAR , 0 , 2 , 0 } , { rtwCAPI_VECTOR , 2 , 2 , 0 } , {
rtwCAPI_VECTOR , 4 , 2 , 0 } , { rtwCAPI_MATRIX_COL_MAJOR , 6 , 2 , 0 } , {
rtwCAPI_VECTOR , 8 , 2 , 0 } } ; static const uint_T rtDimensionArray [ ] = {
1 , 1 , 2 , 1 , 1 , 8 , 4 , 8 , 1 , 4 } ; static const real_T
rtcapiStoredFloats [ ] = { 0.05 , 0.0 } ; static const rtwCAPI_FixPtMap
rtFixPtMap [ ] = { { ( NULL ) , ( NULL ) , rtwCAPI_FIX_RESERVED , 0 , 0 , ( boolean_T ) 0 } , } ; static const rtwCAPI_SampleTimeMap rtSampleTimeMap [ ] = { { ( const void * ) & rtcapiStoredFloats [ 0 ] , ( const void * ) & rtcapiStoredFloats [ 1 ] , ( int8_T ) 0 , ( uint8_T ) 0 } } ; static rtwCAPI_ModelMappingStaticInfo mmiStatic = { { rtBlockSignals , 0 , rtRootInputs , 4 , rtRootOutputs , 5 } , { rtBlockParameters , 8 , rtModelParameters , 6 } , { ( NULL ) , 0 } , { rtDataTypeMap , rtDimensionMap , rtFixPtMap , rtElementMap , rtSampleTimeMap , rtDimensionArray } , "float" , { 168352552U , 2349709062U , 2007525573U , 16642492U } , ( NULL ) , 0 , ( boolean_T ) 0 , rt_LoggedStateIdxList } ; const rtwCAPI_ModelMappingStaticInfo * acc_decision_core_GetCAPIStaticMap ( void ) { return & mmiStatic ; }
#ifndef HOST_CAPI_BUILD
void acc_decision_core_InitializeDataMapInfo ( void ) { rtwCAPI_SetVersion ( ( *
rt_dataMapInfoPtr ) . mmi , 1 ) ; rtwCAPI_SetStaticMap ( ( *
rt_dataMapInfoPtr ) . mmi , & mmiStatic ) ; rtwCAPI_SetLoggingStaticMap ( ( *
rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ; rtwCAPI_SetDataAddressMap ( ( *
rt_dataMapInfoPtr ) . mmi , rtDataAddrMap ) ; rtwCAPI_SetVarDimsAddressMap ( ( *
rt_dataMapInfoPtr ) . mmi , rtVarDimsAddrMap ) ;
rtwCAPI_SetInstanceLoggingInfo ( ( * rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ;
rtwCAPI_SetChildMMIArray ( ( * rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ;
rtwCAPI_SetChildMMIArrayLen ( ( * rt_dataMapInfoPtr ) . mmi , 0 ) ; }
#else
#ifdef __cplusplus
extern "C" {
#endif
void acc_decision_core_host_InitializeDataMapInfo ( acc_decision_core_host_DataMapInfo_T * dataMap , const char * path ) { rtwCAPI_SetVersion ( dataMap -> mmi , 1 ) ; rtwCAPI_SetStaticMap ( dataMap -> mmi , & mmiStatic ) ; rtwCAPI_SetDataAddressMap ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetVarDimsAddressMap ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetPath ( dataMap -> mmi , path ) ; rtwCAPI_SetFullPath ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetChildMMIArray ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetChildMMIArrayLen ( dataMap -> mmi , 0 ) ; }
#ifdef __cplusplus
}
#endif
#endif
