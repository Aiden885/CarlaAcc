#include "sppvt_control_model.h"
#include "rtwtypes.h"
#include "sppvt_control_model_private.h"
#include "mwmathutil.h"
#include "sppvt_control_model_capi.h"
static RegMdlInfo rtMdlInfo_sppvt_control_model [ 40 ] = { { "honz4qowvac" ,
MDL_INFO_NAME_MDLREF_DWORK , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"i5reni0luu" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "ksktbk4ijb" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "ivsdpsvzsq" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "dodxkx1vyd" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "nkdcxyasa5" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "lhn50qno1e" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "kmmsetmsyr" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "f0yls5acve" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "olbjabz3eh" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "lbnzjfg3iq" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "pi3fqpgfc4" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "j0slriomzc" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "etubzmduay" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "ayovtvs4k0" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "i24dvpsv4q" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "lhb0xodtrz" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "e0fkfskrgf" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "moodhcvwbn" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "hjyrn31yiw" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "bejwtmvm5q" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "sppvt_control_model" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , 0 , ( NULL ) } , { "k4hnywv3xt" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "nvhpshikqfm" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * )
"sppvt_control_model" } , { "hzh1clgcok" , MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT ,
0 , - 1 , ( void * ) "sppvt_control_model" } , { "lla4rxpd2v" ,
MDL_INFO_ID_GLOBAL_RTW_CONSTRUCT , 0 , - 1 , ( void * ) "sppvt_control_model"
} , { "mr_sppvt_control_model_GetSimStateDisallowedBlocks" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_extractBitFieldFromCellArrayWithOffset" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_cacheBitFieldToCellArrayWithOffset" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_restoreDataFromMxArrayWithOffset" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_cacheDataToMxArrayWithOffset" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_extractBitFieldFromMxArray" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_cacheBitFieldToMxArray" , MDL_INFO_ID_MODEL_FCN_NAME
, 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_restoreDataFromMxArray" , MDL_INFO_ID_MODEL_FCN_NAME
, 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_cacheDataAsMxArray" , MDL_INFO_ID_MODEL_FCN_NAME , 0
, - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_RegisterSimStateChecksum" ,
MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , {
"mr_sppvt_control_model_SetDWork" , MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , { "mr_sppvt_control_model_GetDWork" , MDL_INFO_ID_MODEL_FCN_NAME , 0 , - 1 , ( void * ) "sppvt_control_model" } , { "sppvt_control_model.h" , MDL_INFO_MODEL_FILENAME , 0 , - 1 , ( NULL ) } , { "sppvt_control_model.c" , MDL_INFO_MODEL_FILENAME , 0 , - 1 , ( void * ) "sppvt_control_model" } } ; nvhpshikqfm nvhpshikqf = { 2.0 , - 3.0 , 0.0 } ; void sppvt_control_model ( const real_T * jlwnlbickz , const real_T * oa2l3hxdft , const real_T * p5d10symtt , const real_T * iij0tvraqk , const real_T * b153mqkknq , const real_T * pgs12lddo2 , const real_T * jypzdwp30j , const real_T * ipeapiysbn , const real_T * olayervcpn , real_T * aavict1um0 , real_T * adxopizfhg , real_T * di0zwxyhhe , real_T * kzejqtvb1e , boolean_T * cpewdcpadj ) { real_T b3hb1yb03f ; real_T kmee142no2 ; boolean_T cb1qbpvacg ; boolean_T hqdpzquaj4 ; boolean_T osphoobpmb ; kmee142no2 = * jlwnlbickz + * p5d10symtt ; kmee142no2 *= * iij0tvraqk ; if ( kmee142no2 > nvhpshikqf . P_0 ) { * aavict1um0 = nvhpshikqf . P_0 ; } else if ( kmee142no2 < nvhpshikqf . P_1 ) { * aavict1um0 = nvhpshikqf . P_1 ; } else { * aavict1um0 = kmee142no2 ; } kmee142no2 = * jlwnlbickz - * b153mqkknq ; * adxopizfhg = kmee142no2 / * oa2l3hxdft ; kmee142no2 = * adxopizfhg - * pgs12lddo2 ; * di0zwxyhhe = kmee142no2 / * oa2l3hxdft ; kmee142no2 = * di0zwxyhhe - * jypzdwp30j ; * kzejqtvb1e = kmee142no2 / * oa2l3hxdft ; kmee142no2 = muDoubleScalarAbs ( * jlwnlbickz ) ; b3hb1yb03f = muDoubleScalarAbs ( * adxopizfhg ) ; osphoobpmb = ( kmee142no2 > * olayervcpn ) ; hqdpzquaj4 = ( b3hb1yb03f <= * ipeapiysbn ) ; cb1qbpvacg = ( * di0zwxyhhe < nvhpshikqf . P_2 ) ; * cpewdcpadj = ( cb1qbpvacg && hqdpzquaj4 && osphoobpmb ) ; } void i24dvpsv4q ( lla4rxpd2v * const ninbu0alvp ) { if ( ! slIsRapidAcceleratorSimulating ( ) ) { slmrRunPluginEvent ( ninbu0alvp -> _mdlRefSfcnS , "sppvt_control_model" , "SIMSTATUS_TERMINATING_MODELREF_ACCEL_EVENT" ) ; } } void moodhcvwbn ( SimStruct * _mdlRefSfcnS , int_T mdlref_TID0 , int_T mdlref_TID1 , lla4rxpd2v * const ninbu0alvp , void * sysRanPtr , int contextTid , rtwCAPI_ModelMappingInfo * rt_ParentMMI , const char_T * rt_ChildPath , int_T rt_ChildMMIIdx , int_T rt_CSTATEIdx ) { ( void ) memset ( ( void * ) ninbu0alvp , 0 , sizeof ( lla4rxpd2v ) ) ; ninbu0alvp -> Timing . mdlref_GlobalTID [ 0 ] = mdlref_TID0 ; ninbu0alvp -> Timing . mdlref_GlobalTID [ 1 ] = mdlref_TID1 ; ninbu0alvp -> _mdlRefSfcnS = ( _mdlRefSfcnS ) ; if ( ! slIsRapidAcceleratorSimulating ( ) ) { slmrRunPluginEvent ( ninbu0alvp -> _mdlRefSfcnS , "sppvt_control_model" , "START_OF_SIM_MODEL_MODELREF_ACCEL_EVENT" ) ; } sppvt_control_model_InitializeDataMapInfo ( ninbu0alvp , sysRanPtr , contextTid ) ; if ( ( rt_ParentMMI != ( NULL ) ) && ( rt_ChildPath != ( NULL ) ) ) { rtwCAPI_SetChildMMI ( * rt_ParentMMI , rt_ChildMMIIdx , & ( ninbu0alvp -> DataMapInfo . mmi ) ) ; rtwCAPI_SetPath ( ninbu0alvp -> DataMapInfo . mmi , rt_ChildPath ) ; rtwCAPI_MMISetContStateStartIndex ( ninbu0alvp -> DataMapInfo . mmi , rt_CSTATEIdx ) ; } } void mr_sppvt_control_model_MdlInfoRegFcn ( SimStruct * mdlRefSfcnS , char_T * modelName , int_T * retVal ) { * retVal = 0 ; { boolean_T regSubmodelsMdlinfo = false ; ssGetRegSubmodelsMdlinfo ( mdlRefSfcnS , & regSubmodelsMdlinfo ) ; if ( regSubmodelsMdlinfo ) { } } * retVal = 0 ; ssRegModelRefMdlInfo ( mdlRefSfcnS , modelName , rtMdlInfo_sppvt_control_model , 40 ) ; * retVal = 1 ; } static void mr_sppvt_control_model_cacheDataAsMxArray ( mxArray * destArray , mwIndex i , int j , const void * srcData , size_t numBytes ) ; static void mr_sppvt_control_model_cacheDataAsMxArray ( mxArray * destArray , mwIndex i , int j , const void * srcData , size_t numBytes ) { mxArray * newArray = mxCreateUninitNumericMatrix ( ( size_t ) 1 , numBytes , mxUINT8_CLASS , mxREAL ) ; memcpy ( ( uint8_T * ) mxGetData ( newArray ) , ( const uint8_T * ) srcData , numBytes ) ; mxSetFieldByNumber ( destArray , i , j , newArray ) ; } static void mr_sppvt_control_model_restoreDataFromMxArray ( void * destData , const mxArray * srcArray , mwIndex i , int j , size_t numBytes ) ; static void mr_sppvt_control_model_restoreDataFromMxArray ( void * destData , const mxArray * srcArray , mwIndex i , int j , size_t numBytes ) { memcpy ( ( uint8_T * ) destData , ( const uint8_T * ) mxGetData ( mxGetFieldByNumber ( srcArray , i , j ) ) , numBytes ) ; } static void mr_sppvt_control_model_cacheBitFieldToMxArray ( mxArray * destArray , mwIndex i , int j , uint_T bitVal ) ; static void mr_sppvt_control_model_cacheBitFieldToMxArray ( mxArray * destArray , mwIndex i , int j , uint_T bitVal ) { mxSetFieldByNumber ( destArray , i , j , mxCreateDoubleScalar ( ( real_T ) bitVal ) ) ; } static uint_T mr_sppvt_control_model_extractBitFieldFromMxArray ( const mxArray * srcArray , mwIndex i , int j , uint_T numBits ) ; static uint_T mr_sppvt_control_model_extractBitFieldFromMxArray ( const mxArray * srcArray , mwIndex i , int j , uint_T numBits ) { const uint_T varVal = ( uint_T ) mxGetScalar ( mxGetFieldByNumber ( srcArray , i , j ) ) ; return varVal & ( ( 1u << numBits ) - 1u ) ; } static void mr_sppvt_control_model_cacheDataToMxArrayWithOffset ( mxArray * destArray , mwIndex i , int j , mwIndex offset , const void * srcData , size_t numBytes ) ; static void mr_sppvt_control_model_cacheDataToMxArrayWithOffset ( mxArray * destArray , mwIndex i , int j , mwIndex offset , const void * srcData , size_t numBytes ) { uint8_T * varData = ( uint8_T * ) mxGetData ( mxGetFieldByNumber ( destArray , i , j ) ) ; memcpy ( ( uint8_T * ) & varData [ offset * numBytes ] , ( const uint8_T * ) srcData , numBytes ) ; } static void mr_sppvt_control_model_restoreDataFromMxArrayWithOffset ( void * destData , const mxArray * srcArray , mwIndex i , int j , mwIndex offset , size_t numBytes ) ; static void mr_sppvt_control_model_restoreDataFromMxArrayWithOffset ( void * destData , const mxArray * srcArray , mwIndex i , int j , mwIndex offset , size_t numBytes ) { const uint8_T * varData = ( const uint8_T * ) mxGetData ( mxGetFieldByNumber ( srcArray , i , j ) ) ; memcpy ( ( uint8_T * ) destData , ( const uint8_T * ) & varData [ offset * numBytes ] , numBytes ) ; } static void mr_sppvt_control_model_cacheBitFieldToCellArrayWithOffset ( mxArray * destArray , mwIndex i , int j , mwIndex offset , uint_T fieldVal ) ; static void mr_sppvt_control_model_cacheBitFieldToCellArrayWithOffset ( mxArray * destArray , mwIndex i , int j , mwIndex offset , uint_T fieldVal ) { mxSetCell ( mxGetFieldByNumber ( destArray , i , j ) , offset , mxCreateDoubleScalar ( ( real_T ) fieldVal ) ) ; } static uint_T mr_sppvt_control_model_extractBitFieldFromCellArrayWithOffset ( const mxArray * srcArray , mwIndex i , int j , mwIndex offset , uint_T numBits ) ; static uint_T mr_sppvt_control_model_extractBitFieldFromCellArrayWithOffset ( const mxArray * srcArray , mwIndex i , int j , mwIndex offset , uint_T numBits ) { const uint_T fieldVal = ( uint_T ) mxGetScalar ( mxGetCell ( mxGetFieldByNumber ( srcArray , i , j ) , offset ) ) ; return fieldVal & ( ( 1u << numBits ) - 1u ) ; } mxArray * mr_sppvt_control_model_GetDWork ( const honz4qowvac * mdlrefDW ) { ( void ) mdlrefDW ; return ( NULL ) ; } void mr_sppvt_control_model_SetDWork ( honz4qowvac * mdlrefDW , const mxArray * ssDW ) { ( void ) ssDW ; ( void ) mdlrefDW ; } void mr_sppvt_control_model_RegisterSimStateChecksum ( SimStruct * S ) { const uint32_T chksum [ 4 ] = { 3903718554U , 2003392538U , 1082102322U , 1044957618U , } ; slmrModelRefRegisterSimStateChecksum ( S , "sppvt_control_model" , & chksum [ 0 ] ) ; } mxArray * mr_sppvt_control_model_GetSimStateDisallowedBlocks ( ) { return ( NULL ) ; }
#if defined(_MSC_VER)
#pragma warning(disable: 4505) //unreferenced local function has been removed
#endif
