/* Include files */

#include "LKS_tcp11_sfun.h"
#include "c3_LKS_tcp11.h"
#include <string.h>
#include "mwmathutil.h"
#define _SF_MEX_LISTEN_FOR_CTRL_C(S)   sf_mex_listen_for_ctrl_c(S);
#ifdef utFree
#undef utFree
#endif

#ifdef utMalloc
#undef utMalloc
#endif

#ifdef __cplusplus

extern "C" void *utMalloc(size_t size);
extern "C" void utFree(void*);

#else

extern void *utMalloc(size_t size);
extern void utFree(void*);

#endif

/* Forward Declarations */

/* Type Definitions */

/* Named Constants */
#define CALL_EVENT                     (-1)

/* Variable Declarations */

/* Variable Definitions */
static real_T _sfTime_;
static emlrtMCInfo c3_emlrtMCI = { 1,  /* lineNo */
  1,                                   /* colNo */
  "TCPClient",                         /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/networklib/+matlabshared/+network/+internal/+coder/TCPClient.p"/* pName */
};

static emlrtMCInfo c3_b_emlrtMCI = { 158,/* lineNo */
  13,                                  /* colNo */
  "Channel",                           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pName */
};

static emlrtMCInfo c3_c_emlrtMCI = { 821,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_d_emlrtMCI = { 819,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_e_emlrtMCI = { 817,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_f_emlrtMCI = { 815,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_g_emlrtMCI = { 813,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_h_emlrtMCI = { 811,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_i_emlrtMCI = { 809,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_j_emlrtMCI = { 807,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_k_emlrtMCI = { 805,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_l_emlrtMCI = { 803,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_m_emlrtMCI = { 801,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_n_emlrtMCI = { 799,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_o_emlrtMCI = { 797,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_p_emlrtMCI = { 795,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_q_emlrtMCI = { 823,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_r_emlrtMCI = { 831,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_s_emlrtMCI = { 357,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_t_emlrtMCI = { 53,/* lineNo */
  5,                                   /* colNo */
  "repmat",                            /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/elmat/repmat.m"/* pName */
};

static emlrtMCInfo c3_u_emlrtMCI = { 364,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtMCInfo c3_v_emlrtMCI = { 139,/* lineNo */
  25,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtMCInfo c3_w_emlrtMCI = { 145,/* lineNo */
  21,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtMCInfo c3_x_emlrtMCI = { 15,/* lineNo */
  9,                                   /* colNo */
  "assertSupportedString",             /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/assertSupportedString.m"/* pName */
};

static emlrtMCInfo c3_y_emlrtMCI = { 1,/* lineNo */
  1,                                   /* colNo */
  "AsyncIOTransportChannel",           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/+asyncIOTransportChannel/+cod"
  "er/AsyncIOTransportChannel.p"       /* pName */
};

static emlrtMCInfo c3_ab_emlrtMCI = { 159,/* lineNo */
  13,                                  /* colNo */
  "CoderTimeAPI",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/coder/coder/lib/+coder/+internal/+time/CoderTimeAPI.m"/* pName */
};

static emlrtMCInfo c3_bb_emlrtMCI = { 249,/* lineNo */
  17,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRSInfo c3_emlrtRSI = { 8,  /* lineNo */
  "MATLAB Function1",                  /* fcnName */
  "#LKS_tcp11:325"                     /* pathName */
};

static emlrtRSInfo c3_b_emlrtRSI = { 12,/* lineNo */
  "MATLAB Function1",                  /* fcnName */
  "#LKS_tcp11:325"                     /* pathName */
};

static emlrtRSInfo c3_c_emlrtRSI = { 14,/* lineNo */
  "MATLAB Function1",                  /* fcnName */
  "#LKS_tcp11:325"                     /* pathName */
};

static emlrtRSInfo c3_d_emlrtRSI = { 1,/* lineNo */
  "ITransport",                        /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/ITransport.p"/* pathName */
};

static emlrtRSInfo c3_e_emlrtRSI = { 1,/* lineNo */
  "ITokenReader",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/ITokenReader.p"/* pathName */
};

static emlrtRSInfo c3_f_emlrtRSI = { 1,/* lineNo */
  "IFilterable",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/IFilterable.p"/* pathName */
};

static emlrtRSInfo c3_g_emlrtRSI = { 269,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_h_emlrtRSI = { 299,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_i_emlrtRSI = { 329,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_j_emlrtRSI = { 330,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_k_emlrtRSI = { 333,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_l_emlrtRSI = { 337,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_m_emlrtRSI = { 1,/* lineNo */
  "TCPClient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/networklib/+matlabshared/+network/+internal/+coder/TCPClient.p"/* pathName */
};

static emlrtRSInfo c3_n_emlrtRSI = { 29,/* lineNo */
  "sprintf",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/sprintf.m"/* pathName */
};

static emlrtRSInfo c3_o_emlrtRSI = { 53,/* lineNo */
  "sprintf",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/sprintf.m"/* pathName */
};

static emlrtRSInfo c3_p_emlrtRSI = { 55,/* lineNo */
  "sprintf",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/sprintf.m"/* pathName */
};

static emlrtRSInfo c3_q_emlrtRSI = { 54,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_r_emlrtRSI = { 175,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_s_emlrtRSI = { 178,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_t_emlrtRSI = { 181,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_u_emlrtRSI = { 182,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_v_emlrtRSI = { 283,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_w_emlrtRSI = { 792,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_x_emlrtRSI = { 434,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_y_emlrtRSI = { 501,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_ab_emlrtRSI = { 509,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_bb_emlrtRSI = { 830,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_cb_emlrtRSI = { 55,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_db_emlrtRSI = { 76,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_eb_emlrtRSI = { 25,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_fb_emlrtRSI = { 36,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_gb_emlrtRSI = { 563,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_hb_emlrtRSI = { 279,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_ib_emlrtRSI = { 292,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_jb_emlrtRSI = { 40,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_kb_emlrtRSI = { 38,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_lb_emlrtRSI = { 1,/* lineNo */
  "AsyncIOTransportChannel",           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/+asyncIOTransportChannel/+cod"
  "er/AsyncIOTransportChannel.p"       /* pathName */
};

static emlrtRSInfo c3_mb_emlrtRSI = { 1,/* lineNo */
  "ByteOrder",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/ByteOrder.p"/* pathName */
};

static emlrtRSInfo c3_nb_emlrtRSI = { 122,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_ob_emlrtRSI = { 124,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_pb_emlrtRSI = { 127,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_qb_emlrtRSI = { 131,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_rb_emlrtRSI = { 209,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_sb_emlrtRSI = { 184,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_tb_emlrtRSI = { 210,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_ub_emlrtRSI = { 228,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_vb_emlrtRSI = { 547,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_wb_emlrtRSI = { 299,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_xb_emlrtRSI = { 307,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_yb_emlrtRSI = { 242,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_ac_emlrtRSI = { 317,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_bc_emlrtRSI = { 447,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_cc_emlrtRSI = { 176,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_dc_emlrtRSI = { 164,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_ec_emlrtRSI = { 260,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_fc_emlrtRSI = { 350,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_gc_emlrtRSI = { 407,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_hc_emlrtRSI = { 91,/* lineNo */
  "strcmp",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pathName */
};

static emlrtRSInfo c3_ic_emlrtRSI = { 167,/* lineNo */
  "strcmp",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pathName */
};

static emlrtRSInfo c3_jc_emlrtRSI = { 240,/* lineNo */
  "strcmp",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pathName */
};

static emlrtRSInfo c3_kc_emlrtRSI = { 241,/* lineNo */
  "strcmp",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pathName */
};

static emlrtRSInfo c3_lc_emlrtRSI = { 242,/* lineNo */
  "strcmp",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pathName */
};

static emlrtRSInfo c3_mc_emlrtRSI = { 16,/* lineNo */
  "lower",                             /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/lower.m"/* pathName */
};

static emlrtRSInfo c3_nc_emlrtRSI = { 10,/* lineNo */
  "eml_string_transform",              /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/eml_string_transform.m"/* pathName */
};

static emlrtRSInfo c3_oc_emlrtRSI = { 137,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_pc_emlrtRSI = { 140,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_qc_emlrtRSI = { 143,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_rc_emlrtRSI = { 148,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_sc_emlrtRSI = { 167,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_tc_emlrtRSI = { 232,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_uc_emlrtRSI = { 558,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_vc_emlrtRSI = { 616,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_wc_emlrtRSI = { 186,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_xc_emlrtRSI = { 187,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_yc_emlrtRSI = { 192,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_ad_emlrtRSI = { 195,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_bd_emlrtRSI = { 197,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_cd_emlrtRSI = { 200,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_dd_emlrtRSI = { 202,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_ed_emlrtRSI = { 203,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_fd_emlrtRSI = { 32,/* lineNo */
  "tic",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/timefun/tic.m"/* pathName */
};

static emlrtRSInfo c3_gd_emlrtRSI = { 7,/* lineNo */
  "getTime",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/coder/coder/lib/+coder/+internal/+time/getTime.m"/* pathName */
};

static emlrtRSInfo c3_hd_emlrtRSI = { 21,/* lineNo */
  "CoderTimeAPI",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/coder/coder/lib/+coder/+internal/+time/CoderTimeAPI.m"/* pathName */
};

static emlrtRSInfo c3_id_emlrtRSI = { 148,/* lineNo */
  "CoderTimeAPI",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/coder/coder/lib/+coder/+internal/+time/CoderTimeAPI.m"/* pathName */
};

static emlrtRSInfo c3_jd_emlrtRSI = { 37,/* lineNo */
  "toc",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/timefun/toc.m"/* pathName */
};

static emlrtRSInfo c3_kd_emlrtRSI = { 249,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_ld_emlrtRSI = { 48,/* lineNo */
  "pause",                             /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/timefun/pause.m"/* pathName */
};

static emlrtRSInfo c3_md_emlrtRSI = { 90,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_nd_emlrtRSI = { 540,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_od_emlrtRSI = { 225,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_pd_emlrtRSI = { 549,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_qd_emlrtRSI = { 204,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_rd_emlrtRSI = { 707,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_sd_emlrtRSI = { 365,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_td_emlrtRSI = { 469,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_ud_emlrtRSI = { 239,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_vd_emlrtRSI = { 242,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_wd_emlrtRSI = { 247,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_xd_emlrtRSI = { 254,/* lineNo */
  "OutputStream",                      /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pathName */
};

static emlrtRSInfo c3_yd_emlrtRSI = { 238,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_ae_emlrtRSI = { 567,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_be_emlrtRSI = { 162,/* lineNo */
  "Stream",                            /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Stream.m"/* pathName */
};

static emlrtRSInfo c3_ce_emlrtRSI = { 607,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_de_emlrtRSI = { 22,/* lineNo */
  "matlabCodegenHandle",               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/matlabCodegenHandle.m"/* pathName */
};

static emlrtRSInfo c3_ee_emlrtRSI = { 447,/* lineNo */
  "tcpclient",                         /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pathName */
};

static emlrtRSInfo c3_fe_emlrtRSI = { 329,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_ge_emlrtRSI = { 334,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_he_emlrtRSI = { 455,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_ie_emlrtRSI = { 195,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_je_emlrtRSI = { 211,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_ke_emlrtRSI = { 212,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_le_emlrtRSI = { 213,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_me_emlrtRSI = { 220,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_ne_emlrtRSI = { 228,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_oe_emlrtRSI = { 14,/* lineNo */
  "warning",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/coder/coder/lib/+coder/+internal/warning.m"/* pathName */
};

static emlrtRSInfo c3_pe_emlrtRSI = { 477,/* lineNo */
  "API",                               /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pathName */
};

static emlrtRSInfo c3_qe_emlrtRSI = { 337,/* lineNo */
  "Channel",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pathName */
};

static emlrtRSInfo c3_re_emlrtRSI = { 633,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_se_emlrtRSI = { 374,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_te_emlrtRSI = { 365,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_ue_emlrtRSI = { 366,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_ve_emlrtRSI = { 367,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_we_emlrtRSI = { 256,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_xe_emlrtRSI = { 259,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_ye_emlrtRSI = { 236,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_af_emlrtRSI = { 237,/* lineNo */
  "BufferChannel",                     /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pathName */
};

static emlrtRSInfo c3_bf_emlrtRSI = { 1,/* lineNo */
  "MATLAB Function1",                  /* fcnName */
  "#LKS_tcp11:325"                     /* pathName */
};

static emlrtRTEInfo c3_emlrtRTEI = { 1,/* lineNo */
  1,                                   /* colNo */
  "AsyncIOTransportChannel",           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/transportlib/+matlabshared/+transportlib/+internal/+asyncIOTransportChannel/+cod"
  "er/AsyncIOTransportChannel.p"       /* pName */
};

static emlrtRTEInfo c3_b_emlrtRTEI = { 12,/* lineNo */
  1,                                   /* colNo */
  "blanks",                            /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/blanks.m"/* pName */
};

static emlrtRTEInfo c3_c_emlrtRTEI = { 1,/* lineNo */
  1,                                   /* colNo */
  "TCPClient",                         /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/networklib/+matlabshared/+network/+internal/+coder/TCPClient.p"/* pName */
};

static emlrtRTEInfo c3_d_emlrtRTEI = { 764,/* lineNo */
  21,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_e_emlrtRTEI = { 762,/* lineNo */
  21,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_f_emlrtRTEI = { 440,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_g_emlrtRTEI = { 446,/* lineNo */
  44,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_h_emlrtRTEI = { 128,/* lineNo */
  57,                                  /* colNo */
  "allOrAny",                          /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/allOrAny.m"/* pName */
};

static emlrtRTEInfo c3_i_emlrtRTEI = { 170,/* lineNo */
  17,                                  /* colNo */
  "Channel",                           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pName */
};

static emlrtRTEInfo c3_j_emlrtRTEI = { 175,/* lineNo */
  85,                                  /* colNo */
  "Channel",                           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pName */
};

static emlrtRTEInfo c3_k_emlrtRTEI = { 175,/* lineNo */
  103,                                 /* colNo */
  "Channel",                           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/Channel.m"/* pName */
};

static emlrtRTEInfo c3_l_emlrtRTEI = { 273,/* lineNo */
  88,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_m_emlrtRTEI = { 775,/* lineNo */
  17,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_n_emlrtRTEI = { 273,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_o_emlrtRTEI = { 274,/* lineNo */
  91,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_p_emlrtRTEI = { 274,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_q_emlrtRTEI = { 427,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_r_emlrtRTEI = { 433,/* lineNo */
  44,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_s_emlrtRTEI = { 129,/* lineNo */
  21,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtRTEInfo c3_t_emlrtRTEI = { 209,/* lineNo */
  13,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtRTEInfo c3_u_emlrtRTEI = { 259,/* lineNo */
  13,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtRTEInfo c3_v_emlrtRTEI = { 260,/* lineNo */
  13,                                  /* colNo */
  "BufferChannel",                     /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/testmeaslib/general/+matlabshared/+asyncio/+buffer/+internal/+coder/BufferChanne"
  "l.m"                                /* pName */
};

static emlrtRTEInfo c3_w_emlrtRTEI = { 206,/* lineNo */
  50,                                  /* colNo */
  "strcmp",                            /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pName */
};

static emlrtRTEInfo c3_x_emlrtRTEI = { 824,/* lineNo */
  38,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_y_emlrtRTEI = { 787,/* lineNo */
  13,                                  /* colNo */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m"/* pName */
};

static emlrtRTEInfo c3_ab_emlrtRTEI = { 70,/* lineNo */
  17,                                  /* colNo */
  "InputStream",                       /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pName */
};

static emlrtRTEInfo c3_bb_emlrtRTEI = { 72,/* lineNo */
  17,                                  /* colNo */
  "InputStream",                       /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pName */
};

static emlrtRTEInfo c3_cb_emlrtRTEI = { 563,/* lineNo */
  102,                                 /* colNo */
  "InputStream",                       /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pName */
};

static emlrtRTEInfo c3_db_emlrtRTEI = { 65,/* lineNo */
  28,                                  /* colNo */
  "repmat",                            /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/elmat/repmat.m"/* pName */
};

static emlrtRTEInfo c3_eb_emlrtRTEI = { 568,/* lineNo */
  13,                                  /* colNo */
  "InputStream",                       /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pName */
};

static emlrtRTEInfo c3_fb_emlrtRTEI = { 89,/* lineNo */
  13,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_gb_emlrtRTEI = { 126,/* lineNo */
  34,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_hb_emlrtRTEI = { 140,/* lineNo */
  21,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_ib_emlrtRTEI = { 205,/* lineNo */
  50,                                  /* colNo */
  "strcmp",                            /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/eml/+coder/+internal/strcmp.m"/* pName */
};

static emlrtRTEInfo c3_jb_emlrtRTEI = { 203,/* lineNo */
  26,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_kb_emlrtRTEI = { 145,/* lineNo */
  25,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_lb_emlrtRTEI = { 149,/* lineNo */
  29,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_mb_emlrtRTEI = { 239,/* lineNo */
  13,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtRTEInfo c3_nb_emlrtRTEI = { 6,/* lineNo */
  20,                                  /* colNo */
  "MATLAB Function1",                  /* fName */
  "#LKS_tcp11:325"                     /* pName */
};

static emlrtRTEInfo c3_ob_emlrtRTEI = { 269,/* lineNo */
  24,                                  /* colNo */
  "tcpclient",                         /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/matlab/networklib/interface/+networklibcoder/tcpclient.m"/* pName */
};

static emlrtRTEInfo c3_pb_emlrtRTEI = { 563,/* lineNo */
  13,                                  /* colNo */
  "InputStream",                       /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pName */
};

static emlrtDCInfo c3_emlrtDCI = { 57, /* lineNo */
  53,                                  /* colNo */
  "sprintf",                           /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/sprintf.m",/* pName */
  4                                    /* checkKind */
};

static emlrtBCInfo c3_emlrtBCI = { 1,  /* iFirst */
  1024,                                /* iLast */
  787,                                 /* lineNo */
  24,                                  /* colNo */
  "",                                  /* aName */
  "API",                               /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/API.m",/* pName */
  0                                    /* checkKind */
};

static emlrtRTEInfo c3_qb_emlrtRTEI = { 200,/* lineNo */
  22,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m"/* pName */
};

static emlrtDCInfo c3_b_emlrtDCI = { 203,/* lineNo */
  34,                                  /* colNo */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m",/* pName */
  1                                    /* checkKind */
};

static emlrtBCInfo c3_b_emlrtBCI = { -1,/* iFirst */
  -1,                                  /* iLast */
  203,                                 /* lineNo */
  34,                                  /* colNo */
  "",                                  /* aName */
  "OutputStream",                      /* fName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/OutputStream.m",/* pName */
  0                                    /* checkKind */
};

static emlrtRSInfo c3_cf_emlrtRSI = { 568,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_df_emlrtRSI = { 72,/* lineNo */
  "InputStream",                       /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/InputStream.m"/* pathName */
};

static emlrtRSInfo c3_ef_emlrtRSI = { 76,/* lineNo */
  "eml_switch_helper",                 /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/eml/eml_switch_helper.m"/* pathName */
};

static emlrtRSInfo c3_ff_emlrtRSI = { 58,/* lineNo */
  "sprintf",                           /* fcnName */
  "/home/aiden/snap/code/app/matlab/toolbox/eml/lib/matlab/strfun/sprintf.m"/* pathName */
};

static char_T c3_cv[128] = { '\x00', '\x01', '\x02', '\x03', '\x04', '\x05',
  '\x06', '\a', '\b', '\t', '\n', '\v', '\f', '\r', '\x0e', '\x0f', '\x10',
  '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
  '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '!', '\"', '#', '$', '%', '&',
  '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
  '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'a', 'b', 'c', 'd', 'e',
  'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
  'v', 'w', 'x', 'y', 'z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',
  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
  'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\x7f' };

static char_T c3_cv1[31] = { 'C', 'o', 'd', 'e', 'r', ':', 't', 'o', 'o', 'l',
  'b', 'o', 'x', ':', 'u', 'n', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd', 'S',
  't', 'r', 'i', 'n', 'g' };

static char_T c3_cv2[13] = { 'l', 'i', 't', 't', 'l', 'e', '-', 'e', 'n', 'd',
  'i', 'a', 'n' };

static char_T c3_cv3[35] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
  'a', 'n', 'n', 'e', 'l', ':', 'u', 'n', 'e', 'x', 'p', 'e', 'c', 't', 'e', 'd',
  'E', 'x', 'c', 'e', 'p', 't', 'i', 'o', 'n' };

static char_T c3_cv4[37] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
  'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'd', 'e', 'r', 'S', 'y', 'n', 'c', 'h',
  'r', 'o', 'n', 'o', 'u', 's', 'E', 'r', 'r', 'o', 'r' };

static char_T c3_cv5[30] = { 'C', 'o', 'd', 'e', 'r', ':', 'b', 'u', 'i', 'l',
  't', 'i', 'n', 's', ':', 'A', 's', 's', 'e', 'r', 't', 'i', 'o', 'n', 'F', 'a',
  'i', 'l', 'e', 'd' };

static char_T c3_cv6[9] = { 'c', 'o', 'm', 'p', 'l', 'e', 't', 'e', 'd' };

/* Function Declarations */
static void initialize_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void initialize_params_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static void mdl_start_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void mdl_terminate_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static void mdl_setup_runtime_resources_c3_LKS_tcp11
  (SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void mdl_cleanup_runtime_resources_c3_LKS_tcp11
  (SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void enable_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void disable_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void sf_gateway_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void ext_mode_exec_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static void c3_update_jit_animation_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static void c3_do_animation_call_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static const mxArray *get_sim_state_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance);
static void set_sim_state_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray *c3_st);
static c3_tcpclient *c3_tcpclient_tcpclient(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_tcpclient *c3_obj);
static void c3_TCPClient_initializeChannel(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_network_internal_TCPClient *c3_obj,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_iobj_0,
  c3_matlabshared_asyncio_internal_Channel *c3_iobj_1);
static void c3_sprintf(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, uint32_T c3_varargin_1, c3_coder_array_char_T_2D *c3_str);
static void c3_API_dispatchInternalError(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, char_T c3_errorID[1024], char_T
  c3_errorText[1024]);
static void c3_API_trimString(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, char_T c3_in[1024], char_T c3_out_data[], int32_T
  c3_out_size[2]);
static boolean_T c3_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, char_T c3_b_data[], int32_T c3_b_size[2]);
static void c3_API_channelErrorIfFailed(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, CoderChannel c3_chImpl, int32_T
  c3_success);
static c3_matlabshared_asyncio_internal_InputStream *c3_InputStream_InputStream
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp,
   c3_matlabshared_asyncio_internal_InputStream *c3_obj, CoderChannel
   c3_channelImpl);
static void c3_InputStream_clearPartialPacket(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_asyncio_internal_InputStream *c3_obj);
static c3_matlabshared_asyncio_internal_OutputStream
  *c3_OutputStream_OutputStream(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, c3_matlabshared_asyncio_internal_OutputStream *c3_obj,
  CoderChannel c3_channelImpl);
static void c3_TCPClient_validateDisconnected(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_network_internal_TCPClient *c3_obj);
static void c3_AsyncIOTransportChannel_writeAsyncRaw
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp,
   c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_obj,
   uint8_T c3_data[88]);
static void c3_b_API_channelErrorIfFailed(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, CoderChannel c3_chImpl);
static emlrtTimespec c3_tic(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp);
static emlrtTimespec c3_getTime(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp);
static real_T c3_toc(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
                     emlrtStack *c3_sp, emlrtTimespec c3_tstart);
static void c3_pause(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
                     emlrtStack *c3_sp, real_T c3_varargin_1);
static void c3_OutputStream_drain(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, c3_matlabshared_asyncio_internal_OutputStream *c3_obj);
static void c3_warning(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp);
static void c3_Channel_close(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_obj);
static real_T c3_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_a__output_of_length_, const char_T *c3_identifier);
static real_T c3_b_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId);
static void c3_c_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_tmpStr, const char_T *c3_identifier,
  c3_coder_array_char_T_2D *c3_y);
static void c3_d_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_u, const emlrtMsgIdentifier
  *c3_parentId, c3_coder_array_char_T_2D *c3_y);
static void c3_e_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_a__output_of_matlabroot_, const char_T *c3_identifier,
  char_T c3_y_data[], int32_T c3_y_size[2]);
static void c3_f_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId, char_T c3_y_data[],
  int32_T c3_y_size[2]);
static boolean_T c3_g_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray
  *c3_a__output_of_coder_internal_ifWhileCondExtrinsic_, const char_T
  *c3_identifier);
static boolean_T c3_h_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId);
static const mxArray *c3_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1,
  const mxArray *c3_input2);
static const mxArray *c3_length(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0);
static const mxArray *c3_ver(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0);
static const mxArray *c3_getfield(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_matlabroot(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp);
static const mxArray *c3_b_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0);
static const mxArray *c3_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_b_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0);
static const mxArray *c3_b_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_c_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0);
static const mxArray *c3_matlabRelease(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp);
static const mxArray *c3_b_getfield(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_b_matlabroot(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp);
static const mxArray *c3_c_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_d_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0);
static const mxArray *c3_c_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_e_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0);
static void c3_b_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static const mxArray *c3_c_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static void c3_d_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1);
static void c3_array_char_T_2D_SetSize(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_coder_array_char_T_2D
  *c3_coderArray, const emlrtRTEInfo *c3_srcLocation, int32_T c3_size0, int32_T
  c3_size1);
static void c3_array_uint8_T_2D_SetSize(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_coder_array_uint8_T_2D
  *c3_coderArray, const emlrtRTEInfo *c3_srcLocation, int32_T c3_size0, int32_T
  c3_size1);
static void c3_array_tcpclient_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_tcpclient *c3_pStruct);
static void c3_array_matlabshared_network_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_network_internal_TCPClient *c3_pStruct);
static void c3_array_char_T_2D_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_char_T_2D *c3_coderArray);
static void c3_array_matlabshared_transportlib(SFc3_LKS_tcp11InstanceStruct
  *chartInstance,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_pStruct);
static void c3_array_matlabshared_asyncio_buff(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_pStruct);
static void c3_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_InputStream *c3_pStruct);
static void c3_array_uint8_T_2D_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_uint8_T_2D *c3_coderArray);
static void c3_b_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_Channel *c3_pStruct);
static void c3_array_char_T_2D_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_char_T_2D *c3_coderArray);
static void c3_b_array_matlabshared_network_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_network_internal_TCPClient *c3_pStruct);
static void c3_array_uint8_T_2D_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_uint8_T_2D *c3_coderArray);
static void c3_c_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_InputStream *c3_pStruct);
static void c3_b_array_matlabshared_asyncio_buff(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_pStruct);
static void c3_b_array_matlabshared_transportlib(SFc3_LKS_tcp11InstanceStruct
  *chartInstance,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_pStruct);
static void c3_d_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_Channel *c3_pStruct);
static void c3_array_tcpclient_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_tcpclient *c3_pStruct);
static void c3_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_Qyu6eoJFT0AYGGE5WaAhtD *c3_pStruct);
static void c3_array_cell_17_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_17 *c3_pStruct);
static void c3_b_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_Qyu6eoJFT0AYGGE5WaAhtD *c3_pStruct);
static void c3_array_cell_17_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_17 *c3_pStruct);
static void c3_array_s_HTCilNNUmm0Yd43AIdnmID_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_HTCilNNUmm0Yd43AIdnmID *c3_pStruct);
static void c3_array_cell_7_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_7 *c3_pStruct);
static void c3_b_array_s_HTCilNNUmm0Yd43AIdnmID_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_HTCilNNUmm0Yd43AIdnmID *c3_pStruct);
static void c3_array_cell_7_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_7 *c3_pStruct);
static void init_dsm_address_info(SFc3_LKS_tcp11InstanceStruct *chartInstance);
static void init_simulink_io_address(SFc3_LKS_tcp11InstanceStruct *chartInstance);

/* Function Definitions */
static void initialize_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  sim_mode_is_external(chartInstance->S);
  chartInstance->c3_tcpClient_not_empty = false;
  chartInstance->c3_tcpClient.TCPClientObj._pobj1.InputStream.matlabCodegenIsDeleted
    = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer.InputStream.matlabCodegenIsDeleted
    = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj1.OutputStream.matlabCodegenIsDeleted
    = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer.OutputStream.matlabCodegenIsDeleted
    = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj1.matlabCodegenIsDeleted = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer.matlabCodegenIsDeleted
    = true;
  chartInstance->c3_tcpClient.TCPClientObj._pobj0.matlabCodegenIsDeleted = true;
  chartInstance->c3_tcpClient.TCPClientObj.matlabCodegenIsDeleted = true;
  chartInstance->c3_tcpClient.matlabCodegenIsDeleted = true;
  chartInstance->c3_doneDoubleBufferReInit = false;
  chartInstance->c3_sfEvent = CALL_EVENT;
  _sfTime_ = sf_get_time(chartInstance->S);
}

static void initialize_params_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  (void)chartInstance;
}

static void mdl_start_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  c3_array_tcpclient_Constructor(chartInstance, &chartInstance->c3_tcpClient);
  sim_mode_is_external(chartInstance->S);
}

static void mdl_terminate_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderChannel c3_e_chImpl;
  CoderChannel c3_f_chImpl;
  CoderChannel c3_g_chImpl;
  CoderChannel c3_h_chImpl;
  CoderChannel c3_i_chImpl;
  CoderChannel c3_j_chImpl;
  CoderChannel c3_k_chImpl;
  CoderChannel c3_l_chImpl;
  CoderChannel c3_m_chImpl;
  CoderChannel c3_n_chImpl;
  CoderChannel c3_o_chImpl;
  CoderChannel c3_p_chImpl;
  CoderChannel c3_q_chImpl;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_bb_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_j_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_p_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_q_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_w_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_y_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_cb_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_db_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_i_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_l_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_m_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_n_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_o_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_s_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_t_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_v_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_ab_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_x_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_r_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_u_obj;
  c3_matlabshared_network_internal_TCPClient *c3_b_obj;
  c3_matlabshared_network_internal_TCPClient *c3_d_obj;
  c3_matlabshared_network_internal_TCPClient *c3_f_obj;
  c3_matlabshared_network_internal_TCPClient *c3_g_obj;
  c3_matlabshared_network_internal_TCPClient *c3_h_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_e_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_k_obj;
  c3_tcpclient *c3_c_obj;
  c3_tcpclient *c3_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_g_st;
  emlrtStack c3_h_st;
  emlrtStack c3_i_st;
  emlrtStack c3_st = { NULL,           /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  int32_T c3_b_success;
  int32_T c3_c_success;
  int32_T c3_d_success;
  int32_T c3_e_success;
  int32_T c3_f_success;
  int32_T c3_g_success;
  int32_T c3_h_success;
  int32_T c3_i_success;
  int32_T c3_j_success;
  int32_T c3_k_success;
  int32_T c3_success;
  boolean_T c3_b_result;
  boolean_T c3_c_result;
  boolean_T c3_d_result;
  boolean_T c3_e_result;
  boolean_T c3_f_result;
  boolean_T c3_result;
  c3_st.tls = chartInstance->c3_fEmlrtCtx;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_g_st.prev = &c3_f_st;
  c3_g_st.tls = c3_f_st.tls;
  c3_h_st.prev = &c3_g_st;
  c3_h_st.tls = c3_g_st.tls;
  c3_i_st.prev = &c3_h_st;
  c3_i_st.tls = c3_h_st.tls;
  c3_obj = &chartInstance->c3_tcpClient;
  if (!c3_obj->matlabCodegenIsDeleted) {
    c3_obj->matlabCodegenIsDeleted = true;
    c3_c_st.site = &c3_de_emlrtRSI;
    c3_c_obj = c3_obj;
    c3_d_st.site = &c3_ee_emlrtRSI;
    c3_d_obj = &c3_c_obj->TCPClientObj;
    c3_e_st.site = &c3_m_emlrtRSI;
    c3_g_obj = c3_d_obj;
    c3_f_st.site = &c3_m_emlrtRSI;
    c3_i_obj = c3_g_obj->AsyncIOChannel;
    c3_g_st.site = &c3_fe_emlrtRSI;
    c3_m_obj = c3_i_obj;
    c3_h_st.site = &c3_yb_emlrtRSI;
    c3_chImpl = c3_m_obj->ChannelImpl;
    c3_success = coderChannelIsOpen(c3_chImpl, &c3_result);
    c3_i_st.site = &c3_ac_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_i_st, c3_chImpl, c3_success);
    if (c3_result) {
      c3_g_st.site = &c3_ge_emlrtRSI;
      c3_d_chImpl = c3_i_obj->ChannelImpl;
      c3_d_success = coderChannelClose(c3_d_chImpl);
      c3_h_st.site = &c3_he_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_h_st, c3_d_chImpl,
        c3_d_success);
    }
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_b_obj = &chartInstance->c3_tcpClient.TCPClientObj;
  if (!c3_b_obj->matlabCodegenIsDeleted) {
    c3_b_obj->matlabCodegenIsDeleted = true;
    c3_c_st.site = &c3_de_emlrtRSI;
    c3_f_obj = c3_b_obj;
    c3_d_st.site = &c3_m_emlrtRSI;
    c3_h_obj = c3_f_obj;
    c3_e_st.site = &c3_m_emlrtRSI;
    c3_l_obj = c3_h_obj->AsyncIOChannel;
    c3_f_st.site = &c3_fe_emlrtRSI;
    c3_n_obj = c3_l_obj;
    c3_g_st.site = &c3_yb_emlrtRSI;
    c3_b_chImpl = c3_n_obj->ChannelImpl;
    c3_b_success = coderChannelIsOpen(c3_b_chImpl, &c3_b_result);
    c3_h_st.site = &c3_ac_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_h_st, c3_b_chImpl,
      c3_b_success);
    if (c3_b_result) {
      c3_f_st.site = &c3_ge_emlrtRSI;
      c3_e_chImpl = c3_l_obj->ChannelImpl;
      c3_e_success = coderChannelClose(c3_e_chImpl);
      c3_g_st.site = &c3_he_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_e_chImpl,
        c3_e_success);
    }
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_e_obj = &chartInstance->c3_tcpClient.TCPClientObj._pobj0;
  if (!c3_e_obj->matlabCodegenIsDeleted) {
    c3_e_obj->matlabCodegenIsDeleted = true;
    c3_c_st.site = &c3_de_emlrtRSI;
    c3_k_obj = c3_e_obj;
    c3_d_st.site = &c3_lb_emlrtRSI;
    c3_Channel_close(chartInstance, &c3_d_st, &c3_k_obj->UnreadDataBuffer);
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_j_obj = &chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer;
  if (!c3_j_obj->matlabCodegenIsDeleted) {
    c3_j_obj->matlabCodegenIsDeleted = true;
    c3_c_st.site = &c3_de_emlrtRSI;
    c3_p_obj = c3_j_obj;
    c3_d_st.site = &c3_ye_emlrtRSI;
    c3_q_obj = c3_p_obj;
    c3_e_st.site = &c3_yb_emlrtRSI;
    c3_c_chImpl = c3_q_obj->ChannelImpl;
    c3_c_success = coderChannelIsOpen(c3_c_chImpl, &c3_c_result);
    c3_f_st.site = &c3_ac_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_c_chImpl,
      c3_c_success);
    if (c3_c_result) {
      c3_d_st.site = &c3_af_emlrtRSI;
      c3_Channel_close(chartInstance, &c3_d_st, c3_p_obj);
    }

    c3_c_st.site = &c3_de_emlrtRSI;
    c3_w_obj = c3_j_obj;
    c3_d_st.site = &c3_ie_emlrtRSI;
    c3_y_obj = c3_w_obj;
    c3_h_chImpl = 0;
    if (!(c3_y_obj->ChannelImpl == c3_h_chImpl)) {
      c3_e_st.site = &c3_je_emlrtRSI;
      c3_bb_obj = c3_y_obj;
      c3_f_st.site = &c3_yb_emlrtRSI;
      c3_j_chImpl = c3_bb_obj->ChannelImpl;
      c3_h_success = coderChannelIsOpen(c3_j_chImpl, &c3_e_result);
      c3_g_st.site = &c3_ac_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_j_chImpl,
        c3_h_success);
      if (c3_e_result) {
        c3_e_st.site = &c3_ke_emlrtRSI;
        c3_warning(chartInstance, &c3_e_st);
        c3_e_st.site = &c3_le_emlrtRSI;
        c3_Channel_close(chartInstance, &c3_e_st, c3_y_obj);
      }

      c3_e_st.site = &c3_me_emlrtRSI;
      c3_n_chImpl = c3_y_obj->ChannelImpl;
      c3_j_success = coderChannelTerm(c3_n_chImpl);
      c3_f_st.site = &c3_pe_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_n_chImpl,
        c3_j_success);
      c3_e_st.site = &c3_ne_emlrtRSI;
      c3_p_chImpl = c3_y_obj->ChannelImpl;
      coderChannelDestroy(c3_p_chImpl);
      c3_q_chImpl = 0;
      c3_y_obj->ChannelImpl = c3_q_chImpl;
    }
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_o_obj = &chartInstance->c3_tcpClient.TCPClientObj._pobj1;
  if (!c3_o_obj->matlabCodegenIsDeleted) {
    c3_o_obj->matlabCodegenIsDeleted = true;
    c3_c_st.site = &c3_de_emlrtRSI;
    c3_s_obj = c3_o_obj;
    c3_d_st.site = &c3_ie_emlrtRSI;
    c3_t_obj = c3_s_obj;
    c3_f_chImpl = 0;
    if (!(c3_t_obj->ChannelImpl == c3_f_chImpl)) {
      c3_e_st.site = &c3_je_emlrtRSI;
      c3_v_obj = c3_t_obj;
      c3_f_st.site = &c3_yb_emlrtRSI;
      c3_g_chImpl = c3_v_obj->ChannelImpl;
      c3_f_success = coderChannelIsOpen(c3_g_chImpl, &c3_d_result);
      c3_g_st.site = &c3_ac_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_g_chImpl,
        c3_f_success);
      if (c3_d_result) {
        c3_e_st.site = &c3_ke_emlrtRSI;
        c3_warning(chartInstance, &c3_e_st);
        c3_e_st.site = &c3_le_emlrtRSI;
        c3_cb_obj = c3_t_obj;
        c3_f_st.site = &c3_fe_emlrtRSI;
        c3_db_obj = c3_cb_obj;
        c3_g_st.site = &c3_yb_emlrtRSI;
        c3_l_chImpl = c3_db_obj->ChannelImpl;
        c3_i_success = coderChannelIsOpen(c3_l_chImpl, &c3_f_result);
        c3_h_st.site = &c3_ac_emlrtRSI;
        c3_API_channelErrorIfFailed(chartInstance, &c3_h_st, c3_l_chImpl,
          c3_i_success);
        if (c3_f_result) {
          c3_f_st.site = &c3_ge_emlrtRSI;
          c3_o_chImpl = c3_cb_obj->ChannelImpl;
          c3_k_success = coderChannelClose(c3_o_chImpl);
          c3_g_st.site = &c3_he_emlrtRSI;
          c3_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_o_chImpl,
            c3_k_success);
        }
      }

      c3_e_st.site = &c3_me_emlrtRSI;
      c3_i_chImpl = c3_t_obj->ChannelImpl;
      c3_g_success = coderChannelTerm(c3_i_chImpl);
      c3_f_st.site = &c3_pe_emlrtRSI;
      c3_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_i_chImpl,
        c3_g_success);
      c3_e_st.site = &c3_ne_emlrtRSI;
      c3_k_chImpl = c3_t_obj->ChannelImpl;
      coderChannelDestroy(c3_k_chImpl);
      c3_m_chImpl = 0;
      c3_t_obj->ChannelImpl = c3_m_chImpl;
    }
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_r_obj =
    &chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer.OutputStream;
  if (!c3_r_obj->matlabCodegenIsDeleted) {
    c3_r_obj->matlabCodegenIsDeleted = true;
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_u_obj = &chartInstance->c3_tcpClient.TCPClientObj._pobj1.OutputStream;
  if (!c3_u_obj->matlabCodegenIsDeleted) {
    c3_u_obj->matlabCodegenIsDeleted = true;
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_x_obj =
    &chartInstance->c3_tcpClient.TCPClientObj._pobj0.UnreadDataBuffer.InputStream;
  if (!c3_x_obj->matlabCodegenIsDeleted) {
    c3_x_obj->matlabCodegenIsDeleted = true;
  }

  c3_b_st.site = &c3_bf_emlrtRSI;
  c3_ab_obj = &chartInstance->c3_tcpClient.TCPClientObj._pobj1.InputStream;
  if (!c3_ab_obj->matlabCodegenIsDeleted) {
    c3_ab_obj->matlabCodegenIsDeleted = true;
  }

  c3_array_tcpclient_Destructor(chartInstance, &chartInstance->c3_tcpClient);
}

static void mdl_setup_runtime_resources_c3_LKS_tcp11
  (SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  static const uint32_T c3_decisionTxtEndIdx = 0U;
  static const uint32_T c3_decisionTxtStartIdx = 0U;
  sfSetAnimationVectors(chartInstance->S, &chartInstance->c3_JITStateAnimation[0],
                        &chartInstance->c3_JITTransitionAnimation[0]);
  covrtCreateStateflowInstanceData(chartInstance->c3_covrtInstance, 1U, 0U, 1U,
    62U);
  covrtChartInitFcn(chartInstance->c3_covrtInstance, 0U, false, false, false);
  covrtStateInitFcn(chartInstance->c3_covrtInstance, 0U, 0U, false, false, false,
                    0U, &c3_decisionTxtStartIdx, &c3_decisionTxtEndIdx);
  covrtTransInitFcn(chartInstance->c3_covrtInstance, 0U, 0, NULL, NULL, 0U, NULL);
  covrtEmlInitFcn(chartInstance->c3_covrtInstance, "", 4U, 0U, 1U, 0U, 1U, 0U,
                  0U, 0U, 0U, 0U, 0U, 0U);
  covrtEmlFcnInitFcn(chartInstance->c3_covrtInstance, 4U, 0U, 0U, "c3_LKS_tcp11",
                     0, -1, 686);
  covrtEmlIfInitFcn(chartInstance->c3_covrtInstance, 4U, 0U, 0U, 223, 244, -1,
                    317, false);
}

static void mdl_cleanup_runtime_resources_c3_LKS_tcp11
  (SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  covrtDeleteStateflowInstanceData(chartInstance->c3_covrtInstance);
}

static void enable_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  _sfTime_ = sf_get_time(chartInstance->S);
}

static void disable_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  _sfTime_ = sf_get_time(chartInstance->S);
}

static void sf_gateway_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  static char_T c3_b_cv[45] = { 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', 'l',
    'i', 'b', ':', 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', ':', 'i', 'n',
    'v', 'a', 'l', 'i', 'd', 'C', 'o', 'n', 'n', 'e', 'c', 't', 'i', 'o', 'n',
    'S', 't', 'a', 't', 'e' };

  static char_T c3_b_cv1[13] = { 'r', 'e', 'm', 'o', 't', 'e', ' ', 's', 'e',
    'r', 'v', 'e', 'r' };

  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  c3_cell_24 c3_args;
  c3_coder_array_char_T_2D c3_b_out;
  c3_matlabshared_asyncio_internal_Channel *c3_f_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_k_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_l_obj;
  c3_matlabshared_network_internal_TCPClient *c3_b_obj;
  c3_matlabshared_network_internal_TCPClient *c3_c_obj;
  c3_matlabshared_network_internal_TCPClient *c3_d_obj;
  c3_matlabshared_network_internal_TCPClient *c3_varargin_1;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co
    *c3_b_varargin_1;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co
    *c3_c_varargin_1;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_e_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_g_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_h_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_i_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_j_obj;
  c3_sssYvN9TzuAOXEmc5gNaOX c3_options;
  c3_tcpclient *c3_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_g_st;
  emlrtStack c3_h_st;
  emlrtStack c3_i_st;
  emlrtStack c3_j_st;
  emlrtStack c3_k_st;
  emlrtStack c3_l_st;
  emlrtStack c3_m_st;
  emlrtStack c3_st = { NULL,           /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_e_y = NULL;
  const mxArray *c3_f_y = NULL;
  const mxArray *c3_g_y = NULL;
  const mxArray *c3_h_y = NULL;
  const mxArray *c3_i_y = NULL;
  const mxArray *c3_j_y = NULL;
  const mxArray *c3_l_y = NULL;
  const mxArray *c3_m_y = NULL;
  const mxArray *c3_n_y = NULL;
  const mxArray *c3_p_y = NULL;
  const mxArray *c3_q_y = NULL;
  const mxArray *c3_y = NULL;
  real_T c3_x[11];
  real_T c3_b_LCState;
  real_T c3_b_brakeState;
  real_T c3_b_ey;
  real_T c3_b_max_theta;
  real_T c3_b_offState;
  real_T c3_b_output_edelta_t;
  real_T c3_b_raw_theta;
  real_T c3_b_systemState;
  real_T c3_b_theta;
  uint64_T c3_b_exampleValue;
  uint64_T c3_exampleValue;
  uint64_T c3_numBytesWritten;
  int32_T c3_b_kstr;
  int32_T c3_b_success;
  int32_T c3_c_success;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i2;
  int32_T c3_i3;
  int32_T c3_i4;
  int32_T c3_i5;
  int32_T c3_i6;
  int32_T c3_kstr;
  int32_T c3_loop_ub;
  int32_T c3_remainingDimsA;
  int32_T c3_success;
  char_T c3_b[13];
  char_T c3_b_s;
  char_T c3_c_s;
  char_T c3_c_x;
  char_T c3_d_s;
  char_T c3_d_x;
  char_T c3_e_s;
  char_T c3_e_x;
  char_T c3_f_s;
  char_T c3_f_x;
  char_T c3_g_s;
  char_T c3_h_s;
  char_T c3_k_y;
  char_T c3_o_y;
  char_T c3_s;
  int8_T c3_b_u;
  int8_T c3_c_u;
  int8_T c3_d_u;
  int8_T c3_u;
  uint8_T c3_b_x[88];
  uint8_T c3_data[88];
  boolean_T c3_b1;
  boolean_T c3_b2;
  boolean_T c3_b3;
  boolean_T c3_b_LDState;
  boolean_T c3_b_b;
  boolean_T c3_b_p;
  boolean_T c3_b_vState;
  boolean_T c3_b_value;
  boolean_T c3_c_p;
  boolean_T c3_d_p;
  boolean_T c3_exitg1;
  boolean_T c3_out;
  boolean_T c3_p;
  boolean_T c3_value;
  c3_st.tls = chartInstance->c3_fEmlrtCtx;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_g_st.prev = &c3_f_st;
  c3_g_st.tls = c3_f_st.tls;
  c3_h_st.prev = &c3_g_st;
  c3_h_st.tls = c3_g_st.tls;
  c3_i_st.prev = &c3_h_st;
  c3_i_st.tls = c3_h_st.tls;
  c3_j_st.prev = &c3_i_st;
  c3_j_st.tls = c3_i_st.tls;
  c3_k_st.prev = &c3_j_st;
  c3_k_st.tls = c3_j_st.tls;
  c3_l_st.prev = &c3_k_st;
  c3_l_st.tls = c3_k_st.tls;
  c3_m_st.prev = &c3_l_st;
  c3_m_st.tls = c3_l_st.tls;
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 10U,
                    *chartInstance->c3_output_edelta_t);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 9U,
                    *chartInstance->c3_max_theta);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 8U,
                    *chartInstance->c3_raw_theta);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 7U,
                    *chartInstance->c3_theta);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 6U, *chartInstance->c3_ey);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 5U,
                    *chartInstance->c3_LCState);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 4U,
                    *chartInstance->c3_brakeState);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 3U, (real_T)
                    *chartInstance->c3_LDState);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 2U, (real_T)
                    *chartInstance->c3_vState);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 1U,
                    *chartInstance->c3_offState);
  covrtSigUpdateFcn(chartInstance->c3_covrtInstance, 0U,
                    *chartInstance->c3_systemState);
  _sfTime_ = sf_get_time(chartInstance->S);
  chartInstance->c3_JITTransitionAnimation[0] = 0U;
  chartInstance->c3_sfEvent = CALL_EVENT;
  c3_b_systemState = *chartInstance->c3_systemState;
  c3_b_offState = *chartInstance->c3_offState;
  c3_b_vState = *chartInstance->c3_vState;
  c3_b_LDState = *chartInstance->c3_LDState;
  c3_b_brakeState = *chartInstance->c3_brakeState;
  c3_b_LCState = *chartInstance->c3_LCState;
  c3_b_ey = *chartInstance->c3_ey;
  c3_b_theta = *chartInstance->c3_theta;
  c3_b_raw_theta = *chartInstance->c3_raw_theta;
  c3_b_max_theta = *chartInstance->c3_max_theta;
  c3_b_output_edelta_t = *chartInstance->c3_output_edelta_t;
  covrtEmlFcnEval(chartInstance->c3_covrtInstance, 4U, 0, 0);
  if (covrtEmlIfEval(chartInstance->c3_covrtInstance, 4U, 0, 0,
                     !chartInstance->c3_tcpClient_not_empty)) {
    c3_b_st.site = &c3_emlrtRSI;
    c3_tcpclient_tcpclient(chartInstance, &c3_b_st, &chartInstance->c3_tcpClient);
    chartInstance->c3_tcpClient_not_empty = true;
  }

  c3_b_st.site = &c3_b_emlrtRSI;
  c3_x[0] = c3_b_systemState;
  c3_x[1] = c3_b_offState;
  c3_x[2] = (real_T)c3_b_vState;
  c3_x[3] = (real_T)c3_b_LDState;
  c3_x[4] = c3_b_brakeState;
  c3_x[5] = c3_b_LCState;
  c3_x[6] = c3_b_ey;
  c3_x[7] = c3_b_theta;
  c3_x[8] = c3_b_raw_theta;
  c3_x[9] = c3_b_max_theta;
  c3_x[10] = c3_b_output_edelta_t;
  memcpy((void *)&c3_data[0], (void *)&c3_x[0], (uint32_T)((size_t)88 * sizeof
          (uint8_T)));
  c3_b_st.site = &c3_c_emlrtRSI;
  c3_obj = &chartInstance->c3_tcpClient;
  c3_c_st.site = &c3_gc_emlrtRSI;
  c3_b_obj = &c3_obj->TCPClientObj;
  c3_d_st.site = &c3_m_emlrtRSI;
  c3_varargin_1 = c3_b_obj;
  c3_e_st.site = &c3_m_emlrtRSI;
  c3_c_obj = c3_varargin_1;
  c3_f_st.site = &c3_m_emlrtRSI;
  c3_d_obj = c3_c_obj;
  c3_g_st.site = &c3_m_emlrtRSI;
  c3_e_obj = c3_d_obj->TransportChannel;
  c3_h_st.site = &c3_lb_emlrtRSI;
  c3_f_obj = c3_e_obj->AsyncIOChannel;
  c3_i_st.site = &c3_yb_emlrtRSI;
  c3_chImpl = c3_f_obj->ChannelImpl;
  c3_success = coderChannelIsOpen(c3_chImpl, &c3_out);
  c3_j_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_j_st, c3_chImpl, c3_success);
  if (c3_out) {
    c3_value = true;
  } else {
    c3_value = false;
  }

  c3_b_value = c3_value;
  if (!c3_b_value) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 45),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 45),
                  false);
    c3_c_y = NULL;
    sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 13),
                  false);
    sf_mex_call(&c3_e_st, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(&c3_e_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_e_st, NULL, "message", 1U, 2U, 14, c3_b_y, 14, c3_c_y)));
  }

  c3_e_st.site = &c3_m_emlrtRSI;
  c3_b_varargin_1 = c3_varargin_1->TransportChannel;
  c3_g_obj = c3_b_varargin_1;
  c3_f_st.site = &c3_lb_emlrtRSI;
  c3_c_varargin_1 = c3_b_varargin_1;
  c3_array_char_T_2D_Constructor(chartInstance, &c3_b_out);
  c3_g_st.site = &c3_lb_emlrtRSI;
  c3_h_obj = c3_c_varargin_1;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_g_st, &c3_b_out, &c3_emlrtRTEI,
    1, c3_h_obj->ByteOrder.size[1]);
  c3_loop_ub = c3_h_obj->ByteOrder.size[1] - 1;
  for (c3_i = 0; c3_i <= c3_loop_ub; c3_i++) {
    c3_b_out.vector.data[c3_i] = c3_h_obj->ByteOrder.vector.data[c3_i];
  }

  c3_g_st.site = &c3_lb_emlrtRSI;
  c3_i_obj = c3_c_varargin_1;
  c3_h_st.site = &c3_mb_emlrtRSI;
  for (c3_i1 = 0; c3_i1 < 13; c3_i1++) {
    c3_b[c3_i1] = c3_i_obj->MachineByteOrder[c3_i1];
  }

  c3_i_st.site = &c3_hc_emlrtRSI;
  c3_j_st.site = &c3_ic_emlrtRSI;
  c3_remainingDimsA = c3_b_out.size[1];
  if (c3_remainingDimsA != 13) {
  } else {
    c3_kstr = 1;
    c3_exitg1 = false;
    while ((!c3_exitg1) && (c3_kstr - 1 < 13)) {
      c3_b_kstr = c3_kstr - 1;
      c3_k_st.site = &c3_jc_emlrtRSI;
      c3_s = c3_b_out.vector.data[c3_b_kstr];
      c3_b_s = c3_s;
      c3_b_b = ((uint8_T)c3_b_s <= 127);
      c3_p = c3_b_b;
      if (!c3_p) {
        c3_d_y = NULL;
        sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_e_y = NULL;
        sf_mex_assign(&c3_e_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_u = MAX_int8_T;
        c3_f_y = NULL;
        sf_mex_assign(&c3_f_y, sf_mex_create("y", &c3_u, 2, 0U, 0, 0U, 0), false);
        sf_mex_call(&c3_k_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_d_y, 14,
                    sf_mex_call(&c3_k_st, NULL, "getString", 1U, 1U, 14,
          sf_mex_call(&c3_k_st, NULL, "message", 1U, 2U, 14, c3_e_y, 14, c3_f_y)));
      }

      c3_k_st.site = &c3_kc_emlrtRSI;
      c3_c_s = c3_b[c3_b_kstr];
      c3_d_s = c3_c_s;
      c3_b1 = ((uint8_T)c3_d_s <= 127);
      c3_b_p = c3_b1;
      if (!c3_b_p) {
        c3_g_y = NULL;
        sf_mex_assign(&c3_g_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_h_y = NULL;
        sf_mex_assign(&c3_h_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_b_u = MAX_int8_T;
        c3_i_y = NULL;
        sf_mex_assign(&c3_i_y, sf_mex_create("y", &c3_b_u, 2, 0U, 0, 0U, 0),
                      false);
        sf_mex_call(&c3_k_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_g_y, 14,
                    sf_mex_call(&c3_k_st, NULL, "getString", 1U, 1U, 14,
          sf_mex_call(&c3_k_st, NULL, "message", 1U, 2U, 14, c3_h_y, 14, c3_i_y)));
      }

      c3_k_st.site = &c3_lc_emlrtRSI;
      c3_c_x = c3_b_out.vector.data[c3_b_kstr];
      c3_l_st.site = &c3_mc_emlrtRSI;
      c3_d_x = c3_c_x;
      c3_m_st.site = &c3_nc_emlrtRSI;
      c3_e_s = c3_d_x;
      c3_f_s = c3_e_s;
      c3_b2 = ((uint8_T)c3_f_s <= 127);
      c3_c_p = c3_b2;
      if (!c3_c_p) {
        c3_j_y = NULL;
        sf_mex_assign(&c3_j_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_l_y = NULL;
        sf_mex_assign(&c3_l_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_c_u = MAX_int8_T;
        c3_m_y = NULL;
        sf_mex_assign(&c3_m_y, sf_mex_create("y", &c3_c_u, 2, 0U, 0, 0U, 0),
                      false);
        sf_mex_call(&c3_m_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_j_y, 14,
                    sf_mex_call(&c3_m_st, NULL, "getString", 1U, 1U, 14,
          sf_mex_call(&c3_m_st, NULL, "message", 1U, 2U, 14, c3_l_y, 14, c3_m_y)));
      }

      c3_k_y = c3_cv[(uint8_T)c3_d_x & 127];
      c3_k_st.site = &c3_lc_emlrtRSI;
      c3_e_x = c3_b[c3_b_kstr];
      c3_l_st.site = &c3_mc_emlrtRSI;
      c3_f_x = c3_e_x;
      c3_m_st.site = &c3_nc_emlrtRSI;
      c3_g_s = c3_f_x;
      c3_h_s = c3_g_s;
      c3_b3 = ((uint8_T)c3_h_s <= 127);
      c3_d_p = c3_b3;
      if (!c3_d_p) {
        c3_n_y = NULL;
        sf_mex_assign(&c3_n_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_p_y = NULL;
        sf_mex_assign(&c3_p_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
          31), false);
        c3_d_u = MAX_int8_T;
        c3_q_y = NULL;
        sf_mex_assign(&c3_q_y, sf_mex_create("y", &c3_d_u, 2, 0U, 0, 0U, 0),
                      false);
        sf_mex_call(&c3_m_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_n_y, 14,
                    sf_mex_call(&c3_m_st, NULL, "getString", 1U, 1U, 14,
          sf_mex_call(&c3_m_st, NULL, "message", 1U, 2U, 14, c3_p_y, 14, c3_q_y)));
      }

      c3_o_y = c3_cv[(uint8_T)c3_f_x & 127];
      if (c3_k_y != c3_o_y) {
        c3_exitg1 = true;
      } else {
        c3_kstr++;
      }
    }
  }

  c3_array_char_T_2D_Destructor(chartInstance, &c3_b_out);
  c3_g_st.site = &c3_lb_emlrtRSI;
  for (c3_i2 = 0; c3_i2 < 88; c3_i2++) {
    c3_b_x[c3_i2] = c3_data[c3_i2];
  }

  memcpy((void *)&c3_data[0], (void *)&c3_b_x[0], (uint32_T)((size_t)88 * sizeof
          (uint8_T)));
  if (c3_c_varargin_1->WriteAsync) {
    c3_g_st.site = &c3_lb_emlrtRSI;
    c3_AsyncIOTransportChannel_writeAsyncRaw(chartInstance, &c3_g_st,
      c3_c_varargin_1, c3_data);
  } else {
    c3_g_st.site = &c3_lb_emlrtRSI;
    c3_j_obj = c3_c_varargin_1;
    for (c3_i3 = 0; c3_i3 < 88; c3_i3++) {
      c3_options.Data[c3_i3] = c3_data[c3_i3];
    }

    c3_h_st.site = &c3_lb_emlrtRSI;
    c3_k_obj = c3_j_obj->AsyncIOChannel;
    c3_i_st.site = &c3_sd_emlrtRSI;
    c3_b_chImpl = c3_k_obj->ChannelImpl;
    c3_args.f1 = "Data";
    c3_args.f2 = "uint8";
    for (c3_i4 = 0; c3_i4 < 88; c3_i4++) {
      c3_data[c3_i4] = c3_options.Data[c3_i4];
    }

    for (c3_i5 = 0; c3_i5 < 88; c3_i5++) {
      c3_args.f4[c3_i5] = c3_data[c3_i5];
    }

    for (c3_i6 = 0; c3_i6 < 88; c3_i6++) {
      c3_data[c3_i6] = c3_args.f4[c3_i6];
    }

    c3_b_success = coderChannelExecute(c3_b_chImpl, "Write", 1, c3_args.f1,
      c3_args.f2, 88, &c3_data[0]);
    c3_j_st.site = &c3_td_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_j_st, c3_b_chImpl,
      c3_b_success);
    c3_h_st.site = &c3_lb_emlrtRSI;
    c3_l_obj = c3_j_obj->AsyncIOChannel;
    c3_exampleValue = c3_l_obj->CustomProps.LatestNumBytesWrittenToDevice;
    c3_i_st.site = &c3_ec_emlrtRSI;
    c3_c_chImpl = c3_l_obj->ChannelImpl;
    c3_b_exampleValue = c3_exampleValue;
    c3_numBytesWritten = c3_b_exampleValue;
    c3_c_success = coderChannelGetPropertyValue(c3_c_chImpl,
      "LatestNumBytesWrittenToDevice", "uint64", 1, &c3_numBytesWritten);
    c3_j_st.site = &c3_fc_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_j_st, c3_c_chImpl,
      c3_c_success);
    c3_j_obj->NumBytesWritten += (real_T)c3_numBytesWritten;
  }

  c3_f_st.site = &c3_lb_emlrtRSI;
  c3_OutputStream_drain(chartInstance, &c3_f_st, &c3_g_obj->
                        AsyncIOChannel->OutputStream);
}

static void ext_mode_exec_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  (void)chartInstance;
}

static void c3_update_jit_animation_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  (void)chartInstance;
}

static void c3_do_animation_call_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  (void)chartInstance;
}

static const mxArray *get_sim_state_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance)
{
  const mxArray *c3_st = NULL;
  const mxArray *c3_y = NULL;
  (void)chartInstance;
  c3_st = NULL;
  c3_y = NULL;
  sf_mex_assign(&c3_y, sf_mex_createcellmatrix(0, 1), false);
  sf_mex_assign(&c3_st, c3_y, false);
  return c3_st;
}

static void set_sim_state_c3_LKS_tcp11(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray *c3_st)
{
  const mxArray *c3_u;
  chartInstance->c3_doneDoubleBufferReInit = true;
  c3_u = sf_mex_dup(c3_st);
  sf_mex_destroy(&c3_u);
  sf_mex_destroy(&c3_st);
}

static c3_tcpclient *c3_tcpclient_tcpclient(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_tcpclient *c3_obj)
{
  static char_T c3_b_cv[39] = { 'n', 'e', 't', 'w', 'o', 'r', 'k', ':', 't', 'c',
    'p', 'c', 'l', 'i', 'e', 'n', 't', ':', 'a', 'l', 'r', 'e', 'a', 'd', 'y',
    'C', 'o', 'n', 'n', 'e', 'c', 't', 'e', 'd', 'E', 'r', 'r', 'o', 'r' };

  static char_T c3_hostName[9] = { '1', '2', '7', '.', '0', '.', '0', '.', '1' };

  static char_T c3_val[5] = { 'u', 'i', 'n', 't', '8' };

  static char_T c3_b_val[4] = { 'D', 'a', 't', 'a' };

  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderChannel c3_e_chImpl;
  c3_cell_17 c3_args;
  c3_coder_array_char_T_2D c3_value;
  c3_matlabshared_asyncio_internal_Channel *c3_eb_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_fb_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_gb_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_hb_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_y_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_inputStream;
  c3_matlabshared_asyncio_internal_InputStream *c3_t_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_outputStream;
  c3_matlabshared_asyncio_internal_OutputStream *c3_s_obj;
  c3_matlabshared_network_internal_TCPClient *c3_ab_obj;
  c3_matlabshared_network_internal_TCPClient *c3_b_this;
  c3_matlabshared_network_internal_TCPClient *c3_bb_obj;
  c3_matlabshared_network_internal_TCPClient *c3_c_obj;
  c3_matlabshared_network_internal_TCPClient *c3_c_this;
  c3_matlabshared_network_internal_TCPClient *c3_cb_obj;
  c3_matlabshared_network_internal_TCPClient *c3_d_obj;
  c3_matlabshared_network_internal_TCPClient *c3_d_this;
  c3_matlabshared_network_internal_TCPClient *c3_e_obj;
  c3_matlabshared_network_internal_TCPClient *c3_e_this;
  c3_matlabshared_network_internal_TCPClient *c3_f_obj;
  c3_matlabshared_network_internal_TCPClient *c3_g_obj;
  c3_matlabshared_network_internal_TCPClient *c3_h_obj;
  c3_matlabshared_network_internal_TCPClient *c3_i_obj;
  c3_matlabshared_network_internal_TCPClient *c3_l_obj;
  c3_matlabshared_network_internal_TCPClient *c3_n_obj;
  c3_matlabshared_network_internal_TCPClient *c3_q_obj;
  c3_matlabshared_network_internal_TCPClient *c3_r_obj;
  c3_matlabshared_network_internal_TCPClient *c3_u_obj;
  c3_matlabshared_network_internal_TCPClient *c3_w_obj;
  c3_matlabshared_network_internal_TCPClient *c3_x_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_db_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_j_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_k_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_m_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_o_obj;
  c3_s_Qyu6eoJFT0AYGGE5WaAhtD c3_options;
  c3_tcpclient *c3_b_obj;
  c3_tcpclient *c3_p_obj;
  c3_tcpclient *c3_this;
  c3_tcpclient *c3_v_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_g_st;
  emlrtStack c3_h_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_y = NULL;
  real_T c3_dv[2];
  real_T c3_dv1[2];
  real_T c3_f_value[2];
  real_T c3_d;
  int32_T c3_b_size[2];
  int32_T c3_b_loop_ub;
  int32_T c3_b_success;
  int32_T c3_c_loop_ub;
  int32_T c3_c_success;
  int32_T c3_d_loop_ub;
  int32_T c3_d_success;
  int32_T c3_e_loop_ub;
  int32_T c3_e_success;
  int32_T c3_f_loop_ub;
  int32_T c3_g_loop_ub;
  int32_T c3_h_loop_ub;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i10;
  int32_T c3_i11;
  int32_T c3_i12;
  int32_T c3_i13;
  int32_T c3_i14;
  int32_T c3_i15;
  int32_T c3_i16;
  int32_T c3_i17;
  int32_T c3_i18;
  int32_T c3_i19;
  int32_T c3_i2;
  int32_T c3_i20;
  int32_T c3_i21;
  int32_T c3_i22;
  int32_T c3_i23;
  int32_T c3_i24;
  int32_T c3_i25;
  int32_T c3_i26;
  int32_T c3_i27;
  int32_T c3_i28;
  int32_T c3_i29;
  int32_T c3_i3;
  int32_T c3_i30;
  int32_T c3_i31;
  int32_T c3_i32;
  int32_T c3_i33;
  int32_T c3_i34;
  int32_T c3_i4;
  int32_T c3_i5;
  int32_T c3_i6;
  int32_T c3_i7;
  int32_T c3_i8;
  int32_T c3_i9;
  int32_T c3_i_loop_ub;
  int32_T c3_j_loop_ub;
  int32_T c3_k_loop_ub;
  int32_T c3_l_loop_ub;
  int32_T c3_loop_ub;
  int32_T c3_m_loop_ub;
  int32_T c3_success;
  uint32_T c3_u;
  char_T c3_b_data[512];
  char_T c3_e_value[9];
  boolean_T c3_bv[2];
  boolean_T c3_bv1[2];
  boolean_T c3_g_value[2];
  boolean_T c3_b;
  boolean_T c3_b_exampleValue;
  boolean_T c3_b_result;
  boolean_T c3_b_value;
  boolean_T c3_c_value;
  boolean_T c3_d_value;
  boolean_T c3_exampleValue;
  boolean_T c3_out;
  boolean_T c3_result;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_g_st.prev = &c3_f_st;
  c3_g_st.tls = c3_f_st.tls;
  c3_h_st.prev = &c3_g_st;
  c3_h_st.tls = c3_g_st.tls;
  c3_b_obj = c3_obj;
  c3_st.site = &c3_g_emlrtRSI;
  c3_this = c3_b_obj;
  c3_b_obj = c3_this;
  c3_st.site = &c3_h_emlrtRSI;
  c3_c_obj = &c3_b_obj->TCPClientObj;
  c3_d_obj = c3_c_obj;
  c3_d_obj->InputBufferSize = rtInf;
  c3_d_obj->OutputBufferSize = rtInf;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_e_obj = c3_d_obj;
  c3_d_obj = c3_e_obj;
  c3_c_st.site = &c3_d_emlrtRSI;
  c3_b_this = c3_d_obj;
  c3_d_obj = c3_b_this;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_f_obj = c3_d_obj;
  c3_d_obj = c3_f_obj;
  c3_c_st.site = &c3_e_emlrtRSI;
  c3_c_this = c3_d_obj;
  c3_d_obj = c3_c_this;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_g_obj = c3_d_obj;
  c3_d_obj = c3_g_obj;
  c3_c_st.site = &c3_f_emlrtRSI;
  c3_d_this = c3_d_obj;
  c3_d_obj = c3_d_this;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_e_this = c3_d_obj;
  c3_d_obj = c3_e_this;
  for (c3_i = 0; c3_i < 9; c3_i++) {
    c3_d_obj->RemoteHost[c3_i] = c3_hostName[c3_i];
  }

  c3_d_obj->RemotePort = 25028.0;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_TCPClient_initializeChannel(chartInstance, &c3_b_st, c3_d_obj,
    &c3_d_obj->_pobj0, &c3_d_obj->_pobj1);
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_h_obj = c3_d_obj;
  c3_h_obj->IsWriteOnly = false;
  c3_h_obj->IsSharingPort = false;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_i_obj = c3_d_obj;
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_j_obj = c3_i_obj->TransportChannel;
  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_j_obj->ByteOrder,
    &c3_emlrtRTEI, 1, c3_b_size[1]);
  c3_loop_ub = c3_b_size[1] - 1;
  for (c3_i1 = 0; c3_i1 <= c3_loop_ub; c3_i1++) {
    c3_j_obj->ByteOrder.vector.data[c3_i1] = c3_b_data[c3_i1];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_j_obj->ByteOrder,
    &c3_emlrtRTEI, 1, 13);
  for (c3_i2 = 0; c3_i2 < 13; c3_i2++) {
    c3_j_obj->ByteOrder.vector.data[c3_i2] = c3_cv2[c3_i2];
  }

  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_i_obj->ByteOrder,
    &c3_c_emlrtRTEI, 1, c3_b_size[1]);
  c3_b_loop_ub = c3_b_size[1] - 1;
  for (c3_i3 = 0; c3_i3 <= c3_b_loop_ub; c3_i3++) {
    c3_i_obj->ByteOrder.vector.data[c3_i3] = c3_b_data[c3_i3];
  }

  c3_array_char_T_2D_Constructor(chartInstance, &c3_value);
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_k_obj = c3_i_obj->TransportChannel;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_value, &c3_emlrtRTEI,
    1, c3_k_obj->ByteOrder.size[1]);
  c3_c_loop_ub = c3_k_obj->ByteOrder.size[1] - 1;
  for (c3_i4 = 0; c3_i4 <= c3_c_loop_ub; c3_i4++) {
    c3_value.vector.data[c3_i4] = c3_k_obj->ByteOrder.vector.data[c3_i4];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_i_obj->ByteOrder,
    &c3_c_emlrtRTEI, 1, c3_value.size[1]);
  c3_d_loop_ub = c3_value.size[1] - 1;
  for (c3_i5 = 0; c3_i5 <= c3_d_loop_ub; c3_i5++) {
    c3_i_obj->ByteOrder.vector.data[c3_i5] = c3_value.vector.data[c3_i5];
  }

  c3_b_st.site = &c3_m_emlrtRSI;
  c3_l_obj = c3_d_obj;
  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_l_obj->NativeDataType,
    &c3_c_emlrtRTEI, 1, c3_b_size[1]);
  c3_e_loop_ub = c3_b_size[1] - 1;
  for (c3_i6 = 0; c3_i6 <= c3_e_loop_ub; c3_i6++) {
    c3_l_obj->NativeDataType.vector.data[c3_i6] = c3_b_data[c3_i6];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_l_obj->NativeDataType,
    &c3_c_emlrtRTEI, 1, 5);
  for (c3_i7 = 0; c3_i7 < 5; c3_i7++) {
    c3_l_obj->NativeDataType.vector.data[c3_i7] = c3_val[c3_i7];
  }

  c3_c_st.site = &c3_m_emlrtRSI;
  c3_m_obj = c3_l_obj->TransportChannel;
  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_m_obj->NativeDataType,
    &c3_emlrtRTEI, 1, c3_b_size[1]);
  c3_f_loop_ub = c3_b_size[1] - 1;
  for (c3_i8 = 0; c3_i8 <= c3_f_loop_ub; c3_i8++) {
    c3_m_obj->NativeDataType.vector.data[c3_i8] = c3_b_data[c3_i8];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_m_obj->NativeDataType,
    &c3_emlrtRTEI, 1, 5);
  for (c3_i9 = 0; c3_i9 < 5; c3_i9++) {
    c3_m_obj->NativeDataType.vector.data[c3_i9] = c3_val[c3_i9];
  }

  c3_b_st.site = &c3_m_emlrtRSI;
  c3_n_obj = c3_d_obj;
  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_n_obj->DataFieldName,
    &c3_c_emlrtRTEI, 1, c3_b_size[1]);
  c3_g_loop_ub = c3_b_size[1] - 1;
  for (c3_i10 = 0; c3_i10 <= c3_g_loop_ub; c3_i10++) {
    c3_n_obj->DataFieldName.vector.data[c3_i10] = c3_b_data[c3_i10];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_n_obj->DataFieldName,
    &c3_c_emlrtRTEI, 1, 4);
  for (c3_i11 = 0; c3_i11 < 4; c3_i11++) {
    c3_n_obj->DataFieldName.vector.data[c3_i11] = c3_b_val[c3_i11];
  }

  c3_c_st.site = &c3_m_emlrtRSI;
  c3_o_obj = c3_n_obj->TransportChannel;
  c3_b_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_o_obj->DataFieldName,
    &c3_emlrtRTEI, 1, c3_b_size[1]);
  c3_h_loop_ub = c3_b_size[1] - 1;
  for (c3_i12 = 0; c3_i12 <= c3_h_loop_ub; c3_i12++) {
    c3_o_obj->DataFieldName.vector.data[c3_i12] = c3_b_data[c3_i12];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_o_obj->DataFieldName,
    &c3_emlrtRTEI, 1, 4);
  for (c3_i13 = 0; c3_i13 < 4; c3_i13++) {
    c3_o_obj->DataFieldName.vector.data[c3_i13] = c3_b_val[c3_i13];
  }

  c3_d_obj->matlabCodegenIsDeleted = false;
  c3_st.site = &c3_i_emlrtRSI;
  c3_p_obj = c3_b_obj;
  c3_b_st.site = &c3_cc_emlrtRSI;
  c3_q_obj = &c3_p_obj->TCPClientObj;
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_r_obj = c3_q_obj;
  c3_outputStream = &c3_r_obj->AsyncIOChannel->OutputStream;
  c3_d_st.site = &c3_m_emlrtRSI;
  c3_s_obj = c3_outputStream;
  c3_s_obj->Timeout = 5.0;
  c3_inputStream = &c3_r_obj->AsyncIOChannel->InputStream;
  c3_d_st.site = &c3_m_emlrtRSI;
  c3_t_obj = c3_inputStream;
  c3_t_obj->Timeout = 5.0;
  c3_q_obj->Timeout = 5.0;
  c3_st.site = &c3_j_emlrtRSI;
  c3_u_obj = &c3_b_obj->TCPClientObj;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_TCPClient_validateDisconnected(chartInstance, &c3_b_st, c3_u_obj);
  c3_u_obj->ConnectTimeout = rtInf;
  c3_st.site = &c3_k_emlrtRSI;
  c3_v_obj = c3_b_obj;
  c3_b_st.site = &c3_dc_emlrtRSI;
  c3_w_obj = &c3_v_obj->TCPClientObj;
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_TCPClient_validateDisconnected(chartInstance, &c3_c_st, c3_w_obj);
  c3_w_obj->TransferDelay = true;
  c3_st.site = &c3_l_emlrtRSI;
  c3_x_obj = &c3_b_obj->TCPClientObj;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_y_obj = c3_x_obj->AsyncIOChannel;
  c3_c_st.site = &c3_yb_emlrtRSI;
  c3_chImpl = c3_y_obj->ChannelImpl;
  c3_success = coderChannelIsOpen(c3_chImpl, &c3_result);
  c3_d_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_d_st, c3_chImpl, c3_success);
  if (c3_result) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    sf_mex_call(&c3_st, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14, sf_mex_call
                (&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call(&c3_st, NULL,
      "message", 1U, 1U, 14, c3_b_y)));
  }

  c3_b_st.site = &c3_m_emlrtRSI;
  c3_ab_obj = c3_x_obj;
  c3_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(chartInstance, &c3_options);
  for (c3_i14 = 0; c3_i14 < 9; c3_i14++) {
    c3_options.HostName[c3_i14] = c3_ab_obj->RemoteHost[c3_i14];
  }

  c3_c_st.site = &c3_m_emlrtRSI;
  c3_d = muDoubleScalarRound(c3_ab_obj->RemotePort);
  if (c3_d < 4.294967296E+9) {
    if (c3_d >= 0.0) {
      c3_u = (uint32_T)c3_d;
    } else {
      c3_u = 0U;
    }
  } else if (c3_d >= 4.294967296E+9) {
    c3_u = MAX_uint32_T;
  } else {
    c3_u = 0U;
  }

  c3_sprintf(chartInstance, &c3_c_st, c3_u, &c3_options.ServiceName);
  c3_options.ConnectTimeout = c3_ab_obj->ConnectTimeout;
  c3_options.IsWriteOnly = c3_ab_obj->IsWriteOnly;
  c3_options.IsSharingPort = c3_ab_obj->IsSharingPort;
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_bb_obj = c3_ab_obj;
  c3_d_st.site = &c3_m_emlrtRSI;
  c3_cb_obj = c3_bb_obj;
  c3_e_st.site = &c3_m_emlrtRSI;
  c3_db_obj = c3_cb_obj->TransportChannel;
  c3_f_st.site = &c3_lb_emlrtRSI;
  c3_eb_obj = c3_db_obj->AsyncIOChannel;
  c3_g_st.site = &c3_yb_emlrtRSI;
  c3_b_chImpl = c3_eb_obj->ChannelImpl;
  c3_b_success = coderChannelIsOpen(c3_b_chImpl, &c3_out);
  c3_h_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_h_st, c3_b_chImpl, c3_b_success);
  if (c3_out) {
    c3_b_value = true;
  } else {
    c3_b_value = false;
  }

  c3_c_value = c3_b_value;
  if (!c3_c_value) {
    c3_d_value = c3_bb_obj->TransferDelay;
  } else {
    c3_d_st.site = &c3_m_emlrtRSI;
    c3_fb_obj = c3_bb_obj->AsyncIOChannel;
    c3_exampleValue = c3_fb_obj->CustomProps.TransferDelay;
    c3_e_st.site = &c3_ec_emlrtRSI;
    c3_c_chImpl = c3_fb_obj->ChannelImpl;
    c3_b_exampleValue = c3_exampleValue;
    c3_d_value = c3_b_exampleValue;
    c3_c_success = coderChannelGetPropertyValue(c3_c_chImpl, "TransferDelay",
      "logical", 1, &c3_d_value);
    c3_f_st.site = &c3_fc_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_c_chImpl,
      c3_c_success);
  }

  c3_options.TransferDelay = c3_d_value;
  c3_c_st.site = &c3_m_emlrtRSI;
  c3_gb_obj = c3_ab_obj->AsyncIOChannel;
  c3_d_st.site = &c3_wb_emlrtRSI;
  c3_hb_obj = c3_gb_obj;
  c3_e_st.site = &c3_yb_emlrtRSI;
  c3_d_chImpl = c3_hb_obj->ChannelImpl;
  c3_d_success = coderChannelIsOpen(c3_d_chImpl, &c3_b_result);
  c3_f_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_d_chImpl, c3_d_success);
  if (!c3_b_result) {
    c3_d_st.site = &c3_xb_emlrtRSI;
    c3_e_chImpl = c3_gb_obj->ChannelImpl;
    c3_array_cell_17_Constructor(chartInstance, &c3_args);
    c3_args.f1 = "HostName";
    c3_args.f2 = "char";
    for (c3_i15 = 0; c3_i15 < 9; c3_i15++) {
      c3_e_value[c3_i15] = c3_options.HostName[c3_i15];
    }

    for (c3_i16 = 0; c3_i16 < 9; c3_i16++) {
      c3_args.f4[c3_i16] = c3_e_value[c3_i16];
    }

    c3_args.f5 = "ServiceName";
    c3_args.f6 = "char";
    c3_args.f7 = c3_options.ServiceName.size[1];
    c3_b = (c3_options.ServiceName.size[1] == 1);
    if (c3_b) {
      c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_value,
        &c3_e_emlrtRTEI, 1, c3_options.ServiceName.size[1] +
        c3_options.ServiceName.size[1]);
      c3_j_loop_ub = c3_options.ServiceName.size[1] - 1;
      for (c3_i18 = 0; c3_i18 <= c3_j_loop_ub; c3_i18++) {
        c3_value.vector.data[c3_i18] = c3_options.ServiceName.vector.data[c3_i18];
      }

      c3_k_loop_ub = c3_options.ServiceName.size[1] - 1;
      for (c3_i19 = 0; c3_i19 <= c3_k_loop_ub; c3_i19++) {
        c3_value.vector.data[c3_i19 + c3_options.ServiceName.size[1]] =
          c3_options.ServiceName.vector.data[c3_i19];
      }
    } else {
      c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_value,
        &c3_d_emlrtRTEI, 1, c3_options.ServiceName.size[1]);
      c3_i_loop_ub = c3_options.ServiceName.size[1] - 1;
      for (c3_i17 = 0; c3_i17 <= c3_i_loop_ub; c3_i17++) {
        c3_value.vector.data[c3_i17] = c3_options.ServiceName.vector.data[c3_i17];
      }
    }

    c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_args.f8,
      &c3_f_emlrtRTEI, 1, c3_value.size[1]);
    c3_l_loop_ub = c3_value.size[1] - 1;
    for (c3_i20 = 0; c3_i20 <= c3_l_loop_ub; c3_i20++) {
      c3_args.f8.vector.data[c3_i20] = c3_value.vector.data[c3_i20];
    }

    c3_args.f9 = "ReceiveSize";
    c3_args.f10 = "double";
    for (c3_i21 = 0; c3_i21 < 2; c3_i21++) {
      c3_args.f12[c3_i21] = 65536.0;
    }

    c3_args.f13 = "SendSize";
    c3_args.f14 = "double";
    for (c3_i22 = 0; c3_i22 < 2; c3_i22++) {
      c3_args.f16[c3_i22] = 65536.0;
    }

    c3_args.f17 = "ConnectTimeout";
    c3_args.f18 = "double";
    c3_f_value[0] = c3_options.ConnectTimeout;
    c3_f_value[1] = c3_options.ConnectTimeout;
    for (c3_i23 = 0; c3_i23 < 2; c3_i23++) {
      c3_args.f20[c3_i23] = c3_f_value[c3_i23];
    }

    c3_args.f21 = "IsWriteOnly";
    c3_args.f22 = "logical";
    c3_g_value[0] = c3_options.IsWriteOnly;
    c3_g_value[1] = c3_options.IsWriteOnly;
    for (c3_i24 = 0; c3_i24 < 2; c3_i24++) {
      c3_args.f24[c3_i24] = c3_g_value[c3_i24];
    }

    c3_args.f25 = "IsSharingPort";
    c3_args.f26 = "logical";
    c3_g_value[0] = c3_options.IsSharingPort;
    c3_g_value[1] = c3_options.IsSharingPort;
    for (c3_i25 = 0; c3_i25 < 2; c3_i25++) {
      c3_args.f28[c3_i25] = c3_g_value[c3_i25];
    }

    c3_args.f29 = "TransferDelay";
    c3_args.f30 = "logical";
    c3_g_value[0] = c3_options.TransferDelay;
    c3_g_value[1] = c3_options.TransferDelay;
    for (c3_i26 = 0; c3_i26 < 2; c3_i26++) {
      c3_args.f32[c3_i26] = c3_g_value[c3_i26];
    }

    for (c3_i27 = 0; c3_i27 < 9; c3_i27++) {
      c3_e_value[c3_i27] = c3_args.f4[c3_i27];
    }

    c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_value,
      &c3_g_emlrtRTEI, 1, c3_args.f8.size[1]);
    c3_m_loop_ub = c3_args.f8.size[1] - 1;
    for (c3_i28 = 0; c3_i28 <= c3_m_loop_ub; c3_i28++) {
      c3_value.vector.data[c3_i28] = c3_args.f8.vector.data[c3_i28];
    }

    for (c3_i29 = 0; c3_i29 < 2; c3_i29++) {
      c3_f_value[c3_i29] = c3_args.f12[c3_i29];
    }

    for (c3_i30 = 0; c3_i30 < 2; c3_i30++) {
      c3_dv[c3_i30] = c3_args.f16[c3_i30];
    }

    for (c3_i31 = 0; c3_i31 < 2; c3_i31++) {
      c3_dv1[c3_i31] = c3_args.f20[c3_i31];
    }

    for (c3_i32 = 0; c3_i32 < 2; c3_i32++) {
      c3_g_value[c3_i32] = c3_args.f24[c3_i32];
    }

    for (c3_i33 = 0; c3_i33 < 2; c3_i33++) {
      c3_bv[c3_i33] = c3_args.f28[c3_i33];
    }

    for (c3_i34 = 0; c3_i34 < 2; c3_i34++) {
      c3_bv1[c3_i34] = c3_args.f32[c3_i34];
    }

    c3_e_success = coderChannelOpen(c3_e_chImpl, 8, c3_args.f1, c3_args.f2, 9,
      &c3_e_value[0], c3_args.f5, c3_args.f6, c3_args.f7, &c3_value.vector.data
      [0], c3_args.f9, c3_args.f10, 1, &c3_f_value[0], c3_args.f13, c3_args.f14,
      1, &c3_dv[0], c3_args.f17, c3_args.f18, 1, &c3_dv1[0], c3_args.f21,
      c3_args.f22, 1, &c3_g_value[0], c3_args.f25, c3_args.f26, 1, &c3_bv[0],
      c3_args.f29, c3_args.f30, 1, &c3_bv1[0]);
    c3_array_cell_17_Destructor(chartInstance, &c3_args);
    c3_e_st.site = &c3_bc_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_e_chImpl,
      c3_e_success);
  }

  c3_array_char_T_2D_Destructor(chartInstance, &c3_value);
  c3_b_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(chartInstance, &c3_options);
  c3_b_obj->matlabCodegenIsDeleted = false;
  return c3_b_obj;
}

static void c3_TCPClient_initializeChannel(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_network_internal_TCPClient *c3_obj,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_iobj_0,
  c3_matlabshared_asyncio_internal_Channel *c3_iobj_1)
{
  static c3_syRUN6P8dnDtW4zHaczOEwG c3_r = { true,/* InitAccess */
    true,                              /* TransferDelay */
    0UL                                /* LatestNumBytesWrittenToDevice */
  };

  static real_T c3_streamLimits[2] = { 0.0, 0.0 };

  static char_T c3_cv9[147] = { '/', 'h', 'o', 'm', 'e', '/', 'a', 'i', 'd', 'e',
    'n', '/', 's', 'n', 'a', 'p', '/', 'c', 'o', 'd', 'e', '/', 'a', 'p', 'p',
    '/', 'm', 'a', 't', 'l', 'a', 'b', '/', 't', 'o', 'o', 'l', 'b', 'o', 'x',
    '/', 's', 'h', 'a', 'r', 'e', 'd', '/', 'a', 's', 'y', 'n', 'c', 'i', 'o',
    'l', 'i', 'b', '/', '+', 'm', 'a', 't', 'l', 'a', 'b', 's', 'h', 'a', 'r',
    'e', 'd', '/', '+', 'a', 's', 'y', 'n', 'c', 'i', 'o', '/', '+', 'i', 'n',
    't', 'e', 'r', 'n', 'a', 'l', '/', '+', 'c', 'o', 'd', 'e', 'r', '/', '.',
    '.', '/', '.', '.', '/', '.', '.', '/', '.', '.', '/', 'b', 'i', 'n', '/',
    'g', 'l', 'n', 'x', 'a', '6', '4', '/', 't', 'e', 's', 't', 'c', 'o', 'd',
    'e', 'r', 'c', 'o', 'n', 'v', 'e', 'r', 't', 'e', 'r', 'a', 'r', 'r', 'a',
    'y', 's' };

  static char_T c3_cv13[46] = { 't', 'o', 'o', 'l', 'b', 'o', 'x', '/', 's', 'h',
    'a', 'r', 'e', 'd', '/', 't', 'e', 's', 't', 'm', 'e', 'a', 's', 'l', 'i',
    'b', '/', 'g', 'e', 'n', 'e', 'r', 'a', 'l', '/', 'b', 'i', 'n', '/', 'g',
    'l', 'n', 'x', 'a', '6', '4' };

  static char_T c3_cv12[44] = { 't', 'e', 's', 't', 'm', 'e', 'a', 's', 'l', 'i',
    'b', ':', 'A', 's', 'y', 'n', 'c', 'i', 'o', 'B', 'u', 'f', 'f', 'e', 'r',
    ':', 'W', 'r', 'o', 'n', 'g', 'M', 'A', 'T', 'L', 'A', 'B', 'V', 'e', 'r',
    's', 'i', 'o', 'n' };

  static char_T c3_cv16[42] = { 't', 'e', 's', 't', 'm', 'e', 'a', 's', 'l', 'i',
    'b', ':', 'A', 's', 'y', 'n', 'c', 'i', 'o', 'B', 'u', 'f', 'f', 'e', 'r',
    ':', 'C', 'a', 'n', 'n', 'o', 't', 'F', 'i', 'n', 'd', 'P', 'l', 'u', 'g',
    'i', 'n' };

  static char_T c3_b_cv5[37] = { 't', 'o', 'o', 'l', 'b', 'o', 'x', '/', 's',
    'h', 'a', 'r', 'e', 'd', '/', 'n', 'e', 't', 'w', 'o', 'r', 'k', 'l', 'i',
    'b', '/', 'b', 'i', 'n', '/', 'g', 'l', 'n', 'x', 'a', '6', '4' };

  static char_T c3_b_cv3[36] = { 'n', 'e', 't', 'w', 'o', 'r', 'k', ':', 't',
    'c', 'p', 'c', 'l', 'i', 'e', 'n', 't', ':', 'W', 'r', 'o', 'n', 'g', 'M',
    'A', 'T', 'L', 'A', 'B', 'V', 'e', 'r', 's', 'i', 'o', 'n' };

  static char_T c3_cv8[35] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'i', 'n', 'v', 'a', 'l', 'i', 'd', 'S', 't',
    'r', 'e', 'a', 'm', 'L', 'i', 'm', 'i', 't', 's' };

  static char_T c3_cv7[34] = { 'n', 'e', 't', 'w', 'o', 'r', 'k', ':', 't', 'c',
    'p', 'c', 'l', 'i', 'e', 'n', 't', ':', 'C', 'a', 'n', 'n', 'o', 't', 'F',
    'i', 'n', 'd', 'P', 'l', 'u', 'g', 'i', 'n' };

  static char_T c3_converterFullName[29] = { 'l', 'i', 'b', 'm', 'w', 'n', 'e',
    't', 'w', 'o', 'r', 'k', 'c', 'o', 'd', 'e', 'r', 'c', 'o', 'n', 'v', 'e',
    'r', 't', 'e', 'r', '.', 's', 'o' };

  static char_T c3_deviceFullName[23] = { 'l', 'i', 'b', 'm', 'w', 't', 'c', 'p',
    'c', 'l', 'i', 'e', 'n', 't', 'd', 'e', 'v', 'i', 'c', 'e', '.', 's', 'o' };

  static char_T c3_b_deviceFullName[14] = { 'l', 'i', 'b', 'm', 'w', 'b', 'u',
    'f', 'f', 'e', 'r', '.', 's', 'o' };

  static char_T c3_b_cv2[8] = { '(', 'R', '2', '0', '2', '4', 'b', ')' };

  static char_T c3_b_cv1[7] = { 'R', 'e', 'l', 'e', 'a', 's', 'e' };

  static char_T c3_b_cv4[7] = { 'g', 'l', 'n', 'x', 'a', '6', '4' };

  static char_T c3_cv10[7] = { 'R', 'e', 'l', 'e', 'a', 's', 'e' };

  static char_T c3_cv15[7] = { 'g', 'l', 'n', 'x', 'a', '6', '4' };

  static char_T c3_b_cv[6] = { 'm', 'a', 't', 'l', 'a', 'b' };

  static char_T c3_cv11[6] = { 'R', '2', '0', '2', '4', 'b' };

  static char_T c3_b_val[5] = { 'u', 'i', 'n', 't', '8' };

  static char_T c3_b_cv6[4] = { 'f', 'i', 'l', 'e' };

  static char_T c3_cv14[4] = { 'f', 'i', 'l', 'e' };

  static char_T c3_val[4] = { 'D', 'a', 't', 'a' };

  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderChannel c3_e_chImpl;
  CoderChannel c3_f_chImpl;
  CoderChannel c3_g_chImpl;
  CoderChannel c3_h_chImpl;
  CoderChannel c3_i_chImpl;
  CoderChannel c3_j_chImpl;
  c3_cell_7 c3_args;
  c3_coder_array_char_T_2D c3_b_value;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_d_this;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_e_this;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_j_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_k_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_l_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_m_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_n_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_o_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_p_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_q_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_b_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_c_obj;
  c3_matlabshared_asyncio_internal_Channel *c3_channel;
  c3_matlabshared_asyncio_internal_Channel *c3_this;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_b_this;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_c_this;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_d_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_e_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_f_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_g_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_h_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_i_obj;
  c3_rtString_6 c3_r1;
  c3_rtString_6 c3_r2;
  c3_s_HTCilNNUmm0Yd43AIdnmID c3_options;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_st;
  const mxArray *c3_b_propValues[1];
  const mxArray *c3_propValues[1];
  const mxArray *c3_ab_y = NULL;
  const mxArray *c3_b_thisMLVersion = NULL;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_bb_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_cb_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_db_y = NULL;
  const mxArray *c3_e_y = NULL;
  const mxArray *c3_eb_y = NULL;
  const mxArray *c3_f_y = NULL;
  const mxArray *c3_fb_y = NULL;
  const mxArray *c3_g_y = NULL;
  const mxArray *c3_h_y = NULL;
  const mxArray *c3_i_y = NULL;
  const mxArray *c3_j_y = NULL;
  const mxArray *c3_k_y = NULL;
  const mxArray *c3_m_y = NULL;
  const mxArray *c3_n_y = NULL;
  const mxArray *c3_o_y = NULL;
  const mxArray *c3_q_y = NULL;
  const mxArray *c3_r_y = NULL;
  const mxArray *c3_s_y = NULL;
  const mxArray *c3_t_y = NULL;
  const mxArray *c3_thisMLVersion = NULL;
  const mxArray *c3_u_y = NULL;
  const mxArray *c3_v_y = NULL;
  const mxArray *c3_w_y = NULL;
  const mxArray *c3_x_y = NULL;
  const mxArray *c3_y = NULL;
  const mxArray *c3_y_y = NULL;
  real_T c3_b_streamLimits[2];
  real_T c3_b_k;
  real_T c3_d;
  real_T c3_d_k;
  int32_T c3_converterFullPathML_size[2];
  int32_T c3_converterPluginPath_size[2];
  int32_T c3_converterPlugin_size[2];
  int32_T c3_deviceFullPathML_size[2];
  int32_T c3_devicePath_size[2];
  int32_T c3_devicePluginPath_size[2];
  int32_T c3_devicePlugin_size[2];
  int32_T c3_out_size[2];
  int32_T c3_thisMatlabRoot_size[2];
  int32_T c3_b_loop_ub;
  int32_T c3_b_success;
  int32_T c3_c_k;
  int32_T c3_c_loop_ub;
  int32_T c3_c_success;
  int32_T c3_d_loop_ub;
  int32_T c3_d_success;
  int32_T c3_e_loop_ub;
  int32_T c3_f_loop_ub;
  int32_T c3_g_loop_ub;
  int32_T c3_h_loop_ub;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i10;
  int32_T c3_i11;
  int32_T c3_i12;
  int32_T c3_i13;
  int32_T c3_i14;
  int32_T c3_i15;
  int32_T c3_i16;
  int32_T c3_i17;
  int32_T c3_i18;
  int32_T c3_i19;
  int32_T c3_i2;
  int32_T c3_i20;
  int32_T c3_i21;
  int32_T c3_i22;
  int32_T c3_i23;
  int32_T c3_i24;
  int32_T c3_i25;
  int32_T c3_i26;
  int32_T c3_i27;
  int32_T c3_i28;
  int32_T c3_i29;
  int32_T c3_i3;
  int32_T c3_i30;
  int32_T c3_i31;
  int32_T c3_i32;
  int32_T c3_i33;
  int32_T c3_i34;
  int32_T c3_i35;
  int32_T c3_i36;
  int32_T c3_i37;
  int32_T c3_i38;
  int32_T c3_i39;
  int32_T c3_i4;
  int32_T c3_i40;
  int32_T c3_i41;
  int32_T c3_i42;
  int32_T c3_i43;
  int32_T c3_i44;
  int32_T c3_i45;
  int32_T c3_i46;
  int32_T c3_i47;
  int32_T c3_i5;
  int32_T c3_i6;
  int32_T c3_i7;
  int32_T c3_i8;
  int32_T c3_i9;
  int32_T c3_i_loop_ub;
  int32_T c3_j_loop_ub;
  int32_T c3_k;
  int32_T c3_k_loop_ub;
  int32_T c3_l_loop_ub;
  int32_T c3_loop_ub;
  int32_T c3_m_loop_ub;
  int32_T c3_n_loop_ub;
  int32_T c3_o_loop_ub;
  int32_T c3_p_loop_ub;
  int32_T c3_q_loop_ub;
  int32_T c3_r_loop_ub;
  int32_T c3_s_loop_ub;
  int32_T c3_success;
  int32_T c3_t_loop_ub;
  int32_T c3_u_loop_ub;
  int32_T c3_v_loop_ub;
  int32_T c3_w_loop_ub;
  int32_T c3_x_loop_ub;
  uint32_T c3_u;
  const char_T *c3_b_propClasses[1] = { "coder.internal.string" };

  const char_T *c3_b_propNames[1] = { "Value" };

  const char_T *c3_propClasses[1] = { "coder.internal.string" };

  const char_T *c3_propNames[1] = { "Value" };

  char_T c3_errorID[1024];
  char_T c3_errorText[1024];
  char_T c3_converterPluginPath_data[581];
  char_T c3_out_data[581];
  char_T c3_converterFullPathML_data[580];
  char_T c3_converterPlugin_data[580];
  char_T c3_devicePluginPath_data[575];
  char_T c3_deviceFullPathML_data[574];
  char_T c3_devicePath_data[574];
  char_T c3_devicePlugin_data[574];
  char_T c3_thisMatlabRoot_data[512];
  char_T c3_value[9];
  char_T c3_b_u[6];
  boolean_T c3_b[2];
  boolean_T c3_x_data[2];
  boolean_T c3_b1;
  boolean_T c3_b2;
  boolean_T c3_b3;
  boolean_T c3_b_b;
  boolean_T c3_exitg1;
  boolean_T c3_invalidStreamLimits;
  boolean_T c3_l_y;
  boolean_T c3_p_y;
  boolean_T c3_result;
  c3_streamLimits[0U] = rtInf;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_array_s_HTCilNNUmm0Yd43AIdnmID_(chartInstance, &c3_options);
  for (c3_i = 0; c3_i < 9; c3_i++) {
    c3_options.HostName[c3_i] = c3_obj->RemoteHost[c3_i];
  }

  c3_st.site = &c3_m_emlrtRSI;
  c3_d = muDoubleScalarRound(c3_obj->RemotePort);
  if (c3_d < 4.294967296E+9) {
    if (c3_d >= 0.0) {
      c3_u = (uint32_T)c3_d;
    } else {
      c3_u = 0U;
    }
  } else if (c3_d >= 4.294967296E+9) {
    c3_u = MAX_uint32_T;
  } else {
    c3_u = 0U;
  }

  c3_sprintf(chartInstance, &c3_st, c3_u, &c3_options.ServiceName);
  c3_st.site = &c3_m_emlrtRSI;
  c3_y = NULL;
  sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 6),
                false);
  c3_b_y = NULL;
  sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 7),
                false);
  c3_b_st.site = &c3_m_emlrtRSI;
  sf_mex_assign(&c3_thisMLVersion, c3_getfield(chartInstance, &c3_b_st, c3_ver
    (chartInstance, &c3_b_st, c3_y), c3_b_y), false);
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_e_emlrt_marshallIn(chartInstance, c3_matlabroot(chartInstance, &c3_b_st),
                        "<output of matlabroot>", c3_thisMatlabRoot_data,
                        c3_thisMatlabRoot_size);
  c3_c_y = NULL;
  sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1, 8),
                false);
  c3_b_st.site = &c3_m_emlrtRSI;
  if (c3_g_emlrt_marshallIn(chartInstance,
       c3_coder_internal_ifWhileCondExtrinsic(chartInstance, &c3_b_st,
        c3_b_strcmp(chartInstance, &c3_b_st, sf_mex_dup(c3_thisMLVersion),
                    c3_c_y)), "<output of coder.internal.ifWhileCondExtrinsic>"))
  {
    c3_deviceFullPathML_size[1] = c3_thisMatlabRoot_size[1] + 62;
    c3_loop_ub = c3_thisMatlabRoot_size[1] - 1;
    for (c3_i1 = 0; c3_i1 <= c3_loop_ub; c3_i1++) {
      c3_deviceFullPathML_data[c3_i1] = c3_thisMatlabRoot_data[c3_i1];
    }

    c3_deviceFullPathML_data[c3_thisMatlabRoot_size[1]] = '/';
    for (c3_i2 = 0; c3_i2 < 37; c3_i2++) {
      c3_deviceFullPathML_data[(c3_i2 + c3_thisMatlabRoot_size[1]) + 1] =
        c3_b_cv5[c3_i2];
    }

    c3_deviceFullPathML_data[c3_thisMatlabRoot_size[1] + 38] = '/';
    for (c3_i3 = 0; c3_i3 < 23; c3_i3++) {
      c3_deviceFullPathML_data[(c3_i3 + c3_thisMatlabRoot_size[1]) + 39] =
        c3_deviceFullName[c3_i3];
    }

    c3_h_y = NULL;
    sf_mex_assign(&c3_h_y, sf_mex_create("y", &c3_deviceFullPathML_data, 10, 0U,
      1, 0U, 2, 1, c3_deviceFullPathML_size[1]), false);
    c3_i_y = NULL;
    sf_mex_assign(&c3_i_y, sf_mex_create("y", c3_b_cv6, 10, 0U, 1, 0U, 2, 1, 4),
                  false);
    c3_b_st.site = &c3_m_emlrtRSI;
    if (c3_g_emlrt_marshallIn(chartInstance,
         c3_b_coder_internal_ifWhileCondExtrinsic(chartInstance, &c3_b_st,
          c3_exist(chartInstance, &c3_b_st, c3_h_y, c3_i_y)),
         "<output of coder.internal.ifWhileCondExtrinsic>")) {
      c3_devicePlugin_size[1] = c3_deviceFullPathML_size[1];
      c3_b_loop_ub = c3_deviceFullPathML_size[1] - 1;
      for (c3_i4 = 0; c3_i4 <= c3_b_loop_ub; c3_i4++) {
        c3_devicePlugin_data[c3_i4] = c3_deviceFullPathML_data[c3_i4];
      }
    } else {
      c3_j_y = NULL;
      sf_mex_assign(&c3_j_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 34),
                    false);
      c3_k_y = NULL;
      sf_mex_assign(&c3_k_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 34),
                    false);
      sf_mex_call(&c3_st, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_j_y, 14,
                  sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
        (&c3_st, NULL, "message", 1U, 1U, 14, c3_k_y)));
    }

    c3_converterFullPathML_size[1] = c3_thisMatlabRoot_size[1] + 68;
    c3_c_loop_ub = c3_thisMatlabRoot_size[1] - 1;
    for (c3_i7 = 0; c3_i7 <= c3_c_loop_ub; c3_i7++) {
      c3_converterFullPathML_data[c3_i7] = c3_thisMatlabRoot_data[c3_i7];
    }

    c3_converterFullPathML_data[c3_thisMatlabRoot_size[1]] = '/';
    for (c3_i8 = 0; c3_i8 < 37; c3_i8++) {
      c3_converterFullPathML_data[(c3_i8 + c3_thisMatlabRoot_size[1]) + 1] =
        c3_b_cv5[c3_i8];
    }

    c3_converterFullPathML_data[c3_thisMatlabRoot_size[1] + 38] = '/';
    for (c3_i9 = 0; c3_i9 < 29; c3_i9++) {
      c3_converterFullPathML_data[(c3_i9 + c3_thisMatlabRoot_size[1]) + 39] =
        c3_converterFullName[c3_i9];
    }

    c3_m_y = NULL;
    sf_mex_assign(&c3_m_y, sf_mex_create("y", &c3_converterFullPathML_data, 10,
      0U, 1, 0U, 2, 1, c3_converterFullPathML_size[1]), false);
    c3_o_y = NULL;
    sf_mex_assign(&c3_o_y, sf_mex_create("y", c3_b_cv6, 10, 0U, 1, 0U, 2, 1, 4),
                  false);
    c3_b_st.site = &c3_m_emlrtRSI;
    if (c3_g_emlrt_marshallIn(chartInstance,
         c3_c_coder_internal_ifWhileCondExtrinsic(chartInstance, &c3_b_st,
          c3_b_exist(chartInstance, &c3_b_st, c3_m_y, c3_o_y)),
         "<output of coder.internal.ifWhileCondExtrinsic>")) {
      c3_converterPlugin_size[1] = c3_converterFullPathML_size[1];
      c3_d_loop_ub = c3_converterFullPathML_size[1] - 1;
      for (c3_i13 = 0; c3_i13 <= c3_d_loop_ub; c3_i13++) {
        c3_converterPlugin_data[c3_i13] = c3_converterFullPathML_data[c3_i13];
      }
    } else {
      c3_r_y = NULL;
      sf_mex_assign(&c3_r_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 34),
                    false);
      c3_s_y = NULL;
      sf_mex_assign(&c3_s_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 34),
                    false);
      sf_mex_call(&c3_st, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_r_y, 14,
                  sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
        (&c3_st, NULL, "message", 1U, 1U, 14, c3_s_y)));
    }
  } else {
    c3_d_y = NULL;
    sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_b_cv3, 10, 0U, 1, 0U, 2, 1, 36),
                  false);
    c3_e_y = NULL;
    sf_mex_assign(&c3_e_y, sf_mex_create("y", c3_b_cv3, 10, 0U, 1, 0U, 2, 1, 36),
                  false);
    c3_f_y = NULL;
    sf_mex_assign(&c3_f_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1, 8),
                  false);
    c3_g_y = NULL;
    sf_mex_assign(&c3_g_y, sf_mex_create("y", c3_b_cv4, 10, 0U, 1, 0U, 2, 1, 7),
                  false);
    sf_mex_call(&c3_st, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_d_y, 14,
                sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_st, NULL, "message", 1U, 3U, 14, c3_e_y, 14, c3_f_y, 14, c3_g_y)));
  }

  sf_mex_destroy(&c3_thisMLVersion);
  c3_st.site = &c3_m_emlrtRSI;
  c3_b_obj = c3_iobj_1;
  c3_b_streamLimits[0] = c3_obj->InputBufferSize;
  c3_b_streamLimits[1] = c3_obj->OutputBufferSize;
  c3_c_obj = c3_b_obj;
  c3_b_st.site = &c3_q_emlrtRSI;
  c3_this = c3_c_obj;
  c3_c_obj = c3_this;
  c3_c_obj->CustomProps = c3_r;
  for (c3_i5 = 0; c3_i5 < 2; c3_i5++) {
    c3_b[c3_i5] = muDoubleScalarIsNaN(c3_b_streamLimits[c3_i5]);
  }

  for (c3_i6 = 0; c3_i6 < 2; c3_i6++) {
    c3_x_data[c3_i6] = c3_b[c3_i6];
  }

  c3_l_y = false;
  c3_k = 0;
  c3_exitg1 = false;
  while ((!c3_exitg1) && (c3_k < 2)) {
    c3_b_k = (real_T)c3_k + 1.0;
    if (!c3_x_data[(int32_T)c3_b_k - 1]) {
      c3_b_b = true;
    } else {
      c3_b_b = false;
    }

    if (!c3_b_b) {
      c3_l_y = true;
      c3_exitg1 = true;
    } else {
      c3_k++;
    }
  }

  if (c3_l_y) {
    c3_invalidStreamLimits = true;
  } else {
    for (c3_i10 = 0; c3_i10 < 2; c3_i10++) {
      c3_b[c3_i10] = (c3_b_streamLimits[c3_i10] < 0.0);
    }

    for (c3_i11 = 0; c3_i11 < 2; c3_i11++) {
      c3_x_data[c3_i11] = c3_b[c3_i11];
    }

    c3_p_y = false;
    c3_c_k = 0;
    c3_exitg1 = false;
    while ((!c3_exitg1) && (c3_c_k < 2)) {
      c3_d_k = (real_T)c3_c_k + 1.0;
      if (!c3_x_data[(int32_T)c3_d_k - 1]) {
        c3_b2 = true;
      } else {
        c3_b2 = false;
      }

      if (!c3_b2) {
        c3_p_y = true;
        c3_exitg1 = true;
      } else {
        c3_c_k++;
      }
    }

    if (c3_p_y) {
      c3_invalidStreamLimits = true;
    } else {
      c3_invalidStreamLimits = false;
    }
  }

  if (c3_invalidStreamLimits) {
    c3_n_y = NULL;
    sf_mex_assign(&c3_n_y, sf_mex_create("y", c3_cv8, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_q_y = NULL;
    sf_mex_assign(&c3_q_y, sf_mex_create("y", c3_cv8, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    sf_mex_call(&c3_st, &c3_b_emlrtMCI, "error", 0U, 2U, 14, c3_n_y, 14,
                sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_st, NULL, "message", 1U, 1U, 14, c3_q_y)));
  }

  c3_chImpl = 0;
  c3_c_obj->ChannelImpl = c3_chImpl;
  c3_b1 = (c3_converterPlugin_size[1] == 0);
  if (c3_b1) {
    c3_converterPlugin_size[1] = 147;
    for (c3_i12 = 0; c3_i12 < 147; c3_i12++) {
      c3_converterPlugin_data[c3_i12] = c3_cv9[c3_i12];
    }
  }

  c3_b_st.site = &c3_r_emlrtRSI;
  c3_devicePluginPath_size[1] = c3_devicePlugin_size[1];
  c3_e_loop_ub = c3_devicePlugin_size[1] - 1;
  for (c3_i14 = 0; c3_i14 <= c3_e_loop_ub; c3_i14++) {
    c3_devicePluginPath_data[c3_i14] = c3_devicePlugin_data[c3_i14];
  }

  c3_converterPluginPath_size[1] = c3_converterPlugin_size[1];
  c3_f_loop_ub = c3_converterPlugin_size[1] - 1;
  for (c3_i15 = 0; c3_i15 <= c3_f_loop_ub; c3_i15++) {
    c3_converterPluginPath_data[c3_i15] = c3_converterPlugin_data[c3_i15];
  }

  c3_converterFullPathML_size[1] = c3_devicePluginPath_size[1];
  c3_g_loop_ub = c3_devicePluginPath_size[1] - 1;
  for (c3_i16 = 0; c3_i16 <= c3_g_loop_ub; c3_i16++) {
    c3_converterFullPathML_data[c3_i16] = c3_devicePluginPath_data[c3_i16];
  }

  c3_out_size[1] = c3_converterFullPathML_size[1] + 1;
  c3_h_loop_ub = c3_converterFullPathML_size[1] - 1;
  for (c3_i17 = 0; c3_i17 <= c3_h_loop_ub; c3_i17++) {
    c3_out_data[c3_i17] = c3_converterFullPathML_data[c3_i17];
  }

  c3_out_data[c3_converterFullPathML_size[1]] = '\x00';
  c3_i_loop_ub = c3_out_size[1] - 1;
  for (c3_i18 = 0; c3_i18 <= c3_i_loop_ub; c3_i18++) {
    c3_devicePluginPath_data[c3_i18] = c3_out_data[c3_i18];
  }

  c3_converterFullPathML_size[1] = c3_converterPluginPath_size[1];
  c3_j_loop_ub = c3_converterPluginPath_size[1] - 1;
  for (c3_i19 = 0; c3_i19 <= c3_j_loop_ub; c3_i19++) {
    c3_converterFullPathML_data[c3_i19] = c3_converterPluginPath_data[c3_i19];
  }

  c3_k_loop_ub = c3_converterFullPathML_size[1] - 1;
  for (c3_i20 = 0; c3_i20 <= c3_k_loop_ub; c3_i20++) {
    c3_converterPluginPath_data[c3_i20] = c3_converterFullPathML_data[c3_i20];
  }

  c3_converterPluginPath_data[c3_converterFullPathML_size[1]] = '\x00';
  c3_b_chImpl = coderChannelCreate(&c3_devicePluginPath_data[0],
    &c3_converterPluginPath_data[0], c3_b_streamLimits[0], c3_b_streamLimits[1],
    &c3_errorID[0], &c3_errorText[0]);
  c3_c_chImpl = 0;
  if (c3_b_chImpl == c3_c_chImpl) {
    c3_c_st.site = &c3_v_emlrtRSI;
    c3_API_dispatchInternalError(chartInstance, &c3_c_st, c3_errorID,
      c3_errorText);
  }

  c3_c_obj->ChannelImpl = c3_b_chImpl;
  c3_b_st.site = &c3_s_emlrtRSI;
  c3_d_chImpl = c3_c_obj->ChannelImpl;
  c3_array_cell_7_Constructor(chartInstance, &c3_args);
  c3_args.f1 = "HostName";
  c3_args.f2 = "char";
  for (c3_i21 = 0; c3_i21 < 9; c3_i21++) {
    c3_value[c3_i21] = c3_options.HostName[c3_i21];
  }

  for (c3_i22 = 0; c3_i22 < 9; c3_i22++) {
    c3_args.f4[c3_i22] = c3_value[c3_i22];
  }

  c3_args.f5 = "ServiceName";
  c3_args.f6 = "char";
  c3_args.f7 = c3_options.ServiceName.size[1];
  c3_b3 = (c3_options.ServiceName.size[1] == 1);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_b_value);
  if (c3_b3) {
    c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_b_value,
      &c3_e_emlrtRTEI, 1, c3_options.ServiceName.size[1] +
      c3_options.ServiceName.size[1]);
    c3_m_loop_ub = c3_options.ServiceName.size[1] - 1;
    for (c3_i24 = 0; c3_i24 <= c3_m_loop_ub; c3_i24++) {
      c3_b_value.vector.data[c3_i24] = c3_options.ServiceName.vector.data[c3_i24];
    }

    c3_n_loop_ub = c3_options.ServiceName.size[1] - 1;
    for (c3_i25 = 0; c3_i25 <= c3_n_loop_ub; c3_i25++) {
      c3_b_value.vector.data[c3_i25 + c3_options.ServiceName.size[1]] =
        c3_options.ServiceName.vector.data[c3_i25];
    }
  } else {
    c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_b_value,
      &c3_d_emlrtRTEI, 1, c3_options.ServiceName.size[1]);
    c3_l_loop_ub = c3_options.ServiceName.size[1] - 1;
    for (c3_i23 = 0; c3_i23 <= c3_l_loop_ub; c3_i23++) {
      c3_b_value.vector.data[c3_i23] = c3_options.ServiceName.vector.data[c3_i23];
    }
  }

  c3_b_array_s_HTCilNNUmm0Yd43AIdnmID_(chartInstance, &c3_options);
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_args.f8,
    &c3_q_emlrtRTEI, 1, c3_b_value.size[1]);
  c3_o_loop_ub = c3_b_value.size[1] - 1;
  for (c3_i26 = 0; c3_i26 <= c3_o_loop_ub; c3_i26++) {
    c3_args.f8.vector.data[c3_i26] = c3_b_value.vector.data[c3_i26];
  }

  for (c3_i27 = 0; c3_i27 < 9; c3_i27++) {
    c3_value[c3_i27] = c3_args.f4[c3_i27];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_b_value,
    &c3_r_emlrtRTEI, 1, c3_args.f8.size[1]);
  c3_p_loop_ub = c3_args.f8.size[1] - 1;
  for (c3_i28 = 0; c3_i28 <= c3_p_loop_ub; c3_i28++) {
    c3_b_value.vector.data[c3_i28] = c3_args.f8.vector.data[c3_i28];
  }

  c3_success = coderChannelInit(c3_d_chImpl, 2, c3_args.f1, c3_args.f2, 9,
    &c3_value[0], c3_args.f5, c3_args.f6, c3_args.f7, &c3_b_value.vector.data[0]);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_b_value);
  c3_array_cell_7_Destructor(chartInstance, &c3_args);
  c3_c_st.site = &c3_x_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_c_st, c3_d_chImpl, c3_success);
  c3_b_st.site = &c3_t_emlrtRSI;
  c3_InputStream_InputStream(chartInstance, &c3_b_st, &c3_c_obj->InputStream,
    c3_c_obj->ChannelImpl);
  c3_b_st.site = &c3_u_emlrtRSI;
  c3_OutputStream_OutputStream(chartInstance, &c3_b_st, &c3_c_obj->OutputStream,
    c3_c_obj->ChannelImpl);
  c3_c_obj->matlabCodegenIsDeleted = false;
  c3_obj->AsyncIOChannel = c3_c_obj;
  c3_st.site = &c3_m_emlrtRSI;
  c3_d_obj = c3_iobj_0;
  c3_channel = c3_obj->AsyncIOChannel;
  c3_e_obj = c3_d_obj;
  c3_e_obj->NumBytesWritten = 0.0;
  c3_e_obj->WriteAsync = true;
  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_b_this = c3_e_obj;
  c3_e_obj = c3_b_this;
  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_f_obj = c3_e_obj;
  c3_e_obj = c3_f_obj;
  c3_c_st.site = &c3_mb_emlrtRSI;
  c3_c_this = c3_e_obj;
  c3_e_obj = c3_c_this;
  for (c3_i29 = 0; c3_i29 < 13; c3_i29++) {
    c3_e_obj->MachineByteOrder[c3_i29] = c3_cv2[c3_i29];
  }

  c3_e_obj->AsyncIOChannel = c3_channel;
  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_g_obj = c3_e_obj;
  c3_thisMatlabRoot_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_g_obj->ByteOrder,
    &c3_emlrtRTEI, 1, c3_thisMatlabRoot_size[1]);
  c3_q_loop_ub = c3_thisMatlabRoot_size[1] - 1;
  for (c3_i30 = 0; c3_i30 <= c3_q_loop_ub; c3_i30++) {
    c3_g_obj->ByteOrder.vector.data[c3_i30] = c3_thisMatlabRoot_data[c3_i30];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_g_obj->ByteOrder,
    &c3_emlrtRTEI, 1, 13);
  for (c3_i31 = 0; c3_i31 < 13; c3_i31++) {
    c3_g_obj->ByteOrder.vector.data[c3_i31] = c3_cv2[c3_i31];
  }

  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_h_obj = c3_e_obj;
  c3_thisMatlabRoot_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_h_obj->DataFieldName,
    &c3_emlrtRTEI, 1, c3_thisMatlabRoot_size[1]);
  c3_r_loop_ub = c3_thisMatlabRoot_size[1] - 1;
  for (c3_i32 = 0; c3_i32 <= c3_r_loop_ub; c3_i32++) {
    c3_h_obj->DataFieldName.vector.data[c3_i32] = c3_thisMatlabRoot_data[c3_i32];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_h_obj->DataFieldName,
    &c3_emlrtRTEI, 1, 4);
  for (c3_i33 = 0; c3_i33 < 4; c3_i33++) {
    c3_h_obj->DataFieldName.vector.data[c3_i33] = c3_val[c3_i33];
  }

  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_i_obj = c3_e_obj;
  c3_thisMatlabRoot_size[1] = 0;
  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_i_obj->NativeDataType,
    &c3_emlrtRTEI, 1, c3_thisMatlabRoot_size[1]);
  c3_s_loop_ub = c3_thisMatlabRoot_size[1] - 1;
  for (c3_i34 = 0; c3_i34 <= c3_s_loop_ub; c3_i34++) {
    c3_i_obj->NativeDataType.vector.data[c3_i34] = c3_thisMatlabRoot_data[c3_i34];
  }

  c3_array_char_T_2D_SetSize(chartInstance, &c3_b_st, &c3_i_obj->NativeDataType,
    &c3_emlrtRTEI, 1, 5);
  for (c3_i35 = 0; c3_i35 < 5; c3_i35++) {
    c3_i_obj->NativeDataType.vector.data[c3_i35] = c3_b_val[c3_i35];
  }

  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_j_obj = &c3_e_obj->UnreadDataBuffer;
  c3_k_obj = c3_j_obj;
  c3_c_st.site = &c3_rb_emlrtRSI;
  c3_t_y = NULL;
  sf_mex_assign(&c3_t_y, sf_mex_create("y", c3_cv10, 10, 0U, 1, 0U, 2, 1, 7),
                false);
  c3_d_st.site = &c3_nb_emlrtRSI;
  sf_mex_assign(&c3_b_thisMLVersion, c3_b_getfield(chartInstance, &c3_d_st,
    c3_matlabRelease(chartInstance, &c3_d_st), c3_t_y), false);
  c3_d_st.site = &c3_ob_emlrtRSI;
  c3_e_emlrt_marshallIn(chartInstance, c3_b_matlabroot(chartInstance, &c3_d_st),
                        "<output of matlabroot>", c3_thisMatlabRoot_data,
                        c3_thisMatlabRoot_size);
  for (c3_i36 = 0; c3_i36 < 6; c3_i36++) {
    c3_r1.Value[c3_i36] = c3_cv11[c3_i36];
  }

  c3_u_y = NULL;
  sf_mex_assign(&c3_u_y, sf_mex_create_class_instance("coder.internal.string"),
                false);
  for (c3_i37 = 0; c3_i37 < 6; c3_i37++) {
    c3_b_u[c3_i37] = c3_r1.Value[c3_i37];
  }

  c3_v_y = NULL;
  sf_mex_assign(&c3_v_y, sf_mex_create("y", c3_b_u, 10, 0U, 1, 0U, 2, 1, 6),
                false);
  c3_propValues[0] = c3_v_y;
  sf_mex_set_all_properties(&c3_u_y, 0, 1, &c3_propNames[0], &c3_propClasses[0],
    &c3_propValues[0]);
  sf_mex_assign(&c3_u_y, sf_mex_convert_to_redirect_source(c3_u_y, 0,
    "coder.internal.string"), false);
  c3_d_st.site = &c3_pb_emlrtRSI;
  if (c3_g_emlrt_marshallIn(chartInstance,
       c3_d_coder_internal_ifWhileCondExtrinsic(chartInstance, &c3_d_st,
        c3_c_strcmp(chartInstance, &c3_d_st, sf_mex_dup(c3_b_thisMLVersion),
                    c3_u_y)), "<output of coder.internal.ifWhileCondExtrinsic>"))
  {
    c3_deviceFullPathML_size[1] = c3_thisMatlabRoot_size[1] + 62;
    c3_t_loop_ub = c3_thisMatlabRoot_size[1] - 1;
    for (c3_i39 = 0; c3_i39 <= c3_t_loop_ub; c3_i39++) {
      c3_deviceFullPathML_data[c3_i39] = c3_thisMatlabRoot_data[c3_i39];
    }

    c3_deviceFullPathML_data[c3_thisMatlabRoot_size[1]] = '/';
    for (c3_i40 = 0; c3_i40 < 46; c3_i40++) {
      c3_deviceFullPathML_data[(c3_i40 + c3_thisMatlabRoot_size[1]) + 1] =
        c3_cv13[c3_i40];
    }

    c3_deviceFullPathML_data[c3_thisMatlabRoot_size[1] + 47] = '/';
    for (c3_i42 = 0; c3_i42 < 14; c3_i42++) {
      c3_deviceFullPathML_data[(c3_i42 + c3_thisMatlabRoot_size[1]) + 48] =
        c3_b_deviceFullName[c3_i42];
    }

    c3_bb_y = NULL;
    sf_mex_assign(&c3_bb_y, sf_mex_create("y", &c3_deviceFullPathML_data, 10, 0U,
      1, 0U, 2, 1, c3_deviceFullPathML_size[1]), false);
    c3_cb_y = NULL;
    sf_mex_assign(&c3_cb_y, sf_mex_create("y", c3_cv14, 10, 0U, 1, 0U, 2, 1, 4),
                  false);
    c3_d_st.site = &c3_qb_emlrtRSI;
    if (c3_g_emlrt_marshallIn(chartInstance,
         c3_e_coder_internal_ifWhileCondExtrinsic(chartInstance, &c3_d_st,
          c3_c_exist(chartInstance, &c3_d_st, c3_bb_y, c3_cb_y)),
         "<output of coder.internal.ifWhileCondExtrinsic>")) {
      c3_devicePath_size[1] = c3_deviceFullPathML_size[1];
      c3_u_loop_ub = c3_deviceFullPathML_size[1] - 1;
      for (c3_i43 = 0; c3_i43 <= c3_u_loop_ub; c3_i43++) {
        c3_devicePath_data[c3_i43] = c3_deviceFullPathML_data[c3_i43];
      }
    } else {
      c3_eb_y = NULL;
      sf_mex_assign(&c3_eb_y, sf_mex_create("y", c3_cv16, 10, 0U, 1, 0U, 2, 1,
        42), false);
      c3_fb_y = NULL;
      sf_mex_assign(&c3_fb_y, sf_mex_create("y", c3_cv16, 10, 0U, 1, 0U, 2, 1,
        42), false);
      sf_mex_call(&c3_c_st, &c3_v_emlrtMCI, "error", 0U, 2U, 14, c3_eb_y, 14,
                  sf_mex_call(&c3_c_st, NULL, "getString", 1U, 1U, 14,
        sf_mex_call(&c3_c_st, NULL, "message", 1U, 1U, 14, c3_fb_y)));
    }
  } else {
    for (c3_i38 = 0; c3_i38 < 6; c3_i38++) {
      c3_r2.Value[c3_i38] = c3_cv11[c3_i38];
    }

    c3_w_y = NULL;
    sf_mex_assign(&c3_w_y, sf_mex_create("y", c3_cv12, 10, 0U, 1, 0U, 2, 1, 44),
                  false);
    c3_x_y = NULL;
    sf_mex_assign(&c3_x_y, sf_mex_create("y", c3_cv12, 10, 0U, 1, 0U, 2, 1, 44),
                  false);
    c3_y_y = NULL;
    sf_mex_assign(&c3_y_y, sf_mex_create_class_instance("coder.internal.string"),
                  false);
    for (c3_i41 = 0; c3_i41 < 6; c3_i41++) {
      c3_b_u[c3_i41] = c3_r2.Value[c3_i41];
    }

    c3_ab_y = NULL;
    sf_mex_assign(&c3_ab_y, sf_mex_create("y", c3_b_u, 10, 0U, 1, 0U, 2, 1, 6),
                  false);
    c3_b_propValues[0] = c3_ab_y;
    sf_mex_set_all_properties(&c3_y_y, 0, 1, &c3_b_propNames[0],
      &c3_b_propClasses[0], &c3_b_propValues[0]);
    sf_mex_assign(&c3_y_y, sf_mex_convert_to_redirect_source(c3_y_y, 0,
      "coder.internal.string"), false);
    c3_db_y = NULL;
    sf_mex_assign(&c3_db_y, sf_mex_create("y", c3_cv15, 10, 0U, 1, 0U, 2, 1, 7),
                  false);
    sf_mex_call(&c3_c_st, &c3_w_emlrtMCI, "error", 0U, 2U, 14, c3_w_y, 14,
                sf_mex_call(&c3_c_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_c_st, NULL, "message", 1U, 3U, 14, c3_x_y, 14, c3_y_y, 14, c3_db_y)));
  }

  sf_mex_destroy(&c3_b_thisMLVersion);
  c3_c_st.site = &c3_sb_emlrtRSI;
  c3_d_this = c3_k_obj;
  c3_k_obj = c3_d_this;
  c3_c_st.site = &c3_tb_emlrtRSI;
  c3_l_obj = c3_k_obj;
  c3_k_obj = c3_l_obj;
  c3_d_st.site = &c3_q_emlrtRSI;
  c3_e_this = c3_k_obj;
  c3_k_obj = c3_e_this;
  c3_e_chImpl = 0;
  c3_k_obj->ChannelImpl = c3_e_chImpl;
  c3_d_st.site = &c3_r_emlrtRSI;
  c3_converterPluginPath_size[1] = c3_devicePath_size[1];
  c3_v_loop_ub = c3_devicePath_size[1] - 1;
  for (c3_i44 = 0; c3_i44 <= c3_v_loop_ub; c3_i44++) {
    c3_converterPluginPath_data[c3_i44] = c3_devicePath_data[c3_i44];
  }

  c3_converterFullPathML_size[1] = c3_converterPluginPath_size[1];
  c3_w_loop_ub = c3_converterPluginPath_size[1] - 1;
  for (c3_i45 = 0; c3_i45 <= c3_w_loop_ub; c3_i45++) {
    c3_converterFullPathML_data[c3_i45] = c3_converterPluginPath_data[c3_i45];
  }

  c3_x_loop_ub = c3_converterFullPathML_size[1] - 1;
  for (c3_i46 = 0; c3_i46 <= c3_x_loop_ub; c3_i46++) {
    c3_converterPluginPath_data[c3_i46] = c3_converterFullPathML_data[c3_i46];
  }

  c3_converterPluginPath_data[c3_converterFullPathML_size[1]] = '\x00';
  c3_f_chImpl = coderChannelCreate(&c3_converterPluginPath_data[0],
    "/home/aiden/snap/code/app/matlab/toolbox/shared/asynciolib/+matlabshared/+asyncio/+internal/+coder/../../../../bin/glnxa64/testc"
    "oderconverterarrays", rtInf, 0.0, &c3_errorID[0], &c3_errorText[0]);
  c3_g_chImpl = 0;
  if (c3_f_chImpl == c3_g_chImpl) {
    c3_e_st.site = &c3_v_emlrtRSI;
    c3_API_dispatchInternalError(chartInstance, &c3_e_st, c3_errorID,
      c3_errorText);
  }

  c3_k_obj->ChannelImpl = c3_f_chImpl;
  c3_d_st.site = &c3_s_emlrtRSI;
  c3_h_chImpl = c3_k_obj->ChannelImpl;
  c3_b_success = coderChannelInit(c3_h_chImpl, 0);
  c3_e_st.site = &c3_x_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_h_chImpl, c3_b_success);
  c3_d_st.site = &c3_t_emlrtRSI;
  c3_InputStream_InputStream(chartInstance, &c3_d_st, &c3_k_obj->InputStream,
    c3_k_obj->ChannelImpl);
  c3_d_st.site = &c3_u_emlrtRSI;
  c3_OutputStream_OutputStream(chartInstance, &c3_d_st, &c3_k_obj->OutputStream,
    c3_k_obj->ChannelImpl);
  c3_k_obj->DataEventsDisabled = true;
  for (c3_i47 = 0; c3_i47 < 2; c3_i47++) {
    c3_k_obj->StreamLimits[c3_i47] = c3_streamLimits[c3_i47];
  }

  c3_m_obj = c3_k_obj;
  c3_m_obj->TotalElementsWritten = 0.0;
  c3_c_st.site = &c3_ub_emlrtRSI;
  c3_n_obj = c3_k_obj;
  c3_d_st.site = &c3_vb_emlrtRSI;
  c3_o_obj = c3_n_obj;
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_d_st, &c3_o_obj->PartialPacket,
    &c3_u_emlrtRTEI, 1, 0);
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_d_st, &c3_o_obj->PartialPacket,
    &c3_v_emlrtRTEI, 0, 0);
  c3_n_obj->PartialPacketStart = 0.0;
  c3_n_obj->PartialPacketCount = 0.0;
  c3_k_obj->matlabCodegenIsDeleted = false;
  c3_b_st.site = &c3_lb_emlrtRSI;
  c3_p_obj = &c3_e_obj->UnreadDataBuffer;
  c3_c_st.site = &c3_wb_emlrtRSI;
  c3_q_obj = c3_p_obj;
  c3_d_st.site = &c3_yb_emlrtRSI;
  c3_i_chImpl = c3_q_obj->ChannelImpl;
  c3_c_success = coderChannelIsOpen(c3_i_chImpl, &c3_result);
  c3_e_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_i_chImpl, c3_c_success);
  if (!c3_result) {
    c3_c_st.site = &c3_xb_emlrtRSI;
    c3_j_chImpl = c3_p_obj->ChannelImpl;
    c3_d_success = coderChannelOpen(c3_j_chImpl, 0);
    c3_d_st.site = &c3_bc_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_d_st, c3_j_chImpl,
      c3_d_success);
  }

  c3_e_obj->matlabCodegenIsDeleted = false;
  c3_obj->TransportChannel = c3_e_obj;
}

static void c3_sprintf(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, uint32_T c3_varargin_1, c3_coder_array_char_T_2D *c3_str)
{
  static char_T c3_b_cv[7] = { 's', 'p', 'r', 'i', 'n', 't', 'f' };

  static char_T c3_formatSpec[2] = { '%', 'u' };

  emlrtStack c3_b_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_tmpStr = NULL;
  const mxArray *c3_y = NULL;
  real_T c3_strSize;
  uint32_T c3_b_varargin_1;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_st.site = &c3_n_emlrtRSI;
  c3_b_varargin_1 = c3_varargin_1;
  c3_y = NULL;
  sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 7),
                false);
  c3_b_y = NULL;
  sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_formatSpec, 10, 0U, 1, 0U, 2, 1,
    2), false);
  c3_c_y = NULL;
  sf_mex_assign(&c3_c_y, sf_mex_create("y", &c3_b_varargin_1, 7, 0U, 0, 0U, 0),
                false);
  c3_b_st.site = &c3_o_emlrtRSI;
  sf_mex_assign(&c3_tmpStr, c3_feval(chartInstance, &c3_b_st, c3_y, c3_b_y,
    c3_c_y), false);
  c3_b_st.site = &c3_p_emlrtRSI;
  c3_strSize = c3_emlrt_marshallIn(chartInstance, c3_length(chartInstance,
    &c3_b_st, sf_mex_dup(c3_tmpStr)), "<output of length>");
  if (!(c3_strSize >= 0.0)) {
    emlrtNonNegativeCheckR2012b(c3_strSize, &c3_emlrtDCI, &c3_st);
  }

  c3_b_st.site = &c3_ff_emlrtRSI;
  c3_c_emlrt_marshallIn(chartInstance, &c3_b_st, sf_mex_dup(c3_tmpStr), "tmpStr",
                        c3_str);
  sf_mex_destroy(&c3_tmpStr);
}

static void c3_API_dispatchInternalError(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, char_T c3_errorID[1024], char_T
  c3_errorText[1024])
{
  static char_T c3_cv10[49] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O', 'u',
    't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 's', 'y', 'n', 'c',
    'h', 'r', 'o', 'n', 'o', 'u', 's', 'O', 'u', 't', 'p', 'u', 't', 'N', 'o',
    't', 'P', 'o', 's', 's', 'i', 'b', 'l', 'e' };

  static char_T c3_cv24[49] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O', 'u',
    't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 's', 'y', 'n', 'c',
    'h', 'r', 'o', 'n', 'o', 'u', 's', 'O', 'u', 't', 'p', 'u', 't', 'N', 'o',
    't', 'P', 'o', 's', 's', 'i', 'b', 'l', 'e' };

  static char_T c3_cv23[47] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I', 'n',
    'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 's', 'y', 'n', 'c', 'h',
    'r', 'o', 'n', 'o', 'u', 's', 'I', 'n', 'p', 'u', 't', 'N', 'o', 't', 'P',
    'o', 's', 's', 'i', 'b', 'l', 'e' };

  static char_T c3_cv9[47] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I', 'n',
    'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 's', 'y', 'n', 'c', 'h',
    'r', 'o', 'n', 'o', 'u', 's', 'I', 'n', 'p', 'u', 't', 'N', 'o', 't', 'P',
    'o', 's', 's', 'i', 'b', 'l', 'e' };

  static char_T c3_cv12[46] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O', 'u',
    't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l',
    'd', 'N', 'o', 't', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'O', 'u', 't', 'p',
    'u', 't', 'D', 'a', 't', 'a' };

  static char_T c3_cv26[46] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O', 'u',
    't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l',
    'd', 'N', 'o', 't', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'O', 'u', 't', 'p',
    'u', 't', 'D', 'a', 't', 'a' };

  static char_T c3_cv11[44] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I', 'n',
    'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd',
    'N', 'o', 't', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'I', 'n', 'p', 'u', 't',
    'D', 'a', 't', 'a' };

  static char_T c3_cv25[44] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I', 'n',
    'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd',
    'N', 'o', 't', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'I', 'n', 'p', 'u', 't',
    'D', 'a', 't', 'a' };

  static char_T c3_cv13[39] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S', 't',
    'r', 'e', 'a', 'm', ':', 'c', 'a', 'n', 'n', 'o', 't', 'A', 'd', 'd', 'F',
    'i', 'l', 't', 'e', 'r', 'W', 'h', 'i', 'l', 'e', 'O', 'p', 'e', 'n' };

  static char_T c3_cv21[39] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'C',
    'r', 'e', 'a', 't', 'e', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'e', 'r' };

  static char_T c3_cv27[39] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S', 't',
    'r', 'e', 'a', 'm', ':', 'c', 'a', 'n', 'n', 'o', 't', 'A', 'd', 'd', 'F',
    'i', 'l', 't', 'e', 'r', 'W', 'h', 'i', 'l', 'e', 'O', 'p', 'e', 'n' };

  static char_T c3_cv7[39] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'C',
    'r', 'e', 'a', 't', 'e', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'e', 'r' };

  static char_T c3_b_cv4[37] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C',
    'h', 'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't',
    'L', 'o', 'a', 'd', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'e', 'r' };

  static char_T c3_cv18[37] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'L',
    'o', 'a', 'd', 'C', 'o', 'n', 'v', 'e', 'r', 't', 'e', 'r' };

  static char_T c3_b_cv6[36] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C',
    'h', 'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't',
    'C', 'r', 'e', 'a', 't', 'e', 'D', 'e', 'v', 'i', 'c', 'e' };

  static char_T c3_cv20[36] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'C',
    'r', 'e', 'a', 't', 'e', 'D', 'e', 'v', 'i', 'c', 'e' };

  static char_T c3_cv22[35] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S', 't',
    'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'C', 'r',
    'e', 'a', 't', 'e', 'F', 'i', 'l', 't', 'e', 'r' };

  static char_T c3_cv8[35] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S', 't',
    'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'C', 'r',
    'e', 'a', 't', 'e', 'F', 'i', 'l', 't', 'e', 'r' };

  static char_T c3_b_cv3[34] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C',
    'h', 'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't',
    'L', 'o', 'a', 'd', 'D', 'e', 'v', 'i', 'c', 'e' };

  static char_T c3_cv17[34] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C', 'h',
    'a', 'n', 'n', 'e', 'l', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'L',
    'o', 'a', 'd', 'D', 'e', 'v', 'i', 'c', 'e' };

  static char_T c3_b_cv2[33] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O',
    'u', 't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'n', 'o', 't',
    'S', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd' };

  static char_T c3_b_cv5[33] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S',
    't', 'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'L',
    'o', 'a', 'd', 'F', 'i', 'l', 't', 'e', 'r' };

  static char_T c3_cv15[33] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O', 'u',
    't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'n', 'o', 't', 'S',
    'u', 'p', 'p', 'o', 'r', 't', 'e', 'd' };

  static char_T c3_cv19[33] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'S', 't',
    'r', 'e', 'a', 'm', ':', 'c', 'o', 'u', 'l', 'd', 'N', 'o', 't', 'L', 'o',
    'a', 'd', 'F', 'i', 'l', 't', 'e', 'r' };

  static char_T c3_b_cv[32] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I', 'n',
    'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'n', 'o', 't', 'S', 'u',
    'p', 'p', 'o', 'r', 't', 'e', 'd' };

  static char_T c3_b_cv1[32] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'I',
    'n', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 'n', 'o', 't', 'S',
    'u', 'p', 'p', 'o', 'r', 't', 'e', 'd' };

  static char_T c3_cv14[7] = { 'E', 'r', 'r', 'I', 'D', ':', ' ' };

  static char_T c3_cv16[7] = { ',', ' ', 'M', 's', 'g', ':', ' ' };

  c3_coder_array_char_T_2D c3_x;
  emlrtStack c3_st;
  const mxArray *c3_ab_y = NULL;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_bb_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_cb_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_db_y = NULL;
  const mxArray *c3_e_y = NULL;
  const mxArray *c3_eb_y = NULL;
  const mxArray *c3_f_y = NULL;
  const mxArray *c3_fb_y = NULL;
  const mxArray *c3_g_y = NULL;
  const mxArray *c3_gb_y = NULL;
  const mxArray *c3_h_y = NULL;
  const mxArray *c3_i_y = NULL;
  const mxArray *c3_j_y = NULL;
  const mxArray *c3_k_y = NULL;
  const mxArray *c3_l_y = NULL;
  const mxArray *c3_m_y = NULL;
  const mxArray *c3_n_y = NULL;
  const mxArray *c3_o_y = NULL;
  const mxArray *c3_p_y = NULL;
  const mxArray *c3_q_y = NULL;
  const mxArray *c3_r_y = NULL;
  const mxArray *c3_s_y = NULL;
  const mxArray *c3_t_y = NULL;
  const mxArray *c3_u_y = NULL;
  const mxArray *c3_v_y = NULL;
  const mxArray *c3_w_y = NULL;
  const mxArray *c3_x_y = NULL;
  const mxArray *c3_y = NULL;
  const mxArray *c3_y_y = NULL;
  real_T c3_index;
  int32_T c3_errorID_size[2];
  int32_T c3_u_size[2];
  int32_T c3_ab_kstr;
  int32_T c3_b_kstr;
  int32_T c3_b_loop_ub;
  int32_T c3_b_remainingDimsB;
  int32_T c3_c_kstr;
  int32_T c3_c_loop_ub;
  int32_T c3_c_remainingDimsB;
  int32_T c3_d_kstr;
  int32_T c3_d_loop_ub;
  int32_T c3_d_remainingDimsB;
  int32_T c3_e_kstr;
  int32_T c3_e_loop_ub;
  int32_T c3_e_remainingDimsB;
  int32_T c3_exitg1;
  int32_T c3_f_kstr;
  int32_T c3_f_loop_ub;
  int32_T c3_f_remainingDimsB;
  int32_T c3_g_kstr;
  int32_T c3_g_loop_ub;
  int32_T c3_g_remainingDimsB;
  int32_T c3_h_kstr;
  int32_T c3_h_loop_ub;
  int32_T c3_h_remainingDimsB;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i10;
  int32_T c3_i11;
  int32_T c3_i12;
  int32_T c3_i13;
  int32_T c3_i14;
  int32_T c3_i15;
  int32_T c3_i16;
  int32_T c3_i2;
  int32_T c3_i3;
  int32_T c3_i4;
  int32_T c3_i5;
  int32_T c3_i6;
  int32_T c3_i7;
  int32_T c3_i8;
  int32_T c3_i9;
  int32_T c3_i_kstr;
  int32_T c3_i_loop_ub;
  int32_T c3_i_remainingDimsB;
  int32_T c3_j_kstr;
  int32_T c3_j_loop_ub;
  int32_T c3_j_remainingDimsB;
  int32_T c3_k_kstr;
  int32_T c3_k_loop_ub;
  int32_T c3_k_remainingDimsB;
  int32_T c3_kstr;
  int32_T c3_l_kstr;
  int32_T c3_l_loop_ub;
  int32_T c3_l_remainingDimsB;
  int32_T c3_loop_ub;
  int32_T c3_m_kstr;
  int32_T c3_m_loop_ub;
  int32_T c3_m_remainingDimsB;
  int32_T c3_n_kstr;
  int32_T c3_n_loop_ub;
  int32_T c3_o_kstr;
  int32_T c3_p_kstr;
  int32_T c3_q_kstr;
  int32_T c3_r_kstr;
  int32_T c3_remainingDimsB;
  int32_T c3_s_kstr;
  int32_T c3_t_kstr;
  int32_T c3_u_kstr;
  int32_T c3_v_kstr;
  int32_T c3_w_kstr;
  int32_T c3_x_kstr;
  int32_T c3_y_kstr;
  char_T c3_u_data[2062];
  char_T c3_errorID_data[1024];
  boolean_T c3_b_result;
  boolean_T c3_c_result;
  boolean_T c3_d_result;
  boolean_T c3_e_result;
  boolean_T c3_f_result;
  boolean_T c3_g_result;
  boolean_T c3_h_result;
  boolean_T c3_i_result;
  boolean_T c3_j_result;
  boolean_T c3_k_result;
  boolean_T c3_l_result;
  boolean_T c3_m_result;
  boolean_T c3_n_result;
  boolean_T c3_result;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_st.site = &c3_w_emlrtRSI;
  c3_API_trimString(chartInstance, &c3_st, c3_errorID, c3_errorID_data,
                    c3_errorID_size);
  c3_result = false;
  c3_array_char_T_2D_Constructor(chartInstance, &c3_x);
  c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x, &c3_w_emlrtRTEI, 1,
    c3_errorID_size[1]);
  c3_loop_ub = c3_errorID_size[1] - 1;
  for (c3_i = 0; c3_i <= c3_loop_ub; c3_i++) {
    c3_x.vector.data[c3_i] = c3_errorID_data[c3_i];
  }

  c3_remainingDimsB = c3_x.size[1];
  if (c3_remainingDimsB != 32) {
  } else {
    c3_kstr = 1;
    do {
      c3_exitg1 = 0;
      if (c3_kstr - 1 < 32) {
        c3_b_kstr = c3_kstr - 1;
        if (c3_b_cv[c3_b_kstr] != c3_errorID_data[c3_b_kstr]) {
          c3_exitg1 = 1;
        } else {
          c3_kstr++;
        }
      } else {
        c3_result = true;
        c3_exitg1 = 1;
      }
    } while (c3_exitg1 == 0);
  }

  if (c3_result) {
    c3_index = 0.0;
  } else {
    c3_b_result = false;
    c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x, &c3_w_emlrtRTEI, 1,
      c3_errorID_size[1]);
    c3_b_loop_ub = c3_errorID_size[1] - 1;
    for (c3_i1 = 0; c3_i1 <= c3_b_loop_ub; c3_i1++) {
      c3_x.vector.data[c3_i1] = c3_errorID_data[c3_i1];
    }

    c3_b_remainingDimsB = c3_x.size[1];
    if (c3_b_remainingDimsB != 33) {
    } else {
      c3_c_kstr = 1;
      do {
        c3_exitg1 = 0;
        if (c3_c_kstr - 1 < 33) {
          c3_d_kstr = c3_c_kstr - 1;
          if (c3_cv15[c3_d_kstr] != c3_errorID_data[c3_d_kstr]) {
            c3_exitg1 = 1;
          } else {
            c3_c_kstr++;
          }
        } else {
          c3_b_result = true;
          c3_exitg1 = 1;
        }
      } while (c3_exitg1 == 0);
    }

    if (c3_b_result) {
      c3_index = 1.0;
    } else {
      c3_c_result = false;
      c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x, &c3_w_emlrtRTEI, 1,
        c3_errorID_size[1]);
      c3_c_loop_ub = c3_errorID_size[1] - 1;
      for (c3_i3 = 0; c3_i3 <= c3_c_loop_ub; c3_i3++) {
        c3_x.vector.data[c3_i3] = c3_errorID_data[c3_i3];
      }

      c3_c_remainingDimsB = c3_x.size[1];
      if (c3_c_remainingDimsB != 34) {
      } else {
        c3_e_kstr = 1;
        do {
          c3_exitg1 = 0;
          if (c3_e_kstr - 1 < 34) {
            c3_f_kstr = c3_e_kstr - 1;
            if (c3_cv17[c3_f_kstr] != c3_errorID_data[c3_f_kstr]) {
              c3_exitg1 = 1;
            } else {
              c3_e_kstr++;
            }
          } else {
            c3_c_result = true;
            c3_exitg1 = 1;
          }
        } while (c3_exitg1 == 0);
      }

      if (c3_c_result) {
        c3_index = 2.0;
      } else {
        c3_d_result = false;
        c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x, &c3_w_emlrtRTEI,
          1, c3_errorID_size[1]);
        c3_e_loop_ub = c3_errorID_size[1] - 1;
        for (c3_i7 = 0; c3_i7 <= c3_e_loop_ub; c3_i7++) {
          c3_x.vector.data[c3_i7] = c3_errorID_data[c3_i7];
        }

        c3_d_remainingDimsB = c3_x.size[1];
        if (c3_d_remainingDimsB != 37) {
        } else {
          c3_g_kstr = 1;
          do {
            c3_exitg1 = 0;
            if (c3_g_kstr - 1 < 37) {
              c3_h_kstr = c3_g_kstr - 1;
              if (c3_cv18[c3_h_kstr] != c3_errorID_data[c3_h_kstr]) {
                c3_exitg1 = 1;
              } else {
                c3_g_kstr++;
              }
            } else {
              c3_d_result = true;
              c3_exitg1 = 1;
            }
          } while (c3_exitg1 == 0);
        }

        if (c3_d_result) {
          c3_index = 3.0;
        } else {
          c3_e_result = false;
          c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
            &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
          c3_f_loop_ub = c3_errorID_size[1] - 1;
          for (c3_i8 = 0; c3_i8 <= c3_f_loop_ub; c3_i8++) {
            c3_x.vector.data[c3_i8] = c3_errorID_data[c3_i8];
          }

          c3_e_remainingDimsB = c3_x.size[1];
          if (c3_e_remainingDimsB != 33) {
          } else {
            c3_i_kstr = 1;
            do {
              c3_exitg1 = 0;
              if (c3_i_kstr - 1 < 33) {
                c3_j_kstr = c3_i_kstr - 1;
                if (c3_cv19[c3_j_kstr] != c3_errorID_data[c3_j_kstr]) {
                  c3_exitg1 = 1;
                } else {
                  c3_i_kstr++;
                }
              } else {
                c3_e_result = true;
                c3_exitg1 = 1;
              }
            } while (c3_exitg1 == 0);
          }

          if (c3_e_result) {
            c3_index = 4.0;
          } else {
            c3_f_result = false;
            c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
              &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
            c3_g_loop_ub = c3_errorID_size[1] - 1;
            for (c3_i9 = 0; c3_i9 <= c3_g_loop_ub; c3_i9++) {
              c3_x.vector.data[c3_i9] = c3_errorID_data[c3_i9];
            }

            c3_f_remainingDimsB = c3_x.size[1];
            if (c3_f_remainingDimsB != 36) {
            } else {
              c3_k_kstr = 1;
              do {
                c3_exitg1 = 0;
                if (c3_k_kstr - 1 < 36) {
                  c3_l_kstr = c3_k_kstr - 1;
                  if (c3_cv20[c3_l_kstr] != c3_errorID_data[c3_l_kstr]) {
                    c3_exitg1 = 1;
                  } else {
                    c3_k_kstr++;
                  }
                } else {
                  c3_f_result = true;
                  c3_exitg1 = 1;
                }
              } while (c3_exitg1 == 0);
            }

            if (c3_f_result) {
              c3_index = 5.0;
            } else {
              c3_g_result = false;
              c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
              c3_h_loop_ub = c3_errorID_size[1] - 1;
              for (c3_i10 = 0; c3_i10 <= c3_h_loop_ub; c3_i10++) {
                c3_x.vector.data[c3_i10] = c3_errorID_data[c3_i10];
              }

              c3_g_remainingDimsB = c3_x.size[1];
              if (c3_g_remainingDimsB != 39) {
              } else {
                c3_m_kstr = 1;
                do {
                  c3_exitg1 = 0;
                  if (c3_m_kstr - 1 < 39) {
                    c3_n_kstr = c3_m_kstr - 1;
                    if (c3_cv21[c3_n_kstr] != c3_errorID_data[c3_n_kstr]) {
                      c3_exitg1 = 1;
                    } else {
                      c3_m_kstr++;
                    }
                  } else {
                    c3_g_result = true;
                    c3_exitg1 = 1;
                  }
                } while (c3_exitg1 == 0);
              }

              if (c3_g_result) {
                c3_index = 6.0;
              } else {
                c3_h_result = false;
                c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                  &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                c3_i_loop_ub = c3_errorID_size[1] - 1;
                for (c3_i11 = 0; c3_i11 <= c3_i_loop_ub; c3_i11++) {
                  c3_x.vector.data[c3_i11] = c3_errorID_data[c3_i11];
                }

                c3_h_remainingDimsB = c3_x.size[1];
                if (c3_h_remainingDimsB != 35) {
                } else {
                  c3_o_kstr = 1;
                  do {
                    c3_exitg1 = 0;
                    if (c3_o_kstr - 1 < 35) {
                      c3_p_kstr = c3_o_kstr - 1;
                      if (c3_cv22[c3_p_kstr] != c3_errorID_data[c3_p_kstr]) {
                        c3_exitg1 = 1;
                      } else {
                        c3_o_kstr++;
                      }
                    } else {
                      c3_h_result = true;
                      c3_exitg1 = 1;
                    }
                  } while (c3_exitg1 == 0);
                }

                if (c3_h_result) {
                  c3_index = 7.0;
                } else {
                  c3_i_result = false;
                  c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                    &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                  c3_j_loop_ub = c3_errorID_size[1] - 1;
                  for (c3_i12 = 0; c3_i12 <= c3_j_loop_ub; c3_i12++) {
                    c3_x.vector.data[c3_i12] = c3_errorID_data[c3_i12];
                  }

                  c3_i_remainingDimsB = c3_x.size[1];
                  if (c3_i_remainingDimsB != 47) {
                  } else {
                    c3_q_kstr = 1;
                    do {
                      c3_exitg1 = 0;
                      if (c3_q_kstr - 1 < 47) {
                        c3_r_kstr = c3_q_kstr - 1;
                        if (c3_cv23[c3_r_kstr] != c3_errorID_data[c3_r_kstr]) {
                          c3_exitg1 = 1;
                        } else {
                          c3_q_kstr++;
                        }
                      } else {
                        c3_i_result = true;
                        c3_exitg1 = 1;
                      }
                    } while (c3_exitg1 == 0);
                  }

                  if (c3_i_result) {
                    c3_index = 8.0;
                  } else {
                    c3_j_result = false;
                    c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                      &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                    c3_k_loop_ub = c3_errorID_size[1] - 1;
                    for (c3_i13 = 0; c3_i13 <= c3_k_loop_ub; c3_i13++) {
                      c3_x.vector.data[c3_i13] = c3_errorID_data[c3_i13];
                    }

                    c3_j_remainingDimsB = c3_x.size[1];
                    if (c3_j_remainingDimsB != 49) {
                    } else {
                      c3_s_kstr = 1;
                      do {
                        c3_exitg1 = 0;
                        if (c3_s_kstr - 1 < 49) {
                          c3_t_kstr = c3_s_kstr - 1;
                          if (c3_cv24[c3_t_kstr] != c3_errorID_data[c3_t_kstr])
                          {
                            c3_exitg1 = 1;
                          } else {
                            c3_s_kstr++;
                          }
                        } else {
                          c3_j_result = true;
                          c3_exitg1 = 1;
                        }
                      } while (c3_exitg1 == 0);
                    }

                    if (c3_j_result) {
                      c3_index = 9.0;
                    } else {
                      c3_k_result = false;
                      c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                        &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                      c3_l_loop_ub = c3_errorID_size[1] - 1;
                      for (c3_i14 = 0; c3_i14 <= c3_l_loop_ub; c3_i14++) {
                        c3_x.vector.data[c3_i14] = c3_errorID_data[c3_i14];
                      }

                      c3_k_remainingDimsB = c3_x.size[1];
                      if (c3_k_remainingDimsB != 44) {
                      } else {
                        c3_u_kstr = 1;
                        do {
                          c3_exitg1 = 0;
                          if (c3_u_kstr - 1 < 44) {
                            c3_v_kstr = c3_u_kstr - 1;
                            if (c3_cv25[c3_v_kstr] != c3_errorID_data[c3_v_kstr])
                            {
                              c3_exitg1 = 1;
                            } else {
                              c3_u_kstr++;
                            }
                          } else {
                            c3_k_result = true;
                            c3_exitg1 = 1;
                          }
                        } while (c3_exitg1 == 0);
                      }

                      if (c3_k_result) {
                        c3_index = 10.0;
                      } else {
                        c3_l_result = false;
                        c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                          &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                        c3_m_loop_ub = c3_errorID_size[1] - 1;
                        for (c3_i15 = 0; c3_i15 <= c3_m_loop_ub; c3_i15++) {
                          c3_x.vector.data[c3_i15] = c3_errorID_data[c3_i15];
                        }

                        c3_l_remainingDimsB = c3_x.size[1];
                        if (c3_l_remainingDimsB != 46) {
                        } else {
                          c3_w_kstr = 1;
                          do {
                            c3_exitg1 = 0;
                            if (c3_w_kstr - 1 < 46) {
                              c3_x_kstr = c3_w_kstr - 1;
                              if (c3_cv26[c3_x_kstr] !=
                                  c3_errorID_data[c3_x_kstr]) {
                                c3_exitg1 = 1;
                              } else {
                                c3_w_kstr++;
                              }
                            } else {
                              c3_l_result = true;
                              c3_exitg1 = 1;
                            }
                          } while (c3_exitg1 == 0);
                        }

                        if (c3_l_result) {
                          c3_index = 11.0;
                        } else {
                          c3_m_result = false;
                          c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x,
                            &c3_w_emlrtRTEI, 1, c3_errorID_size[1]);
                          c3_n_loop_ub = c3_errorID_size[1] - 1;
                          for (c3_i16 = 0; c3_i16 <= c3_n_loop_ub; c3_i16++) {
                            c3_x.vector.data[c3_i16] = c3_errorID_data[c3_i16];
                          }

                          c3_m_remainingDimsB = c3_x.size[1];
                          if (c3_m_remainingDimsB != 39) {
                          } else {
                            c3_y_kstr = 1;
                            do {
                              c3_exitg1 = 0;
                              if (c3_y_kstr - 1 < 39) {
                                c3_ab_kstr = c3_y_kstr - 1;
                                if (c3_cv27[c3_ab_kstr] !=
                                    c3_errorID_data[c3_ab_kstr]) {
                                  c3_exitg1 = 1;
                                } else {
                                  c3_y_kstr++;
                                }
                              } else {
                                c3_m_result = true;
                                c3_exitg1 = 1;
                              }
                            } while (c3_exitg1 == 0);
                          }

                          if (c3_m_result) {
                            c3_index = 12.0;
                          } else {
                            c3_st.site = &c3_ef_emlrtRSI;
                            c3_n_result = c3_strcmp(chartInstance, &c3_st,
                              c3_errorID_data, c3_errorID_size);
                            if (c3_n_result) {
                              c3_index = 13.0;
                            } else {
                              c3_index = -1.0;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  c3_array_char_T_2D_Destructor(chartInstance, &c3_x);
  switch ((int32_T)c3_index) {
   case 0:
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 32),
                  false);
    c3_q_y = NULL;
    sf_mex_assign(&c3_q_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 32),
                  false);
    sf_mex_call(c3_sp, &c3_p_emlrtMCI, "error", 0U, 2U, 14, c3_b_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_q_y)));
    break;

   case 1:
    c3_c_y = NULL;
    sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    c3_r_y = NULL;
    sf_mex_assign(&c3_r_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    sf_mex_call(c3_sp, &c3_o_emlrtMCI, "error", 0U, 2U, 14, c3_c_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_r_y)));
    break;

   case 2:
    c3_d_y = NULL;
    sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_b_cv3, 10, 0U, 1, 0U, 2, 1, 34),
                  false);
    c3_s_y = NULL;
    sf_mex_assign(&c3_s_y, sf_mex_create("y", c3_b_cv3, 10, 0U, 1, 0U, 2, 1, 34),
                  false);
    sf_mex_call(c3_sp, &c3_n_emlrtMCI, "error", 0U, 2U, 14, c3_d_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_s_y)));
    break;

   case 3:
    c3_e_y = NULL;
    sf_mex_assign(&c3_e_y, sf_mex_create("y", c3_b_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                  false);
    c3_t_y = NULL;
    sf_mex_assign(&c3_t_y, sf_mex_create("y", c3_b_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                  false);
    sf_mex_call(c3_sp, &c3_m_emlrtMCI, "error", 0U, 2U, 14, c3_e_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_t_y)));
    break;

   case 4:
    c3_f_y = NULL;
    sf_mex_assign(&c3_f_y, sf_mex_create("y", c3_b_cv5, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    c3_u_y = NULL;
    sf_mex_assign(&c3_u_y, sf_mex_create("y", c3_b_cv5, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    sf_mex_call(c3_sp, &c3_l_emlrtMCI, "error", 0U, 2U, 14, c3_f_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_u_y)));
    break;

   case 5:
    c3_g_y = NULL;
    sf_mex_assign(&c3_g_y, sf_mex_create("y", c3_b_cv6, 10, 0U, 1, 0U, 2, 1, 36),
                  false);
    c3_v_y = NULL;
    sf_mex_assign(&c3_v_y, sf_mex_create("y", c3_b_cv6, 10, 0U, 1, 0U, 2, 1, 36),
                  false);
    sf_mex_call(c3_sp, &c3_k_emlrtMCI, "error", 0U, 2U, 14, c3_g_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_v_y)));
    break;

   case 6:
    c3_h_y = NULL;
    sf_mex_assign(&c3_h_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    c3_w_y = NULL;
    sf_mex_assign(&c3_w_y, sf_mex_create("y", c3_cv7, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    sf_mex_call(c3_sp, &c3_j_emlrtMCI, "error", 0U, 2U, 14, c3_h_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_w_y)));
    break;

   case 7:
    c3_i_y = NULL;
    sf_mex_assign(&c3_i_y, sf_mex_create("y", c3_cv8, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_x_y = NULL;
    sf_mex_assign(&c3_x_y, sf_mex_create("y", c3_cv8, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    sf_mex_call(c3_sp, &c3_i_emlrtMCI, "error", 0U, 2U, 14, c3_i_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_x_y)));
    break;

   case 8:
    c3_j_y = NULL;
    sf_mex_assign(&c3_j_y, sf_mex_create("y", c3_cv9, 10, 0U, 1, 0U, 2, 1, 47),
                  false);
    c3_y_y = NULL;
    sf_mex_assign(&c3_y_y, sf_mex_create("y", c3_cv9, 10, 0U, 1, 0U, 2, 1, 47),
                  false);
    sf_mex_call(c3_sp, &c3_h_emlrtMCI, "error", 0U, 2U, 14, c3_j_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_y_y)));
    break;

   case 9:
    c3_k_y = NULL;
    sf_mex_assign(&c3_k_y, sf_mex_create("y", c3_cv10, 10, 0U, 1, 0U, 2, 1, 49),
                  false);
    c3_ab_y = NULL;
    sf_mex_assign(&c3_ab_y, sf_mex_create("y", c3_cv10, 10, 0U, 1, 0U, 2, 1, 49),
                  false);
    sf_mex_call(c3_sp, &c3_g_emlrtMCI, "error", 0U, 2U, 14, c3_k_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_ab_y)));
    break;

   case 10:
    c3_l_y = NULL;
    sf_mex_assign(&c3_l_y, sf_mex_create("y", c3_cv11, 10, 0U, 1, 0U, 2, 1, 44),
                  false);
    c3_bb_y = NULL;
    sf_mex_assign(&c3_bb_y, sf_mex_create("y", c3_cv11, 10, 0U, 1, 0U, 2, 1, 44),
                  false);
    sf_mex_call(c3_sp, &c3_f_emlrtMCI, "error", 0U, 2U, 14, c3_l_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_bb_y)));
    break;

   case 11:
    c3_m_y = NULL;
    sf_mex_assign(&c3_m_y, sf_mex_create("y", c3_cv12, 10, 0U, 1, 0U, 2, 1, 46),
                  false);
    c3_cb_y = NULL;
    sf_mex_assign(&c3_cb_y, sf_mex_create("y", c3_cv12, 10, 0U, 1, 0U, 2, 1, 46),
                  false);
    sf_mex_call(c3_sp, &c3_e_emlrtMCI, "error", 0U, 2U, 14, c3_m_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_cb_y)));
    break;

   case 12:
    c3_n_y = NULL;
    sf_mex_assign(&c3_n_y, sf_mex_create("y", c3_cv13, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    c3_db_y = NULL;
    sf_mex_assign(&c3_db_y, sf_mex_create("y", c3_cv13, 10, 0U, 1, 0U, 2, 1, 39),
                  false);
    sf_mex_call(c3_sp, &c3_d_emlrtMCI, "error", 0U, 2U, 14, c3_n_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 1U, 14, c3_db_y)));
    break;

   case 13:
    c3_o_y = NULL;
    sf_mex_assign(&c3_o_y, sf_mex_create("y", c3_cv3, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_eb_y = NULL;
    sf_mex_assign(&c3_eb_y, sf_mex_create("y", c3_cv3, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_fb_y = NULL;
    sf_mex_assign(&c3_fb_y, sf_mex_create("y", c3_errorText, 10, 0U, 1, 0U, 2, 1,
      1024), false);
    sf_mex_call(c3_sp, &c3_c_emlrtMCI, "error", 0U, 2U, 14, c3_o_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 2U, 14, c3_eb_y, 14, c3_fb_y)));
    break;

   default:
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv3, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_p_y = NULL;
    sf_mex_assign(&c3_p_y, sf_mex_create("y", c3_cv3, 10, 0U, 1, 0U, 2, 1, 35),
                  false);
    c3_u_size[1] = c3_errorID_size[1] + 1038;
    for (c3_i2 = 0; c3_i2 < 7; c3_i2++) {
      c3_u_data[c3_i2] = c3_cv14[c3_i2];
    }

    c3_d_loop_ub = c3_errorID_size[1] - 1;
    for (c3_i4 = 0; c3_i4 <= c3_d_loop_ub; c3_i4++) {
      c3_u_data[c3_i4 + 7] = c3_errorID_data[c3_i4];
    }

    for (c3_i5 = 0; c3_i5 < 7; c3_i5++) {
      c3_u_data[(c3_i5 + c3_errorID_size[1]) + 7] = c3_cv16[c3_i5];
    }

    for (c3_i6 = 0; c3_i6 < 1024; c3_i6++) {
      c3_u_data[(c3_i6 + c3_errorID_size[1]) + 14] = c3_errorText[c3_i6];
    }

    c3_gb_y = NULL;
    sf_mex_assign(&c3_gb_y, sf_mex_create("y", &c3_u_data, 10, 0U, 1, 0U, 2, 1,
      c3_u_size[1]), false);
    sf_mex_call(c3_sp, &c3_q_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 2U, 14, c3_p_y, 14, c3_gb_y)));
    break;
  }
}

static void c3_API_trimString(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, char_T c3_in[1024], char_T c3_out_data[], int32_T
  c3_out_size[2])
{
  real_T c3_b_k;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_k;
  int32_T c3_len;
  int32_T c3_loop_ub;
  boolean_T c3_b;
  boolean_T c3_exitg1;
  (void)chartInstance;
  c3_len = 0;
  c3_k = 0;
  c3_exitg1 = false;
  while ((!c3_exitg1) && (c3_k < 1024)) {
    c3_b_k = (real_T)c3_k + 1.0;
    if (c3_in[(int32_T)c3_b_k - 1] == '\x00') {
      c3_exitg1 = true;
    } else {
      c3_len++;
      c3_k++;
    }
  }

  c3_b = (c3_len < 1);
  if (c3_b) {
    c3_i = -1;
  } else {
    if ((c3_len < 1) || (c3_len > 1024)) {
      emlrtDynamicBoundsCheckR2012b(c3_len, 1, 1024, &c3_emlrtBCI,
        (emlrtConstCTX)c3_sp);
    }

    c3_i = c3_len - 1;
  }

  c3_out_size[0] = 1;
  c3_out_size[1] = c3_i + 1;
  c3_loop_ub = c3_i;
  for (c3_i1 = 0; c3_i1 <= c3_loop_ub; c3_i1++) {
    c3_out_data[c3_i1] = c3_in[c3_i1];
  }
}

static boolean_T c3_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, char_T c3_b_data[], int32_T c3_b_size[2])
{
  c3_coder_array_char_T_2D c3_x;
  int32_T c3_b_kstr;
  int32_T c3_exitg1;
  int32_T c3_i;
  int32_T c3_kstr;
  int32_T c3_loop_ub;
  int32_T c3_remainingDimsB;
  boolean_T c3_bool;
  c3_bool = false;
  c3_array_char_T_2D_Constructor(chartInstance, &c3_x);
  c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_x, &c3_w_emlrtRTEI, 1,
    c3_b_size[1]);
  c3_loop_ub = c3_b_size[1] - 1;
  for (c3_i = 0; c3_i <= c3_loop_ub; c3_i++) {
    c3_x.vector.data[c3_i] = c3_b_data[c3_i];
  }

  c3_remainingDimsB = c3_x.size[1];
  c3_array_char_T_2D_Destructor(chartInstance, &c3_x);
  if (c3_remainingDimsB != 35) {
  } else {
    c3_kstr = 1;
    do {
      c3_exitg1 = 0;
      if (c3_kstr - 1 < 35) {
        c3_b_kstr = c3_kstr - 1;
        if (c3_cv3[c3_b_kstr] != c3_b_data[c3_b_kstr]) {
          c3_exitg1 = 1;
        } else {
          c3_kstr++;
        }
      } else {
        c3_bool = true;
        c3_exitg1 = 1;
      }
    } while (c3_exitg1 == 0);
  }

  return c3_bool;
}

static void c3_API_channelErrorIfFailed(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, CoderChannel c3_chImpl, int32_T
  c3_success)
{
  emlrtStack c3_b_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_y = NULL;
  int32_T c3_errorID_size[2];
  char_T c3_errorID[1024];
  char_T c3_errorID_data[1024];
  char_T c3_errorText[1024];
  boolean_T c3_hasSyncError;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  if (c3_success == 0) {
    coderChannelHasSyncError(c3_chImpl, &c3_hasSyncError);
    if (!c3_hasSyncError) {
      coderChannelGetLastError(c3_chImpl, &c3_errorID[0], &c3_errorText[0]);
      c3_st.site = &c3_y_emlrtRSI;
      c3_API_dispatchInternalError(chartInstance, &c3_st, c3_errorID,
        c3_errorText);
    } else {
      coderChannelGetLastSyncError(c3_chImpl, &c3_errorID[0], &c3_errorText[0]);
      c3_st.site = &c3_ab_emlrtRSI;
      c3_b_st.site = &c3_bb_emlrtRSI;
      c3_API_trimString(chartInstance, &c3_b_st, c3_errorID, c3_errorID_data,
                        c3_errorID_size);
      c3_y = NULL;
      sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                    false);
      c3_b_y = NULL;
      sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                    false);
      c3_c_y = NULL;
      sf_mex_assign(&c3_c_y, sf_mex_create("y", &c3_errorID_data, 10, 0U, 1, 0U,
        2, 1, c3_errorID_size[1]), false);
      c3_d_y = NULL;
      sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_errorText, 10, 0U, 1, 0U, 2,
        1, 1024), false);
      sf_mex_call(&c3_st, &c3_r_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                  sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
        (&c3_st, NULL, "message", 1U, 3U, 14, c3_b_y, 14, c3_c_y, 14, c3_d_y)));
    }
  }
}

static c3_matlabshared_asyncio_internal_InputStream *c3_InputStream_InputStream
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp,
   c3_matlabshared_asyncio_internal_InputStream *c3_obj, CoderChannel
   c3_channelImpl)
{
  CoderChannel c3_b_channelImpl;
  CoderChannel c3_chImpl;
  CoderInputStream c3_b_streamImpl;
  CoderInputStream c3_streamImpl;
  c3_matlabshared_asyncio_internal_InputStream *c3_b_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_c_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_this;
  emlrtStack c3_b_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_y = NULL;
  int32_T c3_i;
  int32_T c3_loop_ub;
  uint8_T c3_coderExampleDataLocal_data[1];
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_b_obj = c3_obj;
  c3_st.site = &c3_cb_emlrtRSI;
  c3_c_obj = c3_b_obj;
  c3_b_channelImpl = c3_channelImpl;
  c3_b_obj = c3_c_obj;
  c3_b_st.site = &c3_eb_emlrtRSI;
  c3_this = c3_b_obj;
  c3_b_obj = c3_this;
  c3_b_st.site = &c3_fb_emlrtRSI;
  c3_chImpl = c3_b_channelImpl;
  c3_streamImpl = coderChannelGetInputStream(c3_chImpl);
  c3_b_streamImpl = 0;
  if (!(c3_streamImpl != c3_b_streamImpl)) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv5, 10, 0U, 1, 0U, 2, 1, 30),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_cv5, 10, 0U, 1, 0U, 2, 1, 30),
                  false);
    sf_mex_call(&c3_b_st, &c3_s_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(&c3_b_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_b_st, NULL, "message", 1U, 1U, 14, c3_b_y)));
  }

  c3_b_obj->StreamImpl = c3_streamImpl;
  c3_b_obj->ChannelImpl = c3_b_channelImpl;
  c3_coderExampleDataLocal_data[0] = 0U;
  c3_st.site = &c3_df_emlrtRSI;
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_st, &c3_b_obj->ExampleData,
    &c3_bb_emlrtRTEI, 1, 1);
  c3_loop_ub = 0;
  for (c3_i = 0; c3_i <= c3_loop_ub; c3_i++) {
    c3_b_obj->ExampleData.vector.data[c3_i] = c3_coderExampleDataLocal_data[c3_i];
  }

  c3_st.site = &c3_db_emlrtRSI;
  c3_InputStream_clearPartialPacket(chartInstance, &c3_st, c3_b_obj);
  c3_b_obj->matlabCodegenIsDeleted = false;
  return c3_b_obj;
}

static void c3_InputStream_clearPartialPacket(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_asyncio_internal_InputStream *c3_obj)
{
  static char_T c3_b_cv[15] = { 'M', 'A', 'T', 'L', 'A', 'B', ':', 'p', 'm', 'a',
    'x', 's', 'i', 'z', 'e' };

  c3_coder_array_uint8_T_2D c3_exampleData;
  c3_coder_array_uint8_T_2D c3_partialPacketInitializer;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_st;
  const mxArray *c3_y = NULL;
  int32_T c3_outsize[2];
  int32_T c3_i;
  int32_T c3_loop_ub;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_st.site = &c3_gb_emlrtRSI;
  c3_array_uint8_T_2D_Constructor(chartInstance, &c3_exampleData);
  c3_b_st.site = &c3_gb_emlrtRSI;
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_b_st, &c3_exampleData,
    &c3_cb_emlrtRTEI, c3_obj->ExampleData.size[0], c3_obj->ExampleData.size[1]);
  c3_loop_ub = c3_obj->ExampleData.size[0] * c3_obj->ExampleData.size[1] - 1;
  for (c3_i = 0; c3_i <= c3_loop_ub; c3_i++) {
    c3_exampleData.vector.data[c3_i] = c3_obj->ExampleData.vector.data[c3_i];
  }

  c3_b_st.site = &c3_hb_emlrtRSI;
  c3_c_st.site = &c3_ib_emlrtRSI;
  c3_outsize[0] = c3_exampleData.size[0];
  if (c3_outsize[0] != c3_exampleData.size[0]) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 15),
                  false);
    sf_mex_call(&c3_c_st, &c3_t_emlrtMCI, "error", 0U, 1U, 14, c3_y);
  }

  c3_array_uint8_T_2D_Destructor(chartInstance, &c3_exampleData);
  c3_array_uint8_T_2D_Constructor(chartInstance, &c3_partialPacketInitializer);
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_c_st,
    &c3_partialPacketInitializer, &c3_db_emlrtRTEI, c3_outsize[0], 0);
  c3_st.site = &c3_cf_emlrtRSI;
  c3_array_uint8_T_2D_SetSize(chartInstance, &c3_st, &c3_obj->PartialPacket,
    &c3_eb_emlrtRTEI, c3_partialPacketInitializer.size[0], 0);
  c3_array_uint8_T_2D_Destructor(chartInstance, &c3_partialPacketInitializer);
  c3_obj->PartialPacketStart = 0.0;
  c3_obj->PartialPacketCount = 0.0;
}

static c3_matlabshared_asyncio_internal_OutputStream
  *c3_OutputStream_OutputStream(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, c3_matlabshared_asyncio_internal_OutputStream *c3_obj,
  CoderChannel c3_channelImpl)
{
  CoderChannel c3_b_channelImpl;
  CoderChannel c3_chImpl;
  CoderOutputStream c3_b_streamImpl;
  CoderOutputStream c3_streamImpl;
  c3_matlabshared_asyncio_internal_OutputStream *c3_b_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_c_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_this;
  emlrtStack c3_b_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_y = NULL;
  (void)chartInstance;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_b_obj = c3_obj;
  c3_st.site = &c3_jb_emlrtRSI;
  c3_c_obj = c3_b_obj;
  c3_b_channelImpl = c3_channelImpl;
  c3_b_obj = c3_c_obj;
  c3_b_obj->Timeout = 10.0;
  c3_b_st.site = &c3_eb_emlrtRSI;
  c3_this = c3_b_obj;
  c3_b_obj = c3_this;
  c3_b_st.site = &c3_kb_emlrtRSI;
  c3_chImpl = c3_b_channelImpl;
  c3_streamImpl = coderChannelGetOutputStream(c3_chImpl);
  c3_b_streamImpl = 0;
  if (!(c3_streamImpl != c3_b_streamImpl)) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv5, 10, 0U, 1, 0U, 2, 1, 30),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_cv5, 10, 0U, 1, 0U, 2, 1, 30),
                  false);
    sf_mex_call(&c3_b_st, &c3_u_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(&c3_b_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_b_st, NULL, "message", 1U, 1U, 14, c3_b_y)));
  }

  c3_b_obj->StreamImpl = c3_streamImpl;
  c3_b_obj->ChannelImpl = c3_b_channelImpl;
  c3_b_obj->matlabCodegenIsDeleted = false;
  return c3_b_obj;
}

static void c3_TCPClient_validateDisconnected(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp,
  c3_matlabshared_network_internal_TCPClient *c3_obj)
{
  static char_T c3_b_cv[45] = { 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', 'l',
    'i', 'b', ':', 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', ':', 'c', 'a',
    'n', 'n', 'o', 't', 'S', 'e', 't', 'W', 'h', 'e', 'n', 'C', 'o', 'n', 'n',
    'e', 'c', 't', 'e', 'd' };

  CoderChannel c3_chImpl;
  c3_matlabshared_asyncio_internal_Channel *c3_d_obj;
  c3_matlabshared_network_internal_TCPClient *c3_b_obj;
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_c_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_y = NULL;
  int32_T c3_success;
  boolean_T c3_b_value;
  boolean_T c3_out;
  boolean_T c3_value;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_st.site = &c3_m_emlrtRSI;
  c3_b_obj = c3_obj;
  c3_b_st.site = &c3_m_emlrtRSI;
  c3_c_obj = c3_b_obj->TransportChannel;
  c3_c_st.site = &c3_lb_emlrtRSI;
  c3_d_obj = c3_c_obj->AsyncIOChannel;
  c3_d_st.site = &c3_yb_emlrtRSI;
  c3_chImpl = c3_d_obj->ChannelImpl;
  c3_success = coderChannelIsOpen(c3_chImpl, &c3_out);
  c3_e_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_chImpl, c3_success);
  if (c3_out) {
    c3_value = true;
  } else {
    c3_value = false;
  }

  c3_b_value = c3_value;
  if (c3_b_value) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 45),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 45),
                  false);
    sf_mex_call(c3_sp, &c3_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14, sf_mex_call
                (c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call(c3_sp, NULL,
      "message", 1U, 1U, 14, c3_b_y)));
  }
}

static void c3_AsyncIOTransportChannel_writeAsyncRaw
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp,
   c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_obj,
   uint8_T c3_data[88])
{
  static char_T c3_b_cv[34] = { 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', 'l',
    'i', 'b', ':', 't', 'r', 'a', 'n', 's', 'p', 'o', 'r', 't', ':', 'w', 'r',
    'i', 't', 'e', 'F', 'a', 'i', 'l', 'e', 'd' };

  static char_T c3_b_cv1[7] = { 't', 'i', 'm', 'e', 'o', 'u', 't' };

  static char_T c3_b_b[4] = { 'd', 'o', 'n', 'e' };

  static char_T c3_b_cv2[4] = { 'd', 'o', 'n', 'e' };

  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderChannel c3_e_chImpl;
  CoderChannel c3_f_chImpl;
  CoderOutputStream c3_b_streamImpl;
  CoderOutputStream c3_c_streamImpl;
  CoderOutputStream c3_d_streamImpl;
  CoderOutputStream c3_e_streamImpl;
  CoderOutputStream c3_f_streamImpl;
  CoderOutputStream c3_g_streamImpl;
  CoderOutputStream c3_h_streamImpl;
  CoderOutputStream c3_i_streamImpl;
  CoderOutputStream c3_j_streamImpl;
  CoderOutputStream c3_k_streamImpl;
  CoderOutputStream c3_l_streamImpl;
  CoderOutputStream c3_streamImpl;
  c3_cell_wrap_22 c3_packets_data[1];
  c3_cell_wrap_22 c3_tmp_data[1];
  c3_cell_wrap_22 c3_r;
  c3_coder_array_char_T_2D c3_x;
  c3_matlabshared_asyncio_internal_OutputStream *c3_b_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_c_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_d_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_e_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_f_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_g_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_h_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_i_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_j_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_k_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_g_st;
  emlrtStack c3_st;
  emlrtTimespec c3_startTic;
  const mxArray *c3_ab_y = NULL;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_bb_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_cb_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_e_y = NULL;
  const mxArray *c3_eb_y = NULL;
  const mxArray *c3_f_y = NULL;
  const mxArray *c3_fb_y = NULL;
  const mxArray *c3_g_y = NULL;
  const mxArray *c3_h_y = NULL;
  const mxArray *c3_i_y = NULL;
  const mxArray *c3_j_y = NULL;
  const mxArray *c3_k_y = NULL;
  const mxArray *c3_l_y = NULL;
  const mxArray *c3_n_y = NULL;
  const mxArray *c3_o_y = NULL;
  const mxArray *c3_p_y = NULL;
  const mxArray *c3_q_y = NULL;
  const mxArray *c3_r_y = NULL;
  const mxArray *c3_s_y = NULL;
  const mxArray *c3_t_y = NULL;
  const mxArray *c3_v_y = NULL;
  const mxArray *c3_w_y = NULL;
  const mxArray *c3_x_y = NULL;
  const mxArray *c3_y = NULL;
  real_T c3_b_count;
  real_T c3_b_packetStartIndex;
  real_T c3_c_count;
  real_T c3_count;
  real_T c3_countWrittenThisIteration;
  real_T c3_d_count;
  real_T c3_et;
  real_T c3_numBytes;
  real_T c3_packetEndIndex;
  real_T c3_packetStartIndex;
  real_T c3_timeoutInSeconds;
  int32_T c3_errorStr_size[2];
  int32_T c3_status_size[2];
  int32_T c3_b_kstr;
  int32_T c3_b_loop_ub;
  int32_T c3_b_remainingDimsA;
  int32_T c3_b_success;
  int32_T c3_c_kstr;
  int32_T c3_c_loop_ub;
  int32_T c3_c_success;
  int32_T c3_d_kstr;
  int32_T c3_d_success;
  int32_T c3_e_success;
  int32_T c3_exitg3;
  int32_T c3_f_success;
  int32_T c3_g_success;
  int32_T c3_h_success;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i2;
  int32_T c3_i3;
  int32_T c3_i4;
  int32_T c3_i5;
  int32_T c3_i6;
  int32_T c3_i7;
  int32_T c3_i8;
  int32_T c3_i9;
  int32_T c3_i_success;
  int32_T c3_ii;
  int32_T c3_j_success;
  int32_T c3_k_success;
  int32_T c3_kstr;
  int32_T c3_l_success;
  int32_T c3_loop_ub;
  int32_T c3_remainingDimsA;
  int32_T c3_success;
  char_T c3_errorStr_data[9];
  char_T c3_status_data[9];
  char_T c3_b_s;
  char_T c3_b_x;
  char_T c3_c_s;
  char_T c3_c_x;
  char_T c3_d_s;
  char_T c3_d_x;
  char_T c3_db_y;
  char_T c3_e_s;
  char_T c3_e_x;
  char_T c3_f_s;
  char_T c3_f_x;
  char_T c3_g_s;
  char_T c3_g_x;
  char_T c3_h_s;
  char_T c3_h_x;
  char_T c3_i_s;
  char_T c3_i_x;
  char_T c3_j_s;
  char_T c3_k_s;
  char_T c3_l_s;
  char_T c3_m_s;
  char_T c3_m_y;
  char_T c3_n_s;
  char_T c3_o_s;
  char_T c3_p_s;
  char_T c3_s;
  char_T c3_u_y;
  char_T c3_y_y;
  int8_T c3_b_u;
  int8_T c3_c_u;
  int8_T c3_d_u;
  int8_T c3_e_u;
  int8_T c3_f_u;
  int8_T c3_g_u;
  int8_T c3_h_u;
  int8_T c3_u;
  uint8_T c3_packet[88];
  boolean_T c3_b;
  boolean_T c3_b1;
  boolean_T c3_b2;
  boolean_T c3_b3;
  boolean_T c3_b4;
  boolean_T c3_b5;
  boolean_T c3_b6;
  boolean_T c3_b7;
  boolean_T c3_b8;
  boolean_T c3_b_bool;
  boolean_T c3_b_done;
  boolean_T c3_b_p;
  boolean_T c3_b_result;
  boolean_T c3_bool;
  boolean_T c3_c_p;
  boolean_T c3_completed;
  boolean_T c3_d_p;
  boolean_T c3_done;
  boolean_T c3_e_p;
  boolean_T c3_exitg1;
  boolean_T c3_exitg2;
  boolean_T c3_f_p;
  boolean_T c3_g_p;
  boolean_T c3_guard1;
  boolean_T c3_h_p;
  boolean_T c3_p;
  boolean_T c3_result;
  boolean_T c3_timeout;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_g_st.prev = &c3_f_st;
  c3_g_st.tls = c3_f_st.tls;
  c3_st.site = &c3_lb_emlrtRSI;
  c3_b_obj = &c3_obj->AsyncIOChannel->OutputStream;
  c3_numBytes = 0.0;
  c3_errorStr_size[1] = 0;
  for (c3_i = 0; c3_i < 88; c3_i++) {
    c3_r.f1[c3_i] = c3_data[c3_i];
  }

  c3_packets_data[0] = c3_r;
  c3_packetStartIndex = 1.0;
  c3_array_char_T_2D_Constructor(chartInstance, &c3_x);
  c3_exitg1 = false;
  while ((!c3_exitg1) && (c3_numBytes < 88.0)) {
    c3_b_st.site = &c3_oc_emlrtRSI;
    c3_c_obj = c3_b_obj;
    c3_c_st.site = &c3_tc_emlrtRSI;
    c3_streamImpl = c3_c_obj->StreamImpl;
    c3_success = coderStreamGetSpaceAvailable(c3_streamImpl, &c3_count);
    c3_d_st.site = &c3_uc_emlrtRSI;
    c3_b_streamImpl = c3_streamImpl;
    c3_b_success = c3_success;
    if (c3_b_success == 0) {
      c3_chImpl = coderStreamGetChannel(c3_b_streamImpl);
      c3_e_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_chImpl);
    }

    c3_guard1 = false;
    if (c3_count == 0.0) {
      c3_b_st.site = &c3_pc_emlrtRSI;
      c3_d_obj = c3_b_obj;
      c3_timeoutInSeconds = c3_d_obj->Timeout;
      c3_status_size[1] = 0;
      c3_timeout = false;
      c3_done = false;
      c3_c_st.site = &c3_wc_emlrtRSI;
      c3_f_obj = c3_d_obj;
      c3_d_st.site = &c3_pc_emlrtRSI;
      c3_g_obj = c3_f_obj;
      c3_e_st.site = &c3_tc_emlrtRSI;
      c3_c_streamImpl = c3_g_obj->StreamImpl;
      c3_c_success = coderStreamGetSpaceAvailable(c3_c_streamImpl, &c3_c_count);
      c3_f_st.site = &c3_uc_emlrtRSI;
      c3_d_streamImpl = c3_c_streamImpl;
      c3_d_success = c3_c_success;
      if (c3_d_success == 0) {
        c3_b_chImpl = coderStreamGetChannel(c3_d_streamImpl);
        c3_g_st.site = &c3_vc_emlrtRSI;
        c3_b_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_b_chImpl);
      }

      c3_completed = (c3_c_count > 0.0);
      c3_c_st.site = &c3_xc_emlrtRSI;
      c3_startTic = c3_tic(chartInstance, &c3_c_st);
      while ((!c3_completed) && (!c3_done) && (!c3_timeout)) {
        c3_c_st.site = &c3_yc_emlrtRSI;
        c3_et = c3_toc(chartInstance, &c3_c_st, c3_startTic);
        if (c3_et < 1.0) {
          c3_c_st.site = &c3_ad_emlrtRSI;
          c3_d_st.site = &c3_kd_emlrtRSI;
          c3_pause(chartInstance, &c3_d_st, 0.0);
        } else {
          c3_c_st.site = &c3_bd_emlrtRSI;
          c3_d_st.site = &c3_kd_emlrtRSI;
          c3_pause(chartInstance, &c3_d_st, 0.005);
        }

        c3_c_st.site = &c3_cd_emlrtRSI;
        c3_et = c3_toc(chartInstance, &c3_c_st, c3_startTic);
        c3_timeout = (c3_et > c3_timeoutInSeconds);
        c3_c_st.site = &c3_dd_emlrtRSI;
        c3_h_obj = c3_d_obj;
        c3_d_st.site = &c3_md_emlrtRSI;
        c3_g_streamImpl = c3_h_obj->StreamImpl;
        c3_g_success = coderStreamIsDeviceDone(c3_g_streamImpl, &c3_result);
        c3_e_st.site = &c3_nd_emlrtRSI;
        c3_h_streamImpl = c3_g_streamImpl;
        c3_h_success = c3_g_success;
        if (c3_h_success == 0) {
          c3_d_chImpl = coderStreamGetChannel(c3_h_streamImpl);
          c3_f_st.site = &c3_vc_emlrtRSI;
          c3_b_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_d_chImpl);
        }

        if (c3_result) {
          c3_b_done = true;
        } else {
          c3_c_st.site = &c3_dd_emlrtRSI;
          c3_i_obj = c3_d_obj;
          c3_d_st.site = &c3_od_emlrtRSI;
          c3_i_streamImpl = c3_i_obj->StreamImpl;
          c3_i_success = coderStreamIsOpen(c3_i_streamImpl, &c3_b_result);
          c3_e_st.site = &c3_pd_emlrtRSI;
          c3_j_streamImpl = c3_i_streamImpl;
          c3_j_success = c3_i_success;
          if (c3_j_success == 0) {
            c3_e_chImpl = coderStreamGetChannel(c3_j_streamImpl);
            c3_f_st.site = &c3_vc_emlrtRSI;
            c3_b_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_e_chImpl);
          }

          if (!c3_b_result) {
            c3_b_done = true;
          } else {
            c3_b_done = false;
          }
        }

        c3_done = c3_b_done;
        c3_c_st.site = &c3_ed_emlrtRSI;
        c3_j_obj = c3_d_obj;
        c3_d_st.site = &c3_pc_emlrtRSI;
        c3_k_obj = c3_j_obj;
        c3_e_st.site = &c3_tc_emlrtRSI;
        c3_k_streamImpl = c3_k_obj->StreamImpl;
        c3_k_success = coderStreamGetSpaceAvailable(c3_k_streamImpl, &c3_d_count);
        c3_f_st.site = &c3_uc_emlrtRSI;
        c3_l_streamImpl = c3_k_streamImpl;
        c3_l_success = c3_k_success;
        if (c3_l_success == 0) {
          c3_f_chImpl = coderStreamGetChannel(c3_l_streamImpl);
          c3_g_st.site = &c3_vc_emlrtRSI;
          c3_b_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_f_chImpl);
        }

        c3_completed = (c3_d_count > 0.0);
      }

      if (c3_completed) {
        c3_status_size[1] = 9;
        for (c3_i4 = 0; c3_i4 < 9; c3_i4++) {
          c3_status_data[c3_i4] = c3_cv6[c3_i4];
        }
      } else if (c3_done) {
        c3_status_size[1] = 4;
        for (c3_i5 = 0; c3_i5 < 4; c3_i5++) {
          c3_status_data[c3_i5] = c3_b_b[c3_i5];
        }
      } else if (c3_timeout) {
        c3_status_size[1] = 7;
        for (c3_i6 = 0; c3_i6 < 7; c3_i6++) {
          c3_status_data[c3_i6] = c3_b_cv1[c3_i6];
        }
      }

      c3_b_st.site = &c3_qc_emlrtRSI;
      c3_c_st.site = &c3_hc_emlrtRSI;
      c3_d_st.site = &c3_ic_emlrtRSI;
      c3_bool = false;
      c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_x,
        &c3_ib_emlrtRTEI, 1, c3_status_size[1]);
      c3_loop_ub = c3_status_size[1] - 1;
      for (c3_i7 = 0; c3_i7 <= c3_loop_ub; c3_i7++) {
        c3_x.vector.data[c3_i7] = c3_status_data[c3_i7];
      }

      c3_remainingDimsA = c3_x.size[1];
      if (c3_remainingDimsA != 9) {
      } else {
        c3_kstr = 1;
        do {
          c3_exitg3 = 0;
          if (c3_kstr - 1 < 9) {
            c3_b_kstr = c3_kstr - 1;
            c3_e_st.site = &c3_jc_emlrtRSI;
            c3_s = c3_status_data[c3_b_kstr];
            c3_b_s = c3_s;
            c3_b1 = ((uint8_T)c3_b_s <= 127);
            c3_p = c3_b1;
            if (!c3_p) {
              c3_d_y = NULL;
              sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_e_y = NULL;
              sf_mex_assign(&c3_e_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_u = MAX_int8_T;
              c3_f_y = NULL;
              sf_mex_assign(&c3_f_y, sf_mex_create("y", &c3_u, 2, 0U, 0, 0U, 0),
                            false);
              sf_mex_call(&c3_e_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_d_y,
                          14, sf_mex_call(&c3_e_st, NULL, "getString", 1U, 1U,
                14, sf_mex_call(&c3_e_st, NULL, "message", 1U, 2U, 14, c3_e_y,
                                14, c3_f_y)));
            }

            c3_e_st.site = &c3_kc_emlrtRSI;
            c3_c_s = c3_cv6[c3_b_kstr];
            c3_d_s = c3_c_s;
            c3_b2 = ((uint8_T)c3_d_s <= 127);
            c3_b_p = c3_b2;
            if (!c3_b_p) {
              c3_g_y = NULL;
              sf_mex_assign(&c3_g_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_h_y = NULL;
              sf_mex_assign(&c3_h_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_b_u = MAX_int8_T;
              c3_i_y = NULL;
              sf_mex_assign(&c3_i_y, sf_mex_create("y", &c3_b_u, 2, 0U, 0, 0U, 0),
                            false);
              sf_mex_call(&c3_e_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_g_y,
                          14, sf_mex_call(&c3_e_st, NULL, "getString", 1U, 1U,
                14, sf_mex_call(&c3_e_st, NULL, "message", 1U, 2U, 14, c3_h_y,
                                14, c3_i_y)));
            }

            c3_e_st.site = &c3_lc_emlrtRSI;
            c3_b_x = c3_status_data[c3_b_kstr];
            c3_f_st.site = &c3_mc_emlrtRSI;
            c3_c_x = c3_b_x;
            c3_g_st.site = &c3_nc_emlrtRSI;
            c3_g_s = c3_c_x;
            c3_h_s = c3_g_s;
            c3_b4 = ((uint8_T)c3_h_s <= 127);
            c3_d_p = c3_b4;
            if (!c3_d_p) {
              c3_k_y = NULL;
              sf_mex_assign(&c3_k_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_n_y = NULL;
              sf_mex_assign(&c3_n_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_d_u = MAX_int8_T;
              c3_p_y = NULL;
              sf_mex_assign(&c3_p_y, sf_mex_create("y", &c3_d_u, 2, 0U, 0, 0U, 0),
                            false);
              sf_mex_call(&c3_g_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_k_y,
                          14, sf_mex_call(&c3_g_st, NULL, "getString", 1U, 1U,
                14, sf_mex_call(&c3_g_st, NULL, "message", 1U, 2U, 14, c3_n_y,
                                14, c3_p_y)));
            }

            c3_m_y = c3_cv[(uint8_T)c3_c_x & 127];
            c3_e_st.site = &c3_lc_emlrtRSI;
            c3_d_x = c3_cv6[c3_b_kstr];
            c3_f_st.site = &c3_mc_emlrtRSI;
            c3_e_x = c3_d_x;
            c3_g_st.site = &c3_nc_emlrtRSI;
            c3_k_s = c3_e_x;
            c3_l_s = c3_k_s;
            c3_b6 = ((uint8_T)c3_l_s <= 127);
            c3_f_p = c3_b6;
            if (!c3_f_p) {
              c3_t_y = NULL;
              sf_mex_assign(&c3_t_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_v_y = NULL;
              sf_mex_assign(&c3_v_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
                1, 31), false);
              c3_f_u = MAX_int8_T;
              c3_x_y = NULL;
              sf_mex_assign(&c3_x_y, sf_mex_create("y", &c3_f_u, 2, 0U, 0, 0U, 0),
                            false);
              sf_mex_call(&c3_g_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_t_y,
                          14, sf_mex_call(&c3_g_st, NULL, "getString", 1U, 1U,
                14, sf_mex_call(&c3_g_st, NULL, "message", 1U, 2U, 14, c3_v_y,
                                14, c3_x_y)));
            }

            c3_u_y = c3_cv[(uint8_T)c3_e_x & 127];
            if (c3_m_y != c3_u_y) {
              c3_exitg3 = 1;
            } else {
              c3_kstr++;
            }
          } else {
            c3_bool = true;
            c3_exitg3 = 1;
          }
        } while (c3_exitg3 == 0);
      }

      if (!c3_bool) {
        c3_errorStr_size[1] = c3_status_size[1];
        c3_b_loop_ub = c3_status_size[1] - 1;
        for (c3_i8 = 0; c3_i8 <= c3_b_loop_ub; c3_i8++) {
          c3_errorStr_data[c3_i8] = c3_status_data[c3_i8];
        }

        c3_b_st.site = &c3_rc_emlrtRSI;
        c3_c_st.site = &c3_hc_emlrtRSI;
        c3_d_st.site = &c3_ic_emlrtRSI;
        c3_b_bool = false;
        c3_array_char_T_2D_SetSize(chartInstance, &c3_d_st, &c3_x,
          &c3_ib_emlrtRTEI, 1, c3_status_size[1]);
        c3_c_loop_ub = c3_status_size[1] - 1;
        for (c3_i9 = 0; c3_i9 <= c3_c_loop_ub; c3_i9++) {
          c3_x.vector.data[c3_i9] = c3_status_data[c3_i9];
        }

        c3_b_remainingDimsA = c3_x.size[1];
        if (c3_b_remainingDimsA != 4) {
        } else {
          c3_c_kstr = 1;
          do {
            c3_exitg3 = 0;
            if (c3_c_kstr - 1 < 4) {
              c3_d_kstr = c3_c_kstr - 1;
              c3_e_st.site = &c3_jc_emlrtRSI;
              c3_e_s = c3_status_data[c3_d_kstr];
              c3_f_s = c3_e_s;
              c3_b3 = ((uint8_T)c3_f_s <= 127);
              c3_c_p = c3_b3;
              if (!c3_c_p) {
                c3_j_y = NULL;
                sf_mex_assign(&c3_j_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_l_y = NULL;
                sf_mex_assign(&c3_l_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_c_u = MAX_int8_T;
                c3_o_y = NULL;
                sf_mex_assign(&c3_o_y, sf_mex_create("y", &c3_c_u, 2, 0U, 0, 0U,
                  0), false);
                sf_mex_call(&c3_e_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14,
                            c3_j_y, 14, sf_mex_call(&c3_e_st, NULL, "getString",
                  1U, 1U, 14, sf_mex_call(&c3_e_st, NULL, "message", 1U, 2U, 14,
                  c3_l_y, 14, c3_o_y)));
              }

              c3_e_st.site = &c3_kc_emlrtRSI;
              c3_i_s = c3_b_cv2[c3_d_kstr];
              c3_j_s = c3_i_s;
              c3_b5 = ((uint8_T)c3_j_s <= 127);
              c3_e_p = c3_b5;
              if (!c3_e_p) {
                c3_q_y = NULL;
                sf_mex_assign(&c3_q_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_r_y = NULL;
                sf_mex_assign(&c3_r_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_e_u = MAX_int8_T;
                c3_s_y = NULL;
                sf_mex_assign(&c3_s_y, sf_mex_create("y", &c3_e_u, 2, 0U, 0, 0U,
                  0), false);
                sf_mex_call(&c3_e_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14,
                            c3_q_y, 14, sf_mex_call(&c3_e_st, NULL, "getString",
                  1U, 1U, 14, sf_mex_call(&c3_e_st, NULL, "message", 1U, 2U, 14,
                  c3_r_y, 14, c3_s_y)));
              }

              c3_e_st.site = &c3_lc_emlrtRSI;
              c3_f_x = c3_status_data[c3_d_kstr];
              c3_f_st.site = &c3_mc_emlrtRSI;
              c3_g_x = c3_f_x;
              c3_g_st.site = &c3_nc_emlrtRSI;
              c3_m_s = c3_g_x;
              c3_n_s = c3_m_s;
              c3_b7 = ((uint8_T)c3_n_s <= 127);
              c3_g_p = c3_b7;
              if (!c3_g_p) {
                c3_w_y = NULL;
                sf_mex_assign(&c3_w_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_ab_y = NULL;
                sf_mex_assign(&c3_ab_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_g_u = MAX_int8_T;
                c3_bb_y = NULL;
                sf_mex_assign(&c3_bb_y, sf_mex_create("y", &c3_g_u, 2, 0U, 0, 0U,
                  0), false);
                sf_mex_call(&c3_g_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14,
                            c3_w_y, 14, sf_mex_call(&c3_g_st, NULL, "getString",
                  1U, 1U, 14, sf_mex_call(&c3_g_st, NULL, "message", 1U, 2U, 14,
                  c3_ab_y, 14, c3_bb_y)));
              }

              c3_y_y = c3_cv[(uint8_T)c3_g_x & 127];
              c3_e_st.site = &c3_lc_emlrtRSI;
              c3_h_x = c3_b_cv2[c3_d_kstr];
              c3_f_st.site = &c3_mc_emlrtRSI;
              c3_i_x = c3_h_x;
              c3_g_st.site = &c3_nc_emlrtRSI;
              c3_o_s = c3_i_x;
              c3_p_s = c3_o_s;
              c3_b8 = ((uint8_T)c3_p_s <= 127);
              c3_h_p = c3_b8;
              if (!c3_h_p) {
                c3_cb_y = NULL;
                sf_mex_assign(&c3_cb_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_eb_y = NULL;
                sf_mex_assign(&c3_eb_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U,
                  2, 1, 31), false);
                c3_h_u = MAX_int8_T;
                c3_fb_y = NULL;
                sf_mex_assign(&c3_fb_y, sf_mex_create("y", &c3_h_u, 2, 0U, 0, 0U,
                  0), false);
                sf_mex_call(&c3_g_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14,
                            c3_cb_y, 14, sf_mex_call(&c3_g_st, NULL, "getString",
                  1U, 1U, 14, sf_mex_call(&c3_g_st, NULL, "message", 1U, 2U, 14,
                  c3_eb_y, 14, c3_fb_y)));
              }

              c3_db_y = c3_cv[(uint8_T)c3_i_x & 127];
              if (c3_y_y != c3_db_y) {
                c3_exitg3 = 1;
              } else {
                c3_c_kstr++;
              }
            } else {
              c3_b_bool = true;
              c3_exitg3 = 1;
            }
          } while (c3_exitg3 == 0);
        }

        if (c3_b_bool) {
          c3_errorStr_size[1] = 0;
        }

        c3_exitg1 = true;
      } else {
        c3_guard1 = true;
      }
    } else {
      c3_guard1 = true;
    }

    if (c3_guard1) {
      c3_b_st.site = &c3_sc_emlrtRSI;
      c3_e_obj = c3_b_obj;
      c3_b_packetStartIndex = c3_packetStartIndex;
      c3_b_count = 0.0;
      c3_packetEndIndex = c3_b_packetStartIndex;
      c3_i1 = (int32_T)(1.0 + (1.0 - c3_b_packetStartIndex));
      emlrtForLoopVectorCheckR2021a(c3_b_packetStartIndex, 1.0, 1.0,
        mxDOUBLE_CLASS, c3_i1, &c3_qb_emlrtRTEI, &c3_b_st);
      c3_ii = 0;
      c3_exitg2 = false;
      while ((!c3_exitg2) && (c3_ii <= c3_i1 - 1)) {
        c3_tmp_data[0] = c3_packets_data[0];
        if (c3_b_packetStartIndex != (real_T)(int32_T)muDoubleScalarFloor
            (c3_b_packetStartIndex)) {
          emlrtIntegerCheckR2012b(c3_b_packetStartIndex, &c3_b_emlrtDCI,
            &c3_b_st);
        }

        c3_i2 = (int32_T)c3_b_packetStartIndex - 1;
        if ((c3_i2 < 0) || (c3_i2 > 0)) {
          emlrtDynamicBoundsCheckR2012b(c3_i2, 0, 0, &c3_b_emlrtBCI, &c3_b_st);
        }

        for (c3_i3 = 0; c3_i3 < 88; c3_i3++) {
          c3_packet[c3_i3] = c3_tmp_data[0].f1[c3_i3];
        }

        c3_c_st.site = &c3_qd_emlrtRSI;
        c3_e_streamImpl = c3_e_obj->StreamImpl;
        c3_e_success = coderOutputStreamWriteTypedDataOld(c3_e_streamImpl,
          &c3_countWrittenThisIteration, 1, "uint8", 88, &c3_packet[0]);
        c3_d_st.site = &c3_rd_emlrtRSI;
        c3_f_streamImpl = c3_e_streamImpl;
        c3_f_success = c3_e_success;
        if (c3_f_success == 0) {
          c3_c_chImpl = coderStreamGetChannel(c3_f_streamImpl);
          c3_e_st.site = &c3_vc_emlrtRSI;
          c3_b_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_c_chImpl);
        }

        if (c3_countWrittenThisIteration == 0.0) {
          c3_exitg2 = true;
        } else {
          c3_b_count += c3_countWrittenThisIteration;
          c3_packetEndIndex++;
          c3_ii++;
        }
      }

      c3_numBytes += c3_b_count;
      c3_packetStartIndex = c3_packetEndIndex;
    }
  }

  c3_array_char_T_2D_Destructor(chartInstance, &c3_x);
  c3_b = (c3_errorStr_size[1] == 0);
  if (!c3_b) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 34),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 34),
                  false);
    c3_c_y = NULL;
    sf_mex_assign(&c3_c_y, sf_mex_create("y", &c3_errorStr_data, 10, 0U, 1, 0U,
      2, 1, c3_errorStr_size[1]), false);
    sf_mex_call(c3_sp, &c3_y_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (c3_sp, NULL, "message", 1U, 2U, 14, c3_b_y, 14, c3_c_y)));
  }

  c3_obj->NumBytesWritten += c3_numBytes;
}

static void c3_b_API_channelErrorIfFailed(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, CoderChannel c3_chImpl)
{
  emlrtStack c3_b_st;
  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_y = NULL;
  int32_T c3_errorID_size[2];
  char_T c3_errorID[1024];
  char_T c3_errorID_data[1024];
  char_T c3_errorText[1024];
  boolean_T c3_hasSyncError;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  coderChannelHasSyncError(c3_chImpl, &c3_hasSyncError);
  if (!c3_hasSyncError) {
    coderChannelGetLastError(c3_chImpl, &c3_errorID[0], &c3_errorText[0]);
    c3_st.site = &c3_y_emlrtRSI;
    c3_API_dispatchInternalError(chartInstance, &c3_st, c3_errorID, c3_errorText);
  } else {
    coderChannelGetLastSyncError(c3_chImpl, &c3_errorID[0], &c3_errorText[0]);
    c3_st.site = &c3_ab_emlrtRSI;
    c3_b_st.site = &c3_bb_emlrtRSI;
    c3_API_trimString(chartInstance, &c3_b_st, c3_errorID, c3_errorID_data,
                      c3_errorID_size);
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_cv4, 10, 0U, 1, 0U, 2, 1, 37),
                  false);
    c3_c_y = NULL;
    sf_mex_assign(&c3_c_y, sf_mex_create("y", &c3_errorID_data, 10, 0U, 1, 0U, 2,
      1, c3_errorID_size[1]), false);
    c3_d_y = NULL;
    sf_mex_assign(&c3_d_y, sf_mex_create("y", c3_errorText, 10, 0U, 1, 0U, 2, 1,
      1024), false);
    sf_mex_call(&c3_st, &c3_r_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(&c3_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_st, NULL, "message", 1U, 3U, 14, c3_b_y, 14, c3_c_y, 14, c3_d_y)));
  }
}

static emlrtTimespec c3_tic(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp)
{
  emlrtStack c3_st;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_st.site = &c3_fd_emlrtRSI;
  return c3_getTime(chartInstance, &c3_st);
}

static emlrtTimespec c3_getTime(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp)
{
  static char_T c3_b_cv[33] = { 'C', 'o', 'd', 'e', 'r', ':', 't', 'o', 'o', 'l',
    'b', 'o', 'x', ':', 'C', 'o', 'd', 'e', 'r', 'T', 'i', 'm', 'e', 'C', 'a',
    'l', 'l', 'F', 'a', 'i', 'l', 'e', 'd' };

  static char_T c3_b_cv1[26] = { 'e', 'm', 'l', 'r', 't', 'C', 'l', 'o', 'c',
    'k', 'G', 'e', 't', 't', 'i', 'm', 'e', 'M', 'o', 'n', 'o', 't', 'o', 'n',
    'i', 'c' };

  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_st;
  emlrtTimespec c3_t;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_y = NULL;
  int32_T c3_b_status;
  int32_T c3_status;
  (void)chartInstance;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_st.site = &c3_gd_emlrtRSI;
  c3_b_st.site = &c3_hd_emlrtRSI;
  c3_status = emlrtClockGettimeMonotonic(&c3_t);
  c3_c_st.site = &c3_id_emlrtRSI;
  c3_b_status = c3_status;
  if (c3_b_status != 0) {
    c3_y = NULL;
    sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    c3_b_y = NULL;
    sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 33),
                  false);
    c3_c_y = NULL;
    sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 26),
                  false);
    c3_d_y = NULL;
    sf_mex_assign(&c3_d_y, sf_mex_create("y", &c3_b_status, 6, 0U, 0, 0U, 0),
                  false);
    sf_mex_call(&c3_c_st, &c3_ab_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                sf_mex_call(&c3_c_st, NULL, "getString", 1U, 1U, 14, sf_mex_call
      (&c3_c_st, NULL, "message", 1U, 3U, 14, c3_b_y, 14, c3_c_y, 14, c3_d_y)));
  }

  return c3_t;
}

static real_T c3_toc(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
                     emlrtStack *c3_sp, emlrtTimespec c3_tstart)
{
  emlrtStack c3_st;
  emlrtTimespec c3_tnow;
  real_T c3_tdiff;
  real_T c3_tdiff_nsec;
  real_T c3_tdiff_sec;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_st.site = &c3_jd_emlrtRSI;
  c3_tnow = c3_getTime(chartInstance, &c3_st);
  c3_tdiff_sec = c3_tnow.tv_sec - c3_tstart.tv_sec;
  c3_tdiff_nsec = c3_tnow.tv_nsec - c3_tstart.tv_nsec;
  c3_tdiff = c3_tdiff_sec + c3_tdiff_nsec / 1.0E+9;
  return c3_tdiff;
}

static void c3_pause(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
                     emlrtStack *c3_sp, real_T c3_varargin_1)
{
  static char_T c3_b_cv[5] = { 'p', 'a', 'u', 's', 'e' };

  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_y = NULL;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_y = NULL;
  sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 5),
                false);
  c3_b_y = NULL;
  sf_mex_assign(&c3_b_y, sf_mex_create("y", &c3_varargin_1, 0, 0U, 0, 0U, 0),
                false);
  c3_st.site = &c3_ld_emlrtRSI;
  c3_b_feval(chartInstance, &c3_st, c3_y, c3_b_y);
}

static void c3_OutputStream_drain(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, c3_matlabshared_asyncio_internal_OutputStream *c3_obj)
{
  static char_T c3_b_cv2[35] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'O',
    'u', 't', 'p', 'u', 't', 'S', 't', 'r', 'e', 'a', 'm', ':', 't', 'i', 'm',
    'e', 'o', 'u', 't', 'I', 'n', 'D', 'r', 'a', 'i', 'n' };

  static char_T c3_b[7] = { 't', 'i', 'm', 'e', 'o', 'u', 't' };

  static char_T c3_b_cv1[7] = { 'i', 'n', 'v', 'a', 'l', 'i', 'd' };

  static char_T c3_b_cv3[7] = { 't', 'i', 'm', 'e', 'o', 'u', 't' };

  static char_T c3_b_cv[4] = { 'd', 'o', 'n', 'e' };

  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderChannel c3_e_chImpl;
  CoderOutputStream c3_b_streamImpl;
  CoderOutputStream c3_c_streamImpl;
  CoderOutputStream c3_d_streamImpl;
  CoderOutputStream c3_e_streamImpl;
  CoderOutputStream c3_f_streamImpl;
  CoderOutputStream c3_g_streamImpl;
  CoderOutputStream c3_h_streamImpl;
  CoderOutputStream c3_i_streamImpl;
  CoderOutputStream c3_j_streamImpl;
  CoderOutputStream c3_streamImpl;
  c3_coder_array_char_T_2D c3_x;
  c3_matlabshared_asyncio_internal_OutputStream *c3_b_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_c_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_d_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_e_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_f_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_g_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_h_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_i_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_st;
  emlrtTimespec c3_startTic;
  const mxArray *c3_ab_y = NULL;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_bb_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_d_y = NULL;
  const mxArray *c3_db_y = NULL;
  const mxArray *c3_e_y = NULL;
  const mxArray *c3_eb_y = NULL;
  const mxArray *c3_f_y = NULL;
  const mxArray *c3_g_y = NULL;
  const mxArray *c3_h_y = NULL;
  const mxArray *c3_i_y = NULL;
  const mxArray *c3_j_y = NULL;
  const mxArray *c3_k_y = NULL;
  const mxArray *c3_l_y = NULL;
  const mxArray *c3_n_y = NULL;
  const mxArray *c3_o_y = NULL;
  const mxArray *c3_p_y = NULL;
  const mxArray *c3_q_y = NULL;
  const mxArray *c3_r_y = NULL;
  const mxArray *c3_s_y = NULL;
  const mxArray *c3_t_y = NULL;
  const mxArray *c3_w_y = NULL;
  const mxArray *c3_x_y = NULL;
  const mxArray *c3_y = NULL;
  const mxArray *c3_y_y = NULL;
  real_T c3_b_count;
  real_T c3_count;
  real_T c3_et;
  real_T c3_timeoutInSeconds;
  int32_T c3_status_size[2];
  int32_T c3_b_kstr;
  int32_T c3_b_loop_ub;
  int32_T c3_b_remainingDimsA;
  int32_T c3_b_success;
  int32_T c3_c_kstr;
  int32_T c3_c_success;
  int32_T c3_d_kstr;
  int32_T c3_d_success;
  int32_T c3_e_success;
  int32_T c3_exitg1;
  int32_T c3_f_success;
  int32_T c3_g_success;
  int32_T c3_h_success;
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i2;
  int32_T c3_i3;
  int32_T c3_i4;
  int32_T c3_i_success;
  int32_T c3_j_success;
  int32_T c3_kstr;
  int32_T c3_loop_ub;
  int32_T c3_remainingDimsA;
  int32_T c3_success;
  char_T c3_status_data[9];
  char_T c3_b_s;
  char_T c3_b_x;
  char_T c3_c_s;
  char_T c3_c_x;
  char_T c3_cb_y;
  char_T c3_d_s;
  char_T c3_d_x;
  char_T c3_e_s;
  char_T c3_e_x;
  char_T c3_f_s;
  char_T c3_f_x;
  char_T c3_g_s;
  char_T c3_g_x;
  char_T c3_h_s;
  char_T c3_h_x;
  char_T c3_i_s;
  char_T c3_i_x;
  char_T c3_j_s;
  char_T c3_k_s;
  char_T c3_l_s;
  char_T c3_m_s;
  char_T c3_m_y;
  char_T c3_n_s;
  char_T c3_o_s;
  char_T c3_p_s;
  char_T c3_s;
  char_T c3_u_y;
  char_T c3_v_y;
  int8_T c3_b_u;
  int8_T c3_c_u;
  int8_T c3_d_u;
  int8_T c3_e_u;
  int8_T c3_f_u;
  int8_T c3_g_u;
  int8_T c3_h_u;
  int8_T c3_u;
  boolean_T c3_b1;
  boolean_T c3_b2;
  boolean_T c3_b3;
  boolean_T c3_b4;
  boolean_T c3_b5;
  boolean_T c3_b6;
  boolean_T c3_b7;
  boolean_T c3_b_b;
  boolean_T c3_b_bool;
  boolean_T c3_b_done;
  boolean_T c3_b_p;
  boolean_T c3_b_result;
  boolean_T c3_bool;
  boolean_T c3_c_p;
  boolean_T c3_completed;
  boolean_T c3_d_p;
  boolean_T c3_done;
  boolean_T c3_e_p;
  boolean_T c3_f_p;
  boolean_T c3_g_p;
  boolean_T c3_h_p;
  boolean_T c3_p;
  boolean_T c3_result;
  boolean_T c3_timeout;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_st.site = &c3_ud_emlrtRSI;
  c3_b_obj = c3_obj;
  c3_timeoutInSeconds = c3_b_obj->Timeout;
  c3_status_size[1] = 0;
  c3_timeout = false;
  c3_done = false;
  c3_b_st.site = &c3_wc_emlrtRSI;
  c3_c_obj = c3_b_obj;
  c3_c_st.site = &c3_ud_emlrtRSI;
  c3_d_obj = c3_c_obj;
  c3_d_st.site = &c3_yd_emlrtRSI;
  c3_streamImpl = c3_d_obj->StreamImpl;
  c3_success = coderStreamGetDataAvailable(c3_streamImpl, &c3_count);
  c3_e_st.site = &c3_ae_emlrtRSI;
  c3_b_streamImpl = c3_streamImpl;
  c3_b_success = c3_success;
  if (c3_b_success == 0) {
    c3_chImpl = coderStreamGetChannel(c3_b_streamImpl);
    c3_f_st.site = &c3_vc_emlrtRSI;
    c3_b_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_chImpl);
  }

  c3_completed = (c3_count == 0.0);
  c3_b_st.site = &c3_xc_emlrtRSI;
  c3_startTic = c3_tic(chartInstance, &c3_b_st);
  while ((!c3_completed) && (!c3_done) && (!c3_timeout)) {
    c3_b_st.site = &c3_yc_emlrtRSI;
    c3_et = c3_toc(chartInstance, &c3_b_st, c3_startTic);
    if (c3_et < 1.0) {
      c3_b_st.site = &c3_ad_emlrtRSI;
      c3_c_st.site = &c3_kd_emlrtRSI;
      c3_pause(chartInstance, &c3_c_st, 0.0);
    } else {
      c3_b_st.site = &c3_bd_emlrtRSI;
      c3_c_st.site = &c3_kd_emlrtRSI;
      c3_pause(chartInstance, &c3_c_st, 0.005);
    }

    c3_b_st.site = &c3_cd_emlrtRSI;
    c3_et = c3_toc(chartInstance, &c3_b_st, c3_startTic);
    c3_timeout = (c3_et > c3_timeoutInSeconds);
    c3_b_st.site = &c3_dd_emlrtRSI;
    c3_e_obj = c3_b_obj;
    c3_c_st.site = &c3_md_emlrtRSI;
    c3_c_streamImpl = c3_e_obj->StreamImpl;
    c3_c_success = coderStreamIsDeviceDone(c3_c_streamImpl, &c3_result);
    c3_d_st.site = &c3_nd_emlrtRSI;
    c3_d_streamImpl = c3_c_streamImpl;
    c3_d_success = c3_c_success;
    if (c3_d_success == 0) {
      c3_b_chImpl = coderStreamGetChannel(c3_d_streamImpl);
      c3_e_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_b_chImpl);
    }

    if (c3_result) {
      c3_b_done = true;
    } else {
      c3_b_st.site = &c3_dd_emlrtRSI;
      c3_f_obj = c3_b_obj;
      c3_c_st.site = &c3_od_emlrtRSI;
      c3_e_streamImpl = c3_f_obj->StreamImpl;
      c3_e_success = coderStreamIsOpen(c3_e_streamImpl, &c3_b_result);
      c3_d_st.site = &c3_pd_emlrtRSI;
      c3_f_streamImpl = c3_e_streamImpl;
      c3_f_success = c3_e_success;
      if (c3_f_success == 0) {
        c3_c_chImpl = coderStreamGetChannel(c3_f_streamImpl);
        c3_e_st.site = &c3_vc_emlrtRSI;
        c3_b_API_channelErrorIfFailed(chartInstance, &c3_e_st, c3_c_chImpl);
      }

      if (!c3_b_result) {
        c3_b_done = true;
      } else {
        c3_b_done = false;
      }
    }

    c3_done = c3_b_done;
    c3_b_st.site = &c3_ed_emlrtRSI;
    c3_g_obj = c3_b_obj;
    c3_c_st.site = &c3_ud_emlrtRSI;
    c3_h_obj = c3_g_obj;
    c3_d_st.site = &c3_yd_emlrtRSI;
    c3_g_streamImpl = c3_h_obj->StreamImpl;
    c3_g_success = coderStreamGetDataAvailable(c3_g_streamImpl, &c3_b_count);
    c3_e_st.site = &c3_ae_emlrtRSI;
    c3_h_streamImpl = c3_g_streamImpl;
    c3_h_success = c3_g_success;
    if (c3_h_success == 0) {
      c3_d_chImpl = coderStreamGetChannel(c3_h_streamImpl);
      c3_f_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_f_st, c3_d_chImpl);
    }

    c3_completed = (c3_b_count == 0.0);
  }

  if (c3_completed) {
    c3_status_size[1] = 9;
    for (c3_i = 0; c3_i < 9; c3_i++) {
      c3_status_data[c3_i] = c3_cv6[c3_i];
    }
  } else if (c3_done) {
    c3_status_size[1] = 4;
    for (c3_i1 = 0; c3_i1 < 4; c3_i1++) {
      c3_status_data[c3_i1] = c3_b_cv[c3_i1];
    }
  } else if (c3_timeout) {
    c3_status_size[1] = 7;
    for (c3_i2 = 0; c3_i2 < 7; c3_i2++) {
      c3_status_data[c3_i2] = c3_b[c3_i2];
    }
  }

  c3_st.site = &c3_vd_emlrtRSI;
  c3_b_st.site = &c3_hc_emlrtRSI;
  c3_c_st.site = &c3_ic_emlrtRSI;
  c3_bool = false;
  c3_array_char_T_2D_Constructor(chartInstance, &c3_x);
  c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_x, &c3_ib_emlrtRTEI, 1,
    c3_status_size[1]);
  c3_loop_ub = c3_status_size[1] - 1;
  for (c3_i3 = 0; c3_i3 <= c3_loop_ub; c3_i3++) {
    c3_x.vector.data[c3_i3] = c3_status_data[c3_i3];
  }

  c3_remainingDimsA = c3_x.size[1];
  if (c3_remainingDimsA != 7) {
  } else {
    c3_kstr = 1;
    do {
      c3_exitg1 = 0;
      if (c3_kstr - 1 < 7) {
        c3_b_kstr = c3_kstr - 1;
        c3_d_st.site = &c3_jc_emlrtRSI;
        c3_s = c3_status_data[c3_b_kstr];
        c3_b_s = c3_s;
        c3_b_b = ((uint8_T)c3_b_s <= 127);
        c3_p = c3_b_b;
        if (!c3_p) {
          c3_y = NULL;
          sf_mex_assign(&c3_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_b_y = NULL;
          sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_u = MAX_int8_T;
          c3_d_y = NULL;
          sf_mex_assign(&c3_d_y, sf_mex_create("y", &c3_u, 2, 0U, 0, 0U, 0),
                        false);
          sf_mex_call(&c3_d_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_y, 14,
                      sf_mex_call(&c3_d_st, NULL, "getString", 1U, 1U, 14,
            sf_mex_call(&c3_d_st, NULL, "message", 1U, 2U, 14, c3_b_y, 14,
                        c3_d_y)));
        }

        c3_d_st.site = &c3_kc_emlrtRSI;
        c3_c_s = c3_b_cv1[c3_b_kstr];
        c3_d_s = c3_c_s;
        c3_b1 = ((uint8_T)c3_d_s <= 127);
        c3_b_p = c3_b1;
        if (!c3_b_p) {
          c3_f_y = NULL;
          sf_mex_assign(&c3_f_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_g_y = NULL;
          sf_mex_assign(&c3_g_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_b_u = MAX_int8_T;
          c3_h_y = NULL;
          sf_mex_assign(&c3_h_y, sf_mex_create("y", &c3_b_u, 2, 0U, 0, 0U, 0),
                        false);
          sf_mex_call(&c3_d_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_f_y, 14,
                      sf_mex_call(&c3_d_st, NULL, "getString", 1U, 1U, 14,
            sf_mex_call(&c3_d_st, NULL, "message", 1U, 2U, 14, c3_g_y, 14,
                        c3_h_y)));
        }

        c3_d_st.site = &c3_lc_emlrtRSI;
        c3_b_x = c3_status_data[c3_b_kstr];
        c3_e_st.site = &c3_mc_emlrtRSI;
        c3_c_x = c3_b_x;
        c3_f_st.site = &c3_nc_emlrtRSI;
        c3_g_s = c3_c_x;
        c3_h_s = c3_g_s;
        c3_b3 = ((uint8_T)c3_h_s <= 127);
        c3_d_p = c3_b3;
        if (!c3_d_p) {
          c3_l_y = NULL;
          sf_mex_assign(&c3_l_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_o_y = NULL;
          sf_mex_assign(&c3_o_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_d_u = MAX_int8_T;
          c3_q_y = NULL;
          sf_mex_assign(&c3_q_y, sf_mex_create("y", &c3_d_u, 2, 0U, 0, 0U, 0),
                        false);
          sf_mex_call(&c3_f_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_l_y, 14,
                      sf_mex_call(&c3_f_st, NULL, "getString", 1U, 1U, 14,
            sf_mex_call(&c3_f_st, NULL, "message", 1U, 2U, 14, c3_o_y, 14,
                        c3_q_y)));
        }

        c3_m_y = c3_cv[(uint8_T)c3_c_x & 127];
        c3_d_st.site = &c3_lc_emlrtRSI;
        c3_d_x = c3_b_cv1[c3_b_kstr];
        c3_e_st.site = &c3_mc_emlrtRSI;
        c3_f_x = c3_d_x;
        c3_f_st.site = &c3_nc_emlrtRSI;
        c3_k_s = c3_f_x;
        c3_m_s = c3_k_s;
        c3_b5 = ((uint8_T)c3_m_s <= 127);
        c3_f_p = c3_b5;
        if (!c3_f_p) {
          c3_s_y = NULL;
          sf_mex_assign(&c3_s_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_w_y = NULL;
          sf_mex_assign(&c3_w_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2, 1,
            31), false);
          c3_f_u = MAX_int8_T;
          c3_y_y = NULL;
          sf_mex_assign(&c3_y_y, sf_mex_create("y", &c3_f_u, 2, 0U, 0, 0U, 0),
                        false);
          sf_mex_call(&c3_f_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_s_y, 14,
                      sf_mex_call(&c3_f_st, NULL, "getString", 1U, 1U, 14,
            sf_mex_call(&c3_f_st, NULL, "message", 1U, 2U, 14, c3_w_y, 14,
                        c3_y_y)));
        }

        c3_u_y = c3_cv[(uint8_T)c3_f_x & 127];
        if (c3_m_y != c3_u_y) {
          c3_exitg1 = 1;
        } else {
          c3_kstr++;
        }
      } else {
        c3_bool = true;
        c3_exitg1 = 1;
      }
    } while (c3_exitg1 == 0);
  }

  if (!c3_bool) {
    c3_st.site = &c3_wd_emlrtRSI;
    c3_b_st.site = &c3_hc_emlrtRSI;
    c3_c_st.site = &c3_ic_emlrtRSI;
    c3_b_bool = false;
    c3_array_char_T_2D_SetSize(chartInstance, &c3_c_st, &c3_x, &c3_ib_emlrtRTEI,
      1, c3_status_size[1]);
    c3_b_loop_ub = c3_status_size[1] - 1;
    for (c3_i4 = 0; c3_i4 <= c3_b_loop_ub; c3_i4++) {
      c3_x.vector.data[c3_i4] = c3_status_data[c3_i4];
    }

    c3_b_remainingDimsA = c3_x.size[1];
    if (c3_b_remainingDimsA != 7) {
    } else {
      c3_c_kstr = 1;
      do {
        c3_exitg1 = 0;
        if (c3_c_kstr - 1 < 7) {
          c3_d_kstr = c3_c_kstr - 1;
          c3_d_st.site = &c3_jc_emlrtRSI;
          c3_e_s = c3_status_data[c3_d_kstr];
          c3_f_s = c3_e_s;
          c3_b2 = ((uint8_T)c3_f_s <= 127);
          c3_c_p = c3_b2;
          if (!c3_c_p) {
            c3_i_y = NULL;
            sf_mex_assign(&c3_i_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_j_y = NULL;
            sf_mex_assign(&c3_j_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_c_u = MAX_int8_T;
            c3_k_y = NULL;
            sf_mex_assign(&c3_k_y, sf_mex_create("y", &c3_c_u, 2, 0U, 0, 0U, 0),
                          false);
            sf_mex_call(&c3_d_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_i_y,
                        14, sf_mex_call(&c3_d_st, NULL, "getString", 1U, 1U, 14,
              sf_mex_call(&c3_d_st, NULL, "message", 1U, 2U, 14, c3_j_y, 14,
                          c3_k_y)));
          }

          c3_d_st.site = &c3_kc_emlrtRSI;
          c3_i_s = c3_b_cv3[c3_d_kstr];
          c3_j_s = c3_i_s;
          c3_b4 = ((uint8_T)c3_j_s <= 127);
          c3_e_p = c3_b4;
          if (!c3_e_p) {
            c3_n_y = NULL;
            sf_mex_assign(&c3_n_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_p_y = NULL;
            sf_mex_assign(&c3_p_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_e_u = MAX_int8_T;
            c3_r_y = NULL;
            sf_mex_assign(&c3_r_y, sf_mex_create("y", &c3_e_u, 2, 0U, 0, 0U, 0),
                          false);
            sf_mex_call(&c3_d_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_n_y,
                        14, sf_mex_call(&c3_d_st, NULL, "getString", 1U, 1U, 14,
              sf_mex_call(&c3_d_st, NULL, "message", 1U, 2U, 14, c3_p_y, 14,
                          c3_r_y)));
          }

          c3_d_st.site = &c3_lc_emlrtRSI;
          c3_e_x = c3_status_data[c3_d_kstr];
          c3_e_st.site = &c3_mc_emlrtRSI;
          c3_g_x = c3_e_x;
          c3_f_st.site = &c3_nc_emlrtRSI;
          c3_l_s = c3_g_x;
          c3_n_s = c3_l_s;
          c3_b6 = ((uint8_T)c3_n_s <= 127);
          c3_g_p = c3_b6;
          if (!c3_g_p) {
            c3_t_y = NULL;
            sf_mex_assign(&c3_t_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_x_y = NULL;
            sf_mex_assign(&c3_x_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_g_u = MAX_int8_T;
            c3_ab_y = NULL;
            sf_mex_assign(&c3_ab_y, sf_mex_create("y", &c3_g_u, 2, 0U, 0, 0U, 0),
                          false);
            sf_mex_call(&c3_f_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_t_y,
                        14, sf_mex_call(&c3_f_st, NULL, "getString", 1U, 1U, 14,
              sf_mex_call(&c3_f_st, NULL, "message", 1U, 2U, 14, c3_x_y, 14,
                          c3_ab_y)));
          }

          c3_v_y = c3_cv[(uint8_T)c3_g_x & 127];
          c3_d_st.site = &c3_lc_emlrtRSI;
          c3_h_x = c3_b_cv3[c3_d_kstr];
          c3_e_st.site = &c3_mc_emlrtRSI;
          c3_i_x = c3_h_x;
          c3_f_st.site = &c3_nc_emlrtRSI;
          c3_o_s = c3_i_x;
          c3_p_s = c3_o_s;
          c3_b7 = ((uint8_T)c3_p_s <= 127);
          c3_h_p = c3_b7;
          if (!c3_h_p) {
            c3_bb_y = NULL;
            sf_mex_assign(&c3_bb_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_db_y = NULL;
            sf_mex_assign(&c3_db_y, sf_mex_create("y", c3_cv1, 10, 0U, 1, 0U, 2,
              1, 31), false);
            c3_h_u = MAX_int8_T;
            c3_eb_y = NULL;
            sf_mex_assign(&c3_eb_y, sf_mex_create("y", &c3_h_u, 2, 0U, 0, 0U, 0),
                          false);
            sf_mex_call(&c3_f_st, &c3_x_emlrtMCI, "error", 0U, 2U, 14, c3_bb_y,
                        14, sf_mex_call(&c3_f_st, NULL, "getString", 1U, 1U, 14,
              sf_mex_call(&c3_f_st, NULL, "message", 1U, 2U, 14, c3_db_y, 14,
                          c3_eb_y)));
          }

          c3_cb_y = c3_cv[(uint8_T)c3_i_x & 127];
          if (c3_v_y != c3_cb_y) {
            c3_exitg1 = 1;
          } else {
            c3_c_kstr++;
          }
        } else {
          c3_b_bool = true;
          c3_exitg1 = 1;
        }
      } while (c3_exitg1 == 0);
    }

    if (c3_b_bool) {
      c3_c_y = NULL;
      sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1,
        35), false);
      c3_e_y = NULL;
      sf_mex_assign(&c3_e_y, sf_mex_create("y", c3_b_cv2, 10, 0U, 1, 0U, 2, 1,
        35), false);
      sf_mex_call(c3_sp, &c3_bb_emlrtMCI, "error", 0U, 2U, 14, c3_c_y, 14,
                  sf_mex_call(c3_sp, NULL, "getString", 1U, 1U, 14, sf_mex_call
        (c3_sp, NULL, "message", 1U, 1U, 14, c3_e_y)));
    }

    c3_st.site = &c3_xd_emlrtRSI;
    c3_i_obj = c3_obj;
    c3_b_st.site = &c3_be_emlrtRSI;
    c3_i_streamImpl = c3_i_obj->StreamImpl;
    c3_i_success = coderStreamFlush(c3_i_streamImpl);
    c3_c_st.site = &c3_ce_emlrtRSI;
    c3_j_streamImpl = c3_i_streamImpl;
    c3_j_success = c3_i_success;
    if (c3_j_success == 0) {
      c3_e_chImpl = coderStreamGetChannel(c3_j_streamImpl);
      c3_d_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_d_st, c3_e_chImpl);
    }
  }

  c3_array_char_T_2D_Destructor(chartInstance, &c3_x);
}

static void c3_warning(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp)
{
  static char_T c3_msgID[37] = { 'a', 's', 'y', 'n', 'c', 'i', 'o', ':', 'C',
    'h', 'a', 'n', 'n', 'e', 'l', ':', 's', 't', 'i', 'l', 'l', 'O', 'p', 'e',
    'n', 'D', 'u', 'r', 'i', 'n', 'g', 'D', 'e', 'l', 'e', 't', 'e' };

  static char_T c3_b_cv[7] = { 'w', 'a', 'r', 'n', 'i', 'n', 'g' };

  static char_T c3_b_cv1[7] = { 'm', 'e', 's', 's', 'a', 'g', 'e' };

  emlrtStack c3_st;
  const mxArray *c3_b_y = NULL;
  const mxArray *c3_c_y = NULL;
  const mxArray *c3_y = NULL;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_y = NULL;
  sf_mex_assign(&c3_y, sf_mex_create("y", c3_b_cv, 10, 0U, 1, 0U, 2, 1, 7),
                false);
  c3_b_y = NULL;
  sf_mex_assign(&c3_b_y, sf_mex_create("y", c3_b_cv1, 10, 0U, 1, 0U, 2, 1, 7),
                false);
  c3_c_y = NULL;
  sf_mex_assign(&c3_c_y, sf_mex_create("y", c3_msgID, 10, 0U, 1, 0U, 2, 1, 37),
                false);
  c3_st.site = &c3_oe_emlrtRSI;
  c3_d_feval(chartInstance, &c3_st, c3_y, c3_c_feval(chartInstance, &c3_st,
              c3_b_y, c3_c_y));
}

static void c3_Channel_close(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_obj)
{
  CoderChannel c3_b_chImpl;
  CoderChannel c3_c_chImpl;
  CoderChannel c3_chImpl;
  CoderChannel c3_d_chImpl;
  CoderInputStream c3_b_streamImpl;
  CoderInputStream c3_streamImpl;
  CoderOutputStream c3_c_streamImpl;
  CoderOutputStream c3_d_streamImpl;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_b_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_c_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_d_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_e_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_f_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_i_obj;
  c3_matlabshared_asyncio_buffer_internal_BufferChannel *c3_j_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_g_obj;
  c3_matlabshared_asyncio_internal_InputStream *c3_h_obj;
  c3_matlabshared_asyncio_internal_OutputStream *c3_k_obj;
  emlrtStack c3_b_st;
  emlrtStack c3_c_st;
  emlrtStack c3_d_st;
  emlrtStack c3_e_st;
  emlrtStack c3_f_st;
  emlrtStack c3_g_st;
  emlrtStack c3_h_st;
  emlrtStack c3_st;
  int32_T c3_b_success;
  int32_T c3_c_success;
  int32_T c3_d_success;
  int32_T c3_e_success;
  int32_T c3_f_success;
  int32_T c3_success;
  boolean_T c3_result;
  c3_st.prev = c3_sp;
  c3_st.tls = c3_sp->tls;
  c3_st.site = &c3_fe_emlrtRSI;
  c3_b_st.prev = &c3_st;
  c3_b_st.tls = c3_st.tls;
  c3_c_st.prev = &c3_b_st;
  c3_c_st.tls = c3_b_st.tls;
  c3_d_st.prev = &c3_c_st;
  c3_d_st.tls = c3_c_st.tls;
  c3_e_st.prev = &c3_d_st;
  c3_e_st.tls = c3_d_st.tls;
  c3_f_st.prev = &c3_e_st;
  c3_f_st.tls = c3_e_st.tls;
  c3_g_st.prev = &c3_f_st;
  c3_g_st.tls = c3_f_st.tls;
  c3_h_st.prev = &c3_g_st;
  c3_h_st.tls = c3_g_st.tls;
  c3_b_obj = c3_obj;
  c3_b_st.site = &c3_yb_emlrtRSI;
  c3_chImpl = c3_b_obj->ChannelImpl;
  c3_success = coderChannelIsOpen(c3_chImpl, &c3_result);
  c3_c_st.site = &c3_ac_emlrtRSI;
  c3_API_channelErrorIfFailed(chartInstance, &c3_c_st, c3_chImpl, c3_success);
  if (c3_result) {
    c3_st.site = &c3_ge_emlrtRSI;
    c3_b_chImpl = c3_obj->ChannelImpl;
    c3_b_success = coderChannelClose(c3_b_chImpl);
    c3_b_st.site = &c3_he_emlrtRSI;
    c3_API_channelErrorIfFailed(chartInstance, &c3_b_st, c3_b_chImpl,
      c3_b_success);
    c3_st.site = &c3_qe_emlrtRSI;
    c3_c_obj = c3_obj;
    c3_b_st.site = &c3_re_emlrtRSI;
    c3_d_obj = c3_c_obj;
    c3_e_obj = c3_d_obj;
    c3_e_obj->TotalElementsWritten = 0.0;
    c3_c_st.site = &c3_se_emlrtRSI;
    c3_f_obj = c3_d_obj;
    c3_d_st.site = &c3_te_emlrtRSI;
    c3_g_obj = &c3_f_obj->InputStream;
    c3_e_st.site = &c3_we_emlrtRSI;
    c3_InputStream_clearPartialPacket(chartInstance, &c3_e_st, c3_g_obj);
    c3_e_st.site = &c3_xe_emlrtRSI;
    c3_h_obj = c3_g_obj;
    c3_f_st.site = &c3_be_emlrtRSI;
    c3_streamImpl = c3_h_obj->StreamImpl;
    c3_c_success = coderStreamFlush(c3_streamImpl);
    c3_g_st.site = &c3_ce_emlrtRSI;
    c3_b_streamImpl = c3_streamImpl;
    c3_d_success = c3_c_success;
    if (c3_d_success == 0) {
      c3_c_chImpl = coderStreamGetChannel(c3_b_streamImpl);
      c3_h_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_h_st, c3_c_chImpl);
    }

    c3_d_st.site = &c3_ue_emlrtRSI;
    c3_i_obj = c3_f_obj;
    c3_e_st.site = &c3_vb_emlrtRSI;
    c3_j_obj = c3_i_obj;
    c3_array_uint8_T_2D_SetSize(chartInstance, &c3_e_st,
      &c3_j_obj->PartialPacket, &c3_u_emlrtRTEI, 1, 0);
    c3_array_uint8_T_2D_SetSize(chartInstance, &c3_e_st,
      &c3_j_obj->PartialPacket, &c3_v_emlrtRTEI, 0, 0);
    c3_i_obj->PartialPacketStart = 0.0;
    c3_i_obj->PartialPacketCount = 0.0;
    c3_d_st.site = &c3_ve_emlrtRSI;
    c3_k_obj = &c3_f_obj->OutputStream;
    c3_e_st.site = &c3_be_emlrtRSI;
    c3_c_streamImpl = c3_k_obj->StreamImpl;
    c3_e_success = coderStreamFlush(c3_c_streamImpl);
    c3_f_st.site = &c3_ce_emlrtRSI;
    c3_d_streamImpl = c3_c_streamImpl;
    c3_f_success = c3_e_success;
    if (c3_f_success == 0) {
      c3_d_chImpl = coderStreamGetChannel(c3_d_streamImpl);
      c3_g_st.site = &c3_vc_emlrtRSI;
      c3_b_API_channelErrorIfFailed(chartInstance, &c3_g_st, c3_d_chImpl);
    }
  }
}

static real_T c3_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_a__output_of_length_, const char_T *c3_identifier)
{
  emlrtMsgIdentifier c3_thisId;
  real_T c3_y;
  c3_thisId.fIdentifier = (const char_T *)c3_identifier;
  c3_thisId.fParent = NULL;
  c3_thisId.bParentIsCell = false;
  c3_y = c3_b_emlrt_marshallIn(chartInstance, sf_mex_dup(c3_a__output_of_length_),
    &c3_thisId);
  sf_mex_destroy(&c3_a__output_of_length_);
  return c3_y;
}

static real_T c3_b_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId)
{
  real_T c3_d;
  real_T c3_y;
  (void)chartInstance;
  sf_mex_import(c3_parentId, sf_mex_dup(c3_u), &c3_d, 1, 0, 0U, 0, 0U, 0);
  c3_y = c3_d;
  sf_mex_destroy(&c3_u);
  return c3_y;
}

static void c3_c_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_tmpStr, const char_T *c3_identifier,
  c3_coder_array_char_T_2D *c3_y)
{
  emlrtMsgIdentifier c3_thisId;
  c3_thisId.fIdentifier = (const char_T *)c3_identifier;
  c3_thisId.fParent = NULL;
  c3_thisId.bParentIsCell = false;
  c3_d_emlrt_marshallIn(chartInstance, c3_sp, sf_mex_dup(c3_tmpStr), &c3_thisId,
                        c3_y);
  sf_mex_destroy(&c3_tmpStr);
}

static void c3_d_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_u, const emlrtMsgIdentifier
  *c3_parentId, c3_coder_array_char_T_2D *c3_y)
{
  static boolean_T c3_bv1[2] = { false, true };

  c3_coder_array_char_T_2D c3_r;
  int32_T c3_iv[2];
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i2;
  int32_T c3_loop_ub;
  boolean_T c3_bv[2];
  c3_array_char_T_2D_Constructor(chartInstance, &c3_r);
  for (c3_i = 0; c3_i < 2; c3_i++) {
    c3_iv[c3_i] = 1 + -2 * c3_i;
  }

  c3_array_char_T_2D_SetSize(chartInstance, c3_sp, &c3_r, (emlrtRTEInfo *)NULL,
    sf_mex_get_dimension(c3_u, 0), sf_mex_get_dimension(c3_u, 1));
  for (c3_i1 = 0; c3_i1 < 2; c3_i1++) {
    c3_bv[c3_i1] = c3_bv1[c3_i1];
  }

  sf_mex_import_vs(c3_parentId, sf_mex_dup(c3_u), c3_r.vector.data, 0, 10, 0U, 1,
                   0U, 2, c3_bv, c3_iv, c3_r.size);
  c3_array_char_T_2D_SetSize(chartInstance, c3_sp, c3_y, (emlrtRTEInfo *)NULL, 1,
    c3_r.size[1]);
  c3_loop_ub = c3_r.size[1] - 1;
  for (c3_i2 = 0; c3_i2 <= c3_loop_ub; c3_i2++) {
    c3_y->vector.data[c3_i2] = c3_r.vector.data[c3_i2];
  }

  sf_mex_destroy(&c3_u);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_r);
}

static void c3_e_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_a__output_of_matlabroot_, const char_T *c3_identifier,
  char_T c3_y_data[], int32_T c3_y_size[2])
{
  emlrtMsgIdentifier c3_thisId;
  c3_thisId.fIdentifier = (const char_T *)c3_identifier;
  c3_thisId.fParent = NULL;
  c3_thisId.bParentIsCell = false;
  c3_f_emlrt_marshallIn(chartInstance, sf_mex_dup(c3_a__output_of_matlabroot_),
                        &c3_thisId, c3_y_data, c3_y_size);
  sf_mex_destroy(&c3_a__output_of_matlabroot_);
}

static void c3_f_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId, char_T c3_y_data[],
  int32_T c3_y_size[2])
{
  static boolean_T c3_bv1[2] = { false, true };

  int32_T c3_iv[2];
  int32_T c3_tmp_size[2];
  int32_T c3_i;
  int32_T c3_i1;
  int32_T c3_i2;
  int32_T c3_loop_ub;
  char_T c3_tmp_data[512];
  boolean_T c3_bv[2];
  (void)chartInstance;
  for (c3_i = 0; c3_i < 2; c3_i++) {
    c3_iv[c3_i] = 1 + 511 * c3_i;
  }

  c3_tmp_size[0] = sf_mex_get_dimension(c3_u, 0);
  c3_tmp_size[1] = sf_mex_get_dimension(c3_u, 1);
  for (c3_i1 = 0; c3_i1 < 2; c3_i1++) {
    c3_bv[c3_i1] = c3_bv1[c3_i1];
  }

  sf_mex_import_vs(c3_parentId, sf_mex_dup(c3_u), &c3_tmp_data, 0, 10, 0U, 1, 0U,
                   2, c3_bv, c3_iv, c3_tmp_size);
  c3_y_size[0] = 1;
  c3_y_size[1] = c3_tmp_size[1];
  c3_loop_ub = c3_tmp_size[1] - 1;
  for (c3_i2 = 0; c3_i2 <= c3_loop_ub; c3_i2++) {
    c3_y_data[c3_i2] = c3_tmp_data[c3_i2];
  }

  sf_mex_destroy(&c3_u);
}

static boolean_T c3_g_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray
  *c3_a__output_of_coder_internal_ifWhileCondExtrinsic_, const char_T
  *c3_identifier)
{
  emlrtMsgIdentifier c3_thisId;
  boolean_T c3_y;
  c3_thisId.fIdentifier = (const char_T *)c3_identifier;
  c3_thisId.fParent = NULL;
  c3_thisId.bParentIsCell = false;
  c3_y = c3_h_emlrt_marshallIn(chartInstance, sf_mex_dup
    (c3_a__output_of_coder_internal_ifWhileCondExtrinsic_), &c3_thisId);
  sf_mex_destroy(&c3_a__output_of_coder_internal_ifWhileCondExtrinsic_);
  return c3_y;
}

static boolean_T c3_h_emlrt_marshallIn(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const mxArray *c3_u, const emlrtMsgIdentifier *c3_parentId)
{
  boolean_T c3_b;
  boolean_T c3_y;
  (void)chartInstance;
  sf_mex_import(c3_parentId, sf_mex_dup(c3_u), &c3_b, 1, 11, 0U, 0, 0U, 0);
  c3_y = c3_b;
  sf_mex_destroy(&c3_u);
  return c3_y;
}

const mxArray *sf_c3_LKS_tcp11_get_eml_resolved_functions_info(void)
{
  const mxArray *c3_nameCaptureInfo = NULL;
  c3_nameCaptureInfo = NULL;
  sf_mex_assign(&c3_nameCaptureInfo, sf_mex_create("nameCaptureInfo", NULL, 0,
    0U, 1, 0U, 2, 0, 1), false);
  return c3_nameCaptureInfo;
}

static const mxArray *c3_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1,
  const mxArray *c3_input2)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "feval", 1U, 3U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1), 14, sf_mex_dup(c3_input2)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  sf_mex_destroy(&c3_input2);
  return c3_m;
}

static const mxArray *c3_length(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "length", 1U, 1U, 14, sf_mex_dup
    (c3_input0)), false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_ver(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "ver", 1U, 1U, 14, sf_mex_dup
    (c3_input0)), false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_getfield(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "getfield", 1U, 2U, 14,
    sf_mex_dup(c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_matlabroot(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "matlabroot", 1U, 0U), false);
  return c3_m;
}

static const mxArray *c3_b_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "strcmp", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL,
    "coder.internal.ifWhileCondExtrinsic", 1U, 1U, 14, sf_mex_dup(c3_input0)),
                false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "exist", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_b_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL,
    "coder.internal.ifWhileCondExtrinsic", 1U, 1U, 14, sf_mex_dup(c3_input0)),
                false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_b_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "exist", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_c_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL,
    "coder.internal.ifWhileCondExtrinsic", 1U, 1U, 14, sf_mex_dup(c3_input0)),
                false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_matlabRelease(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "matlabRelease", 1U, 0U), false);
  return c3_m;
}

static const mxArray *c3_b_getfield(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "getfield", 1U, 2U, 14,
    sf_mex_dup(c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_b_matlabroot(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "matlabroot", 1U, 0U), false);
  return c3_m;
}

static const mxArray *c3_c_strcmp(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "strcmp", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_d_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL,
    "coder.internal.ifWhileCondExtrinsic", 1U, 1U, 14, sf_mex_dup(c3_input0)),
                false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static const mxArray *c3_c_exist(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "exist", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static const mxArray *c3_e_coder_internal_ifWhileCondExtrinsic
  (SFc3_LKS_tcp11InstanceStruct *chartInstance, const emlrtStack *c3_sp, const
   mxArray *c3_input0)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL,
    "coder.internal.ifWhileCondExtrinsic", 1U, 1U, 14, sf_mex_dup(c3_input0)),
                false);
  sf_mex_destroy(&c3_input0);
  return c3_m;
}

static void c3_b_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  (void)chartInstance;
  sf_mex_call(c3_sp, NULL, "feval", 0U, 2U, 14, sf_mex_dup(c3_input0), 14,
              sf_mex_dup(c3_input1));
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
}

static const mxArray *c3_c_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance,
  const emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  const mxArray *c3_m = NULL;
  (void)chartInstance;
  c3_m = NULL;
  sf_mex_assign(&c3_m, sf_mex_call(c3_sp, NULL, "feval", 1U, 2U, 14, sf_mex_dup
    (c3_input0), 14, sf_mex_dup(c3_input1)), false);
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
  return c3_m;
}

static void c3_d_feval(SFc3_LKS_tcp11InstanceStruct *chartInstance, const
  emlrtStack *c3_sp, const mxArray *c3_input0, const mxArray *c3_input1)
{
  (void)chartInstance;
  sf_mex_call(c3_sp, NULL, "feval", 0U, 2U, 14, sf_mex_dup(c3_input0), 14,
              sf_mex_dup(c3_input1));
  sf_mex_destroy(&c3_input0);
  sf_mex_destroy(&c3_input1);
}

static void c3_array_char_T_2D_SetSize(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_coder_array_char_T_2D
  *c3_coderArray, const emlrtRTEInfo *c3_srcLocation, int32_T c3_size0, int32_T
  c3_size1)
{
  int32_T c3_newCapacity;
  int32_T c3_newNumel;
  char_T *c3_newData;
  (void)chartInstance;
  c3_coderArray->size[0] = c3_size0;
  c3_coderArray->size[1] = c3_size1;
  c3_newNumel = (int32_T)emlrtSizeMulR2012b((size_t)(uint32_T)(int32_T)
    emlrtSizeMulR2012b((size_t)1U, (size_t)(uint32_T)c3_coderArray->size[0],
                       c3_srcLocation, (void *)c3_sp), (size_t)(uint32_T)
    c3_coderArray->size[1], c3_srcLocation, (void *)c3_sp);
  if (c3_newNumel > c3_coderArray->vector.allocated) {
    c3_newCapacity = c3_coderArray->vector.allocated;
    if (c3_newCapacity < 16) {
      c3_newCapacity = 16;
    }

    while (c3_newCapacity < c3_newNumel) {
      if (c3_newCapacity > 1073741823) {
        c3_newCapacity = MAX_int32_T;
      } else {
        c3_newCapacity <<= 1;
      }
    }

    c3_newData = (char_T *)emlrtMallocMex(sizeof(char_T) * (uint32_T)
      c3_newCapacity);
    if ((void *)c3_newData == NULL) {
      emlrtHeapAllocationErrorR2012b(c3_srcLocation, (void *)c3_sp);
    }

    if ((void *)c3_newData == NULL) {
      emlrtHeapAllocationErrorR2012b(c3_srcLocation, (void *)c3_sp);
    }

    if (c3_coderArray->vector.data != NULL) {
      memcpy(c3_newData, c3_coderArray->vector.data, sizeof(char_T) * (uint32_T)
             c3_coderArray->vector.numel);
      if (c3_coderArray->vector.owner) {
        emlrtFreeMex(c3_coderArray->vector.data);
      }
    }

    c3_coderArray->vector.data = c3_newData;
    c3_coderArray->vector.allocated = c3_newCapacity;
    c3_coderArray->vector.owner = true;
  }

  c3_coderArray->vector.numel = c3_newNumel;
}

static void c3_array_uint8_T_2D_SetSize(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, const emlrtStack *c3_sp, c3_coder_array_uint8_T_2D
  *c3_coderArray, const emlrtRTEInfo *c3_srcLocation, int32_T c3_size0, int32_T
  c3_size1)
{
  int32_T c3_newCapacity;
  int32_T c3_newNumel;
  uint8_T *c3_newData;
  (void)chartInstance;
  c3_coderArray->size[0] = c3_size0;
  c3_coderArray->size[1] = c3_size1;
  c3_newNumel = (int32_T)emlrtSizeMulR2012b((size_t)(uint32_T)(int32_T)
    emlrtSizeMulR2012b((size_t)1U, (size_t)(uint32_T)c3_coderArray->size[0],
                       c3_srcLocation, (void *)c3_sp), (size_t)(uint32_T)
    c3_coderArray->size[1], c3_srcLocation, (void *)c3_sp);
  if (c3_newNumel > c3_coderArray->vector.allocated) {
    c3_newCapacity = c3_coderArray->vector.allocated;
    if (c3_newCapacity < 16) {
      c3_newCapacity = 16;
    }

    while (c3_newCapacity < c3_newNumel) {
      if (c3_newCapacity > 1073741823) {
        c3_newCapacity = MAX_int32_T;
      } else {
        c3_newCapacity <<= 1;
      }
    }

    c3_newData = (uint8_T *)emlrtMallocMex(sizeof(uint8_T) * (uint32_T)
      c3_newCapacity);
    if ((void *)c3_newData == NULL) {
      emlrtHeapAllocationErrorR2012b(c3_srcLocation, (void *)c3_sp);
    }

    if ((void *)c3_newData == NULL) {
      emlrtHeapAllocationErrorR2012b(c3_srcLocation, (void *)c3_sp);
    }

    if (c3_coderArray->vector.data != NULL) {
      memcpy(c3_newData, c3_coderArray->vector.data, sizeof(uint8_T) * (uint32_T)
             c3_coderArray->vector.numel);
      if (c3_coderArray->vector.owner) {
        emlrtFreeMex(c3_coderArray->vector.data);
      }
    }

    c3_coderArray->vector.data = c3_newData;
    c3_coderArray->vector.allocated = c3_newCapacity;
    c3_coderArray->vector.owner = true;
  }

  c3_coderArray->vector.numel = c3_newNumel;
}

static void c3_array_tcpclient_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_tcpclient *c3_pStruct)
{
  c3_array_matlabshared_network_inte(chartInstance, &c3_pStruct->TCPClientObj);
}

static void c3_array_matlabshared_network_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_network_internal_TCPClient *c3_pStruct)
{
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->ByteOrder);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->NativeDataType);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->DataFieldName);
  c3_array_matlabshared_transportlib(chartInstance, &c3_pStruct->_pobj0);
  c3_b_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->_pobj1);
}

static void c3_array_char_T_2D_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_char_T_2D *c3_coderArray)
{
  (void)chartInstance;
  c3_coderArray->vector.data = (char_T *)NULL;
  c3_coderArray->vector.numel = 0;
  c3_coderArray->vector.allocated = 0;
  c3_coderArray->vector.owner = true;
  c3_coderArray->size[0] = 0;
  c3_coderArray->size[1] = 0;
}

static void c3_array_matlabshared_transportlib(SFc3_LKS_tcp11InstanceStruct
  *chartInstance,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_pStruct)
{
  c3_array_matlabshared_asyncio_buff(chartInstance,
    &c3_pStruct->UnreadDataBuffer);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->ByteOrder);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->NativeDataType);
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->DataFieldName);
}

static void c3_array_matlabshared_asyncio_buff(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_pStruct)
{
  c3_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->InputStream);
  c3_array_uint8_T_2D_Constructor(chartInstance, &c3_pStruct->PartialPacket);
}

static void c3_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_InputStream *c3_pStruct)
{
  c3_array_uint8_T_2D_Constructor(chartInstance, &c3_pStruct->PartialPacket);
  c3_array_uint8_T_2D_Constructor(chartInstance, &c3_pStruct->ExampleData);
}

static void c3_array_uint8_T_2D_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_uint8_T_2D *c3_coderArray)
{
  (void)chartInstance;
  c3_coderArray->vector.data = (uint8_T *)NULL;
  c3_coderArray->vector.numel = 0;
  c3_coderArray->vector.allocated = 0;
  c3_coderArray->vector.owner = true;
  c3_coderArray->size[0] = 0;
  c3_coderArray->size[1] = 0;
}

static void c3_b_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_Channel *c3_pStruct)
{
  c3_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->InputStream);
}

static void c3_array_char_T_2D_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_char_T_2D *c3_coderArray)
{
  (void)chartInstance;
  if (c3_coderArray->vector.owner && (c3_coderArray->vector.data != (char_T *)
       NULL)) {
    emlrtFreeMex(c3_coderArray->vector.data);
  }
}

static void c3_b_array_matlabshared_network_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_network_internal_TCPClient *c3_pStruct)
{
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->ByteOrder);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->NativeDataType);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->DataFieldName);
  c3_b_array_matlabshared_transportlib(chartInstance, &c3_pStruct->_pobj0);
  c3_d_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->_pobj1);
}

static void c3_array_uint8_T_2D_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_coder_array_uint8_T_2D *c3_coderArray)
{
  (void)chartInstance;
  if (c3_coderArray->vector.owner && (c3_coderArray->vector.data != (uint8_T *)
       NULL)) {
    emlrtFreeMex(c3_coderArray->vector.data);
  }
}

static void c3_c_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_InputStream *c3_pStruct)
{
  c3_array_uint8_T_2D_Destructor(chartInstance, &c3_pStruct->PartialPacket);
  c3_array_uint8_T_2D_Destructor(chartInstance, &c3_pStruct->ExampleData);
}

static void c3_b_array_matlabshared_asyncio_buff(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_buffer_internal_BufferChannel
  *c3_pStruct)
{
  c3_c_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->InputStream);
  c3_array_uint8_T_2D_Destructor(chartInstance, &c3_pStruct->PartialPacket);
}

static void c3_b_array_matlabshared_transportlib(SFc3_LKS_tcp11InstanceStruct
  *chartInstance,
  c3_matlabshared_transportlib_internal_asyncIOTransportChannel_co *c3_pStruct)
{
  c3_b_array_matlabshared_asyncio_buff(chartInstance,
    &c3_pStruct->UnreadDataBuffer);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->ByteOrder);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->NativeDataType);
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->DataFieldName);
}

static void c3_d_array_matlabshared_asyncio_inte(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_matlabshared_asyncio_internal_Channel *c3_pStruct)
{
  c3_c_array_matlabshared_asyncio_inte(chartInstance, &c3_pStruct->InputStream);
}

static void c3_array_tcpclient_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_tcpclient *c3_pStruct)
{
  c3_b_array_matlabshared_network_inte(chartInstance, &c3_pStruct->TCPClientObj);
}

static void c3_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_Qyu6eoJFT0AYGGE5WaAhtD *c3_pStruct)
{
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->ServiceName);
}

static void c3_array_cell_17_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_17 *c3_pStruct)
{
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->f8);
}

static void c3_b_array_s_Qyu6eoJFT0AYGGE5WaAhtD_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_Qyu6eoJFT0AYGGE5WaAhtD *c3_pStruct)
{
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->ServiceName);
}

static void c3_array_cell_17_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_17 *c3_pStruct)
{
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->f8);
}

static void c3_array_s_HTCilNNUmm0Yd43AIdnmID_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_HTCilNNUmm0Yd43AIdnmID *c3_pStruct)
{
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->ServiceName);
}

static void c3_array_cell_7_Constructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_7 *c3_pStruct)
{
  c3_array_char_T_2D_Constructor(chartInstance, &c3_pStruct->f8);
}

static void c3_b_array_s_HTCilNNUmm0Yd43AIdnmID_(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_s_HTCilNNUmm0Yd43AIdnmID *c3_pStruct)
{
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->ServiceName);
}

static void c3_array_cell_7_Destructor(SFc3_LKS_tcp11InstanceStruct
  *chartInstance, c3_cell_7 *c3_pStruct)
{
  c3_array_char_T_2D_Destructor(chartInstance, &c3_pStruct->f8);
}

static void init_dsm_address_info(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  (void)chartInstance;
}

static void init_simulink_io_address(SFc3_LKS_tcp11InstanceStruct *chartInstance)
{
  chartInstance->c3_covrtInstance = (CovrtStateflowInstance *)
    sfrtGetCovrtInstance(chartInstance->S);
  chartInstance->c3_fEmlrtCtx = (void *)sfrtGetEmlrtCtx(chartInstance->S);
  chartInstance->c3_systemState = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 0);
  chartInstance->c3_offState = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 1);
  chartInstance->c3_vState = (boolean_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 2);
  chartInstance->c3_LDState = (boolean_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 3);
  chartInstance->c3_brakeState = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 4);
  chartInstance->c3_LCState = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 5);
  chartInstance->c3_ey = (real_T *)ssGetInputPortSignal_wrapper(chartInstance->S,
    6);
  chartInstance->c3_theta = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 7);
  chartInstance->c3_raw_theta = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 8);
  chartInstance->c3_max_theta = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 9);
  chartInstance->c3_output_edelta_t = (real_T *)ssGetInputPortSignal_wrapper
    (chartInstance->S, 10);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SFunction Glue Code */
void sf_c3_LKS_tcp11_get_check_sum(mxArray *plhs[])
{
  ((real_T *)mxGetPr((plhs[0])))[0] = (real_T)(4047674644U);
  ((real_T *)mxGetPr((plhs[0])))[1] = (real_T)(2905884686U);
  ((real_T *)mxGetPr((plhs[0])))[2] = (real_T)(1452224625U);
  ((real_T *)mxGetPr((plhs[0])))[3] = (real_T)(1277644295U);
}

mxArray *sf_c3_LKS_tcp11_third_party_uses_info(void)
{
  mxArray * mxcell3p = mxCreateCellMatrix(1,4);
  mxSetCell(mxcell3p, 0, mxCreateString(
             "matlabshared.network.internal.coder.TCPClient"));
  mxSetCell(mxcell3p, 1, mxCreateString(
             "matlabshared.asyncio.buffer.internal.coder.BufferChannel"));
  mxSetCell(mxcell3p, 2, mxCreateString(
             "matlabshared.asyncio.internal.coder.API"));
  mxSetCell(mxcell3p, 3, mxCreateString("coder.internal.time.CoderTimeAPI"));
  return(mxcell3p);
}

mxArray *sf_c3_LKS_tcp11_jit_fallback_info(void)
{
  const char *infoFields[] = { "fallbackType", "fallbackReason",
    "hiddenFallbackType", "hiddenFallbackReason", "incompatibleSymbol" };

  mxArray *mxInfo = mxCreateStructMatrix(1, 1, 5, infoFields);
  mxArray *fallbackType = mxCreateString("late");
  mxArray *fallbackReason = mxCreateString("ir_vars");
  mxArray *hiddenFallbackType = mxCreateString("");
  mxArray *hiddenFallbackReason = mxCreateString("");
  mxArray *incompatibleSymbol = mxCreateString("tcpClient");
  mxSetField(mxInfo, 0, infoFields[0], fallbackType);
  mxSetField(mxInfo, 0, infoFields[1], fallbackReason);
  mxSetField(mxInfo, 0, infoFields[2], hiddenFallbackType);
  mxSetField(mxInfo, 0, infoFields[3], hiddenFallbackReason);
  mxSetField(mxInfo, 0, infoFields[4], incompatibleSymbol);
  return mxInfo;
}

mxArray *sf_c3_LKS_tcp11_updateBuildInfo_args_info(void)
{
  mxArray *mxBIArgs = mxCreateCellMatrix(1,0);
  return mxBIArgs;
}

static const mxArray *sf_get_sim_state_info_c3_LKS_tcp11(void)
{
  const char *infoFields[] = { "chartChecksum", "varInfo" };

  mxArray *mxInfo = mxCreateStructMatrix(1, 1, 2, infoFields);
  mxArray *mxVarInfo = sf_mex_decode(
    "eNpjYPT0ZQACPiCOAGI2IOYAYiYGCGCF8hmhfEa4OAtcXAGISyoLUkHixUXJnilAOi8xF8xPLK3"
    "wzEvLZwAA+RkJgQ=="
    );
  mxArray *mxChecksum = mxCreateDoubleMatrix(1, 4, mxREAL);
  sf_c3_LKS_tcp11_get_check_sum(&mxChecksum);
  mxSetField(mxInfo, 0, infoFields[0], mxChecksum);
  mxSetField(mxInfo, 0, infoFields[1], mxVarInfo);
  return mxInfo;
}

static const char* sf_get_instance_specialization(void)
{
  return "sZTsivma5imC3KtlYYVCReG";
}

static void sf_opaque_initialize_c3_LKS_tcp11(void *chartInstanceVar)
{
  initialize_params_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*)
    chartInstanceVar);
  initialize_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
}

static void sf_opaque_enable_c3_LKS_tcp11(void *chartInstanceVar)
{
  enable_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
}

static void sf_opaque_disable_c3_LKS_tcp11(void *chartInstanceVar)
{
  disable_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
}

static void sf_opaque_gateway_c3_LKS_tcp11(void *chartInstanceVar)
{
  sf_gateway_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
}

static const mxArray* sf_opaque_get_sim_state_c3_LKS_tcp11(SimStruct* S)
{
  return get_sim_state_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct *)
    sf_get_chart_instance_ptr(S));     /* raw sim ctx */
}

static void sf_opaque_set_sim_state_c3_LKS_tcp11(SimStruct* S, const mxArray *st)
{
  set_sim_state_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*)
    sf_get_chart_instance_ptr(S), st);
}

static void sf_opaque_cleanup_runtime_resources_c3_LKS_tcp11(void
  *chartInstanceVar)
{
  if (chartInstanceVar!=NULL) {
    SimStruct *S = ((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar)->S;
    if (sim_mode_is_rtw_gen(S) || sim_mode_is_external(S)) {
      sf_clear_rtw_identifier(S);
      unload_LKS_tcp11_optimization_info();
    }

    mdl_cleanup_runtime_resources_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*)
      chartInstanceVar);
    utFree(chartInstanceVar);
    if (ssGetUserData(S)!= NULL) {
      sf_free_ChartRunTimeInfo(S);
    }

    ssSetUserData(S,NULL);
  }
}

static void sf_opaque_mdl_start_c3_LKS_tcp11(void *chartInstanceVar)
{
  mdl_start_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
  if (chartInstanceVar) {
    sf_reset_warnings_ChartRunTimeInfo(((SFc3_LKS_tcp11InstanceStruct*)
      chartInstanceVar)->S);
  }
}

static void sf_opaque_mdl_terminate_c3_LKS_tcp11(void *chartInstanceVar)
{
  mdl_terminate_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*) chartInstanceVar);
}

extern unsigned int sf_machine_global_initializer_called(void);
static void mdlProcessParameters_c3_LKS_tcp11(SimStruct *S)
{
  mdlProcessParamsCommon(S);
  if (sf_machine_global_initializer_called()) {
    initialize_params_c3_LKS_tcp11((SFc3_LKS_tcp11InstanceStruct*)
      sf_get_chart_instance_ptr(S));
  }
}

const char* sf_c3_LKS_tcp11_get_post_codegen_info(void)
{
  int i;
  const char* encStrCodegen [20] = {
    "eNrdWN1uG0UUHkdpVdpSFVUqvUCid3CD1CpC6gVCpGu7WE2I1XUC7U01nj32jjIzu50fJ4ZX6EU",
    "F4l14FK54AV4AiRvOrNeOtTHOzlooKSttJjO735xzvj1/Y9Lq7RO87uD97j4h13G8gfcWmV3Xyn",
    "lr6Z6tb5Ovynn/JiEspdr21Cgj4RfLEhiDit1oxE8DscrJPtVUmgZyFZXwAkwmnOWZClOeqxFoU",
    "Aw3yDNtg+QaLp3g6rjrFPOSzfcpZ2mcZk4kT3FDmhwoMf03ubmzfZTY5hqY7QIkNtWZG6ddQcfr",
    "WdD2JEqBHRsng7kyYGOXe1PNvhOW5wI6p8B6yliKLJgL7I0ttRDZ02AP4SaeozOZC05Vfa5TamL",
    "I0TssHOYJ/j1wFtmrKRflDbmiNtOcio4Ukffwmti+QD330a1FMM9W2jYM3XjM1dizq50Ehfajn9",
    "TgahRlE9B0DAcqMAa9dZ3T4gMv/LJ+DPa8WzaKQSdnX8U0whZyOxNkyDSU22UqokKYMOwgy/dgA",
    "qKQ36aWNsDO5AeAjeHJIDui2kdBYCQ5xd84KLFRphJe/wtPKqgi4X6HybMGnEvvUpAgzQvVFxtd",
    "5JPO2ExGGEbtvb2a8s5je8qCHlEGtXOfptwAKlz4VaDchBs6FB6NLNnCyto7gGoMJWbkVPsk08f",
    "IcWiSPePKR0IYGpIxtMFCkTA66N1HVLiaOkuDGc67x6HBjBUmF7E+fhqBGWUpJL6ecAH7YPwGpn",
    "Z+xjqyi9ZOuJ22wTDN87qR5AwkWEg8S4NpDofqWGUnqqszGZcdwRq/AsCsQbXCsvAUy5KedlH5e",
    "lpreDPwntWkPZPUCjr0vvEMFFYWb6uvhJRhVHUUtm6o0CbYmP+IpV0Zbiy2U9NOEQNJ0Y8+aIX1",
    "ox+V851FLPXMQOOXoliGFXZKu8Oi2YMBl1AsxBR7itm0vLzcR+RM7gdb6+Vu4X+thjiyIe6bJdz",
    "2Cn5uL+HulnO283rvefzasvzx41Xyb9fguQ6OnMOd8TsfP1/Ct1bIJUtj1d4qT7fO8VSunV0r9/",
    "mwIne7ss/1kruff3v450/3/v4j/YV89ivN327iJzcC/fpOOf9k3qctKtnkXLL37357gV98XPELP",
    "zevBoZPJP2Sy2jnuRUvXx5FL+BZsd8PZL2+pKLvfP0h3hYTXZE3Nesl5ZnLz6mbnQUuI27el/i+",
    "TD3r5IFb/2H+uAq4Te0LzW/v+/uP1uQdUnn/7hW2ozo2rXNXza7fSVjd+bScf704e0YpF8mK7rd",
    "8jA3qaNXT/4l//xXI37zOdjx/5Y9nr3Z2FRVTbHdnx4lyua/97zaLRxqoWX2muIw6NB+fXNBXXK",
    "vEt58fDrpfPNmgnv0DuLj1hw==",
    ""
  };

  static char newstr [1377] = "";
  newstr[0] = '\0';
  for (i = 0; i < 20; i++) {
    strcat(newstr, encStrCodegen[i]);
  }

  return newstr;
}

static void mdlSetWorkWidths_c3_LKS_tcp11(SimStruct *S)
{
  const char* newstr = sf_c3_LKS_tcp11_get_post_codegen_info();
  sf_set_work_widths(S, newstr);
  ssSetChecksum0(S,(3944790162U));
  ssSetChecksum1(S,(3740996475U));
  ssSetChecksum2(S,(654349160U));
  ssSetChecksum3(S,(2356175253U));
}

static void mdlRTW_c3_LKS_tcp11(SimStruct *S)
{
  if (sim_mode_is_rtw_gen(S)) {
    ssWriteRTWStrParam(S, "StateflowChartType", "Embedded MATLAB");
  }
}

static void mdlSetupRuntimeResources_c3_LKS_tcp11(SimStruct *S)
{
  SFc3_LKS_tcp11InstanceStruct *chartInstance;
  chartInstance = (SFc3_LKS_tcp11InstanceStruct *)utMalloc(sizeof
    (SFc3_LKS_tcp11InstanceStruct));
  if (chartInstance==NULL) {
    sf_mex_error_message("Could not allocate memory for chart instance.");
  }

  memset(chartInstance, 0, sizeof(SFc3_LKS_tcp11InstanceStruct));
  chartInstance->chartInfo.chartInstance = chartInstance;
  chartInstance->chartInfo.isEMLChart = 1;
  chartInstance->chartInfo.chartInitialized = 0;
  chartInstance->chartInfo.sFunctionGateway = sf_opaque_gateway_c3_LKS_tcp11;
  chartInstance->chartInfo.initializeChart = sf_opaque_initialize_c3_LKS_tcp11;
  chartInstance->chartInfo.mdlStart = sf_opaque_mdl_start_c3_LKS_tcp11;
  chartInstance->chartInfo.mdlTerminate = sf_opaque_mdl_terminate_c3_LKS_tcp11;
  chartInstance->chartInfo.mdlCleanupRuntimeResources =
    sf_opaque_cleanup_runtime_resources_c3_LKS_tcp11;
  chartInstance->chartInfo.enableChart = sf_opaque_enable_c3_LKS_tcp11;
  chartInstance->chartInfo.disableChart = sf_opaque_disable_c3_LKS_tcp11;
  chartInstance->chartInfo.getSimState = sf_opaque_get_sim_state_c3_LKS_tcp11;
  chartInstance->chartInfo.setSimState = sf_opaque_set_sim_state_c3_LKS_tcp11;
  chartInstance->chartInfo.getSimStateInfo = sf_get_sim_state_info_c3_LKS_tcp11;
  chartInstance->chartInfo.zeroCrossings = NULL;
  chartInstance->chartInfo.outputs = NULL;
  chartInstance->chartInfo.derivatives = NULL;
  chartInstance->chartInfo.mdlRTW = mdlRTW_c3_LKS_tcp11;
  chartInstance->chartInfo.mdlSetWorkWidths = mdlSetWorkWidths_c3_LKS_tcp11;
  chartInstance->chartInfo.extModeExec = NULL;
  chartInstance->chartInfo.restoreLastMajorStepConfiguration = NULL;
  chartInstance->chartInfo.restoreBeforeLastMajorStepConfiguration = NULL;
  chartInstance->chartInfo.storeCurrentConfiguration = NULL;
  chartInstance->chartInfo.callAtomicSubchartUserFcn = NULL;
  chartInstance->chartInfo.callAtomicSubchartAutoFcn = NULL;
  chartInstance->chartInfo.callAtomicSubchartEventFcn = NULL;
  chartInstance->S = S;
  chartInstance->chartInfo.dispatchToExportedFcn = NULL;
  sf_init_ChartRunTimeInfo(S, &(chartInstance->chartInfo), false, 0);
  init_dsm_address_info(chartInstance);
  init_simulink_io_address(chartInstance);
  if (!sim_mode_is_rtw_gen(S)) {
  }

  mdl_setup_runtime_resources_c3_LKS_tcp11(chartInstance);
}

void c3_LKS_tcp11_method_dispatcher(SimStruct *S, int_T method, void *data)
{
  switch (method) {
   case SS_CALL_MDL_SETUP_RUNTIME_RESOURCES:
    mdlSetupRuntimeResources_c3_LKS_tcp11(S);
    break;

   case SS_CALL_MDL_SET_WORK_WIDTHS:
    mdlSetWorkWidths_c3_LKS_tcp11(S);
    break;

   case SS_CALL_MDL_PROCESS_PARAMETERS:
    mdlProcessParameters_c3_LKS_tcp11(S);
    break;

   default:
    /* Unhandled method */
    sf_mex_error_message("Stateflow Internal Error:\n"
                         "Error calling c3_LKS_tcp11_method_dispatcher.\n"
                         "Can't handle method %d.\n", method);
    break;
  }
}
