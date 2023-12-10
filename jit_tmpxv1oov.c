/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) jit_tmpxv1oov_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

static const casadi_int casadi_s0[4] = {0, 1, 0, 0};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

/* dae:(t[0],x[2],z[0],p[2],u[0])->(ode[2],alg[0],quad[0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, w4, *w5=w+5, w6, w7, w8, w9, w10;
  /* #0: @0 = 41.168 */
  w0 = 4.1167962016176233e+01;
  /* #1: @1 = 12.814 */
  w1 = 1.2813960066864546e+01;
  /* #2: @2 = 9.92623 */
  w2 = 9.9262321922399366e+00;
  /* #3: @3 = 90 */
  w3 = 90.;
  /* #4: @4 = 0.1538 */
  w4 = 1.5379999999999999e-01;
  /* #5: @5 = input[3][0] */
  casadi_copy(arg[3], 2, w5);
  /* #6: {@6, @7} = vertsplit(@5) */
  w6 = w5[0];
  w7 = w5[1];
  /* #7: @4 = (@4*@6) */
  w4 *= w6;
  /* #8: @3 = (@3+@4) */
  w3 += w4;
  /* #9: @4 = 48.43 */
  w4 = 4.8430000000000000e+01;
  /* #10: @6 = 0.5616 */
  w6 = 5.6159999999999999e-01;
  /* #11: @5 = input[1][0] */
  casadi_copy(arg[1], 2, w5);
  /* #12: {@8, @9} = vertsplit(@5) */
  w8 = w5[0];
  w9 = w5[1];
  /* #13: @6 = (@6*@9) */
  w6 *= w9;
  /* #14: @10 = 0.3126 */
  w10 = 3.1259999999999999e-01;
  /* #15: @10 = (@10*@8) */
  w10 *= w8;
  /* #16: @6 = (@6+@10) */
  w6 += w10;
  /* #17: @4 = (@4+@6) */
  w4 += w6;
  /* #18: @3 = (@3-@4) */
  w3 -= w4;
  /* #19: @2 = (@2*@3) */
  w2 *= w3;
  /* #20: @3 = 0.780376 */
  w3 = 7.8037603478544926e-01;
  /* #21: @6 = 28.1566 */
  w6 = 2.8156619881093022e+01;
  /* #22: @4 = (@4-@6) */
  w4 -= w6;
  /* #23: @3 = (@3*@4) */
  w3 *= w4;
  /* #24: @2 = (@2-@3) */
  w2 -= w3;
  /* #25: @3 = 34.7532 */
  w3 = 3.4753203189580567e+01;
  /* #26: @2 = (@2/@3) */
  w2 /= w3;
  /* #27: @1 = (@1-@2) */
  w1 -= w2;
  /* #28: @1 = (@1*@8) */
  w1 *= w8;
  /* #29: @0 = (@0-@1) */
  w0 -= w1;
  /* #30: @1 = 28.7285 */
  w1 = 2.8728518738500913e+01;
  /* #31: @0 = (@0/@1) */
  w0 /= w1;
  /* #32: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #33: @0 = 9.16093 */
  w0 = 9.1609294892935136e+00;
  /* #34: @1 = 55 */
  w1 = 55.;
  /* #35: @8 = 0.507 */
  w8 = 5.0700000000000001e-01;
  /* #36: @8 = (@8*@9) */
  w8 *= w9;
  /* #37: @1 = (@1+@8) */
  w1 += w8;
  /* #38: @8 = 14.5295 */
  w8 = 1.4529466559199653e+01;
  /* #39: @1 = (@1-@8) */
  w1 -= w8;
  /* #40: @0 = (@0*@1) */
  w0 *= w1;
  /* #41: @1 = 1 */
  w1 = 1.;
  /* #42: @8 = 9.16093 */
  w8 = 9.1609294892935136e+00;
  /* #43: @9 = 0.121801 */
  w9 = 1.2180091567530534e-01;
  /* #44: @9 = (@9*@7) */
  w9 *= w7;
  /* #45: @8 = (@8/@9) */
  w8 /= w9;
  /* #46: @1 = (@1+@8) */
  w1 += w8;
  /* #47: @0 = (@0/@1) */
  w0 /= w1;
  /* #48: @1 = 34.7532 */
  w1 = 3.4753203189580567e+01;
  /* #49: @0 = (@0/@1) */
  w0 /= w1;
  /* #50: @2 = (@2-@0) */
  w2 -= w0;
  /* #51: @0 = 2.18391 */
  w0 = 2.1839094897321849e+00;
  /* #52: @2 = (@2/@0) */
  w2 /= w0;
  /* #53: output[0][1] = @2 */
  if (res[0]) res[0][1] = w2;
  return 0;
}

CASADI_SYMBOL_EXPORT int dae(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int dae_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int dae_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dae_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int dae_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dae_release(int mem) {
}

CASADI_SYMBOL_EXPORT void dae_incref(void) {
}

CASADI_SYMBOL_EXPORT void dae_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int dae_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int dae_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real dae_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dae_name_in(casadi_int i) {
  switch (i) {
    case 0: return "t";
    case 1: return "x";
    case 2: return "z";
    case 3: return "p";
    case 4: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dae_name_out(casadi_int i) {
  switch (i) {
    case 0: return "ode";
    case 1: return "alg";
    case 2: return "quad";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dae_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s1;
    case 4: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dae_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int dae_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 12;
  return 0;
}

/* daeF:(t[0],x[2],z[0],p[2],u[0])->(ode[2],alg[0]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, w4, *w5=w+5, w6, w7, w8, w9, w10;
  /* #0: @0 = 41.168 */
  w0 = 4.1167962016176233e+01;
  /* #1: @1 = 12.814 */
  w1 = 1.2813960066864546e+01;
  /* #2: @2 = 9.92623 */
  w2 = 9.9262321922399366e+00;
  /* #3: @3 = 90 */
  w3 = 90.;
  /* #4: @4 = 0.1538 */
  w4 = 1.5379999999999999e-01;
  /* #5: @5 = input[3][0] */
  casadi_copy(arg[3], 2, w5);
  /* #6: {@6, @7} = vertsplit(@5) */
  w6 = w5[0];
  w7 = w5[1];
  /* #7: @4 = (@4*@6) */
  w4 *= w6;
  /* #8: @3 = (@3+@4) */
  w3 += w4;
  /* #9: @4 = 48.43 */
  w4 = 4.8430000000000000e+01;
  /* #10: @6 = 0.5616 */
  w6 = 5.6159999999999999e-01;
  /* #11: @5 = input[1][0] */
  casadi_copy(arg[1], 2, w5);
  /* #12: {@8, @9} = vertsplit(@5) */
  w8 = w5[0];
  w9 = w5[1];
  /* #13: @6 = (@6*@9) */
  w6 *= w9;
  /* #14: @10 = 0.3126 */
  w10 = 3.1259999999999999e-01;
  /* #15: @10 = (@10*@8) */
  w10 *= w8;
  /* #16: @6 = (@6+@10) */
  w6 += w10;
  /* #17: @4 = (@4+@6) */
  w4 += w6;
  /* #18: @3 = (@3-@4) */
  w3 -= w4;
  /* #19: @2 = (@2*@3) */
  w2 *= w3;
  /* #20: @3 = 0.780376 */
  w3 = 7.8037603478544926e-01;
  /* #21: @6 = 28.1566 */
  w6 = 2.8156619881093022e+01;
  /* #22: @4 = (@4-@6) */
  w4 -= w6;
  /* #23: @3 = (@3*@4) */
  w3 *= w4;
  /* #24: @2 = (@2-@3) */
  w2 -= w3;
  /* #25: @3 = 34.7532 */
  w3 = 3.4753203189580567e+01;
  /* #26: @2 = (@2/@3) */
  w2 /= w3;
  /* #27: @1 = (@1-@2) */
  w1 -= w2;
  /* #28: @1 = (@1*@8) */
  w1 *= w8;
  /* #29: @0 = (@0-@1) */
  w0 -= w1;
  /* #30: @1 = 28.7285 */
  w1 = 2.8728518738500913e+01;
  /* #31: @0 = (@0/@1) */
  w0 /= w1;
  /* #32: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #33: @0 = 9.16093 */
  w0 = 9.1609294892935136e+00;
  /* #34: @1 = 55 */
  w1 = 55.;
  /* #35: @8 = 0.507 */
  w8 = 5.0700000000000001e-01;
  /* #36: @8 = (@8*@9) */
  w8 *= w9;
  /* #37: @1 = (@1+@8) */
  w1 += w8;
  /* #38: @8 = 14.5295 */
  w8 = 1.4529466559199653e+01;
  /* #39: @1 = (@1-@8) */
  w1 -= w8;
  /* #40: @0 = (@0*@1) */
  w0 *= w1;
  /* #41: @1 = 1 */
  w1 = 1.;
  /* #42: @8 = 9.16093 */
  w8 = 9.1609294892935136e+00;
  /* #43: @9 = 0.121801 */
  w9 = 1.2180091567530534e-01;
  /* #44: @9 = (@9*@7) */
  w9 *= w7;
  /* #45: @8 = (@8/@9) */
  w8 /= w9;
  /* #46: @1 = (@1+@8) */
  w1 += w8;
  /* #47: @0 = (@0/@1) */
  w0 /= w1;
  /* #48: @1 = 34.7532 */
  w1 = 3.4753203189580567e+01;
  /* #49: @0 = (@0/@1) */
  w0 /= w1;
  /* #50: @2 = (@2-@0) */
  w2 -= w0;
  /* #51: @0 = 2.18391 */
  w0 = 2.1839094897321849e+00;
  /* #52: @2 = (@2/@0) */
  w2 /= w0;
  /* #53: output[0][1] = @2 */
  if (res[0]) res[0][1] = w2;
  return 0;
}

CASADI_SYMBOL_EXPORT int daeF(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int daeF_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int daeF_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void daeF_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int daeF_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void daeF_release(int mem) {
}

CASADI_SYMBOL_EXPORT void daeF_incref(void) {
}

CASADI_SYMBOL_EXPORT void daeF_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int daeF_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int daeF_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real daeF_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* daeF_name_in(casadi_int i) {
  switch (i) {
    case 0: return "t";
    case 1: return "x";
    case 2: return "z";
    case 3: return "p";
    case 4: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* daeF_name_out(casadi_int i) {
  switch (i) {
    case 0: return "ode";
    case 1: return "alg";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* daeF_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s1;
    case 4: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* daeF_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int daeF_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 12;
  return 0;
}

/* step:(t[0],h,x0[2],v0[0],p[2],u[0])->(xf[2],vf[0],qf[0]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+3, *rr;
  const casadi_real **arg1=arg+6, *cr, *cs;
  casadi_real *w0=w+12, w1, w2, *w5=w+16, *w7=w+18, w8, *w9=w+21, *w10=w+23;
  /* #0: @0 = input[2][0] */
  casadi_copy(arg[2], 2, w0);
  /* #1: @1 = input[1][0] */
  w1 = arg[1] ? arg[1][0] : 0;
  /* #2: @2 = 6 */
  w2 = 6.;
  /* #3: @2 = (@1/@2) */
  w2  = (w1/w2);
  /* #4: @3 = 0x1 */
  /* #5: @4 = 0x1 */
  /* #6: @5 = input[4][0] */
  casadi_copy(arg[4], 2, w5);
  /* #7: @6 = 0x1 */
  /* #8: {@7, NULL, NULL} = dae(@3, @0, @4, @5, @6) */
  arg1[0]=0;
  arg1[1]=w0;
  arg1[2]=0;
  arg1[3]=w5;
  arg1[4]=0;
  res1[0]=w7;
  res1[1]=0;
  res1[2]=0;
  if (casadi_f0(arg1, res1, iw, w, 0)) return 1;
  /* #9: @3 = 0x1 */
  /* #10: @8 = 2 */
  w8 = 2.;
  /* #11: @8 = (@1/@8) */
  w8  = (w1/w8);
  /* #12: @9 = (@8*@7) */
  for (i=0, rr=w9, cs=w7; i<2; ++i) (*rr++)  = (w8*(*cs++));
  /* #13: @9 = (@0+@9) */
  for (i=0, rr=w9, cr=w0, cs=w9; i<2; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #14: @4 = 0x1 */
  /* #15: {@10, NULL, NULL} = dae(@3, @9, @4, @5, @6) */
  arg1[0]=0;
  arg1[1]=w9;
  arg1[2]=0;
  arg1[3]=w5;
  arg1[4]=0;
  res1[0]=w10;
  res1[1]=0;
  res1[2]=0;
  if (casadi_f0(arg1, res1, iw, w, 0)) return 1;
  /* #16: @9 = (2.*@10) */
  for (i=0, rr=w9, cs=w10; i<2; ++i) *rr++ = (2.* *cs++ );
  /* #17: @7 = (@7+@9) */
  for (i=0, rr=w7, cs=w9; i<2; ++i) (*rr++) += (*cs++);
  /* #18: @10 = (@8*@10) */
  for (i=0, rr=w10, cs=w10; i<2; ++i) (*rr++)  = (w8*(*cs++));
  /* #19: @10 = (@0+@10) */
  for (i=0, rr=w10, cr=w0, cs=w10; i<2; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #20: @4 = 0x1 */
  /* #21: {@9, NULL, NULL} = dae(@3, @10, @4, @5, @6) */
  arg1[0]=0;
  arg1[1]=w10;
  arg1[2]=0;
  arg1[3]=w5;
  arg1[4]=0;
  res1[0]=w9;
  res1[1]=0;
  res1[2]=0;
  if (casadi_f0(arg1, res1, iw, w, 0)) return 1;
  /* #22: @10 = (2.*@9) */
  for (i=0, rr=w10, cs=w9; i<2; ++i) *rr++ = (2.* *cs++ );
  /* #23: @7 = (@7+@10) */
  for (i=0, rr=w7, cs=w10; i<2; ++i) (*rr++) += (*cs++);
  /* #24: @3 = 0x1 */
  /* #25: @9 = (@1*@9) */
  for (i=0, rr=w9, cs=w9; i<2; ++i) (*rr++)  = (w1*(*cs++));
  /* #26: @9 = (@0+@9) */
  for (i=0, rr=w9, cr=w0, cs=w9; i<2; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #27: @4 = 0x1 */
  /* #28: {@10, NULL, NULL} = dae(@3, @9, @4, @5, @6) */
  arg1[0]=0;
  arg1[1]=w9;
  arg1[2]=0;
  arg1[3]=w5;
  arg1[4]=0;
  res1[0]=w10;
  res1[1]=0;
  res1[2]=0;
  if (casadi_f0(arg1, res1, iw, w, 0)) return 1;
  /* #29: @7 = (@7+@10) */
  for (i=0, rr=w7, cs=w10; i<2; ++i) (*rr++) += (*cs++);
  /* #30: @7 = (@2*@7) */
  for (i=0, rr=w7, cs=w7; i<2; ++i) (*rr++)  = (w2*(*cs++));
  /* #31: @0 = (@0+@7) */
  for (i=0, rr=w0, cs=w7; i<2; ++i) (*rr++) += (*cs++);
  /* #32: output[0][0] = @0 */
  casadi_copy(w0, 2, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int step(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int step_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int step_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void step_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int step_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void step_release(int mem) {
}

CASADI_SYMBOL_EXPORT void step_incref(void) {
}

CASADI_SYMBOL_EXPORT void step_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int step_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int step_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real step_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* step_name_in(casadi_int i) {
  switch (i) {
    case 0: return "t";
    case 1: return "h";
    case 2: return "x0";
    case 3: return "v0";
    case 4: return "p";
    case 5: return "u";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* step_name_out(casadi_int i) {
  switch (i) {
    case 0: return "xf";
    case 1: return "vf";
    case 2: return "qf";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* step_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s2;
    case 2: return casadi_s1;
    case 3: return casadi_s0;
    case 4: return casadi_s1;
    case 5: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* step_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int step_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 13;
  if (sz_res) *sz_res = 8;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 25;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
