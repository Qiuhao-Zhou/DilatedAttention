

#ifndef __DA__
#define __DA__

/*
 * Exported functions
 */
extern "C" int _da_forward_cuda(int N, int C, int H, int W, const float *t, const float *f, float *attention, cudaStream_t stream);
extern "C" int _da_backward_cuda(int N, int C, int H, int W, const float *d_attention, const float *t, const float *f, float *dt, float *df, cudaStream_t stream);
extern "C" int _da_map_forward_cuda(int N, int C, int H, int W, const float *attention, const float *g, float *out, cudaStream_t stream);
extern "C" int _da_map_backward_cuda(int N, int C, int H, int W, const float *dout, const float *attention, const float *g, float *d_attention, float *dg, cudaStream_t stream);

#endif

