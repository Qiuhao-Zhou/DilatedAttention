// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <THC/THC.h>
#include <torch/extension.h>
#include "da.h"

extern THCState *state;

int da_forward_cuda(const at::Tensor& t, const at::Tensor& f, at::Tensor& attention) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  int N, C, H, W;
  N = t.size(0); C = t.size(1); H = t.size(2); W = t.size(3);
  float * t_data = t.data<float>();
  float * f_data = f.data<float>();
  float * attention_data = attention.data<float>();
  return _da_forward_cuda(N, C, H, W, t_data, f_data, attention_data, stream);
}

int da_backward_cuda(const at::Tensor& d_attention, const at::Tensor& t, const at::Tensor& f, at::Tensor& dt, at::Tensor& df) {

  cudaStream_t stream = THCState_getCurrentStream(state);
  int N, C, H, W;
  N = t.size(0); C = t.size(1); H = t.size(2); W = t.size(3);
  float * t_data = t.data<float>();
  float * f_data = f.data<float>();
  float * dt_data = dt.data<float>();
  float * df_data = df.data<float>();
  float * d_attention_data = d_attention.data<float>();
  return _da_backward_cuda(N, C, H, W, d_attention_data, t_data, f_data, dt_data, df_data, stream);
}

int da_map_forward_cuda(const at::Tensor& attention, const at::Tensor& g, at::Tensor& out) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  N = g.size(0); C = g.size(1); H = g.size(2); W = g.size(3);

  const float *attention_data = attention.data<float>();
  const float *g_data = g.data<float>();
  float *out_data = out.data<float>();

  return _da_map_forward_cuda(N, C, H, W, attention_data, g_data, out_data, stream);
}

int da_map_backward_cuda(const at::Tensor& dout, const at::Tensor& attention, const at::Tensor& g,
                     at::Tensor& d_attention, at::Tensor& dg) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  N = dout.size(0); C = dout.size(1); H = dout.size(2); W = dout.size(3);

  const float *dout_data = dout.data<float>();
  const float *attention_data = attention.data<float>();
  const float *g_data = g.data<float>();
  float *d_attention_data = d_attention.data<float>();
  float *dg_data = dg.data<float>();

  return _da_map_backward_cuda(N, C, H, W, dout_data, attention_data, g_data, d_attention_data, dg_data, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("da_forward_cuda", &da_forward_cuda, "DA forward CUDA");
    m.def("da_backward_cuda", &da_backward_cuda, "DA backward CUDA");
    m.def("da_map_forward_cuda", &da_map_forward_cuda, "DA map forward CUDA");
    m.def("da_map_backward_cuda", &da_map_backward_cuda, "DA map backward CUDA");
}

