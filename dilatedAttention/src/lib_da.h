int da_forward_cuda(const at::Tensor& t, const at::Tensor& *f, at::Tensor& relation, int dilation);
int da_backward_cuda(const at::Tensor& dr, const at::Tensor& t, const at::Tensor& f, at::Tensor& dt, at::Tensor& df, int dilation);

int da_map_forward_cuda(const at::Tensor& relation, const at::Tensor& g, at::Tensor& out, int dilation);
int da_map_backward_cuda(const at::Tensor& dout, const at::Tensor& relation, const at::Tensor& g, at::Tensor& dr, at::Tensor& dg, int dilation);
