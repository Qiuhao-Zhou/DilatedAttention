

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "common.h"
#include "da.h"


__global__ void da_forward_kernel(const float *t, const float *f, float *attention, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  #int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < 9) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        float _t = t[(batch * chn + plane) * sp + y*width + x];
        int row = z / 3 - 1;
        int col = z % 3 - 1;
        row = row + y;
        col = col + x;
        if(row>=0 && col >= 0 && row < height && col < width){
           float _f = f[(batch * chn + plane) * sp + row * width + col];
           attention[(batch * chn + plane) * sp + y * width + x] += _t*_f;
           }
        #else{
        #   attention[(batch * chn + plane) * sp + y * width + x] += 0;
        #   }
        
        #if (z < width) {
        #  int i = z;
        #  float _f = f[(batch * chn + plane) * sp + y*width + i];
        #  weight[(batch * len + i) * sp + y*width + x] += _t*_f;
        #} else {
        #  int i = z - width;
        #  int j = i<y ? i : i+1;

        #  float _f = f[(batch * chn + plane) * sp + j*width + x];
        #  weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
        #}
      }
    }
  }
}

__global__ void da_backward_kernel_t(const float *d_attention, const float *t, const float *f, float *dt,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = 9;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
        #dw [batchsize * (h+w-1) * height * width] - > [batchsize * 9 * height * width
        for(int i = 0; i < len; +i){
            int col = i % 3 - 1 + x;
            int row = i / 3 - 1 + y;
            if(col >= 0 && row >= 0 && col < width && row < height){
                float _d_attention = d_attention[(batch * len + i) * sp + y * width + x];
                float _f = f[(batch * chn + plane) * sp + row * width + col];
                dt[(batch * chn + plane) * sp + y * width + x] += _dw * _f;
            }
        }
        #for (int i = 0; i < width; ++i) {
        #  float _dw = dw[(batch * len + i) * sp + y*width + x];
        #  float _f = f[(batch * chn + plane) * sp + y*width + i];
        #  dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
        #}
        #for (int i = 0; i < height; ++i)  {
        #  if (i == y) continue;
        #  int j = i<y ? i : i-1;

        #  float _dw = dw[(batch * len + width + j) * sp + y*width + x];
        #  float _f = f[(batch * chn + plane) * sp + i*width + x];
        #  dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
        #}
    }

  }
}

__global__ void da_backward_kernel_f(const float *d_attention, const float *t, const float *f, float *df, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = 9;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    
    for (int batch = 0; batch < num; ++batch) {
      
      for (int i = 0; i < len; ++i){
          int col = i % 3 - 1 + x;
          int row = i / 3 - 1 + y;
          if(col >= 0 && row >= 0 && col < width && row < height){
              float _d_attention = d_attention[(batch * len + i) * sp + y * width + x];
              float _t = t[(batch * chn + plane) * sp + row * y + x];
              df[(batch * chn + plane) * sp + row * width + col] += _dw * _t;
          }
      #for (int i = 0; i < width; ++i) {
      #  float _dw = dw[(batch * len + x) * sp + y*width + i];
      #  float _t = t[(batch * chn + plane) * sp + y*width + i];
      #  df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
      #}
      #for (int i = 0; i < height; ++i) {
      #  if (i == y) continue;
      #  int j = i>y ? y : y-1;

      #  float _dw = dw[(batch * len + width + j) * sp + i*width + x];
      #  float _t = t[(batch * chn + plane) * sp + i*width + x];
      #  df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
      #}
    }

  }
}


__global__ void da_map_forward_kernel(const float *attention, const float *g, float *out, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = 9;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < len; ++i){
          int col = i % 3 - 1 + x;
          int row = i / 3 - 1 + y;
          if(col >= 0 && row >= 0 && col < width && row < height){
              float _g = g[(batch * chn + plane) * sp + row * width + col];
              float _attention = attention[(batch * len + i) * sp + y * width + x];
              out[(batch * chn + plane) * sp + y*width + x] += _g * _attention;
          }
              
      #for (int i = 0; i < width; ++i) {
      #  float _g = g[(batch * chn + plane) * sp + y*width + i];
      #  float _w = weight[(batch * len + i) * sp + y*width + x];
      #  out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
      #}
      #for (int i = 0; i < height; ++i) {
      #  if (i == y) continue;

      #  int j = i<y ? i : i-1;

      #  float _g = g[(batch * chn + plane) * sp + i*width + x];
      #  float _w = weight[(batch * len + width + j) * sp + y*width + x];
      #  out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
      #}
    }
  }

}

__global__ void da_map_backward_kernel_w(const float *dout, const float *attention, const float *g, float *d_attention,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = 9;
  int z = blockIdx.z;

  if (x < width && y < height && z < len) {

    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        float _dout = dout[(batch * chn + plane) * sp + y*width + x];
        int col = z % 3 - 1 + x;
        int row = z / 3 - 1 + y;
        if(col >= 0 && row >= 0 && col < width && row < height){
            float _g = g[(batch * chn + plane) * sp + row*width + col];
            d_attention[(batch * len + z) * sp + y*width + x] += _dout * _g;
        }
        #if (z < width) {
        #  int i = z;
        #  float _g = g[(batch * chn + plane) * sp + y*width + i];
        #  dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
        #} else {
        #  int i = z - width;
        #  int j = i<y ? i : i+1;

        #  float _g = g[(batch * chn + plane) * sp + j*width + x];
        #  dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
        #}
      }
    }
  }
}

__global__ void da_map_backward_kernel_g(const float *dout, const float *attention, const float *g, float *dg, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = 9;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {

    for (int batch = 0; batch < num; ++batch) {
      
      for (int i = 0; i < len; ++i) {
          int col = i % 3 - 1 + x;
          int row = i / 3 - 1 + y;
          if(col >= 0 && row >= 0 && col < width && row < height){
            float _dout = dout[(batch * chn + plane) * sp + row*width + col];
            float _attention = weight[(batch * len + i) * sp + row*width + col];
            dg[(batch * chn + plane) * sp + y*width + x] += _dout * _attention;
          }
            
      }
      #  float _dout = dout[(batch * chn + plane) * sp + y*width + i];
      #  float _w = weight[(batch * len + x) * sp + y*width + i];
      #  dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
      #}

      #for (int i = 0; i < height; ++i) {
      #  if (i == y) continue;
      #  int j = i>y ? y : y-1;

      #  float _dout = dout[(batch * chn + plane) * sp + i*width + x];
      #  float _w = weight[(batch * len + width + j) * sp + i*width + x];
      #  dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
      #}
    }
  }
}

/*
 * Implementations
 */
extern "C" int _da_forward_cuda(int N, int C, int H, int W, const float *t, 
                                const float *f, float *attention, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = 9;
  dim3 blocks(d1, d2, d3);
  da_forward_kernel<<<blocks, threads, 0, stream>>>(t, f, attention, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}


extern "C" int _da_backward_cuda(int N, int C, int H, int W, const float *d_attention, const float *t, const float *f, float *dt, float *df, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  // printf("%f\n", dw[0]);
  da_backward_kernel_t<<<blocks, threads, 0, stream>>>(d_attention, t, f, dt, N, C, H, W);
  da_backward_kernel_f<<<blocks, threads, 0, stream>>>(d_attention, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}


extern "C" int _da_map_forward_cuda(int N, int C, int H, int W, const float *attention, const float *g, float *out, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  dim3 blocks((W+threads.x-1)/threads.x, (H+threads.y-1)/threads.y, C);
  da_map_forward_kernel<<<blocks, threads, 0, stream>>>(attention, g, out, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _da_map_backward_cuda(int N, int C, int H, int W, const float *dout, const float *attention, const float *g, float *d_attention, float *dg, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = H+W;
  dim3 blocks(d1, d2, d3);
  da_map_backward_kernel_w<<<blocks, threads, 0, stream>>>(dout, attention, g, d_attention, N, C, H, W);

  d3 = C;
  blocks = dim3(d1, d2, d3);
  da_map_backward_kernel_g<<<blocks, threads, 0, stream>>>(dout, attention, g, dg, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}


