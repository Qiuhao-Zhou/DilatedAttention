from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="DilatedAttention",
      ext_modules=[
          CUDAExtension(
              'pyda', 
              ['src/lib_da.cpp', 'src/da.cu'],
              extra_compile_args = ["-std=c++11"]
              ), 
          ],
      cmdclass={'build_ext': BuildExtension})
      
