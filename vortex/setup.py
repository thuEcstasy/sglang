from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='vortex',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='vortex_C',
            sources=[
                'csrc/vortex.cc',
                'csrc/decode.cu',
                'csrc/plan.cu',
                'csrc/transpose.cu',
            ],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_90,code=sm_90'
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
