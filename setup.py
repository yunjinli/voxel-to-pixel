from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxel_to_pixel',
    packages=['voxel_to_pixel'],
    ext_modules=[
        CUDAExtension(
            name='voxel_to_pixel._C',
            sources=[
                'src/voxel_to_pixel.cpp',
                'src/voxel_to_pixel_kernel.cu',
                'src/voxel_to_pixel_backward_kernel.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='lltm',
#     ext_modules=[
#         CUDAExtension('lltm_cuda', [
#             'lltm_cuda.cpp',
#             'lltm_cuda_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })