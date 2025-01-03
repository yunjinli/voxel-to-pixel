import torch
from torch.utils.cpp_extension import load
from . import _C
# Load the compiled CUDA extension

# voxel_to_pixel = load(
#     name='voxel_to_pixel',
#     sources=[
#         'src/voxel_to_pixel.cpp',
#         'src/voxel_to_pixel_kernel.cu',
#     ],
#     extra_cuda_cflags=['--use_fast_math'],  # Optional flags for optimization
# )

# class VoxelToPixel(object):
#     def __init__(self, output_size):
#         """
#         Args:
#             output_size (list[int]): The output resolution [height, width].
#         """
#         super().__init__()
#         self.output_size = output_size

#     def __call__(self, features, coordinates, projection_matrix):
#         return _C.forward(features, coordinates, projection_matrix, self.output_size)

class Voxel2PixelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coordinates, projection_matrix, output_size):
        ctx.save_for_backward(features, coordinates, projection_matrix)
        ctx.output_size = output_size
        output_features, inv_depth_map, depth_buffer = _C.forward(features, coordinates, projection_matrix, output_size)
        ctx.inv_depth_map = inv_depth_map  # Save depth map for debugging if needed
        ctx.depth_buffer = depth_buffer
        return output_features, inv_depth_map

    @staticmethod
    def backward(ctx, grad_output, _grad_depth_map=None):
        features, coordinates, projection_matrix = ctx.saved_tensors
        depth_buffer = ctx.depth_buffer
        # Only propagate gradient for features
        grad_features = _C.backward(
            grad_output, features, coordinates, depth_buffer, projection_matrix, ctx.output_size
        )
        return grad_features, None, None, None  # No gradients for others
    
class VoxelToPixel(torch.nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (list[int]): The output resolution [height, width].
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, features, coordinates, projection_matrix):
        """
        Args:
            features (torch.Tensor): Input 3D features, shape (N, C).
            coordinates (torch.Tensor): Voxel coordinates, shape (N, 3).
            projection_matrix (torch.Tensor): Projection matrix, shape (3, 4).

        Returns:
            output_features (torch.Tensor): Projected 2D features, shape (H, W, C).
            depth_map (torch.Tensor): Depth map, shape (H, W).
        """
        return Voxel2PixelFunction.apply(features, coordinates, projection_matrix, self.output_size)