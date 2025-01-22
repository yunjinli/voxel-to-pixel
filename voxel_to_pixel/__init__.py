import torch
from torch.utils.cpp_extension import load
from . import _C
from typing import List, Tuple

class Voxel2PixelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coordinates, projection_matrix, output_size, batch_size):
        ctx.save_for_backward(features, coordinates, projection_matrix)
        ctx.output_size = output_size
        output_features, inv_depth_map, depth_buffer = _C.forward(features, coordinates, projection_matrix, output_size, batch_size)
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
        return grad_features, None, None, None, None  # No gradients for others
        # return None, None, None, None, None  # No gradients for others
    
class VoxelToPixel(torch.nn.Module):
    def __init__(self, 
                 fx: float = 964.828979 / 4,
                 fy: float = 964.828979 / 4,
                 cx: float = 643.788025 / 4,
                 cy: float = 484.407990 / 4,
                 h: int = 960 // 4,
                 w: int = 1280 // 4,
                 R: List[List] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                 voxel_size: List[int] = [0.75, 0.75, 0.75],
                 pcd_range: List[int] = [0, -25, -25, 50, 25, 25],
                 ):
        """Constructor

        Args:
            fx (float, optional): X focal length of the camera. Defaults to 964.828979/4.
            fy (float, optional): Y focal length of the camera. Defaults to 964.828979/4.
            cx (float, optional): X center of the camera. Defaults to 643.788025/4.
            cy (float, optional): Y center of the camera. Defaults to 484.407990/4.
            h (int, optional): Height of the image. Defaults to 960//4.
            w (int, optional): Width of the image. Defaults to 1280//4.
            R (List[List], optional): Axes alignment before projection using pinhole camera model. Defaults to [[0, -1, 0], [0, 0, -1], [1, 0, 0]].
            voxel_size (List[int], optional): Size of the voxel grid. Defaults to [0.75, 0.75, 0.75].
            pcd_range (List[int], optional): Poincloud ranges [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to [0, -25, -25, 50, 25, 25].
        """        
        super().__init__()
        self.output_size = [h, w]
        K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], requires_grad=False, dtype=torch.float32
        )
        vx = voxel_size[0]
        vy = voxel_size[1]
        vz = voxel_size[2]
        x_offset = vx / 2 + pcd_range[0]
        y_offset = vy / 2 + pcd_range[1]
        z_offset = vz / 2 + pcd_range[2]

        R = torch.tensor(
            R, requires_grad=False, dtype=torch.float32
        )

        V2P = torch.tensor(
            [
                [vx, 0, 0, x_offset],
                [0, vy, 0, y_offset],
                [0, 0, vz, z_offset]
            ], requires_grad=False, dtype=torch.float32
        )

        self.projection_matrix = torch.nn.Parameter(K @ R @ V2P, requires_grad=False)
        
    def forward(self, features: torch.tensor, coordinates: torch.tensor, batch_size: int) -> Tuple[torch.tensor, torch.tensor]:
        """Forward pass of the projection module

        Args:
            features (torch.tensor): Voxel features with shape [N, C].
            coordinates (torch.tensor): Voxel coordinates with shape [N, 4]. Note thae the the order is [batch_id, z, y, x]. 
            batch_size (int): Batch size

        Returns:
            Tuple[torch.tensor, torch.tensor]: (projected_feats: [B, C, H, W], inv_depth_maps: [B, H, W])
        """        
        return Voxel2PixelFunction.apply(features, coordinates, self.projection_matrix, self.output_size, batch_size)
    
class VoxelToPixelV2(torch.nn.Module):
    def __init__(self, 
                #  fx: float = 964.828979 / 4,
                #  fy: float = 964.828979 / 4,
                #  cx: float = 643.788025 / 4,
                #  cy: float = 484.407990 / 4,
                #  h: int = 960 // 4,
                #  w: int = 1280 // 4,
                 R: List[List] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                 voxel_size: List[int] = [0.75, 0.75, 0.75],
                 pcd_range: List[int] = [0, -25, -25, 50, 25, 25],
                 device: str = 'cpu',
                 ):
        """Constructor

        Args:
            R (List[List], optional): Axes alignment before projection using pinhole camera model. Defaults to [[0, -1, 0], [0, 0, -1], [1, 0, 0]].
            voxel_size (List[int], optional): Size of the voxel grid. Defaults to [0.75, 0.75, 0.75].
            pcd_range (List[int], optional): Poincloud ranges [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to [0, -25, -25, 50, 25, 25].
        """        
        super().__init__()
        # K = torch.tensor(
        #     [
        #         [fx, 0, cx],
        #         [0, fy, cy],
        #         [0, 0, 1]
        #     ], requires_grad=False, dtype=torch.float32
        # )
        vx = voxel_size[0]
        vy = voxel_size[1]
        vz = voxel_size[2]
        x_offset = vx / 2 + pcd_range[0]
        y_offset = vy / 2 + pcd_range[1]
        z_offset = vz / 2 + pcd_range[2]

        self.R = torch.tensor(
            R, requires_grad=False, dtype=torch.float32
        ).to(device)

        self.V2P = torch.tensor(
            [
                [vx, 0, 0, x_offset],
                [0, vy, 0, y_offset],
                [0, 0, vz, z_offset]
            ], requires_grad=False, dtype=torch.float32
        ).to(device)

        # self.projection_matrix = torch.nn.Parameter(K @ R @ V2P, requires_grad=False)
        
    def forward(self, features: torch.tensor, coordinates: torch.tensor, K: torch.tensor, batch_size: int, h: int, w: int) -> Tuple[torch.tensor, torch.tensor]:
        """Forward pass of the projection module

        Args:
            features (torch.tensor): Voxel features with shape [N, C].
            coordinates (torch.tensor): Voxel coordinates with shape [N, 4]. Note thae the the order is [batch_id, z, y, x]. 
            batch_size (int): Batch size

        Returns:
            Tuple[torch.tensor, torch.tensor]: (projected_feats: [B, C, H, W], inv_depth_maps: [B, H, W])
        """
        output_size = [h, w]
        
        projection_matrix = K.to(features.device) @ self.R @ self.V2P
           
        return Voxel2PixelFunction.apply(features, coordinates, projection_matrix, output_size, batch_size)