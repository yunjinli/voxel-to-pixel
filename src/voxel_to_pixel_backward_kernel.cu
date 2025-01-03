#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void backward_kernel(
    const float *grad_output, const float *depth_buffer, const float *features,
    const float *coordinates, float *grad_features, const float *projection_matrix,
    int N, int C, int H, int W)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // Load coordinates
    int batch_id = int(coordinates[idx * 4]);
    float z = coordinates[idx * 4 + 1];
    float y = coordinates[idx * 4 + 2];
    float x = coordinates[idx * 4 + 3];

    // Apply the projection matrix (assume saved in shared memory for performance)
    // extern __shared__ float projection_matrix[];
    float u = projection_matrix[0] * x + projection_matrix[1] * y + projection_matrix[2] * z + projection_matrix[3];
    float v = projection_matrix[4] * x + projection_matrix[5] * y + projection_matrix[6] * z + projection_matrix[7];
    float depth = projection_matrix[8] * x + projection_matrix[9] * y + projection_matrix[10] * z + projection_matrix[11];

    // Map to pixel coordinates
    int u_d = int(u / depth);
    int v_d = int(v / depth);
    if (u_d < 0 || u_d > W)
        return;
    if (v_d < 0 || v_d > H)
        return;
    if (depth < 0)
        return;
    // int pixel_idx = v_d * W + u_d;
    int pixel_idx = batch_id * (W * H) + v_d * W + u_d;
    // Check depth buffer for the closest voxel at this pixel
    // if (__int_as_float(depth_buffer[pixel_idx]) != depth)
    if (depth_buffer[pixel_idx] != depth)
        return;

    // Backpropagate feature gradients
    for (int c = 0; c < C; ++c)
    {
        // atomicAdd(&grad_features[idx * C + c], grad_output[pixel_idx * C + c]);
        atomicAdd(&grad_features[idx * C + c], grad_output[batch_id * (W * H * C) + c * (W * H) + v_d * W + u_d]);
    }
}

torch::Tensor backward(
    torch::Tensor grad_output, torch::Tensor features, torch::Tensor coordinates,
    torch::Tensor depth_buffer, torch::Tensor projection_matrix, std::vector<int64_t> output_size)
{

    int N = features.size(0);
    int C = features.size(1);
    int H = output_size[0];
    int W = output_size[1];

    auto grad_features = torch::zeros_like(features);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    backward_kernel<<<blocks, threads, sizeof(float) * 12>>>(
        grad_output.data_ptr<float>(), depth_buffer.data_ptr<float>(), features.data_ptr<float>(),
        coordinates.data_ptr<float>(), grad_features.data_ptr<float>(), projection_matrix.data_ptr<float>(),
        N, C, H, W);

    return grad_features;
}