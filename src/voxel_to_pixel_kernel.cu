#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float atomicMinFloat(float *addr, float value)
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void depth_computation_kernel(
    const float *coordinates, const float *projection_matrix,
    float *depth_buffer,
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

    // Apply projection matrix
    float u = projection_matrix[0] * x + projection_matrix[1] * y + projection_matrix[2] * z + projection_matrix[3];
    float v = projection_matrix[4] * x + projection_matrix[5] * y + projection_matrix[6] * z + projection_matrix[7];
    float depth = projection_matrix[8] * x + projection_matrix[9] * y + projection_matrix[10] * z + projection_matrix[11];

    int u_d = int(u / depth);
    int v_d = int(v / depth);
    // printf("(%d, %d) = %f\n", u_d, v_d, depth);
    if (u_d < 0 || u_d >= W)
        return;
    if (v_d < 0 || v_d >= H)
        return;
    if (depth < 0)
        return;
    // printf("(%d, %d) = %f\n", u_d, v_d, depth);
    // Depth selection and feature update
    int pixel_idx = batch_id * (W * H) + v_d * W + u_d;

    // atomicMin(&depth_buffer[pixel_idx], int(depth));
    // if (depth_buffer[pixel_idx] == int(depth))
    // {
    //     for (int c = 0; c < C; c++)
    //     {
    //         output_features[pixel_idx * C + c] = features[idx * C + c];
    //     }
    // }
    // Atomic operation to store the nearest depth
    // atomicMin(&depth_buffer[pixel_idx], __float_as_int(depth));
    atomicMinFloat(&depth_buffer[pixel_idx], depth);
    // printf("%f and %f\n", depth_buffer[pixel_idx], depth);
    // Wait for depth buffer to be updated
    __syncthreads();

    // Write features if current voxel is closest
    // if (__int_as_float(depth_buffer[pixel_idx]) == depth)
    // if (depth_buffer[pixel_idx] == depth)
    // if (depth_buffer[pixel_idx] == int(depth * 1000))
    // {
    //     for (int c = 0; c < C; ++c)
    //     {
    //         // atomicAdd(&output_features[pixel_idx * C + c], features[idx * C + c]);
    //         // output_features[pixel_idx * C + c] = features[idx * C + c];
    //         // atomicAdd(&output_features[batch_id * (W * H * C) + c * (W * H) + v_d * W + u_d], features[idx * C + c]);
    //         output_features[batch_id * (W * H * C) + c * (W * H) + v_d * W + u_d] = features[idx * C + c];
    //     }
    //     inv_depth_map[pixel_idx] = 1.0 / depth;
    // }
}

__global__ void feature_update_kernel(
    const float *features, const float *coordinates, const float *projection_matrix,
    float *depth_buffer, float *inv_depth_map, float *output_features,
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

    // Apply projection matrix
    float u = projection_matrix[0] * x + projection_matrix[1] * y + projection_matrix[2] * z + projection_matrix[3];
    float v = projection_matrix[4] * x + projection_matrix[5] * y + projection_matrix[6] * z + projection_matrix[7];
    float depth = projection_matrix[8] * x + projection_matrix[9] * y + projection_matrix[10] * z + projection_matrix[11];

    int u_d = int(u / depth);
    int v_d = int(v / depth);
    // printf("(%d, %d) = %f\n", u_d, v_d, depth);
    if (u_d < 0 || u_d >= W)
        return;
    if (v_d < 0 || v_d >= H)
        return;
    if (depth < 0)
        return;
    // printf("(%d, %d) = %f\n", u_d, v_d, depth);
    // Depth selection and feature update
    int pixel_idx = batch_id * (W * H) + v_d * W + u_d;

    if (depth_buffer[pixel_idx] == depth)
    {
        for (int c = 0; c < C; ++c)
        {
            // output_features[batch_id * (W * H * C) + c * (W * H) + v_d * W + u_d] = features[idx * C + c];
            atomicAdd(&output_features[batch_id * (W * H * C) + c * (W * H) + v_d * W + u_d], features[idx * C + c]);
        }
        atomicAdd(&inv_depth_map[pixel_idx], 1.0 / depth);

        // inv_depth_map[pixel_idx] = 1.0 / depth;
    }
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor features, torch::Tensor coordinates, torch::Tensor projection_matrix, std::vector<int64_t> output_size, int batch_size)
{
    // Extract dimensions
    int N = features.size(0);
    int C = features.size(1);
    int H = output_size[0];
    int W = output_size[1];

    // auto output_features = torch::zeros({H, W, C}, features.options());
    auto output_features = torch::zeros({batch_size, C, H, W}, features.options());
    auto inv_depth_map = torch::zeros({batch_size, H, W}, features.options());
    auto depth_buffer = torch::full({batch_size, H, W}, INT_MAX, torch::TensorOptions().dtype(torch::kF32).device(features.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    depth_computation_kernel<<<blocks, threads>>>(
        coordinates.data_ptr<float>(), projection_matrix.data_ptr<float>(),
        depth_buffer.data_ptr<float>(),
        N, C, H, W);

    feature_update_kernel<<<blocks, threads>>>(
        features.data_ptr<float>(), coordinates.data_ptr<float>(), projection_matrix.data_ptr<float>(),
        depth_buffer.data_ptr<float>(), inv_depth_map.data_ptr<float>(), output_features.data_ptr<float>(),
        N, C, H, W);

    return std::make_tuple(output_features, inv_depth_map, depth_buffer);
}
