#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
    torch::Tensor features, torch::Tensor coordinates,
    torch::Tensor projection_matrix, std::vector<int64_t> output_size);

torch::Tensor backward(torch::Tensor grad_output, torch::Tensor features,
                       torch::Tensor coordinates, torch::Tensor depth_buffer,
                       torch::Tensor projection_matrix,
                       std::vector<int64_t> output_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Voxel to Pixel projection (CUDA)");
  m.def("backward", &backward, "Voxel to Pixel projection (CUDA)");
}
