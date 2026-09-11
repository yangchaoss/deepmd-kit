// SPDX-License-Identifier: Apache-2.0
// A small real DPA4C graph-lower seam: edge-gradient -> E/F virial assembly.
// The official wrapper fixes this source, compiler, and dispatcher name.  It
// is not an empty marker: the CUDA kernel consumes the live DPA4C edge gradient
// and edge geometry and writes force, atom-virial, and frame-virial tensors
// that the ASE EFS path reads.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename scalar_t>
__global__ void edge_force_virial_kernel(
    const scalar_t* __restrict__ g_e,
    const scalar_t* __restrict__ edge_vec,
    const int64_t* __restrict__ edge_index,
    const bool* __restrict__ edge_mask,
    const int64_t* __restrict__ edge_frame,
    int64_t edge_count,
    scalar_t* __restrict__ force,
    scalar_t* __restrict__ atom_virial,
    scalar_t* __restrict__ virial) {
  const int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge >= edge_count || !edge_mask[edge]) {
    return;
  }
  const int64_t src = edge_index[edge];
  const int64_t dst = edge_index[edge_count + edge];
  const int64_t frame = edge_frame[edge];
  const scalar_t* ge = g_e + edge * 3;
  const scalar_t* ev = edge_vec + edge * 3;
  for (int component = 0; component < 3; ++component) {
    atomicAdd(force + dst * 3 + component, ge[component]);
    atomicAdd(force + src * 3 + component, -ge[component]);
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const scalar_t value = -ge[row] * ev[col];
      atomicAdd(atom_virial + src * 9 + row * 3 + col, value);
      atomicAdd(virial + frame * 9 + row * 3 + col, value);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> edge_force_virial_cuda(
    const torch::Tensor& g_e,
    const torch::Tensor& edge_vec,
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_mask,
    const torch::Tensor& edge_frame,
    int64_t node_capacity,
    int64_t frame_count) {
  TORCH_CHECK(g_e.is_cuda(), "g_e must be on the CUDA/PPU device");
  TORCH_CHECK(edge_vec.is_cuda() && edge_index.is_cuda() && edge_mask.is_cuda() && edge_frame.is_cuda(),
              "all inputs must be on the CUDA/PPU device");
  TORCH_CHECK(g_e.dim() == 2 && g_e.size(1) == 3, "g_e must have shape (E, 3)");
  TORCH_CHECK(edge_vec.sizes() == g_e.sizes(), "edge_vec shape mismatch");
  TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2 && edge_index.size(1) == g_e.size(0),
              "edge_index must have shape (2, E)");
  TORCH_CHECK(edge_mask.dim() == 1 && edge_mask.size(0) == g_e.size(0), "edge_mask shape mismatch");
  TORCH_CHECK(edge_frame.dim() == 1 && edge_frame.size(0) == g_e.size(0), "edge_frame shape mismatch");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64 && edge_frame.scalar_type() == torch::kInt64,
              "edge indices must be int64");
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool, "edge_mask must be bool");
  TORCH_CHECK(g_e.scalar_type() == torch::kFloat32 || g_e.scalar_type() == torch::kFloat64,
              "only FP32/FP64 are supported by the prototype");
  TORCH_CHECK(node_capacity >= 0 && frame_count >= 0, "negative output size");

  auto force = torch::zeros({node_capacity, 3}, g_e.options());
  auto atom_virial = torch::zeros({node_capacity, 3, 3}, g_e.options());
  auto virial = torch::zeros({frame_count, 3, 3}, g_e.options());
  const auto edge_count = g_e.size(0);
  const int blocks = static_cast<int>((edge_count + 255) / 256);
  if (blocks > 0) {
    AT_DISPATCH_FLOATING_TYPES(g_e.scalar_type(), "dpa4c_edge_force_virial", [&] {
      edge_force_virial_kernel<scalar_t><<<blocks, 256>>>(
          g_e.data_ptr<scalar_t>(),
          edge_vec.data_ptr<scalar_t>(),
          edge_index.data_ptr<int64_t>(),
          edge_mask.data_ptr<bool>(),
          edge_frame.data_ptr<int64_t>(),
          edge_count,
          force.data_ptr<scalar_t>(),
          atom_virial.data_ptr<scalar_t>(),
          virial.data_ptr<scalar_t>());
    });
  }
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "dpa4c edge_force_virial launch failed: ", cudaGetErrorString(error));
  return {force, atom_virial, virial};
}

}  // namespace

TORCH_LIBRARY(dpa4c_contest, m) {
  m.def("edge_force_virial(Tensor g_e, Tensor edge_vec, Tensor edge_index, Tensor edge_mask, Tensor edge_frame, int node_capacity, int frame_count) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(dpa4c_contest, CUDA, m) {
  m.impl("edge_force_virial", edge_force_virial_cuda);
}
