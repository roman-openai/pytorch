#pragma once

#include <chrono>

#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/empty.h>

namespace c10d::cuda::detail {

class BarrierHandle {
 public:
  BarrierHandle(std::chrono::milliseconds timeout)
      : comm_{at::empty({3}, at::TensorOptions().dtype(at::kInt))},
        timeout_{timeout} {}

  void run();

  void abort() {
    comm_[0] = 1;
  }

  int32_t status() {
    return comm_[1].item<int32_t>();
  }

 private:
  // (abort, cycles)
  const at::Tensor comm_;
  const std::chrono::milliseconds timeout_;
};

BarrierHandle barrier(std::chrono::milliseconds timeout);

} // namespace c10d::cuda::detail
