#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/cuda/Barrier.cuh>

namespace c10d::cuda::detail {

__global__ void kernel_barrier(int32_t* value, size_t timeout_ms) {
  size_t start = c10d::symmetric_memory::global_timer_ns();
  size_t timeout_ns = timeout_ms * 1e6; // Convert milliseconds to nanoseconds
  while (true) {
    // Atomically read the value
    int current_value = atomicAdd(&value[0], 0);
    // Check if the value is equal to the expected value
    if (current_value == 1) {
      value[1] = 1;
      return;
    }

    if (timeout_ms > 0) {
      // Check if timeout has been reached
      size_t now = c10d::symmetric_memory::global_timer_ns();
      if ((now - start) > timeout_ns) {
        value[1] = 2;
        return;
      }
    }

    // sleep for 1ms
    __nanosleep(1000000);
  }
}

void BarrierHandle::run() {
  kernel_barrier<<<1, 1>>>(comm_.mutable_data_ptr<int32_t>(), timeout_.count());
}

BarrierHandle barrier(std::chrono::milliseconds timeout) {
  BarrierHandle handle{timeout};
  return handle;
}

} // namespace c10d::cuda::detail
