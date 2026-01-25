#include <vector>
#include <musa_fp16.h>

#include "../tester/utils.h"

#define CUDA_CHECK(call)                                                                    \
    {                                                                                       \
        musaError_t err = call;                                                             \
        if (err != musaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__                    \
                      << " - " << musaGetErrorString(err) << "\n";                          \
            exit(-1);                                                                       \
        }                                                                                   \
    }

template <typename T>
__global__ void gpu_trace(const T* input, T* output, size_t skip, size_t N) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid = threadIdx.x;
    const unsigned mask = __activemask();

    T sum = 0;
    for (size_t i = idx; i < N; i += blockDim.x * gridDim.x) {
        // 初始化时获取全部的对角线元素，之后就是课上说的求和问题
        sum += input[i * skip];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    if (tid % 32 == 0) {
        atomicAdd(output, sum);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // TODO: Implement the trace function
    T *musa_input, *musa_output;
    size_t size_bytes = rows * cols * sizeof(T);
    size_t value_cnt = std::min(rows, cols);
    CUDA_CHECK(musaMalloc(&musa_input, size_bytes));
    CUDA_CHECK(musaMalloc(&musa_output, sizeof(T)));
    CUDA_CHECK(musaMemset(musa_output, 0, sizeof(T)));
    CUDA_CHECK(musaMemcpy(musa_input, h_input.data(), size_bytes, musaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim(std::max(1ul, (value_cnt + block_dim.x - 1) / block_dim.x));
    gpu_trace<<<grid_dim, block_dim>>>(musa_input, musa_output, cols + 1, value_cnt);
    T output;
    CUDA_CHECK(musaMemcpy(&output, musa_output, sizeof(T), musaMemcpyDeviceToHost));
    CUDA_CHECK(musaFree(musa_input));
    CUDA_CHECK(musaFree(musa_output));
    return output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
