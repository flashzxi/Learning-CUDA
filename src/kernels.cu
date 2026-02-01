#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../tester/utils.h"
#include <cassert>
#include <cmath>

#define BlockSizeQ 32
#define BlockSizeKV 32
#define BlockDimX 16
#define BlockDimY 16

template<typename T> __device__ __forceinline__ T lowest();

template<> __device__ __forceinline__ float lowest<float>() { return -INFINITY; }

template<>
__device__ __forceinline__ __half lowest<half>() {
    return __ushort_as_half(0xFC00);
}

__device__ __forceinline__ bool is_neg_inf(half x) {
    return (__half_as_ushort(x) == 0xFC00);
}

__device__ __forceinline__ bool is_neg_inf(float x) {
    return isinf(x) && x < 0.0f;
}

#define CUDA_CHECK(call)                                                                    \
    {                                                                                       \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__                    \
                      << " - " << cudaGetErrorString(err) << "\n";                          \
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
    T *cuda_input, *cuda_output;
    size_t size_bytes = rows * cols * sizeof(T);
    size_t value_cnt = std::min(rows, cols);
    CUDA_CHECK(cudaMalloc(&cuda_input, size_bytes));
    CUDA_CHECK(cudaMalloc(&cuda_output, sizeof(T)));
    CUDA_CHECK(cudaMemset(cuda_output, 0, sizeof(T)));
    CUDA_CHECK(cudaMemcpy(cuda_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim(std::max(1ul, (value_cnt + block_dim.x - 1) / block_dim.x));
    gpu_trace<<<grid_dim, block_dim>>>(cuda_input, cuda_output, cols + 1, value_cnt);
    T output;
    CUDA_CHECK(cudaMemcpy(&output, cuda_output, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cuda_input));
    CUDA_CHECK(cudaFree(cuda_output));
    return output;
}

// 单个block的线程合作计算 C = A * B
template <typename T, bool TransposeB>
__device__ void block_gemm_shared(
    const T* A,   // shared mem: [m, k]
    const T* B,   // shared mem: [k, n] or [n, k]
    T* C,         // shared mem: [m, n]
    int m, int k, int n, float factor
) {
    int cRow = threadIdx.y;
    int cCol = threadIdx.x;

    int dimRow = blockDim.y;
    int dimCol = blockDim.x;

    int row_cnt = (m + dimRow - 1) / dimRow;
    int col_cnt = (n + dimCol - 1) / dimCol;

    // 4个应该够用了
    // 真实值应该是ceil(BlockM / blockDim.y) ceil(BlockN / blockDim.x)
    constexpr int max_size_row = 8;
    constexpr int max_size_col = 8;
    float acc[max_size_row][max_size_col];

#pragma unroll
    for (int i = 0; i < max_size_row; ++i)
#pragma unroll
        for (int j = 0; j < max_size_col; ++j)
            acc[i][j] = 0.0f;

    for (int i = 0; i < row_cnt && i * dimRow + cRow < m; ++i) {
        int row = i * dimRow + cRow;
        for (int j = 0; j < col_cnt && j * dimCol + cCol < n; ++j) {
            int col = j * dimCol + cCol;
            for (int kk = 0; kk < k; ++kk) {
                if constexpr (!TransposeB) {
                    acc[i][j] += float(A[row * k + kk]) * float(B[kk * n + col]);
                } else {
                    acc[i][j] += float(A[row * k + kk]) * float(B[col * k + kk]);
                }
            }
            acc[i][j] *= factor;
        }
    }

    for (int i = 0; i < row_cnt; ++i) {
        int row = cRow + i * dimRow;
        for (int j = 0; j < col_cnt; ++j) {
            int col = cCol + j * dimCol;
            if (row < m && col < n) {
                C[row * n + col] = (T)acc[i][j];
            }
        }
    }
    __syncthreads();
}

template<typename T>
struct FlashAttentionParam {
    T* q;
    T* k;
    T* v;
    T* O;
    int batch_size;
    int target_seq_len;
    int src_seq_len;
    int query_heads;
    int kv_heads;
    int head_dim;
    bool is_causal;
    size_t size_per_q_batch;
    size_t size_per_q_head;
    size_t size_per_kv_batch;
    size_t size_per_kv_head;
    float factor;

    FlashAttentionParam(
        T* q,
        T* k,
        T* v,
        T* O,
        int batch_size,
        int target_seq_len,
        int src_seq_len,
        int query_heads,
        int kv_heads,
        int head_dim,
        bool is_causal):
    q(q),
    k(k),
    v(v),
    O(O),
    batch_size(batch_size),
    target_seq_len(target_seq_len),
    src_seq_len(src_seq_len),
    query_heads(query_heads),
    kv_heads(kv_heads),
    head_dim(head_dim),
    is_causal(is_causal),
    factor(1.0f / sqrt(head_dim))
    {
        size_per_q_batch = target_seq_len * query_heads * head_dim;
        size_per_q_head = target_seq_len * head_dim;

        size_per_kv_batch = src_seq_len * kv_heads * head_dim;
        size_per_kv_head = src_seq_len * head_dim;

    }
};

// 从hbm load 连续的len 个元素到smem
template <typename T>
__device__ void copy_mem(
    const T* fmem,
    T* tmem,
    int len
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.y * blockDim.x;

    for (int idx = tid; idx < len; idx += num_threads) {
        tmem[idx] = fmem[idx];
    }

    __syncthreads();
}

template <typename T>
__device__ void copy_mem_stride(
        const T* fmem,
        T* tmem,
        int len,
        int head_dim,
        int stride_in_element
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.y * blockDim.x;

    for (int idx = tid; idx < len; idx += num_threads) {
        int row_idx = idx / head_dim;
        int col_idx = idx % head_dim;
        tmem[idx] = fmem[row_idx * stride_in_element + col_idx];
    }

    __syncthreads();
}

template <typename T>
__device__ void copy_mem_stride_out(
        const T* fmem,
        T* tmem,
        int len,
        int head_dim,
        int stride
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.y * blockDim.x;

    for (int idx = tid; idx < len; idx += num_threads) {
        int row_idx = idx / head_dim;
        int col_idx = idx % head_dim;
        tmem[row_idx * stride + col_idx] = fmem[idx];
    }

    __syncthreads();
}

// 务必16字节对齐
template <typename T>
__device__ void
clear_smem_vec(T* smem, int len) {
    using Vec = int4;  // 16 bytes
    Vec* vsmem = reinterpret_cast<Vec*>(smem);

    int vec_len = len * sizeof(T) / sizeof(Vec);
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    for (int i = tid; i < vec_len; i += stride) {
        vsmem[i] = make_int4(0, 0, 0, 0);
    }
    __syncthreads();
}

template <typename T>
__device__ void
reset_mem(T* smem, int len, T value) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    for (int i = tid; i < len; i += stride) {
        smem[i] = value;
    }
    __syncthreads();
}

// causal mask
template <typename T>
__device__ void mask_S(
    T* smem_s_block,
    int block_start_row_idx,
    int block_start_col_idx,
    int block_rows_cnt,
    int block_cols_cnt
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int num_threads = blockDim.x * blockDim.y;
    int total_elems = block_rows_cnt * block_cols_cnt;

    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int i = idx / block_cols_cnt;  // row in block
        int j = idx % block_cols_cnt;  // col in block

        int q = block_start_row_idx + i;
        int k = block_start_col_idx + j;

        if (k > q) {
            smem_s_block[i * block_cols_cnt + j] = lowest<T>();
        }
    }

    __syncthreads();
}

template<typename T>
__device__ void row_max(
    const T* smem,   // [row_cnt][col_cnt]
    T* o,            // [row_cnt]
    int row_cnt,
    int col_cnt
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_threads = blockDim.x;
    int row_stride  = blockDim.y;

    // 32 x 32 使用 16 x 16 的 thread 处理
    for (int row = ty; row < row_cnt; row += row_stride) {
        float local_max = lowest<float>();

        for (int col = tx; col < col_cnt; col += col_threads) {
            float v = float(smem[row * col_cnt + col]);
            local_max = v > local_max ? v : local_max;
        }

        unsigned mask = __activemask();
        for (int offset = col_threads >> 1; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(mask, local_max, offset, col_threads);
            local_max = other > local_max ? other : local_max;
        }

        if (tx == 0) {
            o[row] = T(local_max);
        }
    }
    __syncthreads();
}

template <typename T>
__device__ void row_sum(
    const T* smem,   // [row_cnt][col_cnt]
    T* o,            // [row_cnt]
    int row_cnt,
    int col_cnt
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_threads = blockDim.x;   // 期望为 1/2/4/8/16/32
    int row_stride  = blockDim.y;

    // 每个 ty 负责若干行：row = ty + k*row_stride
    for (int row = ty; row < row_cnt; row += row_stride) {
        float local_sum = T(0);

        for (int col = tx; col < col_cnt; col += col_threads) {
            local_sum += float(smem[row * col_cnt + col]);
        }
        unsigned mask = __activemask();
        for (int offset = col_threads >> 1; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(mask, local_sum, offset, col_threads);
        }

        if (tx == 0) {
            o[row] = local_sum;
        }
    }
    __syncthreads();
}

template<typename T>
__device__ void safe_exp(
    T* smem_s,
    const T* smem_m,
    int row_cnt,
    int col_cnt
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    for (int i = idx; i < row_cnt * col_cnt; i += stride) {
        int row = i / col_cnt;
        if (is_neg_inf(smem_s[i])) {
            smem_s[i] = (T)0;
        } else {
            float x = (float)(smem_s[i] - smem_m[row]);
            float e = expf(x);
            smem_s[i] = (T)e;
        }
    }
    __syncthreads();
}
// 计算m_new l_new
template<typename T>
__device__ void calc_new_m_l(
        const T* smem_l_ori,
        const T* smem_l,
        T* smem_l_new,
        const T* smem_m_ori,
        const T* smem_m,
        T* smem_m_new,
        int row_cnt
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    for (int i = idx; i < row_cnt; i += stride) {
        smem_m_new[i] = smem_m_ori[i] > smem_m[i] ? smem_m_ori[i] : smem_m[i];
        float h1 = smem_m_ori[i] - smem_m_new[i];
        float h2 = smem_m[i] - smem_m_new[i];
        smem_l_new[i] = T(expf(h1) * float(smem_l_ori[i]) + expf(h2) * float(smem_l[i]));
    }

    __syncthreads();
}

template<typename T>
__device__ void calc_O(
        const T* smem_l_new,
        const T* smem_l_ori,
        const T* smem_m,
        const T* smem_m_new,
        const T* smem_m_ori,
        const T* smem_p,  // m * n
        const T* smem_v,  // n * k
        T* smem_O_ori,
        T* smem_pv,
        int m, int k, int n
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    block_gemm_shared<T, false>(smem_p, smem_v, smem_pv, m, n, k, 1.0f);

    for (int i = idx; i < m * k; i += stride) {
        int row_idx = i / k;
        smem_O_ori[i] = T(
                    (1.0f / float(smem_l_new[row_idx]) *
                    ( float(smem_l_ori[row_idx]) * expf(float(smem_m_ori[row_idx] - smem_m_new[row_idx])) * float(smem_O_ori[i])
                      + expf(float(smem_m[row_idx] - smem_m_new[row_idx])) * float(smem_pv[i]) ))
                );
    }

    __syncthreads();
}

template<typename T>
class SharedMemAllocator {
public:
    __device__ SharedMemAllocator(T* base): _base_ptr(base), _offset(0) {}

    __device__ T* allocate(size_t size) {
        T* ret = _base_ptr + _offset;
        _offset += size;
        return ret;
    }

private:
    T* _base_ptr;
    size_t _offset;
};

/**
 * 一个block内的线程合作计算 (Q_block * K^T) *V
 * 其中： Q_block = Q[BlockSizeQ * block_idx_q : BlockSizeQ * (blockId + 1), h * hIdx : h * (hIdx + 1)]
 *
 * KV会在循环内被拆成小块
 *      K/V_block = [BlockSizeKV * blockIdN : BlockSize * (blockId + 1), h * hIdx : h * (hIdx + 1)]
**/
template<typename T>
void __device__ flash_block(FlashAttentionParam<T> param, int batchIdx, int headIdx, int block_idx_q) {
    extern __shared__ char smem_[];
    T* smem_as_T = reinterpret_cast<T*>(smem_);
    SharedMemAllocator<T> allocator = SharedMemAllocator<T>(smem_as_T);

    // load Q 分块
    T* q_ptr_batch = param.q + param.size_per_q_batch * batchIdx;
    T* q_ptr_block = q_ptr_batch + block_idx_q * param.head_dim * param.query_heads * BlockSizeQ;
    T* q_ptr_head = q_ptr_block + headIdx * param.head_dim;

    int q_block_real_len = param.target_seq_len - block_idx_q * BlockSizeQ < BlockSizeQ ?
            param.target_seq_len - block_idx_q * BlockSizeQ : BlockSizeQ;

    int64_t q_block_size = q_block_real_len * param.head_dim;
    T* smem_q_block = allocator.allocate(BlockSizeQ * param.head_dim);
    copy_mem_stride(q_ptr_head, smem_q_block, q_block_size, param.head_dim, param.query_heads * param.head_dim);

    // 给KV分内存
    int kv_head_idx = headIdx * param.kv_heads / param.query_heads;
//    int kv_head_idx = headIdx % param.kv_heads;

    T* k_ptr_batch = param.k + param.size_per_kv_batch * batchIdx;
    T* v_ptr_batch = param.v + param.size_per_kv_batch * batchIdx;

    T* smem_k_block = allocator.allocate(BlockSizeKV * param.head_dim);
    T* smem_v_block = allocator.allocate(BlockSizeKV * param.head_dim);

    // 中间变量分配内存
    // o的size会变，取最大的BlockSizeQ * BlockSizeKV
    T* smem_o_block = allocator.allocate(BlockSizeQ * param.head_dim);
    // 初始化
    clear_smem_vec(smem_o_block, BlockSizeQ * param.head_dim);

    T* smem_PV = allocator.allocate(BlockSizeQ * param.head_dim);
    T* smem_m = allocator.allocate(BlockSizeQ);
    T* smem_m_ori = allocator.allocate(BlockSizeQ);
    reset_mem(smem_m_ori, q_block_real_len, lowest<T>());
    T* smem_m_new = allocator.allocate(BlockSizeQ);

    T* smem_l = allocator.allocate(BlockSizeQ);
    T* smem_l_new = allocator.allocate(BlockSizeQ);
    T* smem_l_ori = allocator.allocate(BlockSizeQ);
    clear_smem_vec(smem_l_ori, BlockSizeQ);

    T* smem_P =  allocator.allocate(BlockSizeQ * BlockSizeKV);

    int head_dim = param.head_dim;
    float factor = param.factor;
    int kv_block_idx = 0;
    for (int64_t i = 0; i < param.src_seq_len; i += BlockSizeKV, kv_block_idx++) {
        // load K V 分块
        T* k_ptr_block = k_ptr_batch + i * param.head_dim * param.kv_heads;
        T* k_ptr_head = k_ptr_block + kv_head_idx * param.head_dim;
        T* v_ptr_block = v_ptr_batch + i * param.head_dim * param.kv_heads;
        T* v_ptr_head = v_ptr_block + kv_head_idx * param.head_dim;
        int kv_block_real_len = param.src_seq_len - kv_block_idx * BlockSizeKV < BlockSizeKV ?
                param.src_seq_len - kv_block_idx * BlockSizeKV : BlockSizeKV;
        size_t kv_block_real_size = kv_block_real_len * param.head_dim;
        copy_mem_stride(k_ptr_head, smem_k_block, kv_block_real_size, param.head_dim, param.kv_heads * param.head_dim);
        copy_mem_stride(v_ptr_head, smem_v_block, kv_block_real_size, param.head_dim, param.kv_heads * param.head_dim);
        // 计算S block = Q block x K block
        block_gemm_shared<T, true>(smem_q_block, smem_k_block, smem_P, q_block_real_len, head_dim, kv_block_real_len, factor);
        // 设置掩码
        if (param.is_causal) {
            mask_S(smem_P, block_idx_q * BlockSizeQ, kv_block_idx * BlockSizeKV, q_block_real_len, kv_block_real_len);
        }
        // 维护m
        row_max(smem_P, smem_m, q_block_real_len, kv_block_real_len);
        // 计算 P
        safe_exp(smem_P, smem_m, q_block_real_len, kv_block_real_len);
        row_sum(smem_P, smem_l, q_block_real_len, kv_block_real_len);
        calc_new_m_l(smem_l_ori, smem_l, smem_l_new, smem_m_ori, smem_m, smem_m_new, q_block_real_len);
        // 计算 O = P x V
        calc_O(smem_l_new, smem_l_ori, smem_m, smem_m_new, smem_m_ori,
               smem_P, smem_v_block, smem_o_block, smem_PV, q_block_real_len, param.head_dim, kv_block_real_len
        );

        // 写回m, l
        copy_mem(smem_m_new, smem_m_ori, q_block_real_len);
        copy_mem(smem_l_new, smem_l_ori, q_block_real_len);
    }
    // 写出O
    T* O_hbm = param.O
            + batchIdx * param.size_per_q_batch
            + block_idx_q * BlockSizeQ * param.head_dim * param.query_heads
            + headIdx * param.head_dim;
    copy_mem_stride_out(smem_o_block, O_hbm, q_block_size, param.head_dim, param.query_heads * param.head_dim);
}

template<typename T>
void __global__ flash_kernel(FlashAttentionParam<T> param) {
    // 单个block负责一个Q的一个batch的一个head的一个row Block
    const int block_idx_q = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    flash_block(param, batch_idx, head_idx, block_idx_q);
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
    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);

    // 参考 https://zhuanlan.zhihu.com/p/676655352
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    size_t smem_size = prop.sharedMemPerBlock;
    // 为了简单起见，假设Q分块和KV分块一样大
    dim3 block_dim(BlockDimX, BlockDimY);
    const int num_m_block = (target_seq_len + BlockSizeQ - 1) / BlockSizeQ;
    dim3 grid(num_m_block, batch_size, query_heads);

    size_t total_shared_mem_used =
            sizeof(T) * (3 * BlockSizeQ * head_dim + 2 * BlockSizeKV * head_dim + 1 * BlockSizeKV * BlockSizeQ + 6 * BlockSizeQ);

    assert(total_shared_mem_used < smem_size);

    cudaStream_t s0 = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s0));

    T *cuda_q, *cuda_k, *cuda_v, *cuda_o;
    CUDA_CHECK(cudaMallocAsync(&cuda_q, sizeof(T) * batch_size * target_seq_len * query_heads * head_dim, s0));
    CUDA_CHECK(cudaMallocAsync(&cuda_o, sizeof(T) * batch_size * target_seq_len * query_heads * head_dim, s0));
    CUDA_CHECK(cudaMallocAsync(&cuda_k, sizeof(T) * batch_size * src_seq_len * kv_heads * head_dim, s0));
    CUDA_CHECK(cudaMallocAsync(&cuda_v, sizeof(T) * batch_size * src_seq_len * kv_heads * head_dim, s0));

    CUDA_CHECK(cudaMemcpyAsync(cuda_q, h_q.data(), sizeof(T) * batch_size * target_seq_len * query_heads * head_dim, cudaMemcpyHostToDevice, s0))
    CUDA_CHECK(cudaMemcpyAsync(cuda_k, h_k.data(), sizeof(T) * batch_size * src_seq_len * kv_heads * head_dim, cudaMemcpyHostToDevice, s0))
    CUDA_CHECK(cudaMemcpyAsync(cuda_v, h_v.data(), sizeof(T) * batch_size * src_seq_len * kv_heads * head_dim, cudaMemcpyHostToDevice, s0))


    FlashAttentionParam<T> param(
            cuda_q,
            cuda_k,
            cuda_v,
            cuda_o,
            batch_size,
            target_seq_len,
            src_seq_len,
            query_heads,
            kv_heads,
            head_dim,
            is_causal);

    flash_kernel<<<grid, block_dim, total_shared_mem_used, s0>>>(param);
    CUDA_CHECK(cudaMemcpyAsync(h_o.data(), param.O, sizeof(T) * batch_size * target_seq_len * query_heads * head_dim, cudaMemcpyDeviceToHost, s0))

    CUDA_CHECK(cudaStreamSynchronize(s0));

    CUDA_CHECK(cudaFreeAsync(cuda_q, s0));
    CUDA_CHECK(cudaFreeAsync(cuda_k, s0));
    CUDA_CHECK(cudaFreeAsync(cuda_v, s0));
    CUDA_CHECK(cudaFreeAsync(cuda_o, s0));
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamDestroy(s0));
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
