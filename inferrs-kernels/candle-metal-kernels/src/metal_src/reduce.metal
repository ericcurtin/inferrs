#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

METAL_FUNC uint nonzero(uint n) {
    return n == 0 ? 1 : n;
}

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr ushort granularity() {
    return nonzero<vec_elements<T>::value>();
}

METAL_FUNC uint next_p2(uint x) {
    return 1 << (32 - clz(x - 1));
}

METAL_FUNC uint prev_p2(uint x) {
    return 1 << (31 - clz(x));
}

constant uint MAX_SHARED_MEM = 32767;

template<typename T>
METAL_FUNC uint max_shared_mem(uint n) {
    return min(n, div_ceil<MAX_SHARED_MEM, sizeof(T)>());
}


template<ushort D, typename IndexT>
struct strided_indexer {
    constant const IndexT *dims;
    constant const IndexT *strides;
    strided_indexer<D - 1, IndexT> next {dims, strides};

    METAL_FUNC IndexT operator()(IndexT idx) const {
        IndexT dim = dims[D - 1];
        IndexT i = (idx % dim) * strides[D - 1];
        idx /= dim;
        return i + next(idx);
    }
};

template<typename IndexT>
struct strided_indexer<1, IndexT> {
    constant const IndexT *dims;
    constant const IndexT *strides;

    METAL_FUNC IndexT operator()(IndexT idx) const {
        return idx * strides[0];
    }
};

template<ushort D, typename IndexT>
METAL_FUNC IndexT get_strided_idx_fallback(
    IndexT idx,
    constant const IndexT &num_dims,
    constant const IndexT *dims,
    constant const IndexT *strides
) {
    strided_indexer<D, IndexT> next {dims, strides};

    IndexT strided_i = 0;
    for (IndexT d = D; d < num_dims; d++) {
        IndexT dim_idx = num_dims - 1 - d;
        IndexT dim = dims[dim_idx];
        strided_i += (idx % dim) * strides[dim_idx];
        idx /= dim;
    }
    return strided_i + next(idx);
}

template<typename IndexT>
METAL_FUNC IndexT get_strided_index_t(
    IndexT idx,
    constant const IndexT &num_dims,
    constant const IndexT *dims,
    constant const IndexT *strides
) {
    switch (num_dims) {
        case 1: return strided_indexer<1, IndexT>{dims, strides}(idx);
        case 2: return strided_indexer<2, IndexT>{dims, strides}(idx);
        case 3: return strided_indexer<3, IndexT>{dims, strides}(idx);
        case 4: return strided_indexer<4, IndexT>{dims, strides}(idx);
        //case 5: return strided_indexer<5, IndexT>{dims, strides}(idx);
        //case 6: return strided_indexer<6, IndexT>{dims, strides}(idx);
        default: return get_strided_idx_fallback<4, IndexT>(idx, num_dims, dims, strides);
    }
}

template<typename IndexT, bool STRIDED>
struct indexer_t {
    typedef IndexT I;
};

template<typename IndexT>
struct indexer_t<IndexT, false> {
    typedef IndexT I;

    const IndexT last_dim = 0;

    METAL_FUNC IndexT operator()(IndexT i) const {
        return i;
    }
};

template<typename IndexT>
struct indexer_t<IndexT, true> {
    typedef IndexT I;

    constant const IndexT &num_dims;
    constant const IndexT *dims;
    constant const IndexT *strides;
    const IndexT last_dim;

    METAL_FUNC IndexT operator()(IndexT i) const {
        return get_strided_index_t(i, num_dims, dims, strides);
    }
};

struct Divide {
    template<typename T>
    METAL_FUNC T operator()(T a, T b) { return a / b; }
    METAL_FUNC float  operator()(float  a, float  b) { return fast::divide(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return divide(a, b); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a, bfloat  b) { return static_cast<bfloat>(fast::divide(a, b)); }
    #endif
};

struct Exp {
    template<typename T>
    METAL_FUNC T operator()(T a) { return fast::exp(a); }
    METAL_FUNC float  operator()(float  a) { return fast::exp(a); }
    METAL_FUNC half   operator()(half   a) { return exp(a); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a) { return static_cast<bfloat>(fast::exp(a)); }
    #endif
};


// Keeps track of the index of the value in the reduction operation (argmin, argmax, etc.)
// and the value itself. The index is also used to break ties in the reduction operation.
template <typename T>
struct indexed {
    uint i;
    T val;

    constexpr indexed<T>() threadgroup = default;
};

template <typename T>
struct is_indexed_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_indexed_t = is_indexed_type<T>::value;

template <typename T>
struct is_indexed_type<indexed<T>> {
    static constant constexpr bool value = true;
};

template <typename T>
constexpr constant bool not_indexed_t = !is_indexed_t<T>;

template<typename T>
constexpr METAL_FUNC bool operator<(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
constexpr METAL_FUNC bool operator>(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
struct _numeric_limits_impl<indexed<T>> {
    static constexpr METAL_FUNC indexed<T> lowest() {
        return indexed<T>{ 0, numeric_limits<T>::lowest() };
    }

    static constexpr METAL_FUNC indexed<T> max() {
        return indexed<T>{ 0, numeric_limits<T>::max() };
    }
};

#if __METAL_VERSION__ >= 220
METAL_FUNC int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(simd_shuffle_down(as_type<uint2>(data), delta));
}
#endif


#if defined(__HAVE_BFLOAT__)
// Metal does not have simd_shuffle_down for bfloat16
METAL_FUNC bfloat simd_shuffle_down(bfloat value, ushort delta) {
    return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(value), delta));
}
#endif

template <typename T>
METAL_FUNC indexed<T> simd_shuffle_down(indexed<T> iv, ushort delta) {
    return indexed<T> {
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    };
}

template<typename T>
struct Sum {
    static constexpr METAL_FUNC T init() {
        return 0;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_sum(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a + b;
    }
};

template<typename T>
struct Mul {
    static constexpr METAL_FUNC T init() {
        return 1;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_product(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a * b;
    }
};

template<typename T>
struct Min {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::max();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_min(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a < b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::min(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return min(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return min(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return min(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return min(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::min(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template<typename T>
struct Max {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_max(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a > b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::max(a, b); }
    METAL_FUNC half operator()(half a, half b) { return max(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return max(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return max(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return max(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::max(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template <typename T>
constexpr constant bool is_simd_t = __is_valid_simdgroup_type<T>::value;

template <typename T, typename _E = void>
struct is_valid_simd_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_valid_simd_t = is_valid_simd_type<T>::value;

template <typename T>
struct is_valid_simd_type<T, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <typename T>
struct is_valid_simd_type<indexed<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

#if __METAL_VERSION__ >= 220
template <>
struct is_valid_simd_type<int64_t> {
    static constant constexpr bool value = true;
};
#endif

#if defined(__HAVE_BFLOAT__)
template <>
struct is_valid_simd_type<bfloat> {
    static constant constexpr bool value = true;
};
#endif

template <typename T, typename _E = void>
struct is_simd_op {
    static constant constexpr bool value = false;
};
template <typename T>
struct is_simd_op<Sum<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Mul<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Min<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Max<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Helper struct for applying operators.
// The overloaded operator() function is used to apply an operator to two values.
template<typename OP, typename T>
struct operation;

// Specialization for scalar values.
template<typename OP, typename T>
struct operation {
    OP op;

    METAL_FUNC T operator()(T a, T b) {
        return op(a, b);
    }
};

// Specialization for indexed values.
template<typename OP, typename T>
struct operation<OP, indexed<T>> {
    OP op;

    METAL_FUNC indexed<T> operator()(indexed<T> a, indexed<T> b) {
        return op(a, b);
    }
    METAL_FUNC indexed<T> operator()(indexed<T> a, T b, uint idx) {
        return this->operator()(a, indexed<T>{ idx, b });
    }
};

// Load elements from global memory into shared memory.
// Handles both indexed and non-indexed types by using operate.
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT,
    typename _E = void
>
struct loader;

template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT
>
struct loader<T, R, OP, BLOCKSIZE, Indexer, IndexT, typename metal::enable_if_t<not_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        Indexer indexer,
        constant IndexT &src_numel,
        constant IndexT &el_per_block,
        device const T *src,
        const IndexT offset,
        const uint tid
    ) {
        const IndexT idx = tid + offset;
        const IndexT stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (IndexT i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[indexer(i)]);
        }
        return value;
    }
};

// Indexed
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT
>
struct loader<T, R, OP, BLOCKSIZE, Indexer, IndexT, typename metal::enable_if_t<is_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        Indexer indexer,
        constant IndexT &src_numel,
        constant IndexT &el_per_block,
        device const T *src,
        const IndexT offset,
        const uint tid
    ) {
        const IndexT idx = tid + offset;
        const IndexT stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (IndexT i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[indexer(i)], i % indexer.last_dim);
        }
        return value;
    }
};

template<
    typename OP,
    ushort BLOCKSIZE,
    typename T,
    typename _E = void
>
struct simdgroup_reducer;

// Specialization for built-in simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    METAL_FUNC T operator()(T value) {
        return OP::simd_op(value);
    }
};

// Specialization for custom (non-built-in) simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    operation<OP, T> op;

    METAL_FUNC T operator()(T value) {
        if (BLOCKSIZE >= 32) value = op(value, simd_shuffle_down(value, 16));
        if (BLOCKSIZE >= 16) value = op(value, simd_shuffle_down(value,  8));
        if (BLOCKSIZE >=  8) value = op(value, simd_shuffle_down(value,  4));
        if (BLOCKSIZE >=  4) value = op(value, simd_shuffle_down(value,  2));
        if (BLOCKSIZE >=  2) value = op(value, simd_shuffle_down(value,  1));
        return value;
    }
};

template<typename T, typename OP, ushort BLOCKSIZE>
struct block_reducer {
    simdgroup_reducer<OP, BLOCKSIZE, T> simd_reduce;
    operation<OP, T> operate;
    threadgroup T *shared;

    block_reducer(threadgroup T shared[BLOCKSIZE]) {
        this->shared = shared;
    }

    METAL_FUNC T operator()(T value, const uint tid) {
        if (BLOCKSIZE >= 64) {
            // Only store in threadgroup shared memory if needed.
            shared[tid] = value;
            // Threadgroup barrier is needed to ensure that all threads have written to shared memory
            threadgroup_barrier(mem_flags::mem_none);
        }

        #pragma clang loop unroll(full)
        for (ushort s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
            if (tid < s) shared[tid] = operate(shared[tid], shared[tid + s]);
            threadgroup_barrier(mem_flags::mem_none);
        }
        if (tid < 32) {
            // Last shared memory reduce can be done without tid < s check.
            if (BLOCKSIZE >= 64) {
                value = operate(shared[tid], shared[tid + 32]);
                simdgroup_barrier(mem_flags::mem_none);
            }
            // Remaining 32 threads can be reduced with simdgroup_reduce.
            value = simd_reduce(value);
        }
        return value;
    }
};

template<typename T, typename _E = void>
struct storer;

template<typename T>
struct storer<T, typename metal::enable_if_t<not_indexed_t<T>>> {
    device T *dst;
    const uint tid;
    const uint dst_id;

    METAL_FUNC void operator()(T value) {
        if (tid == 0) {
            dst[dst_id] = value;
        }
    }
};

template<typename T>
struct storer<T, typename metal::enable_if_t<is_indexed_t<T>>> {
    device uint *dst;
    const uint tid;
    const uint dst_id;

    METAL_FUNC void operator()(T value) {
        if (tid == 0) {
            dst[dst_id] = value.i;
        }
    }
};

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT = typename Indexer::IndexT
>
METAL_FUNC void reduce(
    Indexer indexer,
    constant IndexT &src_numel,
    constant IndexT &el_per_block,
    device const T *src,
    device R *dst,
    threadgroup R shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    loader<T, R, OP, BLOCKSIZE, Indexer, IndexT> load;
    block_reducer<R, OP, BLOCKSIZE> reduce(shared);
    storer<R> store { dst, tid, dst_id };

    // Calculate offset for the threadgroup of current thread
    const IndexT offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    auto value = load(OP::init(), indexer, src_numel, el_per_block, src, offset, tid);

    // Complete reduction
    R result = reduce(value, tid);

    store(result);
}

#define reduce_switch(CASE_MACRO, OP, T, R, INDEXER)    \
    switch (max_shared_mem<T>(block_dim)) {             \
        CASE_MACRO(OP, T, R, 1024, INDEXER)             \
        CASE_MACRO(OP, T, R,  512, INDEXER)             \
        CASE_MACRO(OP, T, R,  256, INDEXER)             \
        CASE_MACRO(OP, T, R,  128, INDEXER)             \
        CASE_MACRO(OP, T, R,   64, INDEXER)             \
        CASE_MACRO(OP, T, R,   32, INDEXER)             \
        CASE_MACRO(OP, T, R,   16, INDEXER)             \
        CASE_MACRO(OP, T, R,    8, INDEXER)             \
        CASE_MACRO(OP, T, R,    4, INDEXER)             \
        CASE_MACRO(OP, T, R,    2, INDEXER)             \
        CASE_MACRO(OP, T, R,    1, INDEXER)             \
    }

#define reduce_case(OP, T, R, N, INDEXER)                               \
case N: {                                                               \
    threadgroup T shared[N];                                            \
    reduce<T, R, OP<R>, N>(                                             \
        INDEXER, src_numel, el_per_block, src, dst, shared, tid, dst_id \
    );                                                                  \
    break;                                                              \
}

#define impl_reduce_inner(OP, NAME, T)              \
kernel void NAME(                                   \
    constant uint &src_numel,                       \
    constant uint &num_dims,                        \
    constant uint *dims,                            \
    constant uint &el_per_block,                    \
    device const T *src,                            \
    device T *dst,                                  \
    uint tid [[ thread_index_in_threadgroup ]],     \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]]    \
) {                                                 \
    indexer_t<uint, false> indexer;                 \
    reduce_switch(reduce_case, OP, T, T, indexer)   \
}

#define impl_reduce_strided(OP, NAME, T)            \
kernel void NAME##_strided(                         \
    constant uint &src_numel,                       \
    constant uint &num_dims,                        \
    constant uint *dims,                            \
    constant uint *strides,                         \
    constant uint &el_per_block,                    \
    device const T *src,                            \
    device T *dst,                                  \
    uint tid [[ thread_index_in_threadgroup ]],     \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]]    \
) {                                                 \
    indexer_t<uint, true> indexer {                 \
        num_dims, dims, strides, dims[num_dims - 1] \
    };                                              \
    reduce_switch(reduce_case, OP, T, T, indexer)   \
}

#define impl_reduce(OP, NAME, T)                    \
impl_reduce_inner(OP, NAME, T)                      \
impl_reduce_strided(OP, NAME, T)

template<
    typename T,
    typename ReductionOp,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT = typename Indexer::IndexT
>
METAL_FUNC void reduce(
    Indexer indexer,
    constant IndexT &src_numel,
    constant IndexT &el_per_block,
    device const T *src,
    device uint *dst,
    threadgroup indexed<T> shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using I = indexed<T>;
    loader<T, I, ReductionOp, BLOCKSIZE, Indexer, IndexT> load;
    block_reducer<I, ReductionOp, BLOCKSIZE> reduce(shared);
    storer<I> store { dst, tid, dst_id };

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    auto value = load(
        ReductionOp::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );

    // Complete reduction
    I result = reduce(value, tid);

    // Return index of reduce result
    store(result);
}

#define arg_reduce_case(OP, T, R, N, INDEXER)           \
case N: {                                               \
    using I = indexed<R>;                               \
    threadgroup I shared[N];                            \
    reduce<T, OP<I>, N>(                                \
        indexer,                                        \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_arg_reduce_inner(OP, NAME, T)              \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant uint *dims,                                \
    constant uint &el_per_block,                        \
    device const T *src,                                \
    device uint *dst,                                   \
    uint tid [[ thread_index_in_threadgroup ]],         \
    uint dst_id [[ threadgroup_position_in_grid ]],     \
    uint block_dim [[ threads_per_threadgroup ]]        \
) {                                                     \
    indexer_t<uint, false> indexer {                    \
        dims[num_dims - 1]                              \
    };                                                  \
    reduce_switch(arg_reduce_case, OP, T, T, indexer)   \
}                                                       \

#define impl_arg_reduce_strided(OP, NAME, T)            \
kernel void NAME##_strided(                             \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant uint *dims,                                \
    constant uint *strides,                             \
    constant uint &el_per_block,                        \
    device const T *src,                                \
    device uint *dst,                                   \
    uint tid [[ thread_index_in_threadgroup ]],         \
    uint dst_id [[ threadgroup_position_in_grid ]],     \
    uint block_dim [[ threads_per_threadgroup ]]        \
) {                                                     \
    indexer_t<uint, true> indexer {                     \
        num_dims, dims, strides, dims[num_dims - 1]     \
    };                                                  \
    reduce_switch(arg_reduce_case, OP, T, T, indexer)   \
}

#define impl_arg_reduce(OP, NAME, T)                    \
impl_arg_reduce_inner(OP, NAME, T)                      \
impl_arg_reduce_strided(OP, NAME, T)

// Contains the intermediate results for the online softmax calculation.
// m: max
// d: sum of the exponentials
template <typename T>
struct MD {
    T m;
    float d;

    constexpr MD<T>() = default;
    constexpr MD<T>() threadgroup = default;
};

// Enable operations for softmax MD
template<typename OP, typename T>
struct operation<OP, MD<T>> {
    OP op;

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        return op(a, b);
    }

    METAL_FUNC MD<T> operator()(MD<T> a, T b) {
        return this->operator()(a, MD<T>{ b, static_cast<T>(1.0) });
    }
};

template <typename T>
METAL_FUNC MD<T> simd_shuffle_down(MD<T> md, ushort delta) {
    return MD<T> {
        simd_shuffle_down(md.m, delta),
        simd_shuffle_down(md.d, delta)
    };
}

// Enable simd_shuffle_down for softmax MD
template <typename T>
struct is_valid_simd_type<MD<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template<typename T>
struct MDReduceOp {
    Exp fast_exp;

    static constexpr METAL_FUNC MD<T> init() {
        return MD<T>{ numeric_limits<T>::lowest(), 0 };
    }

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d * fast_exp(smaller_m.m - bigger_m.m);
        res.m = bigger_m.m;
        return res;
    }
};

template<typename T, ushort BLOCKSIZE>
struct finalize_softmax {
    Divide fast_divide;
    Exp fast_exp;

    METAL_FUNC void operator()(
        device const T *src,
        device T *dst,
        threadgroup MD<T> &md_total,
        const uint thread_id,
        const uint stop_idx
    ) {
        const float d_total_inverse = fast_divide(1.0, md_total.d);
        for (uint idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
            dst[idx] = static_cast<T>(fast_exp(src[idx] - md_total.m) * d_total_inverse);
        }
    }
};


// Welford's algorithm approach for an online softmax implementation.
// Same as the Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867.pdf
template<typename T, ushort BLOCKSIZE>
METAL_FUNC void softmax(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    threadgroup MD<T> shared[BLOCKSIZE],
    threadgroup MD<T> &md_total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using MDReduceOp = MDReduceOp<T>;
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    loader<T, MD<T>, MDReduceOp, BLOCKSIZE, Indexer, uint> load;
    block_reducer<MD<T>, MDReduceOp, BLOCKSIZE> reduce(shared);
    finalize_softmax<T, BLOCKSIZE> softmax_finalize;

    // Calculate offset for the threadgroup of current thread;
    const uint offset = dst_id * el_per_block;

    // Calculate partial result for current thread
    MD<T> md_partial = MD<T> { numeric_limits<T>::lowest(), 0 };
    md_partial = load(
        md_partial,
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );

    // Reduce in shared memory
    MD<T> md = reduce(md_partial, tid);

    if (tid == 0) md_total = md;
    threadgroup_barrier(mem_flags::mem_none);

    // Finalize softmax
    const uint thread_id = tid + offset;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    softmax_finalize(src, dst, md_total, thread_id, stop_idx);
}

#define softmax_case(T, N)                              \
case N: {                                               \
    threadgroup MD<T> shared[N];                        \
    threadgroup MD<T> md_total;                         \
    softmax<T, N>(                                      \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        md_total,                                       \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_softmax(NAME, T)                           \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &el_per_block,                        \
    device const T *src,                                \
    device T *dst,                                      \
    uint tid [[ thread_index_in_threadgroup ]],         \
    uint dst_id [[ threadgroup_position_in_grid ]],     \
    uint block_dim [[ threads_per_threadgroup ]]        \
) {                                                     \
    switch (max_shared_mem<T>(block_dim)) {             \
        softmax_case(T, 1024);                          \
        softmax_case(T,  512);                          \
        softmax_case(T,  256);                          \
        softmax_case(T,  128);                          \
        softmax_case(T,   64);                          \
        softmax_case(T,   32);                          \
        softmax_case(T,   16);                          \
        softmax_case(T,    8);                          \
        softmax_case(T,    4);                          \
        softmax_case(T,    2);                          \
        softmax_case(T,    1);                          \
    }                                                   \
}


template<typename T>
METAL_FUNC void rmsnorm(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    constant float &eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp = 0;
    while (idx < stop_idx) {
        tmp = tmp + float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(shared_memory[0] / float(el_to_sum_per_block) + eps);
    float inv_norm = 1.0f / norm;
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = float(src[idx]) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

template<typename T>
struct RMS {
    uint count;
    T mean;

    constexpr RMS<T>() = default;
    constexpr RMS<T>() threadgroup = default;
};

template<typename T>
struct RMSLoadOp {
    static constexpr METAL_FUNC RMS<T> init() {
        return { 0, 0 };
    }

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        a.mean += (b.mean * b.mean);
        a.count += 1;
        return a;
    }
};

template<typename T>
struct RMSReduceOp {
    static constexpr METAL_FUNC RMS<T> init() {
        return { 0, 0 };
    }

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        uint new_count = a.count + b.count;
        uint nb_over_n = b.count / new_count;
        T delta = b.mean - a.mean;
        //a.mean += delta * nb_over_n;
        a.mean += b.mean + delta * delta * a.count * nb_over_n;
        // *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
        a.count = new_count;
        return a;
    }
};

template<typename OP, typename T>
struct operation<OP, RMS<T>> {
    OP op;

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        return op(a, b);
    }

    template<typename U>
    METAL_FUNC RMS<T> operator()(RMS<T> a, U b) {
        return this->operator()(a, RMS<T>{ 0, static_cast<T>(b) });
    }
};

template <typename T>
METAL_FUNC RMS<T> simd_shuffle_down(RMS<T> rms, ushort delta) {
    return RMS<T> {
        simd_shuffle_down(rms.count, delta),
        simd_shuffle_down(rms.mean, delta)
    };
}

template <typename T>
struct is_valid_simd_type<RMS<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Kernels
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void rms_norm(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    constant float &eps,
    threadgroup RMS<float> shared[BLOCKSIZE],
    threadgroup float &total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, RMS<float>, RMSLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<RMS<float>, RMSReduceOp<float>, BLOCKSIZE> reduce(shared);

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    // Load with reduction from global memory into shared memory
    RMS<float> value = load(
        RMSLoadOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    RMS<float> result = RMS<float> { value.count, static_cast<float>(value.mean) };

    // Complete reduction
    result = reduce(result, tid);
    if (tid == 0) {
        total = rsqrt(fast_divide(result.mean, float(el_per_block)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (alpha == nullptr) {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            dst[i] = src[i] * static_cast<T>(total);
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            T val = src[i] * static_cast<T>(total);
            val *= alpha[i - offset];
            dst[i] = val;
        }
    }
}


#define rms_norm_case(T, N)                             \
case N: {                                               \
    threadgroup RMS<float> shared[N];                   \
    threadgroup float total;                            \
    rms_norm<T, N>(                                     \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        alpha,                                          \
        eps,                                            \
        shared,                                         \
        total,                                          \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_rms_norm(NAME, T)                          \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &el_per_block,                        \
    device const T *src,                                \
    device T *dst,                                      \
    device const T *alpha,                              \
    constant float &eps,                                \
    uint tid [[ thread_index_in_threadgroup ]],         \
    uint dst_id [[ threadgroup_position_in_grid ]],     \
    uint block_dim [[ threads_per_threadgroup ]]        \
) {                                                     \
    switch (max_shared_mem<float>(block_dim)) {         \
        rms_norm_case(T, 1024);                         \
        rms_norm_case(T,  512);                         \
        rms_norm_case(T,  256);                         \
        rms_norm_case(T,  128);                         \
        rms_norm_case(T,   64);                         \
        rms_norm_case(T,   32);                         \
        rms_norm_case(T,   16);                         \
        rms_norm_case(T,    8);                         \
        rms_norm_case(T,    4);                         \
        rms_norm_case(T,    2);                         \
        rms_norm_case(T,    1);                         \
    }                                                   \
}

/// Fused RMSNorm + residual add: dst[i] = rms_norm(src[i]) * alpha[i] + residual[i]
///
/// Reduces one dispatch vs the standard (rms_norm → separate add).
/// Used in Gemma4 decoder layers after attention/MLP output:
///   xs = post_norm(attn_out) + residual
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void rms_norm_add(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    device const T *residual,
    constant float &eps,
    threadgroup RMS<float> shared[BLOCKSIZE],
    threadgroup float &total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, RMS<float>, RMSLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<RMS<float>, RMSReduceOp<float>, BLOCKSIZE> reduce(shared);

    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    RMS<float> value = load(
        RMSLoadOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    RMS<float> result = RMS<float> { value.count, static_cast<float>(value.mean) };
    result = reduce(result, tid);
    if (tid == 0) {
        total = rsqrt(fast_divide(result.mean, float(el_per_block)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
        T val = src[i] * static_cast<T>(total);
        val *= alpha[i - offset];
        dst[i] = val + residual[i];
    }
}

#define rms_norm_add_case(T, N)                             \
case N: {                                               \
    threadgroup RMS<float> shared[N];                   \
    threadgroup float total;                            \
    rms_norm_add<T, N>(                                 \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        alpha,                                          \
        residual,                                       \
        eps,                                            \
        shared,                                         \
        total,                                          \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_rms_norm_add(NAME, T)                          \
kernel void NAME(                                           \
    constant uint &src_numel,                               \
    constant uint &el_per_block,                            \
    device const T *src,                                    \
    device T *dst,                                          \
    device const T *alpha,                                  \
    device const T *residual,                               \
    constant float &eps,                                    \
    uint tid [[ thread_index_in_threadgroup ]],             \
    uint dst_id [[ threadgroup_position_in_grid ]],         \
    uint block_dim [[ threads_per_threadgroup ]]            \
) {                                                         \
    switch (max_shared_mem<float>(block_dim)) {             \
        rms_norm_add_case(T, 1024);                         \
        rms_norm_add_case(T,  512);                         \
        rms_norm_add_case(T,  256);                         \
        rms_norm_add_case(T,  128);                         \
        rms_norm_add_case(T,   64);                         \
        rms_norm_add_case(T,   32);                         \
        rms_norm_add_case(T,   16);                         \
        rms_norm_add_case(T,    8);                         \
        rms_norm_add_case(T,    4);                         \
        rms_norm_add_case(T,    2);                         \
        rms_norm_add_case(T,    1);                         \
    }                                                       \
}

/// Fused RMSNorm + residual add + scalar multiply:
///   dst[i] = (rms_norm(src[i]) * alpha[i] + residual[i]) * scale
///
/// Saves two dispatches vs the three-dispatch sequence (rms_norm + add + mul).
/// Used in Gemma4 decoder layers for the PLI path:
///   xs = (post_pli_norm(pli_out) + residual) * layer_scalar
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void rms_norm_add_scale(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    device const T *residual,
    constant float &eps,
    constant float &scale,
    threadgroup RMS<float> shared[BLOCKSIZE],
    threadgroup float &total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, RMS<float>, RMSLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<RMS<float>, RMSReduceOp<float>, BLOCKSIZE> reduce(shared);

    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    RMS<float> value = load(
        RMSLoadOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    RMS<float> result = RMS<float> { value.count, static_cast<float>(value.mean) };
    result = reduce(result, tid);
    if (tid == 0) {
        total = rsqrt(fast_divide(result.mean, float(el_per_block)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
        T val = src[i] * static_cast<T>(total);
        val *= alpha[i - offset];
        val = (val + residual[i]) * static_cast<T>(scale);
        dst[i] = val;
    }
}

#define rms_norm_add_scale_case(T, N)                           \
case N: {                                                       \
    threadgroup RMS<float> shared[N];                           \
    threadgroup float total;                                    \
    rms_norm_add_scale<T, N>(                                   \
        src_numel,                                              \
        el_per_block,                                           \
        src,                                                    \
        dst,                                                    \
        alpha,                                                  \
        residual,                                               \
        eps,                                                    \
        scale,                                                  \
        shared,                                                 \
        total,                                                  \
        tid,                                                    \
        dst_id);                                                \
    break;                                                      \
}

#define impl_rms_norm_add_scale(NAME, T)                        \
kernel void NAME(                                               \
    constant uint &src_numel,                                   \
    constant uint &el_per_block,                                \
    device const T *src,                                        \
    device T *dst,                                              \
    device const T *alpha,                                      \
    device const T *residual,                                   \
    constant float &eps,                                        \
    constant float &scale,                                      \
    uint tid [[ thread_index_in_threadgroup ]],                 \
    uint dst_id [[ threadgroup_position_in_grid ]],             \
    uint block_dim [[ threads_per_threadgroup ]]                \
) {                                                             \
    switch (max_shared_mem<float>(block_dim)) {                 \
        rms_norm_add_scale_case(T, 1024);                       \
        rms_norm_add_scale_case(T,  512);                       \
        rms_norm_add_scale_case(T,  256);                       \
        rms_norm_add_scale_case(T,  128);                       \
        rms_norm_add_scale_case(T,   64);                       \
        rms_norm_add_scale_case(T,   32);                       \
        rms_norm_add_scale_case(T,   16);                       \
        rms_norm_add_scale_case(T,    8);                       \
        rms_norm_add_scale_case(T,    4);                       \
        rms_norm_add_scale_case(T,    2);                       \
        rms_norm_add_scale_case(T,    1);                       \
    }                                                           \
}

template<typename T>
struct LayerNormValue {
    uint count;
    T mean;
    T m2;

    constexpr LayerNormValue<T>() = default;
    constexpr LayerNormValue<T>() threadgroup = default;
};

template<typename T>
struct LNLoadOp {
    static constexpr METAL_FUNC LayerNormValue<T> init() {
        return { 0, 0, 0 };
    }

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        a.count += 1;
        T delta1 = b.mean - a.mean;
        a.mean += delta1 / a.count;
        T delta2 = b.mean - a.mean;
        a.m2 += delta1 * delta2;
        return a;
    }
};

template<typename T>
struct LNReduceOp {
    static constexpr METAL_FUNC LayerNormValue<T> init() {
        return { 0, 0, 0 };
    }

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        if (b.count == 0) {
            return a;
        }
        uint new_count = a.count + b.count;
        T nb_over_n = b.count / T(new_count);
        T delta = b.mean - a.mean;
        a.mean += delta * nb_over_n;
        a.m2 += b.m2 + delta * delta * a.count * nb_over_n;
        a.count = new_count;
        return a;
    }
};

template<typename OP, typename T>
struct operation<OP, LayerNormValue<T>> {
    OP op;

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        return op(a, b);
    }

    template<typename U>
    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, U b) {
        return this->operator()(a, LayerNormValue<T>{ 0, static_cast<T>(b), static_cast<T>(b) });
    }
};

template <typename T>
METAL_FUNC LayerNormValue<T> simd_shuffle_down(LayerNormValue<T> lnv, ushort delta) {
    return LayerNormValue<T> {
        simd_shuffle_down(lnv.count, delta),
        simd_shuffle_down(lnv.mean, delta),
        simd_shuffle_down(lnv.m2, delta)
    };
}

template <typename T>
struct is_valid_simd_type<LayerNormValue<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Kernels
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void layer_norm(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    device const T *beta,
    constant float &eps,
    threadgroup LayerNormValue<float> shared[BLOCKSIZE],
    threadgroup float &mu,
    threadgroup float &sigma,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, LayerNormValue<float>, LNLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<LayerNormValue<float>, LNReduceOp<float>, BLOCKSIZE> reduce(shared);

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    // Load with reduction from global memory into shared memory
    LayerNormValue<float> value = load(
        LNReduceOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    LayerNormValue<float> result = LayerNormValue<float> { value.count, static_cast<float>(value.mean), static_cast<float>(value.m2) };

    // Complete reduction
    result = reduce(result, tid);
    if (tid == 0) {
        mu = result.mean;
        sigma = rsqrt(fast_divide(result.m2, float(result.count)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (alpha == nullptr || beta == nullptr) {
        if (alpha == nullptr) {
            #pragma clang loop unroll(full)
            for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
                T val = src[i];
                T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
                dst[i] = normalized + beta[i - offset];
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
                T val = src[i];
                T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
                dst[i] = normalized * alpha[i - offset];
            }
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            T val = src[i];
            T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
            dst[i] = static_cast<T>(fma(normalized, alpha[i - offset], beta[i - offset]));
        }
    }
}

#define layer_norm_case(T, N)                           \
case N: {                                               \
    threadgroup LayerNormValue<float> shared[N];        \
    threadgroup float mu;                               \
    threadgroup float sigma;                            \
    layer_norm<T, N>(                                   \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        alpha,                                          \
        beta,                                           \
        eps,                                            \
        shared,                                         \
        mu,                                             \
        sigma,                                          \
        tid,                                            \
        dst_id,                                         \
        lane_id);                                       \
    break;                                              \
}

#define impl_layer_norm(NAME, T)                        \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &el_per_block,                        \
    device const T *src,                                \
    device T *dst,                                      \
    device const T *alpha,                              \
    device const T *beta,                               \
    constant float &eps,                                \
    uint tid [[ thread_index_in_threadgroup ]],         \
    uint dst_id [[ threadgroup_position_in_grid ]],     \
    uint lane_id [[thread_index_in_simdgroup]],         \
    uint block_dim [[ threads_per_threadgroup ]]        \
) {                                                     \
    switch (max_shared_mem<float>(block_dim)) {         \
        layer_norm_case(T, 1024);                       \
        layer_norm_case(T,  512);                       \
        layer_norm_case(T,  256);                       \
        layer_norm_case(T,  128);                       \
        layer_norm_case(T,   64);                       \
        layer_norm_case(T,   32);                       \
        layer_norm_case(T,   16);                       \
        layer_norm_case(T,    8);                       \
        layer_norm_case(T,    4);                       \
        layer_norm_case(T,    2);                       \
        layer_norm_case(T,    1);                       \
    }                                                   \
}

template<typename T>
METAL_FUNC void ropei(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid
) {
    if (2 * tid >= bh * td) {
        return;
    }
    size_t rope_idx = tid % (td / 2);
    if (stride_b > 0) {
      size_t b_idx = (2 * tid) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s;
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c;
}

template<typename T>
METAL_FUNC void rope(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= bh * td) {
        return;
    }
    size_t i_bh = idx / (td / 2);
    size_t i_td = idx - (td / 2) * i_bh;
    size_t i_t = i_td / (d / 2);
    size_t i_d = i_td - (d / 2) * i_t;
    size_t i1 = i_bh * td + i_t * d + i_d;
    size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
METAL_FUNC void rope_thd(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= b * t * h * d) {
        return;
    }
    const size_t i_bth = idx / (d / 2);
    const size_t i_d = idx - (d / 2) * i_bth;
    const size_t i_t = (i_bth / h) % t;
    const size_t i1 = i_bth * d + i_d;
    const size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      const size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

#define ROPE(FN_NAME, FN_NAME_I, FN_NAME_THD, TYPENAME) \
kernel void FN_NAME_I( \
    constant size_t &bh, \
    constant size_t &td, \
    constant size_t &stride_b, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    ropei<TYPENAME>(bh, td, stride_b, src, cos, sin, dst, tid); \
}\
kernel void FN_NAME( \
    constant size_t &bh, \
    constant size_t &td, \
    constant size_t &d, \
    constant size_t &stride_b, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope<TYPENAME>(bh, td, d, stride_b, src, cos, sin, dst, idx); \
}\
kernel void FN_NAME_THD( \
    constant size_t &b, \
    constant size_t &t, \
    constant size_t &h, \
    constant size_t &d, \
    constant size_t &stride_b, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope_thd<TYPENAME>(b, t, h, d, stride_b, src, cos, sin, dst, idx); \
}\

impl_rms_norm(rmsnorm_f32, float)
impl_rms_norm(rmsnorm_f16, half)
impl_rms_norm_add(rmsnorm_add_f32, float)
impl_rms_norm_add(rmsnorm_add_f16, half)
impl_rms_norm_add_scale(rmsnorm_add_scale_f32, float)
impl_rms_norm_add_scale(rmsnorm_add_scale_f16, half)
impl_layer_norm(layernorm_f32, float)
impl_layer_norm(layernorm_f16, half)
ROPE(rope_f32, rope_i_f32, rope_thd_f32, float)
ROPE(rope_f16, rope_i_f16, rope_thd_f16, half)

impl_reduce(Sum, fast_sum_f32, float)
impl_reduce(Sum, fast_sum_u32, uint)
impl_reduce(Sum, fast_sum_f16, half)
impl_reduce(Sum, fast_sum_u8, uint8_t)

impl_reduce(Mul, fast_mul_f32, float)
impl_reduce(Mul, fast_mul_u32, uint)
impl_reduce(Mul, fast_mul_f16, half)
impl_reduce(Mul, fast_mul_u8, uint8_t)

impl_reduce(Max, fast_max_f32, float)
impl_reduce(Max, fast_max_u32, uint)
impl_reduce(Max, fast_max_f16, half)
impl_reduce(Max, fast_max_u8, uint8_t)

impl_reduce(Min, fast_min_f32, float)
impl_reduce(Min, fast_min_u32, uint)
impl_reduce(Min, fast_min_f16, half)
impl_reduce(Min, fast_min_u8, uint8_t)

impl_arg_reduce(Min, fast_argmin_f32, float)
impl_arg_reduce(Min, fast_argmin_f16, half)
impl_arg_reduce(Min, fast_argmin_u32, uint)
impl_arg_reduce(Min, fast_argmin_u8, uint8_t)

impl_arg_reduce(Max, fast_argmax_f32, float)
impl_arg_reduce(Max, fast_argmax_f16, half)
impl_arg_reduce(Max, fast_argmax_u32, uint)
impl_arg_reduce(Max, fast_argmax_u8, uint8_t)

impl_softmax(softmax_f32, float)
impl_softmax(softmax_f16, half)

#if __METAL_VERSION__ >= 220
impl_reduce(Sum, fast_sum_i64, int64_t)
impl_reduce(Mul, fast_mul_i64, int64_t)
impl_reduce(Min, fast_min_i64, int64_t)
impl_reduce(Max, fast_max_i64, int64_t)

impl_arg_reduce(Min, fast_argmin_i64, int64_t)
impl_arg_reduce(Max, fast_argmax_i64, int64_t)
#endif

#if defined(__HAVE_BFLOAT__)
impl_reduce(Sum, fast_sum_bf16, bfloat)
impl_reduce(Mul, fast_mul_bf16, bfloat)
impl_reduce(Max, fast_max_bf16, bfloat)
impl_reduce(Min, fast_min_bf16, bfloat)

impl_arg_reduce(Min, fast_argmin_bf16, bfloat)
impl_arg_reduce(Max, fast_argmax_bf16, bfloat)

impl_softmax(softmax_bf16, bfloat)

impl_rms_norm(rmsnorm_bf16, bfloat)
impl_layer_norm(layernorm_bf16, bfloat)
ROPE(rope_bf16, rope_i_bf16, rope_thd_bf16, bfloat)
impl_rms_norm_add(rmsnorm_add_bf16, bfloat)
impl_rms_norm_add_scale(rmsnorm_add_scale_bf16, bfloat)

/// RMSNorm with BF16 input and F32 output.
///
/// Eliminates the separate BF16→F32 to_dtype dispatch that is otherwise
/// needed before Q4K GEMV kernels (which require F32 activations).
///
/// For Gemma4 E4B decode: saves 1 dispatch per MLP layer (42 layers =
/// 42 fewer encoder roundtrips per decode step) by fusing the norm
/// output conversion into the normalization kernel.
///
/// API matches rmsnorm_bf16: same grid layout (one TG per sequence element),
/// same alpha (BF16 weight), just different output type (float).
#define rmsnorm_bf16i_f32o_case(N)                              \
case N: {                                                       \
    threadgroup RMS<float> shared[N];                           \
    threadgroup float total;                                     \
    const uint offset = dst_id * el_per_block;                  \
    const uint stop_idx = min(el_per_block + offset, src_numel);\
    const uint idx = tid + offset;                              \
    using Indexer = indexer_t<uint, false>;                     \
    Indexer indexer;                                            \
    Divide fast_divide;                                         \
    loader<bfloat, RMS<float>, RMSLoadOp<float>, N,            \
           Indexer, uint> load_fn;                              \
    block_reducer<RMS<float>, RMSReduceOp<float>, N> reduce(shared); \
    RMS<float> value = load_fn(RMSLoadOp<float>::init(),        \
        indexer, src_numel, el_per_block, src, offset, tid);   \
    RMS<float> result = {value.count, (float)value.mean};       \
    result = reduce(result, tid);                               \
    if (tid == 0) total = rsqrt(fast_divide(result.mean,        \
                                float(el_per_block)) + eps);   \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
    for (uint i = idx; i < stop_idx; i += N) {                 \
        float val = float(src[i]) * total;                      \
        val *= float(alpha[i - offset]);                        \
        dst[i] = val;                                           \
    }                                                           \
    break;                                                      \
}

[[host_name("rmsnorm_bf16i_f32o")]]
kernel void rmsnorm_bf16i_f32o(
    constant uint  &src_numel     [[ buffer(0) ]],
    constant uint  &el_per_block  [[ buffer(1) ]],
    device const bfloat *src      [[ buffer(2) ]],
    device       float  *dst      [[ buffer(3) ]],
    device const bfloat *alpha    [[ buffer(4) ]],
    constant float &eps           [[ buffer(5) ]],
    uint tid     [[ thread_index_in_threadgroup ]],
    uint dst_id  [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    switch (max_shared_mem<float>(block_dim)) {
        rmsnorm_bf16i_f32o_case(1024);
        rmsnorm_bf16i_f32o_case( 512);
        rmsnorm_bf16i_f32o_case( 256);
        rmsnorm_bf16i_f32o_case( 128);
        rmsnorm_bf16i_f32o_case(  64);
        rmsnorm_bf16i_f32o_case(  32);
        rmsnorm_bf16i_f32o_case(  16);
        rmsnorm_bf16i_f32o_case(   8);
        rmsnorm_bf16i_f32o_case(   4);
        rmsnorm_bf16i_f32o_case(   2);
        rmsnorm_bf16i_f32o_case(   1);
    }
}

/// Fused double-RMSNorm kernel: post_attn_norm_add + pre_ffn_norm_f32_out.
///
/// Eliminates one Metal dispatch per decoder layer during single-token decode.
///
/// Computation (per row):
///   bf16_out[i] = rms_norm(src[i]) * alpha1[i] + residual[i]   [BF16]
///   f32_out[i]  = rms_norm(bf16_out[i]) * alpha2[i]              [F32]
///
/// Uses the same loader/reducer infrastructure as rmsnorm_bf16i_f32o.
/// Two threadgroup passes separated by a device memory write to bf16_out.
/// No large intermediate shared memory — relies on device memory round-trip.
///
/// Grid: one threadgroup per row (for decode: 1 row = 1 threadgroup).
/// Block size: min(max_threads, next_pow2(el_per_block / 2)).
///
/// Buffer layout:
///   0: src_numel   (uint)       total elements = rows × el_per_block
///   1: el_per_block (uint)      hidden_size (e.g. 2048)
///   2: src         (bfloat*)    attn_out [BF16, read]
///   3: alpha1      (bfloat*)    post_attn_norm weight [BF16, read]
///   4: residual    (bfloat*)    residual [BF16, read]
///   5: bf16_out    (bfloat*)    xs = norm1(src)*w1 + residual [BF16, write+read]
///   6: alpha2      (bfloat*)    pre_ffn_norm weight [BF16, read]
///   7: f32_out     (float*)     norm2(xs)*w2 [F32, write]
///   8: eps         (float)      epsilon for both norms
/// Macro for rmsnorm_add_bf16i_f32o: two-pass double RMSNorm.
/// Uses a single shared array for both reductions (reused after barrier).
#define rmsnorm_add_bf16i_f32o_case(N)                                        \
case N: {                                                                     \
    threadgroup RMS<float> shared[N];                                         \
    threadgroup float total;                                                  \
    const uint offset = dst_id * el_per_block;                                \
    const uint stop   = min(el_per_block + offset, src_numel);               \
    using Indexer = indexer_t<uint, false>;                                   \
    Indexer indexer;                                                          \
    Divide fast_div;                                                          \
    /* --- Pass 1: RMS of src, write bf16_out --- */                          \
    {                                                                         \
        loader<bfloat, RMS<float>, RMSLoadOp<float>, N, Indexer, uint> load; \
        block_reducer<RMS<float>, RMSReduceOp<float>, N> reduce(shared);     \
        RMS<float> v = load(RMSLoadOp<float>::init(), indexer,               \
                            src_numel, el_per_block, src, offset, tid);       \
        RMS<float> r = {v.count, (float)v.mean};                              \
        r = reduce(r, tid);                                                   \
        if (tid == 0) total = rsqrt(fast_div(r.mean, float(el_per_block)) +  \
                                    eps);                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint i = tid + offset; i < stop; i += N) {                      \
            float val = float(src[i]) * total * float(alpha1[i - offset]);   \
            bf16_out[i] = bfloat(val + float(residual[i]));                  \
        }                                                                     \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_device);                               \
    /* --- Pass 2: RMS of bf16_out, write f32_out --- */                      \
    {                                                                         \
        loader<bfloat, RMS<float>, RMSLoadOp<float>, N, Indexer, uint> load; \
        block_reducer<RMS<float>, RMSReduceOp<float>, N> reduce(shared);     \
        RMS<float> v = load(RMSLoadOp<float>::init(), indexer,               \
                            src_numel, el_per_block, bf16_out, offset, tid); \
        RMS<float> r = {v.count, (float)v.mean};                              \
        r = reduce(r, tid);                                                   \
        if (tid == 0) total = rsqrt(fast_div(r.mean, float(el_per_block)) +  \
                                    eps);                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
        for (uint i = tid + offset; i < stop; i += N) {                      \
            f32_out[i] = float(bf16_out[i]) * total * float(alpha2[i - offset]);\
        }                                                                     \
    }                                                                         \
    break;                                                                    \
}

[[host_name("rmsnorm_add_bf16i_f32o")]]
kernel void rmsnorm_add_bf16i_f32o(
    constant uint  &src_numel     [[ buffer(0) ]],
    constant uint  &el_per_block  [[ buffer(1) ]],
    device const bfloat *src      [[ buffer(2) ]],
    device const bfloat *alpha1   [[ buffer(3) ]],
    device const bfloat *residual [[ buffer(4) ]],
    device       bfloat *bf16_out [[ buffer(5) ]],
    device const bfloat *alpha2   [[ buffer(6) ]],
    device       float  *f32_out  [[ buffer(7) ]],
    constant float      &eps      [[ buffer(8) ]],
    uint tid     [[ thread_index_in_threadgroup ]],
    uint dst_id  [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    switch (max_shared_mem<float>(block_dim)) {
        rmsnorm_add_bf16i_f32o_case(1024);
        rmsnorm_add_bf16i_f32o_case( 512);
        rmsnorm_add_bf16i_f32o_case( 256);
        rmsnorm_add_bf16i_f32o_case( 128);
        rmsnorm_add_bf16i_f32o_case(  64);
        rmsnorm_add_bf16i_f32o_case(  32);
        rmsnorm_add_bf16i_f32o_case(  16);
        rmsnorm_add_bf16i_f32o_case(   8);
        rmsnorm_add_bf16i_f32o_case(   4);
        rmsnorm_add_bf16i_f32o_case(   2);
        rmsnorm_add_bf16i_f32o_case(   1);
    }
}

/// Partial-RoPE kernel for BF16 tensors.
///
/// Applied to a single-token decode tensor with shape [1, n_heads, 1, head_dim].
/// Only the first `rotary_dim` features are rotated; the remaining
/// `head_dim - rotary_dim` features are copied unchanged.
///
/// This fuses: narrow(rotary) + contiguous + rope + slice_set(rot)
///             + narrow(pass)  + contiguous + slice_set(pass)
/// into a single kernel dispatch — eliminating 4 intermediate compute dispatches
/// per head (Q or K) per global attention layer.
///
/// Thread mapping: one thread per output element (n_heads × head_dim total).
///   tid = head_idx * head_dim + d_idx
///
/// For d_idx < rotary_dim: apply RoPE.
///   rot_pair = d_idx % (rotary_dim/2)
///   if d_idx < rotary_dim/2: dst = src_d * cos[rot_pair] - src_{d + rotary_dim/2} * sin[rot_pair]
///   else:                    dst = src_d * sin[rot_pair - rotary_dim/2]
///                                    + src_{d - rotary_dim/2} * cos[rot_pair - rotary_dim/2]
/// For d_idx >= rotary_dim: pass through unchanged.
///
/// Note: src tensor has shape [1, n_heads, 1, head_dim] with contiguous strides.
///       cos/sin shape: [1, rotary_dim/2].
[[host_name("partial_rope_bf16")]]
kernel void partial_rope_bf16(
    device const bfloat  * src,        // [n_heads, head_dim] (after squeezing batch/seq)
    device const bfloat  * cos,        // [rotary_dim/2]
    device const bfloat  * sin,        // [rotary_dim/2]
    device       bfloat  * dst,        // [n_heads, head_dim] output
    constant uint32_t    & n_heads,    // number of heads
    constant uint32_t    & head_dim,   // total head dimension
    constant uint32_t    & rotary_dim, // number of dims to rotate (< head_dim)
    uint2 tid [[thread_position_in_grid]]   // x = head_idx, y = d_idx
) {
    const uint h = tid.x;
    const uint d = tid.y;
    if (h >= n_heads || d >= head_dim) { return; }

    const uint src_off = h * head_dim + d;
    const uint dst_off = h * head_dim + d;

    if (d >= rotary_dim) {
        // Passthrough region: copy unchanged.
        dst[dst_off] = src[src_off];
    } else {
        // RoPE region.
        const uint rdim2 = rotary_dim / 2;  // half of rotary_dim (avoid "half" keyword)
        const uint rot_pair = d % rdim2;
        const float c = (float)cos[rot_pair];
        const float s = (float)sin[rot_pair];
        if (d < rdim2) {
            // First rdim2 elements: x*cos - y*sin
            const float x = (float)src[h * head_dim + d];
            const float y = (float)src[h * head_dim + d + rdim2];
            dst[dst_off] = (bfloat)(x * c - y * s);
        } else {
            // Second rdim2 elements: x*sin + y*cos
            const float x = (float)src[h * head_dim + d - rdim2];
            const float y = (float)src[h * head_dim + d];
            dst[dst_off] = (bfloat)(x * s + y * c);
        }
    }
}

/// Fused RMSNorm + partial-RoPE for BF16 tensors (single-token decode).
///
/// Combines `rms_norm` + `partial_rope_bf16` into one dispatch, saving 1
/// dispatch per head per global attention layer.
///
/// One threadgroup per head, with `head_dim` threads per threadgroup.
/// Requires `head_dim ≤ 1024` (threadgroup size limit).
///
/// src:        [n_heads, head_dim]  BF16 (flattened from [1, n_heads, 1, head_dim])
/// norm_weight:[head_dim]           BF16 (per-element scale from RMSNorm)
/// cos:        [rotary_dim/2]       BF16
/// sin:        [rotary_dim/2]       BF16
/// dst:        [n_heads, head_dim]  BF16 output (pre-allocated buffer)
[[host_name("rms_norm_partial_rope_bf16")]]
kernel void rms_norm_partial_rope_bf16(
    device const bfloat   * src,           // [n_heads, head_dim]
    device const bfloat   * norm_weight,   // [head_dim]
    device const bfloat   * cos,           // [rotary_dim/2]
    device const bfloat   * sin,           // [rotary_dim/2]
    device       bfloat   * dst,           // [n_heads, head_dim] output
    constant uint32_t     & n_heads,
    constant uint32_t     & head_dim,
    constant uint32_t     & rotary_dim,    // must be <= head_dim, even
    constant float        & eps,
    uint  tgpig  [[threadgroup_position_in_grid]],  // head_idx
    uint  tiisg  [[thread_index_in_threadgroup]],   // d_idx within head
    uint  tgsize [[threads_per_threadgroup]]
) {
    const uint h = tgpig;
    if (h >= n_heads) return;

    const uint d = tiisg;
    const device bfloat * src_head = src + h * head_dim;
    device bfloat       * dst_head = dst + h * head_dim;

    // --- Pass 1: compute sum of squares for this head ---
    threadgroup float shared_sos[1024];
    float local_sos = 0.0f;
    if (d < head_dim) {
        float v = float(src_head[d]);
        local_sos = v * v;
    }
    shared_sos[d < 1024 ? d : 0] = local_sos;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction.
    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (d < stride) shared_sos[d] += shared_sos[d + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(shared_sos[0] / float(head_dim) + eps);

    // --- Pass 2: normalize + scale + partial RoPE ---
    if (d >= head_dim) return;

    // Normalized + scaled value.
    const float normed = float(src_head[d]) * inv_rms * float(norm_weight[d]);

    if (d >= rotary_dim) {
        // Passthrough region: write normalized value directly.
        dst_head[d] = (bfloat)normed;
    } else {
        // RoPE region.
        const uint rdim2 = rotary_dim / 2;
        const uint rot_pair = d % rdim2;
        const float c = float(cos[rot_pair]);
        const float s = float(sin[rot_pair]);
        if (d < rdim2) {
            // First rdim2: x*cos - y*sin (x=normed, y=normed pair)
            const float y_normed = float(src_head[d + rdim2]) * inv_rms * float(norm_weight[d + rdim2]);
            dst_head[d] = (bfloat)(normed * c - y_normed * s);
        } else {
            // Second rdim2: x*sin + y*cos
            const float x_normed = float(src_head[d - rdim2]) * inv_rms * float(norm_weight[d - rdim2]);
            dst_head[d] = (bfloat)(x_normed * s + normed * c);
        }
    }
}

/// Fused Q+K+V rms_norm (+partial_rope for Q/K) in a single dispatch.
///
/// Extends rms_norm_partial_rope_qk_bf16 to also normalize V (no RoPE, identity weight).
/// Dispatches n_q_heads + 2*n_kv_heads threadgroups:
///   [0, n_q_heads): Q heads — norm + partial_rope with q_norm_weight
///   [n_q_heads, n_q_heads+n_kv_heads): K heads — norm + partial_rope with k_norm_weight
///   [n_q_heads+n_kv_heads, n_q_heads+2*n_kv_heads): V heads — norm only (rotary_dim=0)
///
/// q_src, k_src: [n_{q,kv}_heads, head_dim] BF16
/// v_src:        [n_kv_heads, head_dim]      BF16
/// q_norm_weight, k_norm_weight: [head_dim]  BF16
/// cos: [rotary_dim/2] BF16, sin: [rotary_dim/2] BF16
/// q_dst, k_dst, v_dst: output buffers
[[host_name("rms_norm_partial_rope_qkv_bf16")]]
kernel void rms_norm_partial_rope_qkv_bf16(
    device const bfloat   * q_src,
    device const bfloat   * k_src,
    device const bfloat   * v_src,
    device const bfloat   * q_norm_weight,
    device const bfloat   * k_norm_weight,
    device const bfloat   * cos,
    device const bfloat   * sin,
    device       bfloat   * q_dst,
    device       bfloat   * k_dst,
    device       bfloat   * v_dst,
    constant uint32_t     & n_q_heads,
    constant uint32_t     & n_kv_heads,
    constant uint32_t     & head_dim,
    constant uint32_t     & rotary_dim,
    constant float        & eps,
    uint  tgpig  [[threadgroup_position_in_grid]],
    uint  tiisg  [[thread_index_in_threadgroup]],
    uint  tgsize [[threads_per_threadgroup]]
) {
    // Classify this threadgroup: Q (0), K (1), or V (2)
    const bool is_q = tgpig < n_q_heads;
    const bool is_v = !is_q && tgpig >= n_q_heads + n_kv_heads;
    // local head index within the type
    const uint local_h = is_q ? tgpig
        : (is_v ? tgpig - n_q_heads - n_kv_heads : tgpig - n_q_heads);

    const device bfloat * src_head = is_q ? (q_src + local_h * head_dim)
        : is_v ? (v_src + local_h * head_dim)
        : (k_src + local_h * head_dim);
    device bfloat * dst_head = is_q ? (q_dst + local_h * head_dim)
        : is_v ? (v_dst + local_h * head_dim)
        : (k_dst + local_h * head_dim);
    // V uses all-ones weight (identity), so we hardcode weight=1 for V.
    const device bfloat * norm_weight = is_q ? q_norm_weight
        : is_v ? nullptr  // not used for V
        : k_norm_weight;

    const uint d = tiisg;

    // --- Pass 1: compute sum of squares ---
    threadgroup float shared_sos[1024];
    float local_sos = 0.0f;
    if (d < head_dim) {
        float v = float(src_head[d]);
        local_sos = v * v;
    }
    shared_sos[d < 1024 ? d : 0] = local_sos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (d < stride) shared_sos[d] += shared_sos[d + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(shared_sos[0] / float(head_dim) + eps);

    // --- Pass 2: normalize + optional RoPE ---
    if (d >= head_dim) return;

    const float w = (is_v || norm_weight == nullptr) ? 1.0f : float(norm_weight[d]);
    const float normed = float(src_head[d]) * inv_rms * w;

    if (is_v || d >= rotary_dim) {
        dst_head[d] = (bfloat)normed;
    } else {
        const uint rdim2 = rotary_dim / 2;
        const uint rot_pair = d % rdim2;
        const float c = float(cos[rot_pair]);
        const float s = float(sin[rot_pair]);
        if (d < rdim2) {
            const float w2 = (norm_weight == nullptr) ? 1.0f : float(norm_weight[d + rdim2]);
            const float y_normed = float(src_head[d + rdim2]) * inv_rms * w2;
            dst_head[d] = (bfloat)(normed * c - y_normed * s);
        } else {
            const float w2 = (norm_weight == nullptr) ? 1.0f : float(norm_weight[d - rdim2]);
            const float x_normed = float(src_head[d - rdim2]) * inv_rms * w2;
            dst_head[d] = (bfloat)(x_normed * s + normed * c);
        }
    }
}

/// Fused Q+K rms_norm + partial_rope in a single dispatch.
///
/// Combines two `rms_norm_partial_rope_bf16` dispatches (one for Q, one for K)
/// into a single dispatch with `n_q_heads + n_kv_heads` threadgroups.
/// Threadgroups [0, n_q_heads) process Q heads using q_norm_weight.
/// Threadgroups [n_q_heads, n_q_heads+n_kv_heads) process K heads using k_norm_weight.
/// Both use the same cos/sin tables and rotary_dim.
///
/// q_src:        [n_q_heads, head_dim]   BF16
/// k_src:        [n_kv_heads, head_dim]  BF16
/// q_norm_weight:[head_dim]              BF16
/// k_norm_weight:[head_dim]              BF16
/// cos:          [rotary_dim/2]          BF16
/// sin:          [rotary_dim/2]          BF16
/// q_dst:        [n_q_heads, head_dim]   BF16
/// k_dst:        [n_kv_heads, head_dim]  BF16
[[host_name("rms_norm_partial_rope_qk_bf16")]]
kernel void rms_norm_partial_rope_qk_bf16(
    device const bfloat   * q_src,          // [n_q_heads, head_dim]
    device const bfloat   * k_src,          // [n_kv_heads, head_dim]
    device const bfloat   * q_norm_weight,  // [head_dim]
    device const bfloat   * k_norm_weight,  // [head_dim]
    device const bfloat   * cos,            // [rotary_dim/2]
    device const bfloat   * sin,            // [rotary_dim/2]
    device       bfloat   * q_dst,          // [n_q_heads, head_dim]
    device       bfloat   * k_dst,          // [n_kv_heads, head_dim]
    constant uint32_t     & n_q_heads,
    constant uint32_t     & n_kv_heads,
    constant uint32_t     & head_dim,
    constant uint32_t     & rotary_dim,
    constant float        & eps,
    uint  tgpig  [[threadgroup_position_in_grid]],  // head_idx (0..n_q_heads+n_kv_heads)
    uint  tiisg  [[thread_index_in_threadgroup]],   // d_idx within head
    uint  tgsize [[threads_per_threadgroup]]
) {
    const bool is_q = tgpig < n_q_heads;
    const uint local_h = is_q ? tgpig : tgpig - n_q_heads;
    const device bfloat * src_head = is_q
        ? (q_src + local_h * head_dim)
        : (k_src + local_h * head_dim);
    device bfloat * dst_head = is_q
        ? (q_dst + local_h * head_dim)
        : (k_dst + local_h * head_dim);
    const device bfloat * norm_weight = is_q ? q_norm_weight : k_norm_weight;

    const uint d = tiisg;

    // --- Pass 1: compute sum of squares for this head ---
    threadgroup float shared_sos[1024];
    float local_sos = 0.0f;
    if (d < head_dim) {
        float v = float(src_head[d]);
        local_sos = v * v;
    }
    shared_sos[d < 1024 ? d : 0] = local_sos;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (d < stride) shared_sos[d] += shared_sos[d + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(shared_sos[0] / float(head_dim) + eps);

    // --- Pass 2: normalize + scale + partial RoPE ---
    if (d >= head_dim) return;

    const float normed = float(src_head[d]) * inv_rms * float(norm_weight[d]);

    if (d >= rotary_dim) {
        dst_head[d] = (bfloat)normed;
    } else {
        const uint rdim2 = rotary_dim / 2;
        const uint rot_pair = d % rdim2;
        const float c = float(cos[rot_pair]);
        const float s = float(sin[rot_pair]);
        if (d < rdim2) {
            const float y_normed = float(src_head[d + rdim2]) * inv_rms * float(norm_weight[d + rdim2]);
            dst_head[d] = (bfloat)(normed * c - y_normed * s);
        } else {
            const float x_normed = float(src_head[d - rdim2]) * inv_rms * float(norm_weight[d - rdim2]);
            dst_head[d] = (bfloat)(x_normed * s + normed * c);
        }
    }
}
#endif
