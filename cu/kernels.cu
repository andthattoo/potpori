// potpori's kernels: Nanbeige4.3-33B and its DFlash2 drafter.
//
// Conventions
// - Activations are f32, row-major, one row per token. Matrices are `[n_out, n_in]` row-major,
//   either the checkpoint's bf16 (`u16`) or its 4-bit NVFP4 form. Arithmetic is f32.
// - Every reduction has a fixed order that depends only on the row being computed, never on
//   how many tokens are in the launch. Feeding a token alone or inside a chunk therefore gives
//   the same bits, which is what lets a rejected draft be overwritten without drift.

typedef unsigned short u16;
typedef unsigned char u8;

#define GEMM_THREADS 128
#define GEMM_ROWS 8
#define HEAD_DIM 128
#define NORM_THREADS 256

static __device__ __forceinline__ float bf(u16 h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

template <int T>
static __device__ __forceinline__ void gemm_impl(float* out, const float* x, const u16* w,
                                                 int n_out, int n_in, int s, int out_stride,
                                                 int add) {
    __shared__ float part[GEMM_ROWS][T][GEMM_THREADS / 32];
    int r0 = blockIdx.x * GEMM_ROWS;
    // Rows past the end repeat the last one; their results are dropped.
    const u16* wr[GEMM_ROWS];
    for (int r = 0; r < GEMM_ROWS; r++) wr[r] = w + (size_t)min(r0 + r, n_out - 1) * n_in;
    for (int s0 = 0; s0 < s; s0 += T) {
        const float* xs[T];
        for (int t = 0; t < T; t++) xs[t] = s0 + t < s ? x + (size_t)(s0 + t) * n_in : nullptr;
        float acc[GEMM_ROWS][T];
        for (int r = 0; r < GEMM_ROWS; r++)
            for (int t = 0; t < T; t++) acc[r][t] = 0.f;
        for (int j = threadIdx.x; j < n_in; j += GEMM_THREADS) {
            float xv[T];
            for (int t = 0; t < T; t++) xv[t] = xs[t] ? xs[t][j] : 0.f;
            for (int r = 0; r < GEMM_ROWS; r++) {
                float wv = bf(wr[r][j]);
                for (int t = 0; t < T; t++) acc[r][t] += wv * xv[t];
            }
        }
        for (int r = 0; r < GEMM_ROWS; r++)
            for (int t = 0; t < T; t++) {
                float v = warp_sum(acc[r][t]);
                if ((threadIdx.x & 31) == 0) part[r][t][threadIdx.x >> 5] = v;
            }
        __syncthreads();
        // One thread per (row, token) adds the four warp sums and stores.
        int r = threadIdx.x / T, t = threadIdx.x % T;
        if (r < GEMM_ROWS && r0 + r < n_out && s0 + t < s) {
            float v = (part[r][t][0] + part[r][t][1]) + (part[r][t][2] + part[r][t][3]);
            float* o = out + (size_t)(s0 + t) * out_stride + r0 + r;
            *o = add ? *o + v : v;
        }
        __syncthreads();
    }
}


// Twice the value of a 4-bit float (sign, 2-bit exponent, 1-bit mantissa): 0, .5, 1, 1.5, 2,
// 3, 4, 6 and their negatives.
static __device__ __forceinline__ float fp4x2(unsigned int n) {
    int v = (int)((0xC8643210u >> ((n & 7u) * 4u)) & 15u);
    return (float)((n & 8u) ? -v : v);
}

// 1024 times the value of an 8-bit float (sign, 4-bit exponent biased by 7, 3-bit mantissa).
static __device__ __forceinline__ float fp8x1024(unsigned int b) {
    unsigned int e = (b >> 3) & 15u, m = b & 7u;
    int v = (int)(e ? (8u | m) << e : m << 1);
    return (float)((b & 0x80u) ? -v : v);
}

// The 4-bit matrix product. Weights are NVFP4: two 4-bit floats per byte, low nibble first,
// `[n_out, n_in / 2]`; one 8-bit float scale per 16 weights, `[n_out, n_in / 16]`; and one
// f32 scale for the matrix, folded with the decoders' constant factors into `gs`:
//   w[r, j] = fp4(w4[r, j]) * fp8(sc[r, j / 16]) * global,   gs = global / 2048.
// A thread owns every 128th run of eight columns: one 4-byte load per row, decoded in
// registers. The order of every sum is fixed by the row, as in the bf16 product.
template <int T>
static __device__ __forceinline__ void gemm4_impl(float* out, const float* x, const u8* w,
                                                  const u8* sc, float gs, int n_out, int n_in,
                                                  int s, int out_stride, int add) {
    __shared__ float part[GEMM_ROWS][T][GEMM_THREADS / 32];
    int r0 = blockIdx.x * GEMM_ROWS;
    const unsigned int* wr[GEMM_ROWS];
    const u8* sr[GEMM_ROWS];
    for (int r = 0; r < GEMM_ROWS; r++) {
        size_t row = (size_t)min(r0 + r, n_out - 1);
        wr[r] = (const unsigned int*)(w + row * (n_in / 2));
        sr[r] = sc + row * (n_in / 16);
    }
    for (int s0 = 0; s0 < s; s0 += T) {
        const float* xs[T];
        for (int t = 0; t < T; t++) xs[t] = s0 + t < s ? x + (size_t)(s0 + t) * n_in : nullptr;
        float acc[GEMM_ROWS][T];
        for (int r = 0; r < GEMM_ROWS; r++)
            for (int t = 0; t < T; t++) acc[r][t] = 0.f;
        for (int c = threadIdx.x; c < n_in / 8; c += GEMM_THREADS) {
            float xv[8][T];
            for (int j = 0; j < 8; j++)
                for (int t = 0; t < T; t++) xv[j][t] = xs[t] ? xs[t][c * 8 + j] : 0.f;
            for (int r = 0; r < GEMM_ROWS; r++) {
                unsigned int bits = wr[r][c];
                float scale = fp8x1024(sr[r][c >> 1]);
                float p[T];
                for (int t = 0; t < T; t++) p[t] = 0.f;
                for (int j = 0; j < 8; j++) {
                    float wv = fp4x2(bits >> (4 * j));
                    for (int t = 0; t < T; t++) p[t] += wv * xv[j][t];
                }
                for (int t = 0; t < T; t++) acc[r][t] += scale * p[t];
            }
        }
        for (int r = 0; r < GEMM_ROWS; r++)
            for (int t = 0; t < T; t++) {
                float v = warp_sum(acc[r][t]);
                if ((threadIdx.x & 31) == 0) part[r][t][threadIdx.x >> 5] = v;
            }
        __syncthreads();
        int r = threadIdx.x / T, t = threadIdx.x % T;
        if (r < GEMM_ROWS && r0 + r < n_out && s0 + t < s) {
            float v = ((part[r][t][0] + part[r][t][1]) + (part[r][t][2] + part[r][t][3])) * gs;
            float* o = out + (size_t)(s0 + t) * out_stride + r0 + r;
            *o = add ? *o + v : v;
        }
        __syncthreads();
    }
}

extern "C" {

// out[t, :] = table[ids[t], :].  grid (ceil(hidden/256), s), block 256.
__global__ void embed(float* out, const u16* table, const int* ids, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i < hidden) out[(size_t)t * hidden + i] = bf(table[(size_t)ids[t] * hidden + i]);
}

// out[r, :] = x[r, :] * w / sqrt(mean(x[r, :]^2) + eps).  grid (rows), block 256.
// `out` may be `x`.
__global__ void rmsnorm(float* out, const float* x, const float* w, float eps, int n) {
    __shared__ float part[NORM_THREADS / 32];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* outr = out + (size_t)blockIdx.x * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += NORM_THREADS) acc += xr[i] * xr[i];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    float total = 0.f;
    for (int k = 0; k < NORM_THREADS / 32; k++) total += part[k];
    float scale = 1.f / sqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += NORM_THREADS) outr[i] = xr[i] * scale * w[i];
}

// out[t, r] (+)= sum_j w[r, j] * x[t, j]: the kernel nearly all the time is spent in.
// One block computes GEMM_ROWS output rows for T tokens at a time. A thread owns every 128th
// column: it loads that column of x once per token, then walks the rows, so each weight costs
// one 2-byte load and T multiply-adds on registers. With one token the kernel is bound by how
// fast the weights can be read; with eight it does eight times the arithmetic on the same
// reads. Tokens past `s` in the last group are fed zeros and never stored.
// grid (ceil(n_out / GEMM_ROWS)), block 128.
#define GEMM_WRAPPER(NAME, T)                                                                 \
    __global__ void NAME(float* out, const float* x, const u16* w, int n_out, int n_in,      \
                         int s, int out_stride, int add) {                                    \
        gemm_impl<T>(out, x, w, n_out, n_in, s, out_stride, add);                             \
    }

GEMM_WRAPPER(gemm_t1, 1)
GEMM_WRAPPER(gemm_t2, 2)
GEMM_WRAPPER(gemm_t4, 4)
GEMM_WRAPPER(gemm_t8, 8)

// The same product with 4-bit weights; see gemm4_impl.  grid (ceil(n_out / GEMM_ROWS)),
// block 128.
#define GEMM4_WRAPPER(NAME, T)                                                                \
    __global__ void NAME(float* out, const float* x, const u8* w, const u8* sc, float gs,    \
                         int n_out, int n_in, int s, int out_stride, int add) {              \
        gemm4_impl<T>(out, x, w, sc, gs, n_out, n_in, s, out_stride, add);                    \
    }

GEMM4_WRAPPER(gemm4_t1, 1)
GEMM4_WRAPPER(gemm4_t2, 2)
GEMM4_WRAPPER(gemm4_t4, 4)
GEMM4_WRAPPER(gemm4_t8, 8)

// out[t, i] = silu(gu[t, i]) * gu[t, n + i]: gate and up halves of a SwiGLU MLP, computed by
// one gemm over the stacked matrices.  grid (ceil(n/256), s), block 256.
__global__ void silu_mul(float* out, const float* gu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gu + (size_t)blockIdx.y * 2 * n;
    float g = row[i];
    out[(size_t)blockIdx.y * n + i] = g / (1.f + expf(-g)) * row[n + i];
}

// Rotary position embedding in place on x[s, heads, 128], token t at position pos0 + t.
// `cs` is [positions, 2, 64]: cosines then sines.  grid (heads, s), block 64.
__global__ void rope(float* x, const float* cs, int heads, int pos0) {
    int i = threadIdx.x;
    float* v = x + ((size_t)blockIdx.y * heads + blockIdx.x) * HEAD_DIM;
    const float* row = cs + (size_t)(pos0 + blockIdx.y) * HEAD_DIM;
    float c = row[i], sn = row[HEAD_DIM / 2 + i];
    float a = v[i], b = v[i + HEAD_DIM / 2];
    v[i] = a * c - b * sn;
    v[i + HEAD_DIM / 2] = b * c + a * sn;
}

// Attention for s query tokens over a key/value cache laid out [position, kv_heads, 128].
// Token t attends to positions [0, len0 + t * step): step 1 is causal, step 0 lets every
// token see the same span. `scratch` is [s, heads, cap] floats.  grid (heads, s), block 128.
__global__ void attn(float* out, const float* q, const float* kc, const float* vc,
                     float* scratch, int heads, int kv_heads, int len0, int step, int cap,
                     float scale) {
    __shared__ float sq[HEAD_DIM];
    __shared__ float part[HEAD_DIM / 32];
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    int kvh = h / (heads / kv_heads);
    int len = len0 + t * step;
    size_t row = (size_t)t * heads + h;
    float* sc = scratch + row * cap;
    sq[d] = q[row * HEAD_DIM + d];
    __syncthreads();

    float best = -INFINITY;
    for (int j = d; j < len; j += HEAD_DIM) {
        const float* kj = kc + ((size_t)j * kv_heads + kvh) * HEAD_DIM;
        float dot = 0.f;
        for (int i = 0; i < HEAD_DIM; i++) dot += sq[i] * kj[i];
        dot *= scale;
        sc[j] = dot;
        best = fmaxf(best, dot);
    }
    best = warp_max(best);
    if ((d & 31) == 0) part[d >> 5] = best;
    __syncthreads();
    best = fmaxf(fmaxf(part[0], part[1]), fmaxf(part[2], part[3]));
    __syncthreads();

    float sum = 0.f;
    for (int j = d; j < len; j += HEAD_DIM) {
        float e = expf(sc[j] - best);
        sc[j] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    if ((d & 31) == 0) part[d >> 5] = sum;
    __syncthreads();
    sum = (part[0] + part[1]) + (part[2] + part[3]);

    float o = 0.f;
    const float* vd = vc + (size_t)kvh * HEAD_DIM + d;
    size_t stride = (size_t)kv_heads * HEAD_DIM;
    for (int j = 0; j < len; j++) o += sc[j] * vd[(size_t)j * stride];
    out[row * HEAD_DIM + d] = o / sum;
}

// The drafter's grouped two-tap convolution along a block of tokens:
//   y[p, c] = (base[0, c] + coef[p, 0, g]) * x[p, c] + (base[1, c] + coef[p, 1, g]) * x[p-1, c]
// with g = c / group and the second term dropped at the first row. A sublayer has two such
// convolutions, one on its input (side 0) and one on its output (side 1): `coef` is
// [rows, 2 sides, 2 taps, groups] and `base` is [2 sides, 2 taps, hidden]. `add` accumulates
// y into `out`.  grid (ceil(hidden/256), rows), block 256.
__global__ void conv2(float* out, const float* x, const float* coef, const float* base,
                      int hidden, int group, int side, int add) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    int p = blockIdx.y;
    int groups = hidden / group, g = c / group;
    const float* cp = coef + ((size_t)p * 2 + side) * 2 * groups;
    base += (size_t)side * 2 * hidden;
    float y = (base[c] + cp[g]) * x[(size_t)p * hidden + c];
    if (p > 0) y += (base[hidden + c] + cp[groups + g]) * x[(size_t)(p - 1) * hidden + c];
    float* o = out + (size_t)p * hidden + c;
    *o = add ? *o + y : y;
}

}  // extern "C"
