// The matrix products on tensor cores, for RTX 30-series cards and newer (sm_80).
//
// Both kernels compute  out[t, r] (+)= sum_j w[r, j] * x[t, j]  like `gemm` and `gemm4` in
// kernels.cu, with one difference in arithmetic: every element of x is rounded to bf16
// (nearest, ties to even) as it is loaded, because the instruction multiplies bf16 by bf16.
// Products and sums are f32.
//
// The instruction is mma.sync m16n8k16: a warp multiplies a 16x16 tile of A by a 16x8 tile of
// B. Here A is 16 output rows by 16 input columns of the weights and B is those 16 columns of
// 8 tokens, so one instruction covers one NVFP4 scale group for a whole drafted block.
//
// A block owns 32 output rows and 8 tokens, and walks the input columns in chunks. For each
// chunk all 256 threads copy the weights and x into shared memory with contiguous loads, then
// the 8 warps run the instructions from there: warp w takes row tile w % 2 and a quarter of
// the chunk's column groups, w / 2. Each warp keeps its own partial tile for the whole matrix
// and the four partials of a row tile are added at the end in a fixed order. Tokens are always
// taken in tiles of 8, padded with zeros, so an output element goes through the same
// instructions whether its token is fed alone or in a chunk.
//
// Which lane holds which element (PTX ISA, "Matrix Fragments for mma.m16n8k16"), with
// g = lane / 4 and t = lane % 4:
//   A: a[0] = row g,     columns 2t, 2t+1      a[1] = row g + 8, columns 2t, 2t+1
//      a[2] = row g,     columns 2t+8, 2t+9    a[3] = row g + 8, columns 2t+8, 2t+9
//   B: b[0] = token g, columns 2t, 2t+1        b[1] = token g, columns 2t+8, 2t+9
//   D: d[0], d[1] = row g, tokens 2t, 2t+1     d[2], d[3] = row g + 8, tokens 2t, 2t+1
// Each a[] and b[] register packs two bf16 values, the lower column in the low half.
//
// grid (ceil(n_out / 32), ceil(s / 8)), block 256.

#include "common.cuh"

#define TC_THREADS 256
#define TC_ROWS 32
// Column groups of 16 per chunk; a multiple of 4.
#define TC_GROUPS 16
#define TC_COLS (TC_GROUPS * 16)

// d = A * B + d
static __device__ __forceinline__ void mma(float d[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(d[0]), "f"(d[1]),
          "f"(d[2]), "f"(d[3]));
}

// f32 to bf16 bits, to nearest with ties to even.
static __device__ __forceinline__ unsigned bf16_bits(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

// Twice the value of a 4-bit float, as bf16 bits: 0, 1, 2, 3, 4, 6, 8, 12 and negatives.
static __device__ __forceinline__ unsigned fp4_bf16(unsigned n) {
    unsigned sh = (n & 7u) * 8u;
    unsigned hi = (unsigned)(0x4141404040403F00ull >> sh) & 0xFFu;
    unsigned lo = (unsigned)(0x4000C08040008000ull >> sh) & 0xFFu;
    return (hi << 8) | lo | ((n & 8u) << 12);
}

// A byte of NVFP4 weights (two columns, low nibble first), packed for A.
static __device__ __forceinline__ unsigned fp4_pair(unsigned byte) {
    return fp4_bf16(byte & 15u) | (fp4_bf16(byte >> 4) << 16);
}


// Add a row tile's four partials and store this block's rows and tokens.
static __device__ __forceinline__ void tc_finish(float (*part)[32][4], const float acc[4],
                                                 float* out, float gs, int n_out, int s,
                                                 int out_stride, int add) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = 0; i < 4; i++) part[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp > 1) return;
    int g = lane >> 2, t = lane & 3;
    for (int i = 0; i < 4; i++) {
        int row = blockIdx.x * TC_ROWS + warp * 16 + g + (i >= 2 ? 8 : 0);
        int tok = blockIdx.y * 8 + 2 * t + (i & 1);
        if (row >= n_out || tok >= s) continue;
        float v = (part[warp][lane][i] + part[warp + 2][lane][i]) +
                  (part[warp + 4][lane][i] + part[warp + 6][lane][i]);
        v *= gs;
        float* o = out + (size_t)tok * out_stride + row;
        *o = add ? *o + v : v;
    }
}

extern "C" {

// 4-bit weights; see gemm4_impl in kernels.cu for the format. The group scale is applied to
// each instruction's result, so A holds the 4-bit values exactly.
__global__ void gemm4_tc(float* out, const float* x, const u8* w, const u8* sc, float gs,
                         int n_out, int n_in, int s, int out_stride, int add) {
    // Rows are one word longer than their data so that a warp's reads, which step from row to
    // row, fall in different banks.
    __shared__ unsigned sw[TC_ROWS][TC_GROUPS * 2 + 1];
    __shared__ float ss[TC_ROWS][TC_GROUPS + 1];
    __shared__ u16 sx[8][TC_COLS + 2];
    __shared__ float part[8][32][4];
    // Every byte of weights decoded to its two bf16 values, packed for A.
    __shared__ unsigned lut[256];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int g = lane >> 2, t = lane & 3;
    int r0 = blockIdx.x * TC_ROWS;
    int groups = n_in / 16;
    // This thread's rows of A and its warp's quarter of each chunk.
    int ra = (warp & 1) * 16 + g, rb = ra + 8, q0 = (warp >> 1) * (TC_GROUPS / 4);
    const u8* wa = (const u8*)sw[ra];
    const u8* wb = (const u8*)sw[rb];
    const unsigned* xb = (const unsigned*)sx[g];
    lut[tid] = fp4_pair((unsigned)tid);

    // What this thread copies per chunk: four words of weights, two scales, a column of x.
    const int NW = TC_ROWS * TC_GROUPS * 2 / TC_THREADS, NS = TC_ROWS * TC_GROUPS / TC_THREADS;
    const unsigned* wsrc[NW];
    const u8* ssrc[NS];
    for (int i = 0; i < NW; i++) {
        int idx = tid + TC_THREADS * i;
        wsrc[i] = (const unsigned*)(w + (size_t)min(r0 + idx / (TC_GROUPS * 2), n_out - 1) *
                                            (n_in / 2)) + idx % (TC_GROUPS * 2);
    }
    for (int i = 0; i < NS; i++) {
        int idx = tid + TC_THREADS * i;
        ssrc[i] = sc + (size_t)min(r0 + idx / TC_GROUPS, n_out - 1) * groups + idx % TC_GROUPS;
    }
    unsigned rw[NW];
    float rs[NS];
    u16 rx[8];
    // Read chunk `g0` into registers. The loads are issued a chunk ahead, so they are in
    // flight while the previous chunk is multiplied.
    auto fetch = [&](int g0) {
        int count = min(TC_GROUPS, groups - g0);
        for (int i = 0; i < NW; i++)
            rw[i] = (tid + TC_THREADS * i) % (TC_GROUPS * 2) < count * 2 ? wsrc[i][g0 * 2] : 0u;
        for (int i = 0; i < NS; i++)
            rs[i] = (tid + TC_THREADS * i) % TC_GROUPS < count ? fp8x1024(ssrc[i][g0]) : 0.f;
        int col = g0 * 16 + tid;
        for (int i = 0; i < 8; i++) {
            int tok = blockIdx.y * 8 + i;
            rx[i] = (u16)bf16_bits((tok < s && col < n_in) ? x[(size_t)tok * n_in + col] : 0.f);
        }
    };

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    fetch(0);
    for (int g0 = 0; g0 < groups; g0 += TC_GROUPS) {
        for (int i = 0; i < NW; i++) {
            int idx = tid + TC_THREADS * i;
            sw[idx / (TC_GROUPS * 2)][idx % (TC_GROUPS * 2)] = rw[i];
        }
        for (int i = 0; i < NS; i++) {
            int idx = tid + TC_THREADS * i;
            ss[idx / TC_GROUPS][idx % TC_GROUPS] = rs[i];
        }
        for (int i = 0; i < 8; i++) sx[i][tid] = rx[i];
        __syncthreads();
        int count = min(TC_GROUPS, groups - g0);
        if (g0 + TC_GROUPS < groups) fetch(g0 + TC_GROUPS);
        for (int c = q0; c < min(q0 + TC_GROUPS / 4, count); c++) {
            unsigned a[4] = {lut[wa[c * 8 + t]], lut[wb[c * 8 + t]], lut[wa[c * 8 + t + 4]],
                             lut[wb[c * 8 + t + 4]]};
            unsigned b[2] = {xb[c * 8 + t], xb[c * 8 + t + 4]};
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            mma(d, a, b);
            float fa = ss[ra][c], fb = ss[rb][c];
            acc[0] += fa * d[0];
            acc[1] += fa * d[1];
            acc[2] += fb * d[2];
            acc[3] += fb * d[3];
        }
        __syncthreads();
    }
    tc_finish(part, acc, out, gs, n_out, s, out_stride, add);
}

}  // extern "C"
