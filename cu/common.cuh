// Shared by kernels.cu and kernels_tc.cu.
#pragma once

typedef unsigned short u16;
typedef unsigned char u8;

// 1024 times the value of an 8-bit float (sign, 4-bit exponent biased by 7, 3-bit mantissa).
static __device__ __forceinline__ float fp8x1024(unsigned int b) {
    unsigned int e = (b >> 3) & 15u, m = b & 7u;
    int v = (int)(e ? (8u | m) << e : m << 1);
    return (float)((b & 0x80u) ? -v : v);
}
