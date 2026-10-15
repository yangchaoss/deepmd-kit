// SPDX-License-Identifier: LGPL-3.0-or-later
//
// avx512 instantiation of the compressed DPA4C CPU kernels.
//
// The build compiles this translation unit with that level's instruction-set
// flags, and the vector width enters the shared body as a compile-time
// constant: with a run-time lane count GCC emits a generic vectorized loop
// with a peel and a tail, which on the short channel blocks of this kernel
// costs more than the block itself.

#include <immintrin.h>

namespace deepmd_dpa4c_cpu {
namespace avx512 {
namespace {
// Only the 3*8 + 5*4 live moments of the fixed C64/L2 layout.
// Unlike the generic loop, discarded angular lanes are not evaluated.
struct ForwardAngularPacked44 {
  __m512 a = _mm512_setzero_ps();
  __m512 b = _mm512_setzero_ps();
  __m512 c = _mm512_setzero_ps();

  inline __attribute__((always_inline)) void update(
      const float* amplitude, float chi_squared, const float* basis) {
    const __m512 amps = _mm512_broadcast_f32x8(_mm256_loadu_ps(amplitude));
    const __m256 w8 = _mm256_mul_ps(_mm256_set1_ps(chi_squared),
                                   _mm256_loadu_ps(basis));
    const __m512 weights = _mm512_broadcast_f32x8(w8);
    const __m512i wa = _mm512_setr_epi32(
        0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1);
    const __m512i wb = _mm512_setr_epi32(
        2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4);
    const __m512i wc = _mm512_setr_epi32(
        5,5,5,5,6,6,6,6,7,7,7,7,5,5,5,5);
    const __m512i ab = _mm512_setr_epi32(
        0,1,2,3,4,5,6,7,0,1,2,3,0,1,2,3);
    a = _mm512_fmadd_ps(_mm512_permutexvar_ps(wa, weights), amps, a);
    b = _mm512_fmadd_ps(_mm512_permutexvar_ps(wb, weights),
                         _mm512_permutexvar_ps(ab, amps), b);
    c = _mm512_mask3_fmadd_ps(_mm512_permutexvar_ps(wc, weights),
                              _mm512_broadcast_f32x4(_mm_loadu_ps(amplitude)),
                              c, 0x0fff);
  }

  inline __attribute__((always_inline)) void store(float* output) const {
    _mm256_storeu_ps(output + 0*16, _mm512_castps512_ps256(a));
    _mm256_storeu_ps(output + 1*16, _mm512_extractf32x8_ps(a, 1));
    _mm256_storeu_ps(output + 2*16, _mm512_castps512_ps256(b));
    const __m256 upper_b = _mm512_extractf32x8_ps(b, 1);
    _mm_storeu_ps(output + 3*16, _mm256_castps256_ps128(upper_b));
    _mm_storeu_ps(output + 4*16, _mm256_extractf128_ps(upper_b, 1));
    _mm_storeu_ps(output + 5*16, _mm512_castps512_ps128(c));
    _mm_storeu_ps(output + 6*16, _mm512_extractf32x4_ps(c, 1));
    _mm_storeu_ps(output + 7*16, _mm512_extractf32x4_ps(c, 2));
  }
};
}  // namespace
}  // namespace avx512
}  // namespace deepmd_dpa4c_cpu

#define DPA4C_CPU_ISA avx512
#define DPA4C_CPU_BLOCK 16
#include "graph_compress_cpu_kernel.h"
