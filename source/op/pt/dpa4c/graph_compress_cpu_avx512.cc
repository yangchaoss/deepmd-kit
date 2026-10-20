// SPDX-License-Identifier: LGPL-3.0-or-later
//
// avx512 instantiation of the compressed DPA4C CPU kernels.
//
// The build compiles this translation unit with that level's instruction-set
// flags, and the vector width enters the shared body as a compile-time
// constant: with a run-time lane count GCC emits a generic vectorized loop
// with a peel and a tail, which on the short channel blocks of this kernel
// costs more than the block itself.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <new>

namespace deepmd_dpa4c_cpu {
namespace avx512 {
namespace {
// Optional per-worker, per-scan memoization. The spline table and FP control
// must remain unchanged inside a node; no callback occurs in the edge loop.
// Never share this object with another scan, table, or worker.
template <bool Gradient>
class SplineMemo64 {
 public:
  struct Probe { std::uint64_t key; unsigned index; bool hit; };
  struct Deleter {
    void operator()(SplineMemo64* p) const noexcept {
      const int saved_errno = errno;
      delete p;
      errno = saved_errno;
    }
  };
  using Owner = std::unique_ptr<SplineMemo64, Deleter>;

  static Owner create() noexcept {
    const int saved_errno = errno;
    const char* enabled_env = std::getenv("DP_DPA4C_SPLINE_MEMO");
    const bool feature = enabled_env && std::strcmp(enabled_env, "1") == 0;
    const char* trace_env = std::getenv("DP_DPA4C_SPLINE_MEMO_TRACE");
    const bool trace = trace_env && std::strcmp(trace_env, "1") == 0;
    const std::uint32_t control = _mm_getcsr() & 0xffc0u;
    const bool admissible = (control & 0x1f80u) == 0x1f80u &&
                            (control & 0x6000u) == 0;
    SplineMemo64* memo = feature && admissible
        ? new (std::nothrow) SplineMemo64(control, trace) : nullptr;
    if (trace)
      std::fprintf(stderr,
          "DP_SPLINE_MEMO admission phase=%s feature=%u control=%u allocated=%u\n",
          Gradient ? "backward" : "forward", feature ? 1u : 0u, control,
          memo ? 1u : 0u);
    errno = saved_errno;
    return Owner(memo);
  }

  bool begin_node() noexcept {
    if ((_mm_getcsr() & 0xffc0u) != control_) {
      disabled_ = true;
    }
    return !disabled_;
  }
  Probe lookup(std::int64_t interval, float dx) noexcept {
    std::uint32_t dx_bits;
    std::memcpy(&dx_bits, &dx, sizeof(dx_bits));
    const std::uint64_t key =
        (std::uint64_t(static_cast<std::uint32_t>(interval)) << 32) | dx_bits;
    std::uint64_t hash = key;
    hash = (hash ^ (hash >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 31;
    const unsigned index = static_cast<unsigned>(hash & 63u);
    const bool hit = valid[index] && keys[index] == key;
    if (hit) ++hits;
    else ++misses;
    return {key, index, hit};
  }
  float* value(unsigned index) noexcept {
    return payload + index * 64 * (Gradient ? 2 : 1);
  }
  float* slope(unsigned index) noexcept {
    return Gradient ? value(index) + 64 : nullptr;
  }
  void commit(const Probe& probe) noexcept {
    if (probe.hit) return;
    if (valid[probe.index]) ++evictions;
    keys[probe.index] = probe.key;
    valid[probe.index] = 1;
  }
  void report(const char* phase) const noexcept {
    if (!trace_) return;
    const int saved_errno = errno;
    std::fprintf(stderr,
        "DP_SPLINE_MEMO phase=%s hits=%llu misses=%llu evictions=%llu disabled=%u\n",
        phase ? phase : "unknown", static_cast<unsigned long long>(hits),
        static_cast<unsigned long long>(misses),
        static_cast<unsigned long long>(evictions), disabled_ ? 1u : 0u);
    errno = saved_errno;
  }

  std::uint64_t hits = 0, misses = 0, evictions = 0;

 private:
  explicit SplineMemo64(std::uint32_t control, bool trace) noexcept
      : control_(control), trace_(trace) {}
  std::uint32_t control_;
  bool trace_;
  bool disabled_ = false;
  std::uint64_t keys[64];
  std::uint8_t valid[64]{};
  alignas(64) float payload[64 * 64 * (Gradient ? 2 : 1)];
};

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
