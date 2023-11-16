all floating point stuff..


```
__m128i _mm_minpos_epu16 (__m128i a)

__m128i _mm_mpsadbw_epu8 (__m128i a, __m128i b, const int imm8)
__m128i _mm_sad_epu8 (__m128i a, __m128i b)

int _mm_test_all_ones (__m128i a)
int _mm_test_all_zeros (__m128i mask, __m128i a)
int _mm_test_mix_ones_zeros (__m128i mask, __m128i a)
int _mm_testc_si128 (__m128i a, __m128i b)
int _mm_testnzc_si128 (__m128i a, __m128i b)
int _mm_testz_si128 (__m128i a, __m128i b)

__m128i _mm_mulhrs_epi16 (__m128i a, __m128i b)

__m128i _mm_stream_load_si128 (__m128i * mem_addr)
void _mm_stream_si128 (__m128i* mem_addr, __m128i a)

void _mm_maskmoveu_si128 (__m128i a, __m128i mask, char* mem_addr)
```
