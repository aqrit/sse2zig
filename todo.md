all floating point stuff and....

```
export fn _mm_sll_epi32 (a: __m128i, count: __m128i) __m128i {
export fn _mm_sll_epi64 (a: __m128i, count: __m128i) __m128i {
export fn _mm_sra_epi16 (a: __m128i, count: __m128i) __m128i {
export fn _mm_sra_epi32 (a: __m128i, count: __m128i) __m128i {
export fn _mm_srl_epi16 (a: __m128i, count: __m128i) __m128i {
export fn _mm_srl_epi32 (a: __m128i, count: __m128i) __m128i {
export fn _mm_srl_epi64 (a: __m128i, count: __m128i) __m128i {

export fn _mm_packs_epi16 (a: __m128i, b: __m128i) __m128i {
export fn _mm_packs_epi32 (a: __m128i, b: __m128i) __m128i {
export fn _mm_packus_epi16 (a: __m128i, b: __m128i) __m128i {
export fn _mm_packus_epi32 (a: __m128i, b: __m128i) __m128i {

export fn _mm_sign_epi32 (a: __m128i, b: __m128i) __m128i { 
export fn _mm_sign_epi16 (a: __m128i, b: __m128i) __m128i {
export fn _mm_sign_epi8 (a: __m128i, b: __m128i) __m128i {

__m128i _mm_minpos_epu16 (__m128i a)

__m128i _mm_mpsadbw_epu8 (__m128i a, __m128i b, const int imm8)
__m128i _mm_sad_epu8 (__m128i a, __m128i b)

export fn _mm_mulhi_epi16 (a: __m128i, b: __m128i) __m128i {
export fn _mm_mulhi_epu16 (a: __m128i, b: __m128i) __m128i {

int _mm_movemask_epi8 (__m128i a)

int _mm_test_all_ones (__m128i a)
int _mm_test_all_zeros (__m128i mask, __m128i a)
int _mm_test_mix_ones_zeros (__m128i mask, __m128i a)
int _mm_testc_si128 (__m128i a, __m128i b)
int _mm_testnzc_si128 (__m128i a, __m128i b)
int _mm_testz_si128 (__m128i a, __m128i b)

__m128i _mm_mulhrs_epi16 (__m128i a, __m128i b)

__m128i _mm_load_si128 (__m128i const* mem_addr)
__m128i _mm_loadl_epi64 (__m128i const* mem_addr)
__m128i _mm_loadu_si128 (__m128i const* mem_addr)
__m128i _mm_loadu_si16 (void const* mem_addr)
__m128i _mm_loadu_si32 (void const* mem_addr)
__m128i _mm_loadu_si64 (void const* mem_addr)

void _mm_store_si128 (__m128i* mem_addr, __m128i a)
void _mm_storel_epi64 (__m128i* mem_addr, __m128i a)
void _mm_storeu_si128 (__m128i* mem_addr, __m128i a)
void _mm_storeu_si16 (void* mem_addr, __m128i a)
void _mm_storeu_si32 (void* mem_addr, __m128i a)
void _mm_storeu_si64 (void* mem_addr, __m128i a)

__m128i _mm_lddqu_si128 (__m128i const* mem_addr)
__m128i _mm_stream_load_si128 (__m128i * mem_addr)
void _mm_stream_si128 (__m128i* mem_addr, __m128i a)

void _mm_maskmoveu_si128 (__m128i a, __m128i mask, char* mem_addr)
```
