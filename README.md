# sse2zig
x86 [SSE intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6889,6889,6976,4635,4635&techs=SSE_ALL) mapped to [Zig](https://ziglang.org/) vector extensions.

Currently, most SSE4.1 intrinsics and below are implemented (but not tested).

However, there are no plans to implement the following:
* Floating-point MXCSR: get exceptions, set rounding mode, etc.
* Non-temporal memory ops: `_mm_maskmoveu_si128`, `_mm_stream_load_si128`, `_mm_clflush`, etc.
* MMX: any intrinsics referencing the `__m64` data type.
* Over-aligned allocator: `_mm_malloc`/`_mm_free`.

Optional assembly statements are provided for many intrinsics.
They can be enabled by manually setting the appropriate flags at the top of `sse.zig` (`has_sse2`, `has_avx`, etc.).

Warning:
* Big-endian architectures won't work.  
* Using an `SSE4.1` intrinsic in a build targeting `SSE2` will result in slow emulation code.

