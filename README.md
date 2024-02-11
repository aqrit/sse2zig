# sse2zig
x86 [SSE intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6889,6889,6976,4635,4635&techs=SSE_ALL) mapped to [Zig](https://ziglang.org/) vector extensions.

Currently, most integer intrinsics are implemented (but not tested) up through AVX2.    
Most floating-point intrinsics from SSE4.1 and below are implemented (but not tested).    
SSE4.2 string instructions are not yet implemented.

There are no plans to implement the following:
* Floating-point MXCSR: get exceptions, set rounding mode, etc.
* Non-temporal memory ops: `_mm_maskmoveu_si128`, `_mm_stream_load_si128`, `_mm_clflush`, etc.
* MMX: any intrinsics referencing the `__m64` data type.
* Over-aligned allocator: `_mm_malloc`/`_mm_free`.

Optional assembly statements are provided for many intrinsics. They are enabled by default.

Use of assembly statements can be controlled in the root source file:    
```pub const sse2zig_useAsm = false;```


Warning:
* Big-endian architectures won't work.  
* Using `SSE4.1` intrinsics in a build targeting `SSE2` will result in slow emulation code.
