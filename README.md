# sse2zig
Intel SSE intrinsics mapped to ziglang vector extensions.

Optional x86 inline assembly statements are provided for some intrinsics.  
TODO: currently, `asm` statements must be enabled manually by setting the feature detection flags (`has_avx`, `has_sse2`, etc.) at the top of the `sse.zig` file.

Warning: Floating point MXCSR is ignored.

Warning: Loads/Store will not work correctly on Big-Endian machines.  
TODO: add auxiliary loads and stores that are endian-agnostic.  

Warning: If you use an `SSE4.1` intrinsic but set a build target for `SSE2` then you'll probably get slow emulation code.  

TODO:
* add missing intrinsics
* add tests
* add compile-time cpu feature detection
* add asm for neon
