# sse2zig
Intel SSE intrinsics mapped to ziglang vector extensions.

How to handle Floating Pont MXCSR behavior is an open question:
1. Ignore it.
2. Write everything in inline assembly.
3. Both, depending on platform ?

Optional x86 inline assembly statements are provided for some intrinsics.  
TODO: currently, `asm` statements must be enabled manually by setting the feature detection flags (`has_avx`, `has_sse2`, etc.) at the top of the `sse.zig` file.

Warning: Loads/Store will not work correctly on Big-Endian machines.  
TODO: add auxiliary loads and stores that are endian-agnostic.  
TODO: `_mm_movemask_epi8` => `@ptrCast()` might have an endian issue?

Warning: If you use an `SSE4.1` intrinsic but set a build target for `SSE2` then you'll probably get slow emulation code.  
(This is a common gotcha w/Rust, TODO: look at what rust outputs for emulated instructions)

TODO:
* add floating point intrinsics
* add tests
* add compile-time cpu feature detection
* add asm for neon
