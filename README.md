# sse2zig
Intel SSE intrinsics mapped to ziglang vector extensions.

Optional x86 inline assembly statements are provided for some intrinsics.

TODO: currently, `asm` statements must be enabled manually by setting the feature detection flags (`has_avx`, `has_sse2`, etc.) at the top of the `sse.zig` file.

Warning: Any sufficiently complex code will probably not work on big endian machines.
Test carefully. 

TODO: `_mm_movemask_epi8` probably has endian issues.

Warning: If you use an `SSE4.1` intrinsic but set a build target for `SSE2` then you'll probably get slow emulation code.  
(This is a common gotcha w/Rust, TODO: look at what rust outputs for emulated instructions)

TODO:
* add floating point intrinsics
* add tests
* add compile-time cpu feature detection
* add asm for neon
