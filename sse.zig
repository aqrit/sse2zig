// SPDX-License-Identifier: MIT

const builtin = @import("builtin");
const root = @import("root");
const std = @import("std");

/// Use of LLVM builtins are enabled by default.
/// (e.g. @"llvm.x86.ssse3.pshuf.b.128")
/// Disable LLVM builtins by declaring in the root src file:
/// `pub const sse2zig_useBuiltins = false;`
const use_builtins = b: {
    if (builtin.zig_backend != .stage2_llvm) break :b false;
    if (@hasDecl(root, "sse2zig_useBuiltins")) break :b root.sse2zig_useBuiltins;
    break :b true; // enabled by default
};

/// Use of inline assembly statements are enabled by default.
/// Disable asm statements by declaring in the root src file:
/// `pub const sse2zig_useAsm = false;`
const use_asm = b: {
    if (@hasDecl(root, "sse2zig_useAsm")) break :b root.sse2zig_useAsm;
    break :b true; // enabled by default
};

/// stage2_x86_64 is currently missing many assembly mnemonics.
/// It is also has some TODOs related to genBinOp for vector extensions.
// https://github.com/ziglang/zig/blob/master/src/arch/x86_64/Encoding.zig
const bug_stage2_x86_64 = (builtin.zig_backend == .stage2_x86_64);

const is_x86_64 = builtin.target.cpu.arch == .x86_64;

const has_avx2 = use_asm and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
const has_avx = use_asm and std.Target.x86.featureSetHas(builtin.cpu.features, .avx);
const has_pclmul = std.Target.x86.featureSetHas(builtin.cpu.features, .pclmul);
const has_sse4_2 = std.Target.x86.featureSetHas(builtin.cpu.features, .sse4_2);
const has_sse4_1 = std.Target.x86.featureSetHas(builtin.cpu.features, .sse4_1);
const has_ssse3 = std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3);
const has_sse3 = std.Target.x86.featureSetHas(builtin.cpu.features, .sse3);
const has_sse2 = use_asm and std.Target.x86.featureSetHas(builtin.cpu.features, .sse2);
const has_sse = use_asm and std.Target.x86.featureSetHas(builtin.cpu.features, .sse);

pub const has_neon = false;

pub const __m128 = @Vector(4, f32);
pub const __m128d = @Vector(2, f64);
pub const __m128i = @Vector(4, i32);
pub const __m256 = @Vector(8, f32);
pub const __m256d = @Vector(4, f64);
pub const __m256i = @Vector(8, i32);

// helpers to reduce verbosity =========================================

const u64x1 = @Vector(1, u64);
const u32x2 = @Vector(2, u32);
const u16x4 = @Vector(4, u16);
const u8x8 = @Vector(8, u8);
const i64x1 = @Vector(1, i64);
const i32x2 = @Vector(2, i32);
const i16x4 = @Vector(4, i16);
const i8x8 = @Vector(8, i8);
//
const u64x2 = @Vector(2, u64);
const u32x4 = @Vector(4, u32);
const u16x8 = @Vector(8, u16);
const u8x16 = @Vector(16, u8);
const i64x2 = @Vector(2, i64);
const i32x4 = @Vector(4, i32);
const i16x8 = @Vector(8, i16);
const i8x16 = @Vector(16, i8);
//
const u64x4 = @Vector(4, u64);
const u32x8 = @Vector(8, u32);
const u16x16 = @Vector(16, u16);
const u8x32 = @Vector(32, u8);
const i64x4 = @Vector(4, i64);
const i32x8 = @Vector(8, i32);
const i16x16 = @Vector(16, i16);
const i8x32 = @Vector(32, i8);
//
const i64x8 = @Vector(8, i64);
const i32x16 = @Vector(16, i32);
const u32x16 = @Vector(16, u32);
const u16x32 = @Vector(32, u16);

// =====================================================================

inline fn bitCast_u64x2(a: anytype) u64x2 {
    return @bitCast(a);
}
inline fn bitCast_u32x4(a: anytype) u32x4 {
    return @bitCast(a);
}
inline fn bitCast_u16x8(a: anytype) u16x8 {
    return @bitCast(a);
}
inline fn bitCast_u8x16(a: anytype) u8x16 {
    return @bitCast(a);
}
inline fn bitCast_i64x2(a: anytype) i64x2 {
    return @bitCast(a);
}
inline fn bitCast_i32x4(a: anytype) i32x4 {
    return @bitCast(a);
}
inline fn bitCast_i16x8(a: anytype) i16x8 {
    return @bitCast(a);
}
inline fn bitCast_i8x16(a: anytype) i8x16 {
    return @bitCast(a);
}
//
inline fn bitCast_u64x4(a: anytype) u64x4 {
    return @bitCast(a);
}
inline fn bitCast_u32x8(a: anytype) u32x8 {
    return @bitCast(a);
}
inline fn bitCast_u16x16(a: anytype) u16x16 {
    return @bitCast(a);
}
inline fn bitCast_u8x32(a: anytype) u8x32 {
    return @bitCast(a);
}
inline fn bitCast_i64x4(a: anytype) i64x4 {
    return @bitCast(a);
}
inline fn bitCast_i32x8(a: anytype) i32x8 {
    return @bitCast(a);
}
inline fn bitCast_i16x16(a: anytype) i16x16 {
    return @bitCast(a);
}
inline fn bitCast_i8x32(a: anytype) i8x32 {
    return @bitCast(a);
}
// =====================================================================

inline fn intCast_u64x2(a: anytype) u64x2 {
    return @intCast(a);
}
inline fn intCast_u32x4(a: anytype) u32x4 {
    return @intCast(a);
}
inline fn intCast_u16x8(a: anytype) u16x8 {
    return @intCast(a);
}
inline fn intCast_u8x16(a: anytype) u8x16 {
    return @intCast(a);
}
inline fn intCast_i64x2(a: anytype) i64x2 {
    return @intCast(a);
}
inline fn intCast_i32x4(a: anytype) i32x4 {
    return @intCast(a);
}
inline fn intCast_i16x8(a: anytype) i16x8 {
    return @intCast(a);
}
inline fn intCast_i8x16(a: anytype) i8x16 {
    return @intCast(a);
}
//
inline fn intCast_u64x4(a: anytype) u64x4 {
    return @intCast(a);
}
inline fn intCast_u32x8(a: anytype) u32x8 {
    return @intCast(a);
}
inline fn intCast_u16x16(a: anytype) u16x16 {
    return @intCast(a);
}
inline fn intCast_u8x32(a: anytype) u8x32 {
    return @intCast(a);
}
inline fn intCast_i64x4(a: anytype) i64x4 {
    return @intCast(a);
}
inline fn intCast_i32x8(a: anytype) i32x8 {
    return @intCast(a);
}
inline fn intCast_i16x16(a: anytype) i16x16 {
    return @intCast(a);
}
inline fn intCast_i8x32(a: anytype) i8x32 {
    return @intCast(a);
}
//

inline fn intCast_i32x16(a: anytype) i32x16 {
    return @intCast(a);
}
inline fn intCast_i64x8(a: anytype) i64x8 {
    return @intCast(a);
}
inline fn intCast_u32x16(a: anytype) u32x16 {
    return @intCast(a);
}
inline fn intCast_u16x32(a: anytype) u16x32 {
    return @intCast(a);
}
// =====================================================================

/// For setting test values (using hex literals).
inline fn _xx_set_epu16(e7: u16, e6: u16, e5: u16, e4: u16, e3: u16, e2: u16, e1: u16, e0: u16) __m128i {
    return @bitCast(u16x8{ e0, e1, e2, e3, e4, e5, e6, e7 });
}

/// For setting test values (using hex literals).
inline fn _xx_set_epu32(e3: u32, e2: u32, e1: u32, e0: u32) __m128i {
    return @bitCast(u32x4{ e0, e1, e2, e3 });
}

/// For setting test values (using hex literals).
inline fn _xx_set_epu64x(e1: u64, e0: u64) __m128i {
    return @bitCast(u64x2{ e0, e1 });
}

/// For setting test values (using hex literals).
inline fn _xx_set_epu8(e15: u8, e14: u8, e13: u8, e12: u8, e11: u8, e10: u8, e9: u8, e8: u8, e7: u8, e6: u8, e5: u8, e4: u8, e3: u8, e2: u8, e1: u8, e0: u8) __m128i {
    return @bitCast(u8x16{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 });
}

/// For setting test values (using hex literals).
inline fn _xx256_set_epu16(e15: u16, e14: u16, e13: u16, e12: u16, e11: u16, e10: u16, e9: u16, e8: u16, e7: u16, e6: u16, e5: u16, e4: u16, e3: u16, e2: u16, e1: u16, e0: u16) __m256i {
    return @bitCast(u16x16{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 });
}

/// For setting test values (using hex literals).
inline fn _xx256_set_epu32(e7: u32, e6: u32, e5: u32, e4: u32, e3: u32, e2: u32, e1: u32, e0: u32) __m256i {
    return @bitCast(u32x8{ e0, e1, e2, e3, e4, e5, e6, e7 });
}

/// For setting test values (using hex literals).
inline fn _xx256_set_epu64x(e3: u64, e2: u64, e1: u64, e0: u64) __m256i {
    return @bitCast(u64x4{ e0, e1, e2, e3 });
}

/// For setting test values (using hex literals).
inline fn _xx256_set_epu8(e31: u8, e30: u8, e29: u8, e28: u8, e27: u8, e26: u8, e25: u8, e24: u8, e23: u8, e22: u8, e21: u8, e20: u8, e19: u8, e18: u8, e17: u8, e16: u8, e15: u8, e14: u8, e13: u8, e12: u8, e11: u8, e10: u8, e9: u8, e8: u8, e7: u8, e6: u8, e5: u8, e4: u8, e3: u8, e2: u8, e1: u8, e0: u8) __m256i {
    return @bitCast(u8x32{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31 });
}

// =====================================================================
// @min will @truncate the type, if it can.
// @max is probably similarly.
// These helper functions implicit cast back to the expected return type.

inline fn min_i32x4(a: i32x4, b: i32x4) i32x4 {
    return @min(a, b);
}
inline fn min_i32x8(a: i32x8, b: i32x8) i32x8 {
    return @min(a, b);
}
inline fn min_u32x4(a: u32x4, b: u32x4) u32x4 {
    return @min(a, b);
}
inline fn min_u32x8(a: u32x8, b: u32x8) u32x8 {
    return @min(a, b);
}
inline fn min_i16x8(a: i16x8, b: i16x8) i16x8 {
    return @min(a, b);
}
inline fn min_i16x16(a: i16x16, b: i16x16) i16x16 {
    return @min(a, b);
}
inline fn min_u16x8(a: u16x8, b: u16x8) u16x8 {
    return @min(a, b);
}
inline fn min_u16x16(a: u16x16, b: u16x16) u16x16 {
    return @min(a, b);
}
inline fn min_i8x16(a: i8x16, b: i8x16) i8x16 {
    return @min(a, b);
}
inline fn min_i8x32(a: i8x32, b: i8x32) i8x32 {
    return @min(a, b);
}
inline fn min_u8x16(a: u8x16, b: u8x16) u8x16 {
    return @min(a, b);
}
inline fn min_u8x32(a: u8x32, b: u8x32) u8x32 {
    return @min(a, b);
}
inline fn max_i32x4(a: i32x4, b: i32x4) i32x4 {
    return @max(a, b);
}
inline fn max_i32x8(a: i32x8, b: i32x8) i32x8 {
    return @max(a, b);
}
inline fn max_u32x4(a: u32x4, b: u32x4) u32x4 {
    return @max(a, b);
}
inline fn max_u32x8(a: u32x8, b: u32x8) u32x8 {
    return @max(a, b);
}
inline fn max_i16x8(a: i16x8, b: i16x8) i16x8 {
    return @max(a, b);
}
inline fn max_i16x16(a: i16x16, b: i16x16) i16x16 {
    return @max(a, b);
}
inline fn max_u16x8(a: u16x8, b: u16x8) u16x8 {
    return @max(a, b);
}
inline fn max_u16x16(a: u16x16, b: u16x16) u16x16 {
    return @max(a, b);
}
inline fn max_i8x16(a: i8x16, b: i8x16) i8x16 {
    return @max(a, b);
}
inline fn max_i8x32(a: i8x32, b: i8x32) i8x32 {
    return @max(a, b);
}
inline fn max_u8x16(a: u8x16, b: u8x16) u8x16 {
    return @max(a, b);
}
inline fn max_u8x32(a: u8x32, b: u8x32) u8x32 {
    return @max(a, b);
}
//
inline fn min_i32x16(a: i32x16, b: i32x16) i32x16 {
    return @min(a, b);
}
inline fn max_i32x16(a: i32x16, b: i32x16) i32x16 {
    return @min(a, b);
}

// =====================================================================
// Note: `nan = (x != x)` code gen seems good.
// However, it somehow fails under test. comptime_float?
// `@setFloatMode(std.builtin.FloatMode.Strict)` doesn't help.

inline fn isNan_pd(a: __m128d) @Vector(2, u1) {
    const pred = (bitCast_u64x2(a) << @splat(1)) > @as(u64x2, @splat(0xFFE0000000000000));
    return @intFromBool(pred);
}
inline fn isNan_ps(a: __m128) @Vector(4, u1) {
    const pred = (bitCast_u32x4(a) << @splat(1)) > @as(u32x4, @splat(0xFF000000));
    return @intFromBool(pred);
}
inline fn isNan_f64(a: f64) u1 {
    const pred = ((@as(u32, @bitCast(a)) << 1) > 0xFFE0000000000000);
    return @intFromBool(pred);
}
inline fn isNan_f32(a: f32) u1 {
    const pred = ((@as(u32, @bitCast(a)) << 1) > 0xFF000000);
    return @intFromBool(pred);
}

// =====================================================================
// Zig is missing: Round to Nearest, halfway-ties round to even?
// Note: Zig has @floor, @ceil, @trunc, and @round.
// See also: https://github.com/ziglang/zig/issues/9551
//
// Zig doesn't expose a way to change the current rounding mode?

/// Round using the current rounding mode
inline fn RoundCurrentDirection_ps(a: __m128) __m128 {
    return RoundEven_ps(a); // TODO: assumes current mode is round even
}

/// Round using the current rounding mode
inline fn RoundCurrentDirection_f32(a: f32) f32 {
    return RoundEven_f32(a); // TODO: assumes current mode is round even
}

/// Round using the current rounding mode
inline fn RoundCurrentDirection_pd(a: __m128d) __m128d {
    return RoundEven_pd(a); // TODO: assumes current mode is round even
}

/// Round using the current rounding mode
inline fn RoundCurrentDirection_f64(a: f64) f64 {
    return RoundEven_f64(a); // TODO: assumes current mode is round even
}

// See: Henry S. Warren, Hacker's Delight 2nd Edition, pp. 378-380
// TODO: round changes SNaN to QNaN ( 7FF0000000000001 -> 7FF8000000000001 ) ?
inline fn RoundEven_f32(a: f32) f32 {
    // TODO: currently, depends on current rounding mode being RoundEven
    const magic: f32 = 8388608.0; // 0x4B000000
    const bits: u32 = @bitCast(a);
    const sign = bits & 0x80000000;
    const abs = bits ^ sign;
    const round = (@as(f32, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | @as(u32, @bitCast(round)); // restore sign
    const pred = (abs >= @as(u32, @bitCast(magic))); // NaN or whole number
    return if (pred) a else @as(f32, @bitCast(res));
}

inline fn RoundEven_ps(a: __m128) __m128 {
    // TODO: currently, depends on current rounding mode being RoundEven
    const magic: __m128 = @splat(8388608.0); // 0x4B000000
    const bits = bitCast_u32x4(a);
    const sign = bits & @as(u32x4, @splat(0x80000000));
    const abs = bits ^ sign;
    const round = (@as(__m128, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | bitCast_u32x4(round); // restore sign
    const pred = (abs >= bitCast_u32x4(magic)); // NaN or whole number
    return @select(f32, pred, a, @as(__m128, @bitCast(res)));
}

inline fn RoundEven_f64(a: f64) f64 {
    // TODO: currently, depends on current rounding mode being RoundEven
    const magic: f64 = @bitCast(@as(u64, 0x4330000000000000));
    const bits: u64 = @bitCast(a);
    const sign = bits & 0x8000000000000000;
    const abs = bits ^ sign;
    const round = (@as(f64, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | @as(u64, @bitCast(round)); // restore sign
    const pred = (abs >= @as(u64, @bitCast(magic))); // NaN or whole number
    return if (pred) a else @as(f64, @bitCast(res));
}

inline fn RoundEven_pd(a: __m128d) __m128d {
    // TODO: currently, depends on current rounding mode being RoundEven
    const magic: __m128d = @bitCast(@as(u64x2, @splat(0x4330000000000000)));
    const bits = bitCast_u64x2(a);
    const sign = bits & @as(u64x2, @splat(0x8000000000000000));
    const abs = bits ^ sign;
    const round = (@as(__m128d, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | bitCast_u64x2(round); // restore sign
    const pred = (abs >= bitCast_u64x2(magic)); // NaN or whole number
    return @select(f64, pred, a, @as(__m128d, @bitCast(res)));
}

// =====================================================================

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u32x1(pred: u1) u32 {
    return @bitCast(@as(i32, @intCast(@as(i1, @bitCast(pred)))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u64x1(pred: u1) u64 {
    return @bitCast(@as(i64, @intCast(@as(i1, @bitCast(pred)))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u8x16(pred: @Vector(16, u1)) u8x16 {
    return @bitCast(intCast_i8x16(@as(@Vector(16, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u16x8(pred: @Vector(8, u1)) u16x8 {
    return @bitCast(intCast_i16x8(@as(@Vector(8, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u32x4(pred: @Vector(4, u1)) u32x4 {
    return @bitCast(intCast_i32x4(@as(@Vector(4, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u64x2(pred: @Vector(2, u1)) u64x2 {
    return @bitCast(intCast_i64x2(@as(@Vector(2, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u8x32(pred: @Vector(32, u1)) u8x32 {
    return @bitCast(intCast_i8x32(@as(@Vector(32, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u16x16(pred: @Vector(16, u1)) u16x16 {
    return @bitCast(intCast_i16x16(@as(@Vector(16, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u32x8(pred: @Vector(8, u1)) u32x8 {
    return @bitCast(intCast_i32x8(@as(@Vector(8, i1), @bitCast(pred))));
}

/// Fill a lane with all set bits or all zeros
inline fn boolMask_u64x4(pred: @Vector(4, u1)) u64x4 {
    return @bitCast(intCast_i64x4(@as(@Vector(4, i1), @bitCast(pred))));
}

// SSE =================================================================

/// PREFETCHT0: prefetch data into all levels of the cache hierarchy
pub const _MM_HINT_T0 = 3;
/// PREFETCHT1: prefetch data into level 2 cache and higher.
pub const _MM_HINT_T1 = 2;
/// PREFETCHT2: prefetch data into level 3 cache and higher.
pub const _MM_HINT_T2 = 1;
/// PREFETCHNTA non-temporal, data is likely dropped after access.
pub const _MM_HINT_NTA = 0;

pub inline fn _mm_add_ps(a: __m128, b: __m128) __m128 {
    return a + b;
}

pub inline fn _mm_add_ss(a: __m128, b: __m128) __m128 {
    return .{ a[0] + b[0], a[1], a[2], a[3] };
}

pub inline fn _mm_and_ps(a: __m128, b: __m128) __m128 {
    return @bitCast(bitCast_u32x4(a) & bitCast_u32x4(b));
}

pub inline fn _mm_andnot_ps(a: __m128, b: __m128) __m128 {
    return @bitCast(~bitCast_u32x4(a) & bitCast_u32x4(b));
}

/// dst[n] = if ((a[n] == b[n]) and (a[n] != NaN) and b[n] != NaN) -1 else 0;
pub inline fn _mm_cmpeq_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (0),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (0),
        );
        return res;
    } else {
        const notNan = ~(isNan_ps(a) | isNan_ps(b));
        const cmpeq = @intFromBool(a == b);
        return @bitCast(boolMask_u32x4(notNan & cmpeq));
    }
}

/// dst = a; dst[0] = if ((a[0] == b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0;
pub inline fn _mm_cmpeq_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (0),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (0),
        );
        return res;
    } else {
        const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
        const cmpeq = @intFromBool(a[0] == b[0]);
        return .{ @bitCast(boolMask_u32x1(notNan & cmpeq)), a[1], a[2], a[3] };
    }
}

pub inline fn _mm_cmpge_ps(a: __m128, b: __m128) __m128 {
    return _mm_cmple_ps(b, a);
}

pub inline fn _mm_cmpge_ss(a: __m128, b: __m128) __m128 {
    return _mm_cmple_ss(b, a);
}

pub inline fn _mm_cmpgt_ps(a: __m128, b: __m128) __m128 {
    return _mm_cmplt_ps(b, a);
}

pub inline fn _mm_cmpgt_ss(a: __m128, b: __m128) __m128 {
    return _mm_cmplt_ss(b, a);
}

/// dst[n] = if ((a[n] <= b[n]) and (a[n] != NaN) and b[n] != NaN) -1 else 0;
pub inline fn _mm_cmple_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (2),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (2),
        );
        return res;
    } else {
        const notNan = ~(isNan_ps(a) | isNan_ps(b));
        const cmple = @intFromBool(a <= b);
        return @bitCast(boolMask_u32x4(notNan & cmple));
    }
}

/// dst = a; dst[0] = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0;
pub inline fn _mm_cmple_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (2),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (2),
        );
        return res;
    } else {
        const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
        const cmple = @intFromBool(a[0] <= b[0]);
        return .{ @bitCast(boolMask_u32x1(notNan & cmple)), a[1], a[2], a[3] };
    }
}

/// dst[n] = if ((a[n] < b[n]) and (a[n] != NaN) and b[n] != NaN) -1 else 0;
pub inline fn _mm_cmplt_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (1),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (1),
        );
        return res;
    } else {
        const notNan = ~(isNan_ps(a) | isNan_ps(b));
        const cmplt = @intFromBool(a < b);
        return @bitCast(boolMask_u32x4(notNan & cmplt));
    }
}

/// dst = a; dst[0] = if ((a[0] < b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0;
pub inline fn _mm_cmplt_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (1),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (1),
        );
        return res;
    } else {
        const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
        const cmplt = @intFromBool(a[0] < b[0]);
        return .{ @bitCast(boolMask_u32x1(notNan & cmplt)), a[1], a[2], a[3] };
    }
}

/// dst[n] = if ((a[n] != b[n]) or (a[n] == NaN) or b[n] == NaN) -1 else 0;
pub inline fn _mm_cmpneq_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (4),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (4),
        );
        return res;
    } else {
        const isNan = isNan_ps(a) | isNan_ps(b);
        const cmpneq = @intFromBool(a != b);
        return @bitCast(boolMask_u32x4(isNan | cmpneq));
    }
}

/// dst = a; dst[0] = if ((a[0] != b[0]) or (a[0] == NaN) or b[0] == NaN) -1 else 0;
pub inline fn _mm_cmpneq_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (4),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (4),
        );
        return res;
    } else {
        const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
        const cmpneq = @intFromBool(a[0] != b[0]);
        return .{ @bitCast(boolMask_u32x1(isNan | cmpneq)), a[1], a[2], a[3] };
    }
}

pub inline fn _mm_cmpnge_ps(a: __m128, b: __m128) __m128 {
    return _mm_cmpnle_ps(b, a);
}

pub inline fn _mm_cmpnge_ss(a: __m128, b: __m128) __m128 {
    return _mm_cmpnle_ss(b, a);
}

pub inline fn _mm_cmpngt_ps(a: __m128, b: __m128) __m128 {
    return _mm_cmpnlt_ps(b, a);
}

pub inline fn _mm_cmpngt_ss(a: __m128, b: __m128) __m128 {
    return _mm_cmpnlt_ss(b, a);
}

pub inline fn _mm_cmpnle_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (6),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (6),
        );
        return res;
    } else {
        const isNan = isNan_ps(a) | isNan_ps(b);
        const cmpnle = ~@intFromBool(a <= b);
        return @bitCast(boolMask_u32x4(isNan | cmpnle));
    }
}

pub inline fn _mm_cmpnle_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (6),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (6),
        );
        return res;
    } else {
        const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
        const cmpnle = ~@intFromBool(a[0] <= b[0]);
        return .{ @bitCast(boolMask_u32x1(isNan | cmpnle)), a[1], a[2], a[3] };
    }
}

pub inline fn _mm_cmpnlt_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (5),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (5),
        );
        return res;
    } else {
        const isNan = isNan_ps(a) | isNan_ps(b);
        const cmpnlt = ~@intFromBool(a < b);
        return @bitCast(boolMask_u32x4(isNan | cmpnlt));
    }
}

pub inline fn _mm_cmpnlt_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (5),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (5),
        );
        return res;
    } else {
        const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
        const cmpnlt = ~@intFromBool(a[0] < b[0]);
        return .{ @bitCast(boolMask_u32x1(isNan | cmpnlt)), a[1], a[2], a[3] };
    }
}

/// dst[n] = if ((a[n] != NaN) and (b[n] != NaN)) -1 else 0;
pub inline fn _mm_cmpord_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (7),
        );
        return res;
    } else {
        const pred = ~(isNan_ps(a) | isNan_ps(b));
        return @bitCast(boolMask_u32x4(pred));
    }
}

/// dst = a; dst[0] = if ((a[0] != NaN) and (b[0] != NaN)) -1 else 0;
pub inline fn _mm_cmpord_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (7),
        );
        return res;
    } else {
        const pred = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
        return .{ @bitCast(boolMask_u32x1(pred)), a[1], a[2], a[3] };
    }
}

/// dst[n] = if ((a[n] == NaN) or (b[n] == NaN)) -1 else 0;
pub inline fn _mm_cmpunord_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (3),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (3),
        );
        return res;
    } else {
        const pred = isNan_ps(a) | isNan_ps(b);
        return @bitCast(boolMask_u32x4(pred));
    }
}

test "_mm_cmpunord_ps" {
    const a = _mm_set_epi32(-2147483648, 2139095040, 2139095041, -1);
    const b = _mm_set_epi32(4, 3, 2, 1);
    const ref0 = _mm_set_epi32(0, 0, -1, -1);
    const ref1 = _mm_set_epi32(0, 0, -1, -1);
    const res0: __m128i = @bitCast(_mm_cmpunord_ps(@bitCast(a), @bitCast(b)));
    const res1: __m128i = @bitCast(_mm_cmpunord_ps(@bitCast(b), @bitCast(a)));
    try std.testing.expectEqual(ref0, res0);
    try std.testing.expectEqual(ref1, res1);
}

/// dst = a; dst[0] = if ((a[0] == NaN) or (b[0] == NaN)) -1 else 0;
pub inline fn _mm_cmpunord_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vcmpss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (3),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (3),
        );
        return res;
    } else {
        const pred = isNan_f32(a[0]) | isNan_f32(b[0]);
        return .{ @bitCast(boolMask_u32x1(pred)), a[1], a[2], a[3] };
    }
}

/// TODO: the difference between _mm_comieq_ss and _mm_ucomieq_ss is QNaN signaling
pub inline fn _mm_comieq_ss(a: __m128, b: __m128) i32 {
    _mm_comieq_ss(a, b);
}
pub inline fn _mm_comige_ss(a: __m128, b: __m128) i32 {
    _mm_comige_ss(a, b);
}
pub inline fn _mm_comigt_ss(a: __m128, b: __m128) i32 {
    _mm_comigt_ss(a, b);
}
pub inline fn _mm_comile_ss(a: __m128, b: __m128) i32 {
    _mm_comile_ss(a, b);
}
pub inline fn _mm_comilt_ss(a: __m128, b: __m128) i32 {
    _mm_comilt_ss(a, b);
}
pub inline fn _mm_comineq_ss(a: __m128, b: __m128) i32 {
    _mm_comineq_ss(a, b);
}

pub inline fn _mm_cvt_si2ss(a: __m128, b: i32) __m128 {
    return _mm_cvtsi32_ss(a, b);
}

pub inline fn _mm_cvt_ss2si(a: __m128) i32 {
    return _mm_cvtss_si32(a);
}

pub inline fn _mm_cvtsi32_ss(a: __m128, b: i32) __m128 {
    return .{ @floatFromInt(b), a[1], a[2], a[3] };
}

pub inline fn _mm_cvtsi64_ss(a: __m128, b: i64) __m128 {
    return .{ @floatFromInt(b), a[1], a[2], a[3] };
}

/// Return the lowest f32.
pub inline fn _mm_cvtss_f32(a: __m128) f32 {
    return a[0];
}

/// Convert lowest f32 to i32.
pub inline fn _mm_cvtss_si32(a: __m128) i32 {
    if (has_avx) {
        return asm ("vcvtss2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("cvtss2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_f32(a[0]);
        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u32, @bitCast(r)) & 0x7F800000) >= 0x4F000000) {
            return @bitCast(@as(u32, 0x80000000)); // exponent too large
        }
        return @intFromFloat(r);
    }
}

/// Convert lowest f32 to i64.
pub inline fn _mm_cvtss_si64(a: __m128) i64 {
    if (has_avx) {
        return asm ("vcvtss2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("cvtss2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_f32(a[0]);

        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u32, @bitCast(r)) & 0x7F800000) >= 0x5F000000) {
            return @bitCast(@as(u64, 0x8000000000000000)); // exponent too large
        }
        return @intFromFloat(r);
    }
}

pub inline fn _mm_cvtt_ss2si(a: __m128) i32 {
    return _mm_cvttss_si32(a);
}

pub inline fn _mm_cvttss_si32(a: __m128) i32 {
    if (has_avx) {
        return asm ("vcvttss2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("cvttss2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u32, @bitCast(a[0])) & 0x7F800000) >= 0x4F000000) {
            return -2147483648; // exponent too large
        }
        return @intFromFloat(a[0]); // @trunc is inferred
    }
}

pub inline fn _mm_cvttss_si64(a: __m128) i64 {
    if (has_avx) {
        return asm ("vcvttss2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("cvttss2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u32, @bitCast(a[0])) & 0x7F800000) >= 0x5F000000) {
            return -9223372036854775808; // exponent too large
        }
        return @intFromFloat(a[0]); // @trunc is inferred
    }
}

pub inline fn _mm_div_ps(a: __m128, b: __m128) __m128 {
    return a / b;
}

pub inline fn _mm_div_ss(a: __m128, b: __m128) __m128 {
    return .{ a[0] / b[0], a[1], a[2], a[3] };
}

// ## pub inline fn _mm_free (mem_addr: *anyopaque) void {}
// ## pub inline fn _MM_GET_EXCEPTION_MASK () u32 {}
// ## pub inline fn _MM_GET_EXCEPTION_STATE () u32 {}
// ## pub inline fn _MM_GET_FLUSH_ZERO_MODE () u32 {}
// ## pub inline fn _MM_GET_ROUNDING_MODE () u32 {}
// ## pub inline fn _mm_getcsr (void) u32 {}

// TODO: missing load/store and ?... that use mmx __m64 notation

pub inline fn _mm_load_ps(mem_addr: *align(16) const [4]f32) __m128 {
    return .{ mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3] };
}

pub inline fn _mm_load_ps1(mem_addr: *const f32) __m128 {
    return _mm_load1_ps(mem_addr);
}

pub inline fn _mm_load_ss(mem_addr: *align(1) const f32) __m128 {
    return .{ mem_addr.*, 0, 0, 0 };
}

pub inline fn _mm_load1_ps(mem_addr: *const f32) __m128 {
    return @splat(mem_addr.*);
}

pub inline fn _mm_loadr_ps(mem_addr: *align(16) const [4]f32) __m128 {
    return .{ mem_addr[3], mem_addr[2], mem_addr[1], mem_addr[0] };
}

pub inline fn _mm_loadu_ps(mem_addr: *align(1) const [4]f32) __m128 {
    return .{ mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3] };
}

test "_mm_loadu_ps" {
    const arr = [_]f32{ 1.0, 1.5, 2.0, 2.5, 3.0 };
    const ref = _mm_set_ps(3.0, 2.5, 2.0, 1.5);
    try std.testing.expectEqual(ref, _mm_loadu_ps(arr[1..5]));
}

// ## pub inline fn _mm_malloc (size: usize, align: usize) *anyopaque {}

pub inline fn _mm_max_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vmaxps %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse) {
        var res = a;
        asm ("maxps %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_ps(a) | isNan_ps(b);
        const negZero: u32x4 = @splat(0x80000000);
        const bothZero = @intFromBool((bitCast_u32x4(a) | bitCast_u32x4(b) | negZero) == negZero);
        const cmplt = @intFromBool(a < b);
        const mask = boolMask_u32x4(cmplt | bothZero | isNan);
        return @bitCast((bitCast_u32x4(a) & ~mask) | (bitCast_u32x4(b) & mask));
    }
}

test "_mm_max_ps" {
    const a = _mm_set_epi32(-2147483648, 2139095041, 2139095042, -8388607);
    const b = _mm_set_epi32(0, 0, 2139095041, 2139095041);
    const c = _mm_set_epi32(-2147483647, -8388608, 2, -8388608);
    const d = _mm_set_epi32(-2147483648, -8388607, 1, 2139095040);
    const ref0 = _mm_set_epi32(0, 0, 2139095041, 2139095041);
    const ref1 = _mm_set_epi32(-2147483648, 2139095041, 2139095042, -8388607);
    const ref2 = _mm_set_epi32(-2147483648, -8388607, 2, 2139095040);
    const ref3 = _mm_set_epi32(-2147483648, -8388608, 2, 2139095040);
    const res0: __m128i = @bitCast(_mm_max_ps(@bitCast(a), @bitCast(b)));
    const res1: __m128i = @bitCast(_mm_max_ps(@bitCast(b), @bitCast(a)));
    const res2: __m128i = @bitCast(_mm_max_ps(@bitCast(c), @bitCast(d)));
    const res3: __m128i = @bitCast(_mm_max_ps(@bitCast(d), @bitCast(c)));
    try std.testing.expectEqual(ref0, res0);
    try std.testing.expectEqual(ref1, res1);
    try std.testing.expectEqual(ref2, res2);
    try std.testing.expectEqual(ref3, res3);
}

pub inline fn _mm_max_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vmaxss %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse) {
        var res = a;
        asm ("maxss %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
        const a0: u32 = @bitCast(a[0]);
        const b0: u32 = @bitCast(b[0]);
        const bothZero = @intFromBool((a0 | b0 | 0x80000000) == 0x80000000);
        const cmplt = @intFromBool(a0 < b0);
        const mask = boolMask_u32x1(cmplt | bothZero | isNan);
        return .{ @bitCast((a0 & ~mask) | (b0 & mask)), a[1], a[2], a[3] };
    }
}

pub inline fn _mm_min_ps(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vminps %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse) {
        var res = a;
        asm ("minps %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_ps(a) | isNan_ps(b);
        const negZero: u32x4 = @splat(0x80000000);
        const bothZero = @intFromBool((bitCast_u32x4(a) | bitCast_u32x4(b) | negZero) == negZero);
        const cmpgt = @intFromBool(a > b);
        const mask = boolMask_u32x4(cmpgt | bothZero | isNan);
        return @bitCast((bitCast_u32x4(a) & ~mask) | (bitCast_u32x4(b) & mask));
    }
}

test "_mm_min_ps" {
    const a = _mm_set_epi32(-2147483648, 2139095041, 2139095042, -8388607);
    const b = _mm_set_epi32(0, 0, 2139095041, 2139095041);
    const c = _mm_set_epi32(-2147483647, -8388608, 2, -8388608);
    const d = _mm_set_epi32(-2147483648, -8388607, 1, 2139095040);
    const ref0 = _mm_set_epi32(0, 0, 2139095041, 2139095041);
    const ref1 = _mm_set_epi32(-2147483648, 2139095041, 2139095042, -8388607);
    const ref2 = _mm_set_epi32(-2147483647, -8388607, 1, -8388608);
    const ref3 = _mm_set_epi32(-2147483647, -8388608, 1, -8388608);
    const res0: __m128i = @bitCast(_mm_min_ps(@bitCast(a), @bitCast(b)));
    const res1: __m128i = @bitCast(_mm_min_ps(@bitCast(b), @bitCast(a)));
    const res2: __m128i = @bitCast(_mm_min_ps(@bitCast(c), @bitCast(d)));
    const res3: __m128i = @bitCast(_mm_min_ps(@bitCast(d), @bitCast(c)));
    try std.testing.expectEqual(ref0, res0);
    try std.testing.expectEqual(ref1, res1);
    try std.testing.expectEqual(ref2, res2);
    try std.testing.expectEqual(ref3, res3);
}

pub inline fn _mm_min_ss(a: __m128, b: __m128) __m128 {
    if (has_avx) {
        return asm ("vminss %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse) {
        var res = a;
        asm ("minss %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
        const a0: u32 = @bitCast(a[0]);
        const b0: u32 = @bitCast(b[0]);
        const bothZero = @intFromBool((a0 | b0 | 0x80000000) == 0x80000000);
        const cmpgt = @intFromBool(a0 > b0);
        const mask = boolMask_u32x1(cmpgt | bothZero | isNan);
        return .{ @bitCast((a0 & ~mask) | (b0 & mask)), a[1], a[2], a[3] };
    }
}

pub inline fn _mm_move_ss(a: __m128, b: __m128) __m128 {
    return .{ b[0], a[1], a[2], a[3] };
}

/// shuffle to { b[2], b[3], a[2], a[3] }
pub inline fn _mm_movehl_ps(a: __m128, b: __m128) __m128 {
    return @shuffle(f32, a, b, [4]i32{ -3, -4, 2, 3 });
}

test "_mm_movehl_ps" {
    const a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    const b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    const ref = _mm_set_ps(4.0, 3.0, 40.0, 30.0);
    try std.testing.expectEqual(ref, _mm_movehl_ps(a, b));
}

/// shuffle to { a[0], a[1], b[0], b[1] }
pub inline fn _mm_movelh_ps(a: __m128, b: __m128) __m128 {
    return @shuffle(f32, a, b, [4]i32{ 0, 1, -1, -2 });
}

test "_mm_movelh_ps" {
    const a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    const b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    const ref = _mm_set_ps(20.0, 10.0, 2.0, 1.0);
    try std.testing.expectEqual(ref, _mm_movelh_ps(a, b));
}

pub inline fn _mm_movemask_ps(a: __m128) i32 {
    const cmp = @as(i32x4, @splat(0)) > bitCast_i32x4(a);
    return @intCast(@as(u4, @bitCast(cmp)));
}

pub inline fn _mm_mul_ps(a: __m128, b: __m128) __m128 {
    return a * b;
}

pub inline fn _mm_mul_ss(a: __m128, b: __m128) __m128 {
    return .{ a[0] * b[0], a[1], a[2], a[3] };
}

pub inline fn _mm_or_ps(a: __m128, b: __m128) __m128 {
    return @bitCast(bitCast_u32x4(a) | bitCast_u32x4(b));
}

pub inline fn _mm_prefetch(p: *const anyopaque, comptime i: comptime_int) void {
    @prefetch(p, .{ .rw = .read, .locality = i, .cache = .data });
}

/// Approximate reciprocal
/// Note: AMD and Intel CPUs produce result that are numerically different.
/// Note: rcpps should ignore rounding direction, but results only have ~13-bits of precision ?
pub inline fn _mm_rcp_ps(a: __m128) __m128 {
    if (has_avx) {
        return asm ("vrcpps %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("rcpps %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else {
        return @as(__m128, @splat(1.0)) / a;
    }
}

/// Approximate reciprocal
pub inline fn _mm_rcp_ss(a: __m128) __m128 {
    if (has_avx) {
        return asm ("vrcpss %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("rcpss %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else {
        return .{ 1.0 / a[0], a[1], a[2], a[3] };
    }
}

/// Approximate reciprocal square root
pub inline fn _mm_rsqrt_ps(a: __m128) __m128 {
    if (has_avx) {
        return asm ("vrsqrtps %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("rsqrtps %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else {
        return @as(__m128, @splat(1.0)) / @sqrt(a);
    }
}

/// Approximate reciprocal square root of a[0]
pub inline fn _mm_rsqrt_ss(a: __m128) __m128 {
    if (has_avx) {
        return asm ("vrsqrtss %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else if (has_sse) {
        return asm ("rsqrtss %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
        );
    } else {
        return .{ 1.0 / @sqrt(a[0]), a[1], a[2], a[3] };
    }
}

// ## pub inline fn _MM_SET_EXCEPTION_MASK (a: u32) void {}
// ## pub inline fn _MM_SET_EXCEPTION_STATE (a: u32) void {}
// ## pub inline fn _MM_SET_FLUSH_ZERO_MODE (a: u32) void {}

pub inline fn _mm_set_ps(e3: f32, e2: f32, e1: f32, e0: f32) __m128 {
    return .{ e0, e1, e2, e3 };
}

test "_mm_set_ps" {
    var a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), a[0]);
    try std.testing.expectEqual(@as(f32, 2.0), a[1]);
    try std.testing.expectEqual(@as(f32, 3.0), a[2]);
    try std.testing.expectEqual(@as(f32, 4.0), a[3]);

    // test assigning to self (https://github.com/ziglang/zig/issues/18082)
    a = _mm_set_ps(a[3], a[1], a[0], a[2]);
    try std.testing.expectEqual(@as(f32, 3.0), a[0]);
    try std.testing.expectEqual(@as(f32, 1.0), a[1]);
    try std.testing.expectEqual(@as(f32, 2.0), a[2]);
    try std.testing.expectEqual(@as(f32, 4.0), a[3]);
}

pub inline fn _mm_set_ps1(a: f32) __m128 {
    return _mm_set1_ps(a);
}

// ## pub inline fn _MM_SET_ROUNDING_MODE (a: u32) void {}

pub inline fn _mm_set_ss(a: f32) __m128 {
    return .{ a, 0, 0, 0 };
}

pub inline fn _mm_set1_ps(a: f32) __m128 {
    return @splat(a);
}

// ## pub inline fn _mm_setcsr (a: u32) void {}

pub inline fn _mm_setr_ps(e3: f32, e2: f32, e1: f32, e0: f32) __m128 {
    return .{ e3, e2, e1, e0 };
}

pub inline fn _mm_setzero_ps() __m128 {
    return @splat(0);
}

pub inline fn _mm_sfence() void {
    if (has_sse) {
        asm volatile ("sfence" ::: "memory");
    } else {
        _mm_mfence();
    }
}

pub inline fn _mm_shuffle_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    const shuf = [4]i32{ imm8 & 3, (imm8 >> 2) & 3, -((imm8 >> 4) & 3) - 1, -((imm8 >> 6) & 3) - 1 };
    return @shuffle(f32, a, b, shuf);
}

pub inline fn _mm_sqrt_ps(a: __m128) __m128 {
    return @sqrt(a);
}

pub inline fn _mm_sqrt_ss(a: __m128) __m128 {
    return .{ @sqrt(a[0]), a[1], a[2], a[3] };
}

pub inline fn _mm_store_ps(mem_addr: *align(16) [4]f32, a: __m128) void {
    for (0..4) |i| mem_addr[i] = a[i];
}

pub inline fn _mm_store_ps1(mem_addr: *align(16) [4]f32, a: __m128) void {
    _mm_store1_ps(mem_addr, a);
}

pub inline fn _mm_store_ss(mem_addr: *align(1) f32, a: __m128) void {
    mem_addr.* = a[0];
}

pub inline fn _mm_store1_ps(mem_addr: *align(16) [4]f32, a: __m128) void {
    for (0..4) |i| mem_addr[i] = a[0];
}

pub inline fn _mm_storer_ps(mem_addr: *align(16) [4]f32, a: __m128) void {
    for (0..4) |i| mem_addr[i] = a[3 - i];
}

pub inline fn _mm_storeu_ps(mem_addr: *align(1) [4]f32, a: __m128) void {
    for (0..4) |i| mem_addr[i] = a[i];
}

// ## pub inline fn _mm_stream_ps (mem_addr: *align(16) [4]f32, a: __m128) void {}

pub inline fn _mm_sub_ps(a: __m128, b: __m128) __m128 {
    return a - b;
}

pub inline fn _mm_sub_ss(a: __m128, b: __m128) __m128 {
    return .{ a[0] - b[0], a[1], a[2], a[3] };
}

pub inline fn _MM_TRANSPOSE4_PS(row0: *__m128, row1: *__m128, row2: *__m128, row3: *__m128) void {
    const tmp0 = _mm_unpacklo_ps(row0.*, row1.*);
    const tmp2 = _mm_unpacklo_ps(row2.*, row3.*);
    const tmp1 = _mm_unpackhi_ps(row0.*, row1.*);
    const tmp3 = _mm_unpackhi_ps(row2.*, row3.*);
    row0.* = _mm_movelh_ps(tmp0, tmp2);
    row1.* = _mm_movehl_ps(tmp2, tmp0);
    row2.* = _mm_movelh_ps(tmp1, tmp3);
    row3.* = _mm_movehl_ps(tmp3, tmp1);
}

test "_MM_TRANSPOSE4_PS" {
    var a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    var b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    var c = _mm_set_ps(400.0, 300.0, 200.0, 100.0);
    var d = _mm_set_ps(4000.0, 3000.0, 2000.0, 1000.0);
    const a_ref = _mm_set_ps(1000.0, 100.0, 10.0, 1.0);
    const b_ref = _mm_set_ps(2000.0, 200.0, 20.0, 2.0);
    const c_ref = _mm_set_ps(3000.0, 300.0, 30.0, 3.0);
    const d_ref = _mm_set_ps(4000.0, 400.0, 40.0, 4.0);
    _MM_TRANSPOSE4_PS(&a, &b, &c, &d);
    try std.testing.expectEqual(a_ref, a);
    try std.testing.expectEqual(b_ref, b);
    try std.testing.expectEqual(c_ref, c);
    try std.testing.expectEqual(d_ref, d);
}

/// result = if ((a[0] == b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomieq_ss(a: __m128, b: __m128) i32 {
    const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
    const cmpeq = @intFromBool(a[0] == b[0]);
    return notNan & cmpeq;
}

/// result = if ((a[0] >= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomige_ss(a: __m128, b: __m128) i32 {
    return _mm_ucomile_ss(b, a);
}

/// result = if ((a[0] > b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomigt_ss(a: __m128, b: __m128) i32 {
    return _mm_ucomilt_ss(b, a);
}

/// result = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomile_ss(a: __m128, b: __m128) i32 {
    const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
    const cmple = @intFromBool(a[0] <= b[0]);
    return notNan & cmple;
}

/// result = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomilt_ss(a: __m128, b: __m128) i32 {
    const notNan = ~(isNan_f32(a[0]) | isNan_f32(b[0]));
    const cmplt = @intFromBool(a[0] < b[0]);
    return notNan & cmplt;
}

/// result = if ((a[0] != b[0]) or (a[0] == NaN) or b[0] == NaN) 1 else 0;
pub inline fn _mm_ucomineq_ss(a: __m128, b: __m128) i32 {
    const isNan = isNan_f32(a[0]) | isNan_f32(b[0]);
    const cmpneq = @intFromBool(a != b);
    return isNan | cmpneq;
}

pub inline fn _mm_undefined_ps() __m128 {
    // zig `undefined` doesn't compare equal to itself ?
    return @splat(0);
}

/// shuffle { a[2], b[2], a[3], b[3] };
pub inline fn _mm_unpackhi_ps(a: __m128, b: __m128) __m128 {
    if (bug_stage2_x86_64) { // error: unsupported mnemonic: 'unpckhps'
        return .{ a[2], b[2], a[3], b[3] };
    }
    return @shuffle(f32, a, b, [4]i32{ 2, -3, 3, -4 });
}

test "_mm_unpackhi_ps" {
    const a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    const b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    const ref = _mm_set_ps(40.0, 4.0, 30.0, 3.0);
    try std.testing.expectEqual(ref, _mm_unpackhi_ps(a, b));
}

/// shuffle { a[0], b[0], a[1], b[1] };
pub inline fn _mm_unpacklo_ps(a: __m128, b: __m128) __m128 {
    if (bug_stage2_x86_64) { // error: unsupported mnemonic: 'unpcklps'
        return .{ a[0], b[0], a[1], b[1] };
    }
    return @shuffle(f32, a, b, [4]i32{ 0, -1, 1, -2 });
}

test "_mm_unpacklo_ps" {
    const a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    const b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    const ref = _mm_set_ps(20.0, 2.0, 10.0, 1.0);
    try std.testing.expectEqual(ref, _mm_unpacklo_ps(a, b));
}

pub inline fn _mm_xor_ps(a: __m128, b: __m128) __m128 {
    return @bitCast(bitCast_u32x4(a) ^ bitCast_u32x4(b));
}

// SSE2 ================================================================

pub inline fn _mm_add_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) +% bitCast_u16x8(b));
}

test _mm_add_epi16 {
    const a = _mm_set_epi16(0, 7, 6, -32768, -1, 3, 2, 1);
    const b = _mm_set_epi16(0, 249, 5, -32768, 4, -16, 32767, -1);
    const ref = _mm_set_epi16(0, 256, 11, 0, 3, -13, -32767, 0);
    try std.testing.expectEqual(ref, _mm_add_epi16(a, b));
}

pub inline fn _mm_add_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u32x4(a) +% bitCast_u32x4(b));
}

test _mm_add_epi32 {
    const a = _mm_set_epi32(4, -2147483648, 2, 1);
    const b = _mm_set_epi32(2147483647, 3, -1, 65535);
    const ref = _mm_set_epi32(-2147483645, -2147483645, 1, 65536);
    try std.testing.expectEqual(ref, _mm_add_epi32(a, b));
}

pub inline fn _mm_add_epi64(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u64x2(a) +% bitCast_u64x2(b));
}

test _mm_add_epi64 {
    const a = _mm_set_epi64x(9223372036854775807, 1);
    const b = _mm_set_epi64x(2, -1);
    const ref = _mm_set_epi64x(-9223372036854775807, 0);
    try std.testing.expectEqual(ref, _mm_add_epi64(a, b));
}

pub inline fn _mm_add_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) +% bitCast_u8x16(b));
}

test _mm_add_epi8 {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const b = _mm_set_epi8(1, 2, 3, -4, 0, 1, 127, -1, -1, -1, -1, -128, -128, 0, 0, 0);
    const ref = _mm_set_epi8(16, 16, 16, 8, 11, 11, -120, 7, 6, 5, 4, -124, -125, 2, 1, 0);
    try std.testing.expectEqual(ref, _mm_add_epi8(a, b));
}

pub inline fn _mm_add_pd(a: __m128d, b: __m128d) __m128d {
    return a + b;
}

pub inline fn _mm_add_sd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] + b[0], a[1] };
}

pub inline fn _mm_adds_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) +| bitCast_i16x8(b));
}

pub inline fn _mm_adds_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i8x16(a) +| bitCast_i8x16(b));
}

pub inline fn _mm_adds_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) +| bitCast_u16x8(b));
}

pub inline fn _mm_adds_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) +| bitCast_u8x16(b));
}

pub inline fn _mm_and_pd(a: __m128d, b: __m128d) __m128d {
    return @bitCast(bitCast_u64x2(a) & bitCast_u64x2(b));
}

pub inline fn _mm_and_si128(a: __m128i, b: __m128i) __m128i {
    return a & b;
}

pub inline fn _mm_andnot_pd(a: __m128d, b: __m128d) __m128d {
    return @bitCast(~bitCast_u64x2(a) & bitCast_u64x2(b));
}

pub inline fn _mm_andnot_si128(a: __m128i, b: __m128i) __m128i {
    return ~a & b;
}

pub inline fn _mm_avg_epu16(a: __m128i, b: __m128i) __m128i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to pavgw
    const one: u32x8 = @splat(1);
    const c = intCast_u32x8(bitCast_u16x8(a));
    const d = intCast_u32x8(bitCast_u16x8(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(u16x8, @truncate(e)));
}

pub inline fn _mm_avg_epu8(a: __m128i, b: __m128i) __m128i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to pavgb
    const one: u16x16 = @splat(1);
    const c = intCast_u16x16(bitCast_u8x16(a));
    const d = intCast_u16x16(bitCast_u8x16(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(u8x16, @truncate(e)));
}

/// alternative name
pub inline fn _mm_bslli_si128(a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_slli_si128(a, imm8);
}

/// alternative name
pub inline fn _mm_bsrli_si128(a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_srli_si128(a, imm8);
}

pub inline fn _mm_castpd_ps(a: __m128d) __m128 {
    return @bitCast(a);
}

pub inline fn _mm_castpd_si128(a: __m128d) __m128i {
    return @bitCast(a);
}

pub inline fn _mm_castps_pd(a: __m128) __m128d {
    return @bitCast(a);
}

pub inline fn _mm_castps_si128(a: __m128) __m128i {
    return @bitCast(a);
}

pub inline fn _mm_castsi128_pd(a: __m128i) __m128d {
    return @bitCast(a);
}

pub inline fn _mm_castsi128_ps(a: __m128i) __m128 {
    return @bitCast(a);
}

// ## pub inline fn _mm_clflush (p: *const anyopaque) void {}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_u16x8(a) == bitCast_u16x8(b));
    return @bitCast(boolMask_u16x8(pred));
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_u32x4(a) == bitCast_u32x4(b));
    return @bitCast(boolMask_u32x4(pred));
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_u8x16(a) == bitCast_u8x16(b));
    return @bitCast(boolMask_u8x16(pred));
}

/// dst[n] = if ((a[n] == b[n]) and (a[n] != NaN) and b[n] != NaN) -1 else 0;
pub inline fn _mm_cmpeq_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (0),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (0),
        );
        return res;
    } else {
        const notNan = ~(isNan_pd(a) | isNan_pd(b));
        const cmpeq = @intFromBool(a == b);
        return @bitCast(boolMask_u64x2(notNan & cmpeq));
    }
}

/// dst[0] = if ((a[0] == b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmpeq_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (0),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (0),
        );
        return res;
    } else {
        const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
        const cmpeq = @intFromBool(a[0] == b[0]);
        return .{ @bitCast(boolMask_u64x1(notNan & cmpeq)), a[1] };
    }
}

pub inline fn _mm_cmpge_pd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmple_pd(b, a);
}

pub inline fn _mm_cmpge_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmple_sd(b, a);
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_i16x8(a) > bitCast_i16x8(b));
    return @bitCast(boolMask_u16x8(pred));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_i32x4(a) > bitCast_i32x4(b));
    return @bitCast(boolMask_u32x4(pred));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_i8x16(a) > bitCast_i8x16(b));
    return @bitCast(boolMask_u8x16(pred));
}

pub inline fn _mm_cmpgt_pd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmplt_pd(b, a);
}

pub inline fn _mm_cmpgt_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmplt_sd(b, a);
}

pub inline fn _mm_cmple_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (2),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (2),
        );
        return res;
    } else {
        const notNan = ~(isNan_pd(a) | isNan_pd(b));
        const cmple = @intFromBool(a <= b);
        return @bitCast(boolMask_u64x2(notNan & cmple));
    }
}

/// dst[0] = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmple_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (2),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (2),
        );
        return res;
    } else {
        const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
        const cmple = @intFromBool(a[0] <= b[0]);
        return .{ @bitCast(boolMask_u64x1(notNan & cmple)), a[1] };
    }
}

/// dst[n] = if (a[n] < b[n]) -1 else 0;
pub inline fn _mm_cmplt_epi16(a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi16(b, a);
}

/// dst[n] = if (a[n] < b[n]) -1 else 0;
pub inline fn _mm_cmplt_epi32(a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi32(b, a);
}

/// dst[n] = if (a[n] < b[n]) -1 else 0;
pub inline fn _mm_cmplt_epi8(a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi8(b, a);
}

/// dst[n] = if ((a[n] < b[n]) and (a[n] != NaN) and b[n] != NaN) -1 else 0;
pub inline fn _mm_cmplt_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (1),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (1),
        );
        return res;
    } else {
        const notNan = ~(isNan_pd(a) | isNan_pd(b));
        const cmplt = @intFromBool(a < b);
        return @bitCast(boolMask_u64x2(notNan & cmplt));
    }
}

/// dst[0] = if ((a[0] < b[0]) and (a[0] != NaN) and b[0] != NaN) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmplt_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (1),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (1),
        );
        return res;
    } else {
        const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
        const cmplt = @intFromBool(a[0] < b[0]);
        return .{ @bitCast(boolMask_u64x1(notNan & cmplt)), a[1] };
    }
}

/// dst[n] = if ((a[n] != b[n]) or (a[n] == NaN) or b[n] == NaN) -1 else 0;
pub inline fn _mm_cmpneq_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (4),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (4),
        );
        return res;
    } else {
        const isNan = isNan_pd(a) | isNan_pd(b);
        const cmpneq = @intFromBool(a != b);
        return @bitCast(boolMask_u64x2(isNan | cmpneq));
    }
}

/// dst[0] = if ((a[0] != b[0]) or (a[0] == NaN) or b[0] == NaN) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmpneq_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (4),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (4),
        );
        return res;
    } else {
        const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
        const cmpneq = @intFromBool(a[0] != b[0]);
        return .{ @bitCast(boolMask_u64x1(isNan | cmpneq)), a[1] };
    }
}

pub inline fn _mm_cmpnge_pd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmpnle_pd(b, a);
}

pub inline fn _mm_cmpnge_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmpnle_sd(b, a);
}

pub inline fn _mm_cmpngt_pd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmpnlt_pd(b, a);
}

pub inline fn _mm_cmpngt_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_cmpnle_sd(b, a);
}

pub inline fn _mm_cmpnle_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (6),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (6),
        );
        return res;
    } else {
        const isNan = isNan_pd(a) | isNan_pd(b);
        const cmpnle = ~@intFromBool(a <= b);
        return @bitCast(boolMask_u64x2(isNan | cmpnle));
    }
}

pub inline fn _mm_cmpnle_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (6),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (6),
        );
        return res;
    } else {
        const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
        const cmpnle = ~@intFromBool(a[0] <= b[0]);
        return .{ @bitCast(boolMask_u64x1(isNan | cmpnle)), a[1] };
    }
}

pub inline fn _mm_cmpnlt_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (5),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (5),
        );
        return res;
    } else {
        const isNan = isNan_pd(a) | isNan_pd(b);
        const cmpnlt = ~@intFromBool(a < b);
        return @bitCast(boolMask_u64x2(isNan | cmpnlt));
    }
}

pub inline fn _mm_cmpnlt_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (5),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (5),
        );
        return res;
    } else {
        const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
        const cmpnlt = ~@intFromBool(a[0] < b[0]);
        return .{ @bitCast(boolMask_u64x1(isNan | cmpnlt)), a[1] };
    }
}

/// dst[n] = if ((a[n] != NaN) and (b[n] != NaN)) -1 else 0;
pub inline fn _mm_cmpord_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (7),
        );
        return res;
    } else {
        const pred = ~(isNan_pd(a) | isNan_pd(b));
        return @bitCast(boolMask_u64x2(pred));
    }
}

/// dst[0] = if ((a[0] != NaN) and (b[0] != NaN)) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmpord_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmpsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (7),
        );
        return res;
    } else {
        const pred = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
        return .{ @bitCast(boolMask_u64x1(pred)), a[1] };
    }
}

/// dst[n] = if ((a[n] == NaN) or (b[n] == NaN)) -1 else 0;
pub inline fn _mm_cmpunord_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vcmppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (3),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (3),
        );
        return res;
    } else {
        const pred = isNan_pd(a) | isNan_pd(b);
        return @bitCast(boolMask_u64x2(pred));
    }
}

test "_mm_cmpunord_pd" {
    const a = _mm_set_epi64x(-9223372036854775808, 9218868437227405312);
    const b = _mm_set_epi64x(9218868437227405313, -1);
    const c = _mm_set_epi64x(2, 1);
    const ref0 = _mm_set_epi64x(0, 0);
    const ref1 = _mm_set_epi64x(-1, -1);
    const ref2 = _mm_set_epi64x(-1, -1);
    const res0: __m128i = @bitCast(_mm_cmpunord_pd(@bitCast(a), @bitCast(c)));
    const res1: __m128i = @bitCast(_mm_cmpunord_pd(@bitCast(b), @bitCast(c)));
    const res2: __m128i = @bitCast(_mm_cmpunord_pd(@bitCast(c), @bitCast(b)));
    try std.testing.expectEqual(ref0, res0);
    try std.testing.expectEqual(ref1, res1);
    try std.testing.expectEqual(ref2, res2);
}

/// dst[0] = if ((a[0] == NaN) or (b[0] == NaN)) -1 else 0; dst[1] = a[1];
pub inline fn _mm_cmpunord_sd(a: __m128d, b: __m128d) __m128d {
    const pred = isNan_f64(a[0]) | isNan_f64(b[0]);
    return .{ @bitCast(boolMask_u64x1(pred)), a[1] };
}

/// TODO: the difference between _mm_comieq_sd and _mm_ucomieq_sd is QNaN signaling
pub inline fn _mm_comieq_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomieq_sd(a, b);
}
pub inline fn _mm_comige_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomige_sd(a, b);
}
pub inline fn _mm_comigt_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomigt_sd(a, b);
}
pub inline fn _mm_comile_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomile_sd(a, b);
}
pub inline fn _mm_comilt_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomilt_sd(a, b);
}
pub inline fn _mm_comineq_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomineq_sd(a, b);
}

pub inline fn _mm_cvtepi32_pd(a: __m128i) __m128d {
    return .{ @floatFromInt(bitCast_i32x4(a)[0]), @floatFromInt(bitCast_i32x4(a)[1]) };
}

pub inline fn _mm_cvtepi32_ps(a: __m128i) __m128 {
    return @floatFromInt(bitCast_i32x4(a));
}

pub inline fn _mm_cvtpd_epi32(a: __m128d) __m128i {
    if (has_avx) {
        return asm ("vcvtpd2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvtpd2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_pd(a);

        @setRuntimeSafety(false); // intFromFloat is guarded
        const mask: u64x2 = @splat(0x7FF0000000000000);
        const limit: u64x2 = @splat(0x41E0000000000000);
        const indefinite: __m128d = @bitCast(@as(u64x2, @splat(0xC1E0000000000000)));
        const predicate = (bitCast_u32x4(r) & mask) >= limit;
        const clamped = @select(64, predicate, indefinite, r);

        const i = @as(i32x2, @intFromFloat(clamped));
        return @bitCast(i32x4{ i[0], i[1], 0, 0 });
    }
}

pub inline fn _mm_cvtpd_ps(a: __m128d) __m128 {
    return .{ @floatCast(a[0]), @floatCast(a[1]), 0, 0 };
}

/// Convert floats to integers.
pub inline fn _mm_cvtps_epi32(a: __m128) __m128i {
    if (has_avx) {
        return asm ("vcvtps2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvtps2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_ps(a);

        @setRuntimeSafety(false); // intFromFloat is guarded
        const mask: u32x4 = @splat(0x7F800000);
        const limit: u32x4 = @splat(0x4F000000);
        const indefinite: __m128 = @bitCast(@as(u32x4, @splat(0xCF000000)));
        const predicate = (bitCast_u32x4(r) & mask) >= limit;
        const clamped = @select(f32, predicate, indefinite, r);

        return @intFromFloat(clamped);
    }
}

pub inline fn _mm_cvtps_pd(a: __m128) __m128d {
    return .{ @floatCast(a[0]), @floatCast(a[1]) };
}

/// Return the lowest f64.
pub inline fn _mm_cvtsd_f64(a: __m128d) f64 {
    return a[0];
}

pub inline fn _mm_cvtsd_si32(a: __m128d) i32 {
    if (has_avx) {
        return asm ("vcvtsd2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvtsd2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_f64(a[0]);

        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u64, @bitCast(r)) & 0x7FF0000000000000) >= 0x41E0000000000000) {
            return @bitCast(@as(u32, 0x80000000)); // exponent too large
        }
        return @intFromFloat(r);
    }
}

pub inline fn _mm_cvtsd_si64(a: __m128d) i64 {
    return _mm_cvtsd_si64x(a);
}

pub inline fn _mm_cvtsd_si64x(a: __m128d) i64 {
    if (has_avx) {
        return asm ("vcvtsd2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvtsd2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else {
        const r = RoundCurrentDirection_f64(a[0]);

        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u64, @bitCast(r)) & 0x7FF0000000000000) >= 0x43E0000000000000) {
            return @bitCast(@as(u64, 0x8000000000000000)); // exponent too large
        }
        return @intFromFloat(r);
    }
}

pub inline fn _mm_cvtsd_ss(a: __m128, b: __m128d) __m128 {
    return .{ @floatCast(b[0]), a[0], a[1], a[2] };
}

pub inline fn _mm_cvtsi128_si32(a: __m128i) i32 {
    return bitCast_i32x4(a)[0];
}

pub inline fn _mm_cvtsi128_si64(a: __m128i) i64 {
    return bitCast_i64x2(a)[0];
}

pub inline fn _mm_cvtsi128_si64x(a: __m128i) i64 {
    return _mm_cvtsi128_si64(a);
}

pub inline fn _mm_cvtsi32_sd(a: __m128d, b: i32) __m128d {
    return .{ @floatFromInt(b), a[1] };
}

pub inline fn _mm_cvtsi32_si128(a: i32) __m128i {
    const r = i32x4{ a, 0, 0, 0 };
    return @bitCast(r);
}

pub inline fn _mm_cvtsi64_sd(a: __m128d, b: i64) __m128d {
    return .{ @floatFromInt(b), a[1] };
}

pub inline fn _mm_cvtsi64_si128(a: i64) __m128i {
    const r = i64x2{ a, 0 };
    return @bitCast(r);
}

pub inline fn _mm_cvtsi64x_sd(a: __m128d, b: i64) __m128d {
    return _mm_cvtsi64_sd(a, b);
}

pub inline fn _mm_cvtsi64x_si128(a: i64) __m128i {
    return _mm_cvtsi64_si128(a);
}

pub inline fn _mm_cvtss_sd(a: __m128d, b: __m128) __m128d {
    return .{ @floatCast(b[0]), a[1] };
}

pub inline fn _mm_cvttpd_epi32(a: __m128d) __m128i {
    if (has_avx) {
        return asm ("vcvttpd2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvttpd2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        const mask: u64x2 = @splat(0x7FF0000000000000);
        const limit: u64x2 = @splat(0x41E0000000000000);
        const indefinite: __m128d = @bitCast(@as(u64x2, @splat(0xC1E0000000000000)));
        const predicate = (bitCast_u32x4(a) & mask) >= limit;
        const clamped = @select(64, predicate, indefinite, a);

        const i = @as(i32x2, @intFromFloat(clamped));
        return @bitCast(i32x4{ i[0], i[1], 0, 0 });
    }
}

pub inline fn _mm_cvttps_epi32(a: __m128) __m128i {
    if (has_avx) {
        return asm ("vcvttps2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvttps2dq %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        const mask: u32x4 = @splat(0x7F800000);
        const limit: u32x4 = @splat(0x4F000000);
        const indefinite: __m128 = @bitCast(@as(u32x4, @splat(0xCF000000)));
        const predicate = (bitCast_u32x4(a) & mask) >= limit;
        const clamped = @select(f32, predicate, indefinite, a);
        return @bitCast(@as(i32x4, @intFromFloat(clamped))); // @trunc is inferred
    }
}

pub inline fn _mm_cvttsd_si32(a: __m128d) i32 {
    if (has_avx) {
        return asm ("vcvttsd2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvttsd2si %[a], %[ret]"
            : [ret] "=r" (-> i32),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u64, @bitCast(a)) & 0x7FF0000000000000) >= 0x41E0000000000000) {
            return @bitCast(@as(u32, 0x80000000)); // exponent too large
        }
        return @intFromFloat(a); // @trunc is inferred
    }
}

pub inline fn _mm_cvttsd_si64(a: __m128d) i64 {
    return _mm_cvttsd_si64x(a);
}

pub inline fn _mm_cvttsd_si64x(a: __m128d) i64 {
    if (has_avx) {
        return asm ("vcvttsd2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else if (has_sse2) {
        return asm ("cvttsd2si %[a], %[ret]"
            : [ret] "=r" (-> i64),
            : [a] "x" (a),
        );
    } else {
        @setRuntimeSafety(false); // intFromFloat is guarded
        if ((@as(u64, @bitCast(a)) & 0x7FF0000000000000) >= 0x43E0000000000000) {
            return @bitCast(@as(u64, 0x8000000000000000)); // exponent too large
        }
        return @intFromFloat(a); // @trunc is inferred
    }
}

pub inline fn _mm_div_pd(a: __m128d, b: __m128d) __m128d {
    return a / b;
}

pub inline fn _mm_div_sd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] / b[0], a[1] };
}

/// extract u16 then zero-extend to i32
pub inline fn _mm_extract_epi16(a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_u16x8(a)[imm8];
}

pub inline fn _mm_insert_epi16(a: __m128i, i: i16, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i16x8(a);
    r[imm8] = i;
    return @bitCast(r);
}

pub inline fn _mm_lfence() void {
    if (has_sse2) {
        asm volatile ("lfence" ::: "memory");
    } else {
        _mm_mfence();
    }
}

pub inline fn _mm_load_pd(mem_addr: *align(16) const [2]f64) __m128d {
    return .{ mem_addr[0], mem_addr[1] };
}

pub inline fn _mm_load_pd1(mem_addr: *const f64) __m128d {
    return _mm_load1_pd(mem_addr);
}

pub inline fn _mm_load_sd(mem_addr: *align(1) const f64) __m128d {
    return .{ mem_addr.*, 0 };
}

pub inline fn _mm_load_si128(mem_addr: *const __m128i) __m128i {
    return mem_addr.*;
}

pub inline fn _mm_load1_pd(mem_addr: *const f64) __m128d {
    return @splat(mem_addr.*);
}

pub inline fn _mm_loadh_pd(a: __m128d, mem_addr: *align(1) const f64) __m128d {
    return .{ a[0], mem_addr.* };
}

// Despite the signature, this is the same as _mm_loadu_si64
pub inline fn _mm_loadl_epi64(mem_addr: *align(1) const __m128i) __m128i {
    return _mm_loadu_si64(@ptrCast(mem_addr));
}

pub inline fn _mm_loadl_pd(a: __m128d, mem_addr: *align(1) const f64) __m128d {
    return .{ mem_addr.*, a[1] };
}

pub inline fn _mm_loadr_pd(mem_addr: *align(16) const [2]f64) __m128d {
    return .{ mem_addr[1], mem_addr[0] };
}

pub inline fn _mm_loadu_pd(mem_addr: *align(1) const [2]f64) __m128d {
    return .{ mem_addr[0], mem_addr[1] };
}

pub inline fn _mm_loadu_si128(mem_addr: *align(1) const __m128i) __m128i {
    return mem_addr.*;
}

test "_mm_loadu_si128" {
    const arr: [17]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const ref0 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const ref1 = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    try std.testing.expectEqual(ref0, _mm_loadu_si128(@ptrCast(&arr[0])));
    try std.testing.expectEqual(ref1, _mm_loadu_si128(@ptrCast(&arr[1])));
}

pub inline fn _mm_loadu_si16(mem_addr: *const anyopaque) __m128i {
    const word = @as(*align(1) const u16, @ptrCast(mem_addr)).*;
    return @bitCast(u16x8{ word, 0, 0, 0, 0, 0, 0, 0 });
}

test "_mm_loadu_si16" {
    const arr: [3]u8 = .{ 1, 2, 3 };
    const ref0 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1);
    const ref1 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0x0302);
    try std.testing.expectEqual(ref0, _mm_loadu_si16(&arr[0]));
    try std.testing.expectEqual(ref1, _mm_loadu_si16(&arr[1]));
}

pub inline fn _mm_loadu_si32(mem_addr: *const anyopaque) __m128i {
    const dword = @as(*align(1) const u32, @ptrCast(mem_addr)).*;
    return @bitCast(u32x4{ dword, 0, 0, 0 });
}

test "_mm_loadu_si32" {
    const arr: [5]u8 = .{ 1, 2, 3, 4, 5 };
    const ref0 = _mm_set_epi32(0, 0, 0, 0x04030201);
    const ref1 = _mm_set_epi32(0, 0, 0, 0x05040302);
    try std.testing.expectEqual(ref0, _mm_loadu_si32(&arr[0]));
    try std.testing.expectEqual(ref1, _mm_loadu_si32(&arr[1]));
}

pub inline fn _mm_loadu_si64(mem_addr: *const anyopaque) __m128i {
    const qword = @as(*align(1) const u64, @ptrCast(mem_addr)).*;
    return @bitCast(u64x2{ qword, 0 });
}

test "_mm_loadu_si64" {
    const arr: [9]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ref0 = _mm_set_epi64x(0, 0x0807060504030201);
    const ref1 = _mm_set_epi64x(0, 0x0908070605040302);
    try std.testing.expectEqual(ref0, _mm_loadu_si64(&arr[0]));
    try std.testing.expectEqual(ref1, _mm_loadu_si64(&arr[1]));
}

pub inline fn _mm_madd_epi16(a: __m128i, b: __m128i) __m128i {
    const r = intCast_i32x8(bitCast_i16x8(a)) *%
        intCast_i32x8(bitCast_i16x8(b));

    const even = @shuffle(i32, r, undefined, [4]i32{ 0, 2, 4, 6 });
    const odd = @shuffle(i32, r, undefined, [4]i32{ 1, 3, 5, 7 });
    return @bitCast(even +% odd);
}

test "_mm_madd_epi16" {
    const a = _xx_set_epu16(0x8000, 0x7FFF, 0x8000, 0x0000, 0xFFFF, 0x8000, 0x7FFF, 0x7FFF);
    const b = _mm_set_epi16(2, 3, -32768, 4, 128, 32767, 255, 5);
    const ref = _mm_set_epi32(32765, 1073741824, -1073709184, 8519420);
    try std.testing.expectEqual(ref, _mm_madd_epi16(a, b));
    try std.testing.expectEqual(ref, _mm_madd_epi16(b, a));
}

// ## pub inline fn _mm_maskmoveu_si128(a: __m128i, mask: __m128i, mem_addr: *[16]u8) void {}

pub inline fn _mm_max_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_i16x8(bitCast_i16x8(a), bitCast_i16x8(b)));
}

test "_mm_max_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -32768, 0, -1, -32768, 1, 32767);
    const b = _mm_set_epi16(-1, -32768, -32767, -1, 32767, -32768, -2, 32494);
    const ref = _mm_set_epi16(-1, 32767, -32767, 0, 32767, -32768, 1, 32767);
    try std.testing.expectEqual(ref, _mm_max_epi16(a, b));
    try std.testing.expectEqual(ref, _mm_max_epi16(b, a));
}

pub inline fn _mm_max_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_u8x16(bitCast_u8x16(a), bitCast_u8x16(b)));
}

test "_mm_max_epu8" {
    const a = _xx_set_epu8(0x80, 0x7F, 0x80, 0x00, 0xFF, 0x80, 0x01, 0x7F, 0x05, 0x00, 0x00, 0x00, 0x80, 0x04, 0x03, 0x02);
    const b = _xx_set_epu8(0xFF, 0x80, 0x81, 0xFF, 0x7F, 0x80, 0xFE, 0x7E, 0x04, 0x01, 0xFF, 0x02, 0x7F, 0x05, 0x06, 0x07);
    const ref = _xx_set_epu8(0xFF, 0x80, 0x81, 0xFF, 0xFF, 0x80, 0xFE, 0x7F, 0x05, 0x01, 0xFF, 0x02, 0x80, 0x05, 0x06, 0x07);
    try std.testing.expectEqual(ref, _mm_max_epu8(a, b));

    const c = _mm_set1_epi8(1);
    try std.testing.expectEqual(c, _mm_max_epu8(c, c));
}

pub inline fn _mm_max_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vmaxpd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("maxpd %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_pd(a) | isNan_pd(b);
        const negZero: u64x2 = @splat(0x8000000000000000);
        const bothZero = @intFromBool((bitCast_u64x2(a) | bitCast_u64x2(b) | negZero) == negZero);
        const cmplt = @intFromBool(a < b);
        const mask = boolMask_u64x2(cmplt | bothZero | isNan);
        return @bitCast((bitCast_u64x2(a) & ~mask) | (bitCast_u64x2(b) & mask));
    }
}

pub inline fn _mm_max_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vmaxsd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("maxsd %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
        const a0: u64 = @bitCast(a[0]);
        const b0: u64 = @bitCast(b[0]);
        const bothZero = @intFromBool((a0 | b0 | 0x80000000) == 0x80000000);
        const cmplt = @intFromBool(a0 < b0);
        const mask = boolMask_u64x1(cmplt | bothZero | isNan);
        return .{ @bitCast((a0 & ~mask) | (b0 & mask)), a[1] };
    }
}

pub inline fn _mm_mfence() void {
    if (has_sse2) {
        asm volatile ("mfence" ::: "memory");
    } else {
        // `@fence(.SeqCst)` was removed in ziglang/zig#21585
        // garbage work-around
        var x = std.atomic.Value(usize).init(0);
        _ = x.cmpxchgStrong(1, 0, .seq_cst, .seq_cst);
    }
}

pub inline fn _mm_min_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_i16x8(bitCast_i16x8(a), bitCast_i16x8(b)));
}

pub inline fn _mm_min_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_u8x16(bitCast_u8x16(a), bitCast_u8x16(b)));
}

test "_mm_min_epu8" {
    const a = _xx_set_epu8(0xFF, 0x80, 0x7F, 0x01, 0x81, 0x7E, 0x7F, 0x80, 0xFE, 0x80, 0x7F, 0x01, 0x81, 0x7E, 0x7F, 0x80);
    const b = _xx_set_epu8(0x80, 0x00, 0x80, 0x80, 0x80, 0x7F, 0xFF, 0xFF, 0xFF, 0x55, 0x44, 0x22, 0x82, 0x66, 0x77, 0x88);
    const ref = _xx_set_epu8(0x80, 0x00, 0x7F, 0x01, 0x80, 0x7E, 0x7F, 0x80, 0xFE, 0x55, 0x44, 0x01, 0x81, 0x66, 0x77, 0x80);
    try std.testing.expectEqual(ref, _mm_min_epu8(a, b));
}

pub inline fn _mm_min_pd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vminpd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("minpd %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_pd(a) | isNan_pd(b);
        const negZero: u64x2 = @splat(0x8000000000000000);
        const bothZero = @intFromBool((bitCast_u64x2(a) | bitCast_u64x2(b) | negZero) == negZero);
        const cmpgt = @intFromBool(a > b);
        const mask = boolMask_u64x2(cmpgt | bothZero | isNan);
        return @bitCast((bitCast_u64x2(a) & ~mask) | (bitCast_u64x2(b) & mask));
    }
}

pub inline fn _mm_min_sd(a: __m128d, b: __m128d) __m128d {
    if (has_avx) {
        return asm ("vminsd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("minsd %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
        const a0: u64 = @bitCast(a[0]);
        const b0: u64 = @bitCast(b[0]);
        const bothZero = @intFromBool((a0 | b0 | 0x80000000) == 0x80000000);
        const cmpgt = @intFromBool(a0 > b0);
        const mask = boolMask_u64x1(cmpgt | bothZero | isNan);
        return .{ @bitCast((a0 & ~mask) | (b0 & mask)), a[1] };
    }
}

pub inline fn _mm_move_epi64(a: __m128i) __m128i {
    const r = i64x2{ bitCast_i64x2(a)[0], 0 };
    return @bitCast(r);
}

pub inline fn _mm_move_sd(a: __m128d, b: __m128d) __m128d {
    return .{ b[0], a[1] };
}

pub inline fn _mm_movemask_epi8(a: __m128i) i32 {
    const cmp = @as(i8x16, @splat(0)) > bitCast_i8x16(a);
    return @intCast(@as(u16, @bitCast(cmp)));
}

pub inline fn _mm_movemask_pd(a: __m128d) i32 {
    const cmp = @as(i64x2, @splat(0)) > bitCast_i64x2(a);
    return @intCast(@as(u2, @bitCast(cmp)));
}

pub inline fn _mm_mul_epu32(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x2{ 0, 2 };
    const x = intCast_u64x2(@shuffle(u32, bitCast_u32x4(a), undefined, shuf));
    const y = intCast_u64x2(@shuffle(u32, bitCast_u32x4(b), undefined, shuf));
    return @bitCast(x *% y);
}

pub inline fn _mm_mul_pd(a: __m128d, b: __m128d) __m128d {
    return a * b;
}

pub inline fn _mm_mul_sd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] * b[0], a[1] };
}

pub inline fn _mm_mulhi_epi16(a: __m128i, b: __m128i) __m128i {
    const r = (intCast_i32x8(bitCast_i16x8(a)) *% intCast_i32x8(bitCast_i16x8(b)));
    return @bitCast(@as(i16x8, @truncate(r >> @splat(16))));
}

pub inline fn _mm_mulhi_epu16(a: __m128i, b: __m128i) __m128i {
    const r = (intCast_u32x8(bitCast_u16x8(a)) *% intCast_u32x8(bitCast_u16x8(b)));
    return @bitCast(@as(u16x8, @truncate(r >> @splat(16))));
}

pub inline fn _mm_mullo_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) *% bitCast_i16x8(b));
}

pub inline fn _mm_or_pd(a: __m128d, b: __m128d) __m128d {
    return @bitCast(bitCast_u64x2(a) | bitCast_u64x2(b));
}

pub inline fn _mm_or_si128(a: __m128i, b: __m128i) __m128i {
    return a | b;
}

pub inline fn _mm_packs_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x16{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 };
    var ab = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf);
    ab = min_i16x16(ab, @splat(127));
    ab = max_i16x16(ab, @splat(-128));
    return @bitCast(@as(i8x16, @truncate(ab)));
}

pub inline fn _mm_packs_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x8{ 0, 1, 2, 3, -1, -2, -3, -4 };
    var ab = @shuffle(i32, bitCast_i32x4(a), bitCast_i32x4(b), shuf);
    ab = min_i32x8(ab, @splat(32767));
    ab = max_i32x8(ab, @splat(-32768));
    return @bitCast(@as(i16x8, @truncate(ab)));
}

pub inline fn _mm_packus_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x16{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 };
    var ab = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf);
    ab = min_i16x16(ab, @splat(255));
    ab = max_i16x16(ab, @splat(0));
    return @bitCast(@as(i8x16, @truncate(ab)));
}

pub inline fn _mm_pause() void {
    std.atomic.spinLoopHint();
}

pub inline fn _mm_sad_epu8(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsadbw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psadbw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const shuf_lo = i32x8{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const shuf_hi = i32x8{ 8, 9, 10, 11, 12, 13, 14, 15 };

        const max = max_u8x16(bitCast_u8x16(a), bitCast_u8x16(b));
        const min = min_u8x16(bitCast_u8x16(a), bitCast_u8x16(b));
        const abd = intCast_u16x16(max -% min);

        const lo = @reduce(.Add, @shuffle(u16, abd, undefined, shuf_lo));
        const hi = @reduce(.Add, @shuffle(u16, abd, undefined, shuf_hi));
        return @bitCast(u64x2{ lo, hi });
    }
}

pub inline fn _mm_set_epi16(e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m128i {
    const r = i16x8{ e0, e1, e2, e3, e4, e5, e6, e7 };
    return @bitCast(r);
}

pub inline fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) __m128i {
    const r = i32x4{ e0, e1, e2, e3 };
    return @bitCast(r);
}

pub inline fn _mm_set_epi64x(e1: i64, e0: i64) __m128i {
    const r = i64x2{ e0, e1 };
    return @bitCast(r);
}

pub inline fn _mm_set_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m128i {
    const r = i8x16{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 };
    return @bitCast(r);
}

pub inline fn _mm_set_pd(e1: f64, e0: f64) __m128d {
    return .{ e0, e1 };
}

pub inline fn _mm_set_pd1(a: f64) __m128d {
    return _mm_set1_pd(a);
}

pub inline fn _mm_set_sd(a: f64) __m128d {
    return .{ a, 0 };
}

pub inline fn _mm_set1_epi16(a: i16) __m128i {
    return @bitCast(@as(i16x8, @splat(a)));
}

pub inline fn _mm_set1_epi32(a: i32) __m128i {
    return @bitCast(@as(i32x4, @splat(a)));
}

pub inline fn _mm_set1_epi64x(a: i64) __m128i {
    return @bitCast(@as(i64x2, @splat(a)));
}

pub inline fn _mm_set1_epi8(a: i8) __m128i {
    return @bitCast(@as(i8x16, @splat(a)));
}

pub inline fn _mm_set1_pd(a: f64) __m128d {
    return @splat(a);
}

pub inline fn _mm_setr_epi16(e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m128i {
    const r = i16x8{ e7, e6, e5, e4, e3, e2, e1, e0 };
    return @bitCast(r);
}

pub inline fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) __m128i {
    const r = i32x4{ e3, e2, e1, e0 };
    return @bitCast(r);
}

/// not listed in intel intrinsics guide (but may exist w/MSVC ?)
pub inline fn _mm_setr_epi64x(e1: i64, e0: i64) __m128i {
    return _mm_set_epi64x(e0, e1);
}

pub inline fn _mm_setr_epi8(e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m128i {
    const r = i8x16{ e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0 };
    return @bitCast(r);
}

pub inline fn _mm_setr_pd(e1: f64, e0: f64) __m128d {
    return _mm_set_pd(e0, e1);
}

pub inline fn _mm_setzero_pd() __m128d {
    return @splat(0);
}

pub inline fn _mm_setzero_si128() __m128i {
    return @splat(0);
}

/// macro not listed in intel intrinsics guide (but is very common)
pub inline fn _MM_SHUFFLE(comptime e3: comptime_int, comptime e2: comptime_int, comptime e1: comptime_int, comptime e0: comptime_int) comptime_int {
    return (e3 << 6) | (e2 << 4) | (e1 << 2) | e0;
}

pub inline fn _mm_shuffle_epi32(a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf = i32x4{ imm8 & 3, (imm8 >> 2) & 3, (imm8 >> 4) & 3, (imm8 >> 6) & 3 };
    return @bitCast(@shuffle(i32, bitCast_i32x4(a), undefined, shuf));
}

pub inline fn _mm_shuffle_pd(a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {
    return .{ if ((imm8 & 1) == 0) a[0] else a[1], if ((imm8 & 2) == 0) b[0] else b[1] };
}

pub inline fn _mm_shufflehi_epi16(a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf = i32x8{ 0, 1, 2, 3, 4 + (imm8 & 3), 4 + ((imm8 >> 2) & 3), 4 + ((imm8 >> 4) & 3), 4 + ((imm8 >> 6) & 3) };
    return @bitCast(@shuffle(i16, bitCast_i16x8(a), undefined, shuf));
}

pub inline fn _mm_shufflelo_epi16(a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf = i32x8{ imm8 & 3, (imm8 >> 2) & 3, (imm8 >> 4) & 3, (imm8 >> 6) & 3, 4, 5, 6, 7 };
    return @bitCast(@shuffle(i16, bitCast_i16x8(a), undefined, shuf));
}

pub inline fn _mm_sll_epi16(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsllw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psllw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 15) {
            return @splat(0);
        }
        return @bitCast(bitCast_u16x8(a) << @splat(@truncate(shift)));
    }
}

pub inline fn _mm_sll_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpslld %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("pslld %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 31) {
            return @splat(0);
        }
        return @bitCast(bitCast_u32x4(a) << @splat(@truncate(shift)));
    }
}

pub inline fn _mm_sll_epi64(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsllq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psllq %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 63) {
            return @splat(0);
        }
        return @bitCast(bitCast_u64x2(a) << @splat(@truncate(shift)));
    }
}

pub inline fn _mm_slli_epi16(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    const shift: u16x8 = @splat(imm8);
    return @bitCast(bitCast_u16x8(a) << shift);
}

pub inline fn _mm_slli_epi32(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    const shift: u32x4 = @splat(imm8);
    return @bitCast(bitCast_u32x4(a) << shift);
}

pub inline fn _mm_slli_epi64(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 63) {
        return @splat(0);
    }
    const shift: u64x2 = @splat(imm8);
    return @bitCast(bitCast_u64x2(a) << shift);
}

pub inline fn _mm_slli_si128(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    return _mm_alignr_epi8(a, @splat(0), 16 - imm8);
}

pub inline fn _mm_sqrt_pd(a: __m128d) __m128d {
    return @sqrt(a);
}

pub inline fn _mm_sqrt_sd(a: __m128d, b: __m128d) __m128d {
    return .{ @sqrt(b[0]), a[1] };
}

pub inline fn _mm_sra_epi16(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsraw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psraw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        var shift = bitCast_u64x2(count)[0];
        if (shift > 15) {
            shift = 15;
        }
        return @bitCast(bitCast_i16x8(a) >> @splat(@truncate(shift)));
    }
}

pub inline fn _mm_sra_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsrad %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psrad %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        var shift = bitCast_u64x2(count)[0];
        if (shift > 31) {
            shift = 31;
        }
        return @bitCast(bitCast_i32x4(a) >> @splat(@truncate(shift)));
    }
}

pub inline fn _mm_srai_epi16(a: __m128i, comptime imm8: comptime_int) __m128i {
    const shift: i16x8 = @splat(@min(@as(u8, @intCast(imm8)), 15));
    return @bitCast(bitCast_i16x8(a) >> shift);
}

test "_mm_srai_epi16" {
    const a = _mm_set_epi16(-32768, -1, -32511, 32767, 1, 128, 0, -128);
    const ref = _mm_set_epi16(-2048, -1, -2032, 2047, 0, 8, 0, -8);
    try std.testing.expectEqual(ref, _mm_srai_epi16(a, 4));
}

pub inline fn _mm_srai_epi32(a: __m128i, comptime imm8: comptime_int) __m128i {
    const shift: i32x4 = @splat(@min(@as(u8, @intCast(imm8)), 31));
    return @bitCast(bitCast_i32x4(a) >> shift);
}

pub inline fn _mm_srl_epi16(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsrlw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psrlw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 15) {
            return @splat(0);
        }
        return @bitCast(bitCast_u16x8(a) >> @splat(@truncate(shift)));
    }
}

pub inline fn _mm_srl_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsrld %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psrld %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 31) {
            return @splat(0);
        }
        return @bitCast(bitCast_u32x4(a) >> @splat(@truncate(shift)));
    }
}

pub inline fn _mm_srl_epi64(a: __m128i, count: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsrlq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("psrlq %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (count),
        );
        return res;
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 63) {
            return @splat(0);
        }
        return @bitCast(bitCast_u64x2(a) >> @splat(@truncate(shift)));
    }
}

pub inline fn _mm_srli_epi16(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    const shift: u16x8 = @splat(imm8);
    return @bitCast(bitCast_u16x8(a) >> shift);
}

pub inline fn _mm_srli_epi32(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    const shift: u32x4 = @splat(imm8);
    return @bitCast(bitCast_u32x4(a) >> shift);
}

pub inline fn _mm_srli_epi64(a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 63) {
        return @splat(0);
    }
    const shift: u64x2 = @splat(imm8);
    return @bitCast(bitCast_u64x2(a) >> shift);
}

pub inline fn _mm_srli_si128(a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_alignr_epi8(@splat(0), a, imm8);
}

pub inline fn _mm_store_pd(mem_addr: *align(16) [2]f64, a: __m128d) void {
    mem_addr[0] = a[0];
    mem_addr[1] = a[1];
}

pub inline fn _mm_store_pd1(mem_addr: *align(16) [2]f64, a: __m128d) void {
    return _mm_store1_pd(mem_addr, a);
}

pub inline fn _mm_store_sd(mem_addr: *align(1) f64, a: __m128d) void {
    mem_addr.* = a[0];
}

pub inline fn _mm_store_si128(mem_addr: *__m128i, a: __m128i) void {
    mem_addr.* = a;
}

pub inline fn _mm_store1_pd(mem_addr: *align(16) [2]f64, a: __m128d) void {
    mem_addr[0] = a[0];
    mem_addr[1] = a[0];
}

pub inline fn _mm_storeh_pd(mem_addr: *f64, a: __m128d) void {
    mem_addr.* = a[1];
}

// Despite the signature, this function is the same as _mm_storeu_si64
pub inline fn _mm_storel_epi64(mem_addr: *align(1) __m128i, a: __m128i) void {
    return _mm_storeu_si64(@ptrCast(mem_addr), a);
}

pub inline fn _mm_storel_pd(mem_addr: *f64, a: __m128d) void {
    mem_addr.* = a[0];
}

pub inline fn _mm_storer_pd(mem_addr: *align(16) [2]f64, a: __m128d) void {
    mem_addr[0] = a[1];
    mem_addr[1] = a[0];
}

pub inline fn _mm_storeu_pd(mem_addr: *align(1) [2]f64, a: __m128d) void {
    mem_addr[0] = a[0];
    mem_addr[1] = a[1];
}

pub inline fn _mm_storeu_si128(mem_addr: *align(1) __m128i, a: __m128i) void {
    mem_addr.* = a;
}

pub inline fn _mm_storeu_si16(mem_addr: *anyopaque, a: __m128i) void {
    @as(*align(1) u16, @ptrCast(mem_addr)).* = bitCast_u16x8(a)[0];
}

pub inline fn _mm_storeu_si32(mem_addr: *anyopaque, a: __m128i) void {
    @as(*align(1) u32, @ptrCast(mem_addr)).* = bitCast_u32x4(a)[0];
}

pub inline fn _mm_storeu_si64(mem_addr: *anyopaque, a: __m128i) void {
    @as(*align(1) u64, @ptrCast(mem_addr)).* = bitCast_u64x2(a)[0];
}

// ## pub inline fn _mm_stream_pd (mem_addr: *[2]f64, a: __m128d) void {}
// ## pub inline fn _mm_stream_si128 (mem_addr: *__m128i, a: __m128i) void {}
// ## pub inline fn _mm_stream_si32 (mem_addr: *i32, a: i32) void {}
// ## pub inline fn _mm_stream_si64 (mem_addr: *i64, a: i64) void {}

pub inline fn _mm_sub_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) -% bitCast_u16x8(b));
}

pub inline fn _mm_sub_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u32x4(a) -% bitCast_u32x4(b));
}

pub inline fn _mm_sub_epi64(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u64x2(a) -% bitCast_u64x2(b));
}

pub inline fn _mm_sub_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) -% bitCast_u8x16(b));
}

pub inline fn _mm_sub_pd(a: __m128d, b: __m128d) __m128d {
    return a - b;
}

pub inline fn _mm_sub_sd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] - b[0], a[1] };
}

pub inline fn _mm_subs_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) -| bitCast_i16x8(b));
}

pub inline fn _mm_subs_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i8x16(a) -| bitCast_i8x16(b));
}

pub inline fn _mm_subs_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) -| bitCast_u16x8(b));
}

pub inline fn _mm_subs_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) -| bitCast_u8x16(b));
}

/// result = if ((a[0] == b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomieq_sd(a: __m128d, b: __m128d) i32 {
    const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
    const cmpeq = @intFromBool(a[0] == b[0]);
    return notNan & cmpeq;
}

/// result = if ((a[0] >= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomige_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomile_sd(b, a);
}

/// result = if ((a[0] > b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomigt_sd(a: __m128d, b: __m128d) i32 {
    return _mm_ucomilt_sd(b, a);
}

/// result = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomile_sd(a: __m128d, b: __m128d) i32 {
    const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
    const cmple = @intFromBool(a[0] <= b[0]);
    return notNan & cmple;
}

/// result = if ((a[0] <= b[0]) and (a[0] != NaN) and b[0] != NaN) 1 else 0;
pub inline fn _mm_ucomilt_sd(a: __m128d, b: __m128d) i32 {
    const notNan = ~(isNan_f64(a[0]) | isNan_f64(b[0]));
    const cmplt = @intFromBool(a[0] < b[0]);
    return notNan & cmplt;
}

/// result = if ((a[0] != b[0]) or (a[0] == NaN) or b[0] == NaN) 1 else 0;
pub inline fn _mm_ucomineq_sd(a: __m128d, b: __m128d) i32 {
    const isNan = isNan_f64(a[0]) | isNan_f64(b[0]);
    const cmpneq = @intFromBool(a != b);
    return isNan | cmpneq;
}

pub inline fn _mm_undefined_pd() __m128d {
    // zig `undefined` doesn't compare equal to itself ?
    return @splat(0);
}

pub inline fn _mm_undefined_si128() __m128i {
    // zig `undefined` doesn't compare equal to itself ?
    return _mm_setzero_si128();
}

pub inline fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x8{ 4, -5, 5, -6, 6, -7, 7, -8 };
    return @bitCast(@shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf));
}

pub inline fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x4{ 2, -3, 3, -4 };
    return @bitCast(@shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf));
}

pub inline fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x2{ 1, -2 };
    return @bitCast(@shuffle(u64, bitCast_u64x2(a), bitCast_u64x2(b), shuf));
}

pub inline fn _mm_unpackhi_epi8(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x16{ 8, -9, 9, -10, 10, -11, 11, -12, 12, -13, 13, -14, 14, -15, 15, -16 };
    return @bitCast(@shuffle(u8, bitCast_u8x16(a), bitCast_u8x16(b), shuf));
}

pub inline fn _mm_unpackhi_pd(a: __m128d, b: __m128d) __m128d {
    return .{ a[1], b[1] };
}

pub inline fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x8{ 0, -1, 1, -2, 2, -3, 3, -4 };
    return @bitCast(@shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf));
}

pub inline fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x4{ 0, -1, 1, -2 };
    return @bitCast(@shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf));
}

pub inline fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x2{ 0, -1 };
    return @bitCast(@shuffle(u64, bitCast_u64x2(a), bitCast_u64x2(b), shuf));
}

pub inline fn _mm_unpacklo_epi8(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x16{ 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8 };
    return @bitCast(@shuffle(u8, bitCast_u8x16(a), bitCast_u8x16(b), shuf));
}

pub inline fn _mm_unpacklo_pd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0], b[0] };
}

pub inline fn _mm_xor_pd(a: __m128d, b: __m128d) __m128d {
    return @bitCast(bitCast_u64x2(a) ^ bitCast_u64x2(b));
}

pub inline fn _mm_xor_si128(a: __m128i, b: __m128i) __m128i {
    return a ^ b;
}

// SSE3 ================================================================

pub inline fn _mm_addsub_pd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] - b[0], a[1] + b[1] };
}

pub inline fn _mm_addsub_ps(a: __m128, b: __m128) __m128 {
    return .{ a[0] - b[0], a[1] + b[1], a[2] - b[2], a[3] + b[3] };
}

pub inline fn _mm_hadd_pd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] + a[1], b[0] + b[1] };
}

pub inline fn _mm_hadd_ps(a: __m128, b: __m128) __m128 {
    return .{ a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3] };
}

pub inline fn _mm_hsub_pd(a: __m128d, b: __m128d) __m128d {
    return .{ a[0] - a[1], b[0] - b[1] };
}

pub inline fn _mm_hsub_ps(a: __m128, b: __m128) __m128 {
    return .{ a[0] - a[1], a[2] - a[3], b[0] - b[1], b[2] - b[3] };
}

// lddqu is only useful on the P4
pub inline fn _mm_lddqu_si128(mem_addr: *align(1) const __m128i) __m128i {
    return _mm_loadu_si128(mem_addr);
}

pub inline fn _mm_loaddup_pd(mem_addr: *const f64) __m128d {
    const e = mem_addr.*;
    return .{ e, e };
}

pub inline fn _mm_movedup_pd(a: __m128d) __m128d {
    return .{ a[0], a[0] };
}

pub inline fn _mm_movehdup_ps(a: __m128) __m128 {
    return .{ a[1], a[1], a[3], a[3] };
}

pub inline fn _mm_moveldup_ps(a: __m128) __m128 {
    return .{ a[0], a[0], a[2], a[2] };
}

// SSSE3 ===============================================================

/// Absolute Value without saturation.
/// (e.g. the absolute value of `-32768 (0x8000)` is `-32768 (0x8000)`)
pub inline fn _mm_abs_epi16(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i16x8(a)));
}

test "_mm_abs_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -128, -127, 128, 127, -32767, 0);
    const ref = _mm_set_epi16(-32768, 32767, 128, 127, 128, 127, 32767, 0);
    try std.testing.expectEqual(ref, _mm_abs_epi16(a));
}

/// Absolute Value without saturation.
/// (e.g. the absolute value of `-2147483648 (0x80000000)` is `-2147483648 (0x80000000)`)
pub inline fn _mm_abs_epi32(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i32x4(a)));
}

test "_mm_abs_epi32" {
    const a = _mm_set_epi32(-2147483648, 2147483647, 65535, -2147450881);
    const ref = _mm_set_epi32(-2147483648, 2147483647, 65535, 2147450881);
    try std.testing.expectEqual(ref, _mm_abs_epi32(a));
}

/// Absolute Value without saturation.
/// (e.g. the absolute value of `-128 (0x80)` is `-128 (0x80)`)
pub inline fn _mm_abs_epi8(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i8x16(a)));
}

test "_mm_abs_epi8" {
    const a = _mm_set_epi8(-128, 127, -1, 1, 127, -128, 1, -1, -127, -1, -126, 1, 0, -2, -64, 3);
    const ref = _mm_set_epi8(-128, 127, 1, 1, 127, -128, 1, 1, 127, 1, 126, 1, 0, 2, 64, 3);
    try std.testing.expectEqual(ref, _mm_abs_epi8(a));
}

/// Append `a` to the left (hi) of `b`, shift right by `imm8` bytes, then truncate to `__m128i`.
pub inline fn _mm_alignr_epi8(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    if (@as(u8, @intCast(imm8)) > 15) {
        return _mm_alignr_epi8(@splat(0), a, imm8 - 16);
    }

    const shuf = comptime blk: {
        var indices: i32x16 = undefined;
        for (0..16) |i| {
            var x: i32 = @as(i32, @intCast(i)) + imm8;
            if (x > 15) {
                x = -x + 15;
            }
            indices[i] = x;
        }
        break :blk indices;
    };
    return @bitCast(@shuffle(u8, bitCast_u8x16(b), bitCast_u8x16(a), shuf));
}

test "_mm_alignr_epi8" {
    const a = _mm_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);
    const b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const ref0 = _mm_set_epi8(24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9);
    const ref1 = _mm_set_epi8(8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30, 29, 28, 27, 26, 25);
    const ref2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31);
    const ref3 = _mm_setzero_si128();
    try std.testing.expectEqual(ref0, _mm_alignr_epi8(a, b, 9));
    try std.testing.expectEqual(ref1, _mm_alignr_epi8(b, a, 9));
    try std.testing.expectEqual(ref2, _mm_alignr_epi8(a, b, 31));
    try std.testing.expectEqual(ref3, _mm_alignr_epi8(a, b, 32));
    try std.testing.expectEqual(b, _mm_alignr_epi8(b, a, 16));
    try std.testing.expectEqual(a, _mm_alignr_epi8(a, b, 16));
    try std.testing.expectEqual(a, _mm_alignr_epi8(b, a, 0));
    try std.testing.expectEqual(b, _mm_alignr_epi8(a, b, 0));
}

/// Horizontally add `+%` pairs of 16-bit integers, then pack (narrow).
pub inline fn _mm_hadd_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phadd.w.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.phadd.w.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphaddw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phaddw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
        const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
        return @bitCast(even +% odd);
    }
}

test "_mm_hadd_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -32768, 1, -32768, -1, -32768, 0);
    const b = _mm_set_epi16(32767, -32768, 2, 1, 32767, 2, 3, 128);
    const ref = _mm_set_epi16(-1, 3, -32767, 131, -1, -32767, 32767, -32768);
    try std.testing.expectEqual(ref, _mm_hadd_epi16(a, b));
}

/// Horizontally add `+%` pairs of 32-bit integers, then pack (narrow).
pub inline fn _mm_hadd_epi32(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phadd.d.128"(i32x4, i32x4) i32x4;
        }.@"llvm.x86.ssse3.phadd.d.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphaddd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phaddd %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x4{ 0, 2, -1, -3 };
        const shuf_odd = i32x4{ 1, 3, -2, -4 };
        const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
        const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
        return @bitCast(even +% odd);
    }
}

test "_mm_hadd_epi32" {
    const a = _mm_set_epi32(1, -2147483648, -2147483648, -2147483648);
    const b = _mm_set_epi32(-1, 1, 2, 3);
    const ref = _mm_set_epi32(0, 5, -2147483647, 0);
    try std.testing.expectEqual(ref, _mm_hadd_epi32(a, b));
}

/// Horizontally add w/saturation `+|` pairs of signed 16-bit integers, then pack (narrow).
pub inline fn _mm_hadds_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phadd.sw.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.phadd.sw.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphaddsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phaddsw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even +| odd);
    }
}

test "_mm_hadds_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -32768, 1, -32768, -1, -32768, 0);
    const b = _mm_set_epi16(32767, -32768, 2, 1, 32767, 2, 3, 128);
    const ref = _mm_set_epi16(-1, 3, 32767, 131, -1, -32767, -32768, -32768);
    try std.testing.expectEqual(ref, _mm_hadds_epi16(a, b));
}

/// Horizontally subtract `-%` pairs of 16-bit integers, then pack (narrow).
/// e.g. `a[0] = a[0] -% a[1]; a[1] = a[2] - a[3]; // etc.`
pub inline fn _mm_hsub_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phsub.w.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.phsub.w.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphsubw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phsubw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
        const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
        return @bitCast(even -% odd);
    }
}

test "_mm_hsub_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -32768, 1, -32768, -1, -32768, 0);
    const b = _mm_set_epi16(32767, -32768, 2, 1, 32767, 2, 3, 128);
    const ref = _mm_set_epi16(1, -1, -32765, 125, -1, -32767, 32767, -32768);
    try std.testing.expectEqual(ref, _mm_hsub_epi16(a, b));
}

/// Horizontally subtract `-%` pairs of 32-bit integers, then pack (narrow).
/// e.g. `a[0] = a[0] -% a[1]; a[1] = a[2] - a[3]; a[2] = b[0] - b[1]; a[3] = b[2] - b[3]`
pub inline fn _mm_hsub_epi32(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phsub.d.128"(i32x4, i32x4) i32x4;
        }.@"llvm.x86.ssse3.phsub.d.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphsubd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phsubd %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x4{ 0, 2, -1, -3 };
        const shuf_odd = i32x4{ 1, 3, -2, -4 };
        const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
        const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
        return @bitCast(even -% odd);
    }
}

test "_mm_hsub_epi32" {
    const a = _mm_set_epi32(1, -2147483648, -2147483648, -2147483648);
    const b = _mm_set_epi32(-2, 1, 2, 3);
    const ref = _mm_set_epi32(3, 1, 2147483647, 0);
    try std.testing.expectEqual(ref, _mm_hsub_epi32(a, b));
}

/// Horizontally subtract w/saturation `-|` pairs of signed 16-bit integers, then pack (narrow).
/// e.g. `a[0] = a[0] -| a[1]; a[1] = a[2] - a[3]; // etc.`
pub inline fn _mm_hsubs_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.phsub.sw.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.phsub.sw.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphsubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("phsubsw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even -| odd);
    }
}

test "_mm_hsubs_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -32768, 1, -32768, -1, -32768, 0);
    const b = _mm_set_epi16(32767, -32768, 2, 1, 32767, 2, 3, 128);
    const ref = _mm_set_epi16(-32768, -1, -32765, 125, 32767, 32767, 32767, 32767);
    try std.testing.expectEqual(ref, _mm_hsubs_epi16(a, b));
}

/// Widening Multiply, horizontally add w/saturation, then pack.
/// `a[i].u8 * b[i].i8 = t[i].i16;`
/// `r[i] = t[i*2] +| t[(i*2)+1];`
pub inline fn _mm_maddubs_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.pmadd.ub.sw.128"(u8x16, i8x16) i16x8;
        }.@"llvm.x86.ssse3.pmadd.ub.sw.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpmaddubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("pmaddubsw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const lo = _mm_mullo_epi16(_mm_srli_epi16(_mm_slli_epi16(a, 8), 8), _mm_srai_epi16(_mm_slli_epi16(b, 8), 8));
        const hi = _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srai_epi16(b, 8));
        return _mm_adds_epi16(lo, hi);
    }
}

test "_mm_maddubs_epi16" {
    const a = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 3, 99, 55, 44, 33, 22, 11, 0, 1, 2);
    const b = _mm_set_epi8(-128, -128, 127, 127, -128, 127, -1, 127, 99, 88, -5, 66, 55, 1, 1, 4);
    const ref = _mm_set_epi16(-32768, 32767, -255, 12570, 9317, 1287, 605, 9);
    try std.testing.expectEqual(ref, _mm_maddubs_epi16(a, b));
}

/// multiply usually used to "scale" fixed-point values
/// (e.g. divide by constant then round to nearest integer)
//
// for discussion about generating the magic multiplicative inverse
// see: Henry S. Warren, Hacker's Delight 2nd Edition, Chapter 10
pub inline fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.pmul.hr.sw.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.pmul.hr.sw.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpmulhrsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("pmulhrsw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        var r = intCast_i32x8(bitCast_i16x8(a));
        r *%= intCast_i32x8(bitCast_i16x8(b));
        r +%= @splat(1 << 14);
        return @bitCast(@as(i16x8, @truncate(r >> @splat(15))));
    }
}

test "_mm_mulhrs_epi16" {
    if (bug_stage2_x86_64) return error.SkipZigTest; // genBinOp for mul_wrap

    const a = _mm_set_epi16(300, 100, -100, 3, 1, 2, -32768, 32767);
    const div3 = _mm_set1_epi16(10923); // 0x2AAB
    const ref = _mm_set_epi16(100, 33, -33, 1, 0, 1, -10923, 10923);
    try std.testing.expectEqual(ref, _mm_mulhrs_epi16(a, div3));
}

/// Gathers bytes from `a` according to `b` (the shuffle control index)
/// The low 4-bits of each lane of `b` index into `a`
/// Bits 4,5,6 are ignored.
/// If Bit_7 is set then make the destintion zero.
pub inline fn _mm_shuffle_epi8(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.pshuf.b.128"(u8x16, u8x16) u8x16;
        }.@"llvm.x86.ssse3.pshuf.b.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx)) {
        return asm ("vpshufb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3)) {
        var r = a;
        asm ("pshufb %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        var r: u8x16 = undefined;
        const shuf = bitCast_i8x16(b) & @as(i8x16, @splat(0x0F));
        const mask = @intFromBool(bitCast_i8x16(b) < @as(i8x16, @splat(0)));
        inline for (0..16) |i| {
            r[i] = bitCast_u8x16(a)[@intCast(shuf[i])];
        }
        return @bitCast(~boolMask_u8x16(mask) & r);
    }
}

test "_mm_shuffle_epi8" {
    const a = _mm_set_epi8(-128, -1, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, -1, -128, -116, 77, 47, 30, 5, 1);
    const ref = _mm_set_epi8(-128, -1, 14, 13, 12, 11, 10, 9, 0, 0, 0, 14, -128, -1, 6, 2);
    try std.testing.expectEqual(ref, _mm_shuffle_epi8(a, b));
}

/// if (b[i] < 0) a[i] = -(a[i]); if (b[i] == 0) a[i] = 0;
pub inline fn _mm_sign_epi16(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.psign.w.128"(i16x8, i16x8) i16x8;
        }.@"llvm.x86.ssse3.psign.w.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpsignw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_builtins) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("psignw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const zero: i16x8 = @splat(0);
        const r = @select(i16, zero > bitCast_i16x8(b), -%bitCast_i16x8(a), bitCast_i16x8(a));
        return @bitCast(@select(i16, (zero == bitCast_i16x8(b)), zero, r));
    }
}

test "_mm_sign_epi16" {
    const a = _mm_set_epi16(-32768, 32767, -128, -127, 128, 127, -32767, 0);
    const b = _mm_set_epi16(-32768, -1, -128, 127, 0, 127, -1, -1);
    const ref = _mm_set_epi16(-32768, -32767, 128, -127, 0, 127, 32767, 0);
    try std.testing.expectEqual(_mm_abs_epi16(a), _mm_sign_epi16(a, a));
    try std.testing.expectEqual(ref, _mm_sign_epi16(a, b));
}

/// if (b[i] < 0) a[i] = -(a[i]); if (b[i] == 0) a[i] = 0;
pub inline fn _mm_sign_epi32(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.psign.d.128"(i32x4, i32x4) i32x4;
        }.@"llvm.x86.ssse3.psign.d.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpsignd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("psignd %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const zero: i32x4 = @splat(0);
        const r = @select(i32, zero > bitCast_i32x4(b), -%bitCast_i32x4(a), bitCast_i32x4(a));
        return @bitCast(@select(i32, (zero == bitCast_i32x4(b)), zero, r));
    }
}

test "_mm_sign_epi32" {
    const a = _mm_set_epi32(-2147483648, -2147483648, -2147483647, 2147483647);
    const b = _mm_set_epi32(-2147483648, 2147483647, -1, 3);
    const ref0 = _mm_set_epi32(-2147483648, -2147483648, 2147483647, 2147483647);
    const ref1 = _mm_set_epi32(-2147483648, -2147483647, 1, 3);
    try std.testing.expectEqual(_mm_abs_epi32(a), _mm_sign_epi32(a, a));
    try std.testing.expectEqual(ref0, _mm_sign_epi32(a, b));
    try std.testing.expectEqual(ref1, _mm_sign_epi32(b, a));
}

/// if (b[i] < 0) a[i] = -(a[i]); if (b[i] == 0) a[i] = 0;
pub inline fn _mm_sign_epi8(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_ssse3)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.ssse3.psign.b.128"(i8x16, i8x16) i8x16;
        }.@"llvm.x86.ssse3.psign.b.128"(@bitCast(a), @bitCast(b)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpsignb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_ssse3) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("psignb %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const zero: i8x16 = @splat(0);
        const r = @select(i8, zero > bitCast_i8x16(b), -%bitCast_i8x16(a), bitCast_i8x16(a));
        return @bitCast(@select(i8, (zero == bitCast_i8x16(b)), zero, r));
    }
}

test "_mm_sign_epi8" {
    const a = _mm_set_epi8(-128, -128, -128, -127, -127, -127, 127, 127, 80, 80, 2, -2, -1, 1, 1, 0);
    const b = _mm_set_epi8(-1, 0, -1, -127, 0, 127, -128, 0, -80, 80, -2, -1, 0, 1, 0, -1);
    const ref = _mm_set_epi8(-128, 0, -128, 127, 0, -127, -127, 0, -80, 80, -2, 2, 0, 1, 0, 0);
    try std.testing.expectEqual(_mm_abs_epi8(a), _mm_sign_epi8(a, a));
    try std.testing.expectEqual(ref, _mm_sign_epi8(a, b));
}

// SSE4.1 ==============================================================

pub const _MM_FROUND_TO_NEAREST_INT = 0x00;
pub const _MM_FROUND_TO_NEG_INF = 0x01;
pub const _MM_FROUND_TO_POS_INF = 0x02;
pub const _MM_FROUND_TO_ZERO = 0x03;
pub const _MM_FROUND_CUR_DIRECTION = 0x04;
pub const _MM_FROUND_RAISE_EXC = 0x00;
pub const _MM_FROUND_NO_EXC = 0x08;
pub const _MM_FROUND_NINT = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEAREST_INT;
pub const _MM_FROUND_FLOOR = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF;
pub const _MM_FROUND_CEIL = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF;
pub const _MM_FROUND_TRUNC = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO;
pub const _MM_FROUND_RINT = _MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION;
pub const _MM_FROUND_NEARBYINT = _MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION;

/// `@select(i16,imm8,b,a)`.
/// `r[i] = if (((imm8 >> i) & 1) != 0) b[i] else a[i];`
pub inline fn _mm_blend_epi16(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    const mask: @Vector(8, bool) = @bitCast(@as(u8, imm8));
    return @bitCast(@select(i16, mask, bitCast_i16x8(b), bitCast_i16x8(a)));
}

test "_mm_blend_epi16" {
    const a = _mm_set_epi16(-3872, -12096, -20320, -28544, 28768, 20544, 12320, 4096);
    const b = _mm_set_epi16(3854, 3340, 2826, 2312, 1798, 1284, 770, 256);
    const ref0 = _mm_set_epi16(-3872, -12096, -20320, -28544, 28768, 20544, 12320, 4096);
    const ref1 = _mm_set_epi16(3854, 3340, 2826, 2312, 1798, 1284, 770, 4096);
    const ref2 = _mm_set_epi16(-3872, -12096, 2826, 2312, 28768, 1284, 770, 4096);
    const ref3 = _mm_set_epi16(-3872, 3340, -20320, 2312, 28768, 1284, 12320, 256);
    try std.testing.expectEqual(ref0, _mm_blend_epi16(a, b, 0));
    try std.testing.expectEqual(ref1, _mm_blend_epi16(b, a, 1));
    try std.testing.expectEqual(ref2, _mm_blend_epi16(a, b, 54));
    try std.testing.expectEqual(ref3, _mm_blend_epi16(a, b, 85));
}

/// `@select(f64,imm8,b,a)`.
/// `r[i] = if (((imm8 >> i) & 1) != 0) b[i] else a[i];`
pub inline fn _mm_blend_pd(a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {
    return @select(f64, @as(@Vector(2, bool), @bitCast(@as(u2, imm8))), b, a);
}

test "_mm_blend_pd" {
    const a: __m128d = @bitCast(_mm_set_epi64x(-1089641583808049024, 8097560366627688448));
    const b: __m128d = @bitCast(_mm_set_epi64x(1084818905618843912, 506097522914230528));
    const ref0 = _mm_set_epi64x(-1089641583808049024, 8097560366627688448);
    const ref1 = _mm_set_epi64x(1084818905618843912, 8097560366627688448);
    const ref2 = _mm_set_epi64x(1084818905618843912, 8097560366627688448);
    const ref3 = _mm_set_epi64x(1084818905618843912, 506097522914230528);
    try std.testing.expectEqual(ref0, @as(__m128i, @bitCast(_mm_blend_pd(a, b, 0))));
    try std.testing.expectEqual(ref1, @as(__m128i, @bitCast(_mm_blend_pd(b, a, 1))));
    try std.testing.expectEqual(ref2, @as(__m128i, @bitCast(_mm_blend_pd(a, b, 2))));
    try std.testing.expectEqual(ref3, @as(__m128i, @bitCast(_mm_blend_pd(a, b, 3))));
}

/// `@select(f32,imm8,b,a)`.
/// `r[i] = if (((imm8 >> i) & 1) != 0) b[i] else a[i];`
pub inline fn _mm_blend_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    return @select(f32, @as(@Vector(4, bool), @bitCast(@as(u4, imm8))), b, a);
}

test "_mm_blend_ps" {
    const a = _mm_set_ps(4.4, 3.3, 2.2, 1.1);
    const b = _mm_set_ps(8.8, 7.7, 6.6, 5.5);
    const ref0 = _mm_set_ps(4.4, 3.3, 2.2, 1.1);
    const ref1 = _mm_set_ps(8.8, 7.7, 6.6, 1.1);
    const ref2 = _mm_set_ps(8.8, 3.3, 6.6, 1.1);
    const ref3 = _mm_set_ps(8.8, 7.7, 6.6, 5.5);
    try std.testing.expectEqual(ref0, _mm_blend_ps(a, b, 0));
    try std.testing.expectEqual(ref1, _mm_blend_ps(b, a, 1));
    try std.testing.expectEqual(ref2, _mm_blend_ps(a, b, 10));
    try std.testing.expectEqual(ref3, _mm_blend_ps(a, b, 15));
}

/// `r[i] = if ((mask[i] < 0) b[i] else a[i];`
pub inline fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) __m128i {
    const cmp = @as(i8x16, @splat(0)) > bitCast_i8x16(mask);
    return @bitCast(@select(i8, cmp, bitCast_i8x16(b), bitCast_i8x16(a)));
}

test "_mm_blendv_epi8" {
    const a = _mm_set_epi8(-1, 127, -128, 0, -1, -1, 1, 2, 127, -1, -128, -1, 68, 51, 34, 17);
    const b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const ref0 = _mm_set_epi8(-1, 127, -128, 0, -1, -1, 1, 2, 127, -1, -128, -1, 68, 51, 34, 17);
    const ref1 = _mm_set_epi8(-1, 14, -128, 12, -1, -1, 9, 8, 7, -1, -128, -1, 3, 2, 1, 0);
    const ref2 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    try std.testing.expectEqual(ref0, _mm_blendv_epi8(a, b, b));
    try std.testing.expectEqual(ref1, _mm_blendv_epi8(b, a, a));
    try std.testing.expectEqual(ref2, _mm_blendv_epi8(b, a, b));
}

/// `r[i] = if ((mask[i] >> 63) != 0) b[i] else a[i];`
pub inline fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) __m128d {
    const cmp = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
    return @select(f64, cmp, b, a);
}

test "_mm_blendv_pd" {
    const a: __m128d = @bitCast(_mm_set_epi64x(-9223372036854775808, -9007199254740992));
    const b: __m128d = @bitCast(_mm_set_epi64x(9223372036854775807, -9007199254740991));
    const ref0 = _mm_set_epi64x(-9223372036854775808, -9007199254740991);
    const ref1 = _mm_set_epi64x(-9223372036854775808, -9007199254740992);
    const ref2 = _mm_set_epi64x(9223372036854775807, -9007199254740992);
    try std.testing.expectEqual(ref0, @as(__m128i, @bitCast(_mm_blendv_pd(a, b, b))));
    try std.testing.expectEqual(ref1, @as(__m128i, @bitCast(_mm_blendv_pd(b, a, a))));
    try std.testing.expectEqual(ref2, @as(__m128i, @bitCast(_mm_blendv_pd(b, a, b))));
}

/// `r[i] = if ((mask[i] >> 31) != 0) b[i] else a[i];`
pub inline fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) __m128 {
    const cmp = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
    return @select(f32, cmp, b, a);
}

test "_mm_blendv_ps" {
    const a: __m128 = @bitCast(_mm_set_epi32(1065353216, 1073741824, 1077936128, 1082130432));
    const b: __m128 = @bitCast(_mm_set_epi32(-8388608, 2139095041, -2147483648, -33488897));
    const ref0 = _mm_set_epi32(-8388608, 1073741824, -2147483648, -33488897);
    const ref1 = _mm_set_epi32(-8388608, 2139095041, -2147483648, -33488897);
    const ref2 = _mm_set_epi32(1065353216, 2139095041, 1077936128, 1082130432);
    try std.testing.expectEqual(ref0, @as(__m128i, @bitCast(_mm_blendv_ps(a, b, b))));
    try std.testing.expectEqual(ref1, @as(__m128i, @bitCast(_mm_blendv_ps(b, a, a))));
    try std.testing.expectEqual(ref2, @as(__m128i, @bitCast(_mm_blendv_ps(b, a, b))));
}

/// Round up to an integer value, rounding towards positive infinity.
///
/// `@ceil(a)` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_ceil_pd(a: __m128d) __m128d {
    return _mm_round_pd(a, _MM_FROUND_CEIL);
}

/// Round up to an integer value, rounding towards positive infinity.
///
/// `@ceil(a)` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_ceil_ps(a: __m128) __m128 {
    return _mm_round_ps(a, _MM_FROUND_CEIL);
}

/// Round `b[0]` up to an integer value, rounding towards positive infinity.
/// Copy `a[1]` into the upper element of the result.
///
/// .{ @ceil(b[0]), a[1] }` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_ceil_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_round_sd(a, b, _MM_FROUND_CEIL);
}

/// Round `b[0]` up to an integer value, rounding towards positive infinity.
/// Copy the upper 3 elements of `a` into the upper three elements of the result.
///
/// .{ @ceil(b[0]), a[1], a[2], a[3] }` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_ceil_ss(a: __m128, b: __m128) __m128 {
    return _mm_round_ss(a, b, _MM_FROUND_CEIL);
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_i64x2(a) == bitCast_i64x2(b));
    return @bitCast(boolMask_u64x2(pred));
}

test "_mm_cmpeq_epi64" {
    const a = _mm_set_epi64x(9223372036854775807, 1);
    const b = _mm_set_epi64x(8070450532247928831, 1);
    const ref = _mm_set_epi64x(0, -1);
    try std.testing.expectEqual(ref, _mm_cmpeq_epi64(a, b));
}

/// Sign-Extend the low 4 words
pub inline fn _mm_cvtepi16_epi32(a: __m128i) __m128i {
    const x = bitCast_i16x8(a);
    return @bitCast(i32x4{ x[0], x[1], x[2], x[3] });
}

test "_mm_cvtepi16_epi32" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi32(3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi16_epi32(a));
}

/// Sign-Extend the low 2 words
pub inline fn _mm_cvtepi16_epi64(a: __m128i) __m128i {
    const x = bitCast_i16x8(a);
    return @bitCast(i64x2{ x[0], x[1] });
}

test "_mm_cvtepi16_epi64" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi64x(-1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi16_epi64(a));
}

/// Sign-Extend the low 2 dwords
pub inline fn _mm_cvtepi32_epi64(a: __m128i) __m128i {
    const x = bitCast_i32x4(a);
    return @bitCast(i64x2{ x[0], x[1] });
}

test "_mm_cvtepi32_epi64" {
    const a = _mm_set_epi32(3, 2, -1, 0);
    const ref = _mm_set_epi64x(-1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi32_epi64(a));
}

/// Sign-Extend the low 8 bytes
pub inline fn _mm_cvtepi8_epi16(a: __m128i) __m128i {
    const x = bitCast_i8x16(a);
    return @bitCast(i16x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm_cvtepi8_epi16" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi8_epi16(a));
}

/// Sign-Extend the low 4 bytes
pub inline fn _mm_cvtepi8_epi32(a: __m128i) __m128i {
    const x = bitCast_i8x16(a);
    return @bitCast(i32x4{ x[0], x[1], x[2], x[3] });
}

test "_mm_cvtepi8_epi32" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi32(3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi8_epi32(a));
}

/// Sign-Extend the low 2 bytes
// note: error in intel intrinsic guide v3.6.7
pub inline fn _mm_cvtepi8_epi64(a: __m128i) __m128i {
    const x = bitCast_i8x16(a);
    return @bitCast(i64x2{ x[0], x[1] });
}

test "_mm_cvtepi8_epi64" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi64x(-1, 0);
    try std.testing.expectEqual(ref, _mm_cvtepi8_epi64(a));
}

/// Zero-Extend the low 4 words
pub inline fn _mm_cvtepu16_epi32(a: __m128i) __m128i {
    const x = bitCast_u16x8(a);
    return @bitCast(u32x4{ x[0], x[1], x[2], x[3] });
}

test "_mm_cvtepu16_epi32" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi32(3, 2, 65535, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu16_epi32(a));
}

/// Zero-Extend the low 2 words
pub inline fn _mm_cvtepu16_epi64(a: __m128i) __m128i {
    const x = bitCast_u16x8(a);
    return @bitCast(u64x2{ x[0], x[1] });
}

test "_mm_cvtepu16_epi64" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi64x(65535, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu16_epi64(a));
}

/// Zero-Extend the low 2 dwords
pub inline fn _mm_cvtepu32_epi64(a: __m128i) __m128i {
    const x = bitCast_u32x4(a);
    return @bitCast(u64x2{ x[0], x[1] });
}

test "_mm_cvtepu32_epi64" {
    const a = _mm_set_epi32(3, 2, -1, 0);
    const ref = _mm_set_epi64x(4294967295, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu32_epi64(a));
}

/// Zero-Extend the low 8 bytes
pub inline fn _mm_cvtepu8_epi16(a: __m128i) __m128i {
    const x = bitCast_u8x16(a);
    return @bitCast(u16x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm_cvtepu8_epi16" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi16(7, 6, 5, 4, 3, 2, 255, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu8_epi16(a));
}

/// Zero-Extend the low 4 bytes
pub inline fn _mm_cvtepu8_epi32(a: __m128i) __m128i {
    const x = bitCast_u8x16(a);
    return @bitCast(u32x4{ x[0], x[1], x[2], x[3] });
}

test "_mm_cvtepu8_epi32" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi32(3, 2, 255, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu8_epi32(a));
}

/// Zero-Extend the low 2 bytes
pub inline fn _mm_cvtepu8_epi64(a: __m128i) __m128i {
    const x = bitCast_u8x16(a);
    return @bitCast(u64x2{ x[0], x[1] });
}

test "_mm_cvtepu8_epi64" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm_set_epi64x(255, 0);
    try std.testing.expectEqual(ref, _mm_cvtepu8_epi64(a));
}

/// dot product
pub inline fn _mm_dp_pd(a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.dppd"(__m128d, __m128d, u8) __m128d;
        }.@"llvm.x86.sse41.dppd"(a, b, imm8);
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vdppd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("dppd %[c], %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return r;
    } else {
        const dp: f64 = switch (@as(u2, imm8 >> 4)) {
            0 => 0.0,
            1 => a[0] * b[0],
            2 => a[1] * b[1],
            3 => (a[0] * b[0]) + (a[1] * b[1]),
        };
        return switch (@as(u2, imm8 & 0x0F)) { // broadcast
            0 => .{ 0.0, 0.0 },
            1 => .{ dp, 0.0 },
            2 => .{ 0.0, dp },
            3 => .{ dp, dp },
        };
    }
}

test "_mm_dp_pd" {
    const a = _mm_set_pd(5.3, 63.1);
    const b = _mm_set_pd(44.1, 0.1);
    const ref = _mm_set_pd(240.04, 0.0);
    try std.testing.expectEqual(ref, _mm_dp_pd(a, b, 0x32));
}

/// dot product
pub inline fn _mm_dp_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.dpps"(__m128, __m128, u8) __m128;
        }.@"llvm.x86.sse41.dpps"(a, b, imm8);
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vdpps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("dpps %[c], %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return r;
    } else {
        const mask: @Vector(4, bool) = @bitCast(@as(u4, imm8 & 0x0F));
        const broadcast: @Vector(4, bool) = @bitCast(@as(u4, imm8 >> 4));
        const a_m = @select(f32, mask, a, @as(__m128, @splat(0)));
        const b_m = @select(f32, mask, b, @as(__m128, @splat(0)));
        const product = a_m * b_m;
        const sum = (product[3] + product[2]) + (product[1] + product[0]);
        return @select(f32, broadcast, @as(__m128, @splat(sum)), @as(__m128, @splat(0)));
    }
}

test "_mm_dp_ps" {
    const a = _mm_set_ps(5.3, 63.1, 98.5, 66.6);
    const b = _mm_set_ps(44.1, 0.1, 123.5, 1346.0);
    const ref = _mm_set_ps(102048.390625, 102048.390625, 102048.390625, 102048.390625);
    try std.testing.expectEqual(ref, _mm_dp_ps(a, b, 0xFF));
}

/// dst = a[imm8];
pub inline fn _mm_extract_epi32(a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_i32x4(a)[imm8];
}

test "_mm_extract_epi32" {
    const a = _mm_set_epi32(3, -2, 1, 0);
    try std.testing.expectEqual(@as(i32, -2), _mm_extract_epi32(a, 2));
}

/// dst = a[imm8];
pub inline fn _mm_extract_epi64(a: __m128i, comptime imm8: comptime_int) i64 {
    return bitCast_i64x2(a)[imm8];
}

test "_mm_extract_epi64" {
    const a = _mm_set_epi64x(-562949953421410, 0);
    const ref: i64 = -562949953421410;
    try std.testing.expectEqual(ref, _mm_extract_epi64(a, 1));
}

/// Extract u8 then zero-extend to i32.
pub inline fn _mm_extract_epi8(a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_u8x16(a)[imm8];
}

test "_mm_extract_epi8" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, -2, 1, 0);
    try std.testing.expectEqual(@as(i32, 254), _mm_extract_epi8(a, 2));
}

/// Extract f32 then bitCast to i32.
pub inline fn _mm_extract_ps(a: __m128, comptime imm8: comptime_int) i32 {
    return bitCast_i32x4(a)[imm8];
}

test "_mm_extract_ps" {
    const a = _mm_set_ps(2.5, 2.0, 1.5, 1.0);
    try std.testing.expectEqual(@as(i32, 1073741824), _mm_extract_ps(a, 2));
}

/// Round down to an integer value, rounding towards negative infinity.
///
/// `@floor(a)` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_floor_pd(a: __m128d) __m128d {
    return _mm_round_pd(a, _MM_FROUND_FLOOR);
}

/// Round down to an integer value, rounding towards negative infinity.
///
/// `@floor(a)` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_floor_ps(a: __m128) __m128 {
    return _mm_round_ps(a, _MM_FROUND_FLOOR);
}

/// Round `b[0]` down to an integer value, rounding towards negative infinity.
/// Copy `a[1]` into the upper element of the result.
///
/// .{ @floor(b[0]), a[1] }` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_floor_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_round_sd(a, b, _MM_FROUND_FLOOR);
}

/// Round `b[0]` down to an integer value, rounding towards negative infinity.
/// Copy the upper elements of `a` into the upper element of the result.
///
/// .{ @floor(b[0]), a[1], a[2], a[3] }` except may raise exceptions, in theory, if we supported that...
pub inline fn _mm_floor_ss(a: __m128, b: __m128) __m128 {
    return _mm_round_ss(a, b, _MM_FROUND_FLOOR);
}

/// a[imm8] = i;
pub inline fn _mm_insert_epi32(a: __m128i, i: i32, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i32x4(a);
    r[imm8 & 0x03] = i;
    return @bitCast(r);
}

test "_mm_insert_epi32" {
    const a = _mm_set_epi32(0x44444444, 0x33333333, 0x22222222, 0x11111111);
    const ref = _mm_set_epi32(0x44444444, 0x33333333, 0x55555555, 0x11111111);
    try std.testing.expectEqual(ref, _mm_insert_epi32(a, 0x55555555, 1));
}

/// a[imm8] = i;
pub inline fn _mm_insert_epi64(a: __m128i, i: i64, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i64x2(a);
    r[imm8 & 0x01] = i;
    return @bitCast(r);
}

test "_mm_insert_epi64" {
    const a = _mm_set_epi64x(0x2222222222222222, 0x1111111111111111);
    const ref = _mm_set_epi64x(0x3333333333333333, 0x1111111111111111);
    try std.testing.expectEqual(ref, _mm_insert_epi64(a, 0x3333333333333333, 1));
}

/// a[imm8] = i;
pub inline fn _mm_insert_epi8(a: __m128i, i: i32, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i8x16(a);
    r[imm8 & 0x0F] = @truncate(i);
    return @bitCast(r);
}

test "_mm_insert_epi8" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const ref = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 16, 5, 4, 3, 2, 1, 0);
    try std.testing.expectEqual(ref, _mm_insert_epi8(a, 272, 6));
}

/// Copy selected element from `b` into selected element in `a`.
/// Zero elements in `a` according to the mask.
/// The helper `_MM_MK_INSERTPS_NDX()` can generate the `imm8` control value.
pub inline fn _mm_insert_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    if ((use_builtins) and (has_sse4_1)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.sse41.insertps"(__m128, __m128, u8) __m128;
        }.@"llvm.x86.sse41.insertps"(a, b, imm8));
    } else if ((use_asm) and (has_avx)) {
        return asm ("vinsertps %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        var r = a;
        asm ("insertps %[c], %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return r;
    } else {
        var r = a;
        r[(imm8 >> 4) & 3] = b[imm8 >> 6];
        if ((imm8 & 1) != 0) r[0] = 0;
        if ((imm8 & 2) != 0) r[1] = 0;
        if ((imm8 & 4) != 0) r[2] = 0;
        if ((imm8 & 8) != 0) r[3] = 0;
        return r;
    }
}

test "_mm_insert_ps" {
    const a = _mm_set_ps(1.7, 1.5, 1.3, 1.1);
    const b = _mm_set_ps(2.7, 2.5, 2.3, 2.1);
    const ref1 = _mm_set_ps(0.0, 2.3, 1.3, 1.1);
    const ref2 = _mm_set_ps(1.7, 0.0, 1.3, 0.0);
    try std.testing.expectEqual(ref1, _mm_insert_ps(a, b, 0x68));
    try std.testing.expectEqual(ref2, _mm_insert_ps(a, b, 0x65));
}

/// `@max(a, b)`
pub inline fn _mm_max_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_i32x4(bitCast_i32x4(a), bitCast_i32x4(b)));
}

test "_mm_max_epi32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(2147418112, 2147483647, 0, 0);
    try std.testing.expectEqual(ref, _mm_max_epi32(a, b));
}

/// `@max(a, b)`
pub inline fn _mm_max_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_i8x16(bitCast_i8x16(a), bitCast_i8x16(b)));
}

test "_mm_max_epi8" {
    const a = _mm_set_epi8(-128, 0, 0, 0, 0, 0, 0, 0, -1, -2, -128, -128, 127, 2, 1, 1);
    const b = _mm_set_epi8(0, 0, 0, 0, -128, 0, 0, 0, -2, -1, 127, -127, -128, 2, 2, 0);
    const ref = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 127, -127, 127, 2, 2, 1);
    try std.testing.expectEqual(ref, _mm_max_epi8(a, b));
}

/// `@max(a, b)`
pub inline fn _mm_max_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_u16x8(bitCast_u16x8(a), bitCast_u16x8(b)));
}

test "_mm_max_epu16" {
    const a = _mm_set_epi16(32767, 1, -1, -257, -32768, 0, 2, 258);
    const b = _mm_set_epi16(-32768, 0, 32767, -2, 0, -32768, 3, 257);
    const ref = _mm_set_epi16(-32768, 1, -1, -2, -32768, -32768, 3, 258);
    try std.testing.expectEqual(ref, _mm_max_epu16(a, b));
}

/// `@max(a, b)`
pub inline fn _mm_max_epu32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(max_u32x4(bitCast_u32x4(a), bitCast_u32x4(b)));
}

test "_mm_max_epu32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(-2147483647, -1, -2147483648, -2147483648);
    try std.testing.expectEqual(ref, _mm_max_epu32(a, b));
}

/// `@min(a, b)`
pub inline fn _mm_min_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_i32x4(bitCast_i32x4(a), bitCast_i32x4(b)));
}

test "_mm_min_epi32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(-2147483647, -1, -2147483648, -2147483648);
    try std.testing.expectEqual(ref, _mm_min_epi32(a, b));
}

/// `@min(a, b)`
pub inline fn _mm_min_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_i8x16(bitCast_i8x16(a), bitCast_i8x16(b)));
}

test "_mm_min_epi8" {
    const a = _mm_set_epi8(-128, 0, 0, 0, 0, 0, 0, 0, -1, -2, -128, -128, 127, 2, 1, 1);
    const b = _mm_set_epi8(0, 0, 0, 0, -128, 0, 0, 0, -2, -1, 127, -127, -128, 2, 2, 0);
    const ref = _mm_set_epi8(-128, 0, 0, 0, -128, 0, 0, 0, -2, -2, -128, -128, -128, 2, 1, 0);
    try std.testing.expectEqual(ref, _mm_min_epi8(a, b));
}

/// `@min(a, b)`
pub inline fn _mm_min_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_u16x8(bitCast_u16x8(a), bitCast_u16x8(b)));
}

test "_mm_min_epu16" {
    const a = _mm_set_epi16(32767, 1, -1, -257, -32768, 0, 2, 258);
    const b = _mm_set_epi16(-32768, 0, 32767, -2, 0, -32768, 3, 257);
    const ref = _mm_set_epi16(32767, 0, 32767, -257, 0, 0, 2, 257);
    try std.testing.expectEqual(ref, _mm_min_epu16(a, b));
}

/// `@min(a, b)`
pub inline fn _mm_min_epu32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(min_u32x4(bitCast_u32x4(a), bitCast_u32x4(b)));
}

test "_mm_min_epu32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(2147418112, 2147483647, 0, 0);
    try std.testing.expectEqual(ref, _mm_min_epu32(a, b));
}

pub inline fn _mm_minpos_epu16(a: __m128i) __m128i {
    if ((use_builtins) and (has_sse4_1)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.sse41.phminposuw"(u16x8) u16x8;
        }.@"llvm.x86.sse41.phminposuw"(@bitCast(a)));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vphminposuw %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        return asm ("phminposuw %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else {
        const idx = u16x8{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const shuf = [16]i32{ -1, 0, -2, 1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6, -8, 7 };
        const unpacked = @shuffle(u16, bitCast_u16x8(a), idx, shuf);
        const r = @reduce(.Min, bitCast_u32x8(unpacked));
        const res = u32x4{ (r << 16) | (r >> 16), 0, 0, 0 };
        return @bitCast(res);
    }
}

test "_mm_minpos_epu16" {
    const a = _mm_set_epi16(-1, 2, 1, 2, 1, 32767, -32768, -1);
    const ref = _mm_set_epi16(0, 0, 0, 0, 0, 0, 3, 1);
    try std.testing.expectEqual(ref, _mm_minpos_epu16(a));
}

/// Helper to create the immediate value for `_mm_insert_ps(a, b, imm8)`.
/// Copy element at position `src_idx` from `b` over `dst_idx` in `a`.
/// Zero elements in `a` according to the bitset in `zero_mask`
pub inline fn _MM_MK_INSERTPS_NDX(comptime src_idx: comptime_int, comptime dst_idx: comptime_int, comptime zero_mask: comptime_int) comptime_int {
    return (((src_idx) << 6) | ((dst_idx) << 4) | (zero_mask));
}

test "_MM_MK_INSERTPS_NDX" {
    try std.testing.expectEqual(0x68, _MM_MK_INSERTPS_NDX(1, 2, 0x8));
}

pub inline fn _mm_mpsadbw_epu8(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    if ((use_builtins) and (has_sse4_1)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.sse41.mpsadbw"(u8x16, u8x16, u8) u16x8;
        }.@"llvm.x86.sse41.mpsadbw"(@bitCast(a), @bitCast(b), imm8));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vmpsadbw %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        var r = a;
        asm ("mpsadbw %[c], %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return r;
    } else {
        const s0: __m128i = _mm_unpacklo_epi32(_mm_srli_si128(a, 0 + (imm8 & 4)), _mm_srli_si128(a, 2 + (imm8 & 4)));
        const s1: __m128i = _mm_unpacklo_epi32(_mm_srli_si128(a, 1 + (imm8 & 4)), _mm_srli_si128(a, 3 + (imm8 & 4)));
        const s2: __m128i = _mm_shuffle_epi32(b, _MM_SHUFFLE(imm8 & 3, imm8 & 3, imm8 & 3, imm8 & 3));

        const abd0 = _mm_sub_epi8(_mm_max_epu8(s0, s2), _mm_min_epu8(s0, s2));
        const abd1 = _mm_sub_epi8(_mm_max_epu8(s1, s2), _mm_min_epu8(s1, s2));

        const mask = _mm_set1_epi16(0x00FF);
        const hsum0 = _mm_add_epi16(_mm_srli_epi16(abd0, 8), _mm_and_si128(abd0, mask));
        const hsum1 = _mm_add_epi16(_mm_srli_epi16(abd1, 8), _mm_and_si128(abd1, mask));

        const lo = _mm_srli_epi32(_mm_add_epi16(hsum0, _mm_slli_epi32(hsum0, 16)), 16);
        const hi = _mm_slli_epi32(_mm_add_epi16(hsum1, _mm_srli_epi32(hsum1, 16)), 16);

        return _mm_or_si128(lo, hi);
    }
}

test "_mm_mpsadbw_epu8" {
    const a = _xx_set_epu32(0x07018593, 0x56312665, 0xFFFFFFFF, 0);
    const b = _xx_set_epu32(3, 0xFA57C0DE, 1, 0);
    const ref0 = _mm_set_epi16(443, 649, 866, 1020, 765, 510, 255, 0);
    const ref1 = _mm_set_epi16(476, 456, 431, 477, 374, 322, 413, 269);
    try std.testing.expectEqual(ref0, _mm_mpsadbw_epu8(a, b, 0));
    try std.testing.expectEqual(ref1, _mm_mpsadbw_epu8(a, b, 6));
}

pub inline fn _mm_mul_epi32(a: __m128i, b: __m128i) __m128i {
    const x = bitCast_i32x4(a);
    const y = bitCast_i32x4(b);
    return @bitCast(i64x2{ @as(i64, x[0]) *% y[0], @as(i64, x[2]) *% y[2] });
}

test "_mm_mul_epi32" {
    const a = _mm_set_epi32(3, 2, 1, -2);
    const b = _mm_set_epi32(5, -2147483646, 9, -2147483647);
    const ref = _mm_set_epi64x(-4294967292, 4294967294);
    try std.testing.expectEqual(ref, _mm_mul_epi32(a, b));
}

pub inline fn _mm_mullo_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i32x4(a) *% bitCast_i32x4(b));
}

test "_mm_mullo_epi32" {
    const a = _mm_set_epi32(3, 2, 1, -2);
    const b = _mm_set_epi32(715827883, -2147483646, -2147483647, -2147483648);
    const ref = _mm_set_epi32(-2147483647, 4, -2147483647, 0);
    try std.testing.expectEqual(ref, _mm_mullo_epi32(a, b));
}

/// Append `b` to the left (hi) of `a`, then narrow each element to a `u16`.
/// Any value that doesn't fit in a `u16` is changed to `0xFFFF`.
pub inline fn _mm_packus_epi32(a: __m128i, b: __m128i) __m128i {
    if ((use_builtins) and (has_sse4_1)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.sse41.packusdw"(i32x4, i32x4) u16x8;
        }.@"llvm.x86.sse41.packusdw"(a, b));
    } else if ((use_asm) and (has_avx)) {
        return asm ("vpackusdw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        var r = a;
        asm ("packusdw %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
        );
        return r;
    } else {
        const shuf: [8]i32 = .{ 0, 1, 2, 3, -1, -2, -3, -4 };
        var ab = @shuffle(i32, bitCast_i32x4(a), bitCast_i32x4(b), shuf);
        ab = min_i32x8(ab, @splat(65535));
        ab = max_i32x8(ab, @splat(0));
        return @bitCast(@as(i16x8, @truncate(ab)));
    }
}

test "_mm_packus_epi32" {
    if (bug_stage2_x86_64) return error.SkipZigTest; // genBinOp for min

    const a = _xx_set_epu32(0x00000001, 0x0000FFFF, 0x00008000, 0x00000000);
    const b = _xx_set_epu32(0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x0001FFFF);
    const ref = _xx_set_epu16(0xFFFF, 0, 0, 0xFFFF, 1, 0xFFFF, 0x8000, 0);
    try std.testing.expectEqual(ref, _mm_packus_epi32(a, b));
}

pub inline fn _mm_round_pd(a: __m128d, comptime imm8: comptime_int) __m128d {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.round.pd"(__m128d, i32) __m128d;
        }.@"llvm.x86.sse41.round.pd"(a, imm8);
    } else if ((use_asm) and (has_avx)) {
        return asm ("vroundpd %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        return asm ("roundpd %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [c] "i" (imm8),
        );
    } else {
        return switch (imm8 & 0x07) {
            _MM_FROUND_TO_NEAREST_INT => RoundEven_pd(a),
            _MM_FROUND_TO_NEG_INF => @floor(a),
            _MM_FROUND_TO_POS_INF => @ceil(a),
            _MM_FROUND_TO_ZERO => @trunc(a),
            else => RoundCurrentDirection_pd(a),
        };
    }
}

pub inline fn _mm_round_ps(a: __m128, comptime imm8: comptime_int) __m128 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.round.ps"(__m128, i32) __m128;
        }.@"llvm.x86.sse41.round.ps"(a, imm8);
    } else if ((use_asm) and (has_avx)) {
        return asm ("vroundps %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        return asm ("roundps %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [c] "i" (imm8),
        );
    } else {
        return switch (imm8 & 0x07) {
            _MM_FROUND_TO_NEAREST_INT => RoundEven_ps(a),
            _MM_FROUND_TO_NEG_INF => @floor(a),
            _MM_FROUND_TO_POS_INF => @ceil(a),
            _MM_FROUND_TO_ZERO => @trunc(a),
            else => RoundCurrentDirection_ps(a),
        };
    }
}

test "_mm_round_ps" {
    if (bug_stage2_x86_64) return error.SkipZigTest; // genBinOp for cmp_gte

    const a = _mm_set_ps(3.5, 2.5, 1.5, 0.5);
    const b = _mm_set_ps(-3.6, -2.4, -1.5, -0.5);
    const c = _mm_set_ps(1.1, 1.9, 8388607.5, -8388607.5);
    const ref0a = _mm_set_ps(4.0, 2.0, 2.0, 0.0);
    const ref0b = _mm_set_ps(-4.0, -2.0, -2.0, -0.0);
    const ref0c = _mm_set_ps(1.0, 2.0, 8388608.0, -8388608.0);
    const ref1a = _mm_set_ps(3.0, 2.0, 1.0, 0.0);
    const ref1b = _mm_set_ps(-4.0, -3.0, -2.0, -1.0);
    const ref1c = _mm_set_ps(1.0, 1.0, 8388607.0, -8388608.0);
    const ref2a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    const ref2b = _mm_set_ps(-3.0, -2.0, -1.0, -0.0);
    const ref2c = _mm_set_ps(2.0, 2.0, 8388608.0, -8388607.0);
    const ref3a = _mm_set_ps(3.0, 2.0, 1.0, 0.0);
    const ref3b = _mm_set_ps(-3.0, -2.0, -1.0, -0.0);
    const ref3c = _mm_set_ps(1.0, 1.0, 8388607.0, -8388607.0);
    try std.testing.expectEqual(ref0a, _mm_round_ps(a, 0));
    try std.testing.expectEqual(ref0b, _mm_round_ps(b, 0));
    try std.testing.expectEqual(ref0c, _mm_round_ps(c, 0));
    try std.testing.expectEqual(ref1a, _mm_round_ps(a, 1));
    try std.testing.expectEqual(ref1b, _mm_round_ps(b, 1));
    try std.testing.expectEqual(ref1c, _mm_round_ps(c, 1));
    try std.testing.expectEqual(ref2a, _mm_round_ps(a, 2));
    try std.testing.expectEqual(ref2b, _mm_round_ps(b, 2));
    try std.testing.expectEqual(ref2c, _mm_round_ps(c, 2));
    try std.testing.expectEqual(ref3a, _mm_round_ps(a, 3));
    try std.testing.expectEqual(ref3b, _mm_round_ps(b, 3));
    try std.testing.expectEqual(ref3c, _mm_round_ps(c, 3));
    try std.testing.expectEqual(ref0b, _mm_round_ps(b, 4));
    try std.testing.expectEqual(ref0b, _mm_round_ps(b, 5));
    try std.testing.expectEqual(ref0b, _mm_round_ps(b, 6));
    try std.testing.expectEqual(ref0b, _mm_round_ps(b, 7));
}

pub inline fn _mm_round_sd(a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.round.sd"(__m128d, __m128d, i32) __m128d;
        }.@"llvm.x86.sse41.round.sd"(a, b, imm8);
    } else if ((use_asm) and (has_avx)) {
        return asm ("vroundsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        var res = a;
        asm ("roundsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return res;
    } else {
        return switch (imm8 & 0x07) {
            _MM_FROUND_TO_NEAREST_INT => .{ RoundEven_f64(b[0]), a[1] },
            _MM_FROUND_TO_NEG_INF => .{ @floor(b[0]), a[1] },
            _MM_FROUND_TO_POS_INF => .{ @ceil(b[0]), a[1] },
            _MM_FROUND_TO_ZERO => .{ @trunc(b[0]), a[1] },
            else => .{ RoundCurrentDirection_f64(b[0]), a[1] },
        };
    }
}

/// Round the lower f32.
/// The fallback implements only:
///    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
pub inline fn _mm_round_ss(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.round.ss"(__m128, __m128, i32) __m128;
        }.@"llvm.x86.sse41.round.ss"(a, b, imm8);
    } else if ((use_asm) and (has_avx)) {
        return asm ("vroundss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_sse4_1)) {
        var res = a;
        asm ("roundss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return res;
    } else {
        return switch (imm8 & 0x07) {
            _MM_FROUND_TO_NEAREST_INT => .{ RoundEven_f32(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_NEG_INF => .{ @floor(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_POS_INF => .{ @ceil(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_ZERO => .{ @trunc(b[0]), a[1], a[2], a[3] },
            else => .{ RoundCurrentDirection_f32(b[0]), a[1], a[2], a[3] },
        };
    }
}

test "_mm_round_ss" {
    const a = _mm_set_ps(3.0, 2.9, 1.0, 0.5);
    const b = _mm_set_ps(3.1, 2.7, 1.1, -2.5);
    const c = _mm_set_ps(3.2, 2.5, 1.2, 8388607.5);
    const ref1a = _mm_set_ps(3.1, 2.7, 1.1, 0.0);
    const ref1b = _mm_set_ps(3.1, 2.7, 1.1, -3.0);
    const ref1c = _mm_set_ps(3.1, 2.7, 1.1, 8388607.0);
    const ref2a = _mm_set_ps(3.1, 2.7, 1.1, 1.0);
    const ref2b = _mm_set_ps(3.1, 2.7, 1.1, -2.0);
    const ref2c = _mm_set_ps(3.1, 2.7, 1.1, 8388608.0);
    const ref3a = _mm_set_ps(3.1, 2.7, 1.1, 0.0);
    const ref3b = _mm_set_ps(3.1, 2.7, 1.1, -2.0);
    const ref3c = _mm_set_ps(3.1, 2.7, 1.1, 8388607.0);
    const ref0a = _mm_set_ps(3.1, 2.7, 1.1, 0.0);
    const ref0b = _mm_set_ps(3.1, 2.7, 1.1, -2.0);
    const ref0c = _mm_set_ps(3.1, 2.7, 1.1, 8388608.0);
    try std.testing.expectEqual(ref0a, _mm_round_ss(b, a, 0));
    try std.testing.expectEqual(ref0b, _mm_round_ss(b, b, 0));
    try std.testing.expectEqual(ref0c, _mm_round_ss(b, c, 0));
    try std.testing.expectEqual(ref1a, _mm_round_ss(b, a, 1));
    try std.testing.expectEqual(ref1b, _mm_round_ss(b, b, 1));
    try std.testing.expectEqual(ref1c, _mm_round_ss(b, c, 1));
    try std.testing.expectEqual(ref2a, _mm_round_ss(b, a, 2));
    try std.testing.expectEqual(ref2b, _mm_round_ss(b, b, 2));
    try std.testing.expectEqual(ref2c, _mm_round_ss(b, c, 2));
    try std.testing.expectEqual(ref3a, _mm_round_ss(b, a, 3));
    try std.testing.expectEqual(ref3b, _mm_round_ss(b, b, 3));
    try std.testing.expectEqual(ref3c, _mm_round_ss(b, c, 3));
    try std.testing.expectEqual(ref0b, _mm_round_ss(b, b, 4));
    try std.testing.expectEqual(ref0b, _mm_round_ss(b, b, 5));
    try std.testing.expectEqual(ref0b, _mm_round_ss(b, b, 6));
    try std.testing.expectEqual(ref0b, _mm_round_ss(b, b, 7));
}

// ## pub inline fn _mm_stream_load_si128 (mem_addr: *const __m128i) __m128i {}

/// result = if (a == -1) 1 else 0;
pub inline fn _mm_test_all_ones(a: __m128i) i32 {
    return _mm_testc_si128(a, @bitCast(@as(i32x4, @splat(-1))));
}

test "_mm_test_all_ones" {
    const a = _mm_set_epi32(-1, -1, -1, -1);
    const b = _mm_set_epi32(1, 1, 1, 1);
    const c = _mm_set_epi32(0, 0, 0, 0);
    try std.testing.expectEqual(1, _mm_test_all_ones(a));
    try std.testing.expectEqual(0, _mm_test_all_ones(b));
    try std.testing.expectEqual(0, _mm_test_all_ones(c));
}

/// result = if ((a & mask) == 0) 1 else 0;
pub inline fn _mm_test_all_zeros(a: __m128i, mask: __m128i) i32 {
    return _mm_testz_si128(a, mask);
}

test "_mm_test_all_zeros" {
    const a = _mm_set_epi32(0, 8, 0, 0);
    const b = _mm_set_epi32(0, 7, 0, 0);
    const c = _mm_set_epi32(0, 9, 0, 0);
    try std.testing.expectEqual(1, _mm_test_all_zeros(a, b));
    try std.testing.expectEqual(0, _mm_test_all_zeros(a, c));
}

/// result = if (((a & mask) != 0) and ((~a & mask) != 0)) 1 else 0;
// note: error in intel intrinsic guide v3.6.7
pub inline fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) i32 {
    return _mm_testnzc_si128(a, mask);
}

test "_mm_test_mix_ones_zeros" {
    const a = _mm_set_epi32(0, 1, 0, 0);
    const b = _mm_set_epi32(0, 3, 0, 0);
    const c = _mm_set_epi32(0, 2, 0, 0);
    try std.testing.expectEqual(@as(i32, 1), _mm_test_mix_ones_zeros(a, b));
    try std.testing.expectEqual(@as(i32, 0), _mm_test_mix_ones_zeros(a, c));
    try std.testing.expectEqual(@as(i32, 0), _mm_test_mix_ones_zeros(b, a));
}

/// result = if ((~a & b) == 0) 1 else 0;
pub inline fn _mm_testc_si128(a: __m128i, b: __m128i) i32 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.ptestc"(u64x2, u64x2) i32;
        }.@"llvm.x86.sse41.ptestc"(@bitCast(a), @bitCast(b));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vptest %[b],%[a]"
            : [_] "={@ccc}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        return asm ("ptest %[b],%[a]"
            : [_] "={@ccc}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        return _mm_testz_si128(~a, b);
    }
}

test "_mm_testc_si128" {
    const a = _mm_set_epi32(0, 3, 0, 0);
    const b = _mm_set_epi32(0, 4, 0, 0);
    const c = _mm_set_epi32(0, 1, 0, 0);
    try std.testing.expectEqual(0, _mm_testc_si128(a, b));
    try std.testing.expectEqual(1, _mm_testc_si128(a, c));
}

/// result = if (((a & b) != 0) and ((~a & b) != 0)) 1 else 0;
pub inline fn _mm_testnzc_si128(a: __m128i, b: __m128i) i32 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.ptestnzc"(u64x2, u64x2) i32;
        }.@"llvm.x86.sse41.ptestnzc"(@bitCast(a), @bitCast(b));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vptest %[b],%[a]"
            : [_] "={@cca}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        return asm ("ptest %[b],%[a]"
            : [_] "={@cca}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        return @intFromBool((_mm_testz_si128(a, b) | _mm_testc_si128(a, b)) == 0);
    }
}

test "_mm_testnzc_si128" {
    const a = _mm_set_epi32(0, 1, 0, 0);
    const b = _mm_set_epi32(0, 3, 0, 0);
    const c = _mm_set_epi32(0, 2, 0, 0);
    try std.testing.expectEqual(@as(i32, 1), _mm_testnzc_si128(a, b));
    try std.testing.expectEqual(@as(i32, 0), _mm_testnzc_si128(a, c));
    try std.testing.expectEqual(@as(i32, 0), _mm_testnzc_si128(b, a));
}

/// result = if ((a & b) == 0) 1 else 0;
pub inline fn _mm_testz_si128(a: __m128i, b: __m128i) i32 {
    if ((use_builtins) and (has_sse4_1)) {
        return struct {
            extern fn @"llvm.x86.sse41.ptestz"(u64x2, u64x2) i32;
        }.@"llvm.x86.sse41.ptestz"(@bitCast(a), @bitCast(b));
    } else if ((use_asm) and (has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vptest %[b],%[a]"
            : [_] "={@ccz}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if ((use_asm) and (has_sse4_1) and (!bug_stage2_x86_64)) {
        return asm ("ptest %[b],%[a]"
            : [_] "={@ccz}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        return @intFromBool(@reduce(.Or, (a & b)) == 0);
    }
}

test "_mm_testz_si128" {
    const a = _mm_set_epi32(0, 8, 0, 0);
    const b = _mm_set_epi32(0, 7, 0, 0);
    const c = _mm_set_epi32(0, 9, 0, 0);
    try std.testing.expectEqual(1, _mm_testz_si128(a, b));
    try std.testing.expectEqual(0, _mm_testz_si128(a, c));
}

// SSE4.2 ==============================================================

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) __m128i {
    const pred = @intFromBool(bitCast_i64x2(a) > bitCast_i64x2(b));
    return @bitCast(boolMask_u64x2(pred));
}

test "_mm_cmpgt_epi64" {
    const a = _xx_set_epu64x(0x8000000000000001, 2);
    const b = _mm_set_epi64x(0, 1);
    const ref0 = _mm_set_epi64x(0, -1);
    const ref1 = _mm_set_epi64x(-1, 0);
    try std.testing.expectEqual(ref0, _mm_cmpgt_epi64(a, b));
    try std.testing.expectEqual(ref1, _mm_cmpgt_epi64(b, a));
}

/// Software CRC-32C
//
// Modified from: https://github.com/DLTcollab/sse2neon/blob/4a036e60472af7dd60a31421fa01557000b5c96b/sse2neon.h#L8528C11-L8528C21
// Copyright (c) Cuda Chen <clh960524@gmail.com>
//
// which was in turn based on: https://create.stephan-brumme.com/crc32/#half-byte
// Author: unknown
inline fn crc32cSoft(crc: anytype, v: anytype) @TypeOf(crc) {
    // 4-bit-indexed table has a small memory footprint
    // while being faster than a bit-twiddling solution
    // but has a loop-carried dependence...
    const crc32c_table: [16]u32 = .{
        0x00000000, 0x105ec76f, 0x20bd8ede, 0x30e349b1,
        0x417b1dbc, 0x5125dad3, 0x61c69362, 0x7198540d,
        0x82f63b78, 0x92a8fc17, 0xa24bb5a6, 0xb21572c9,
        0xc38d26c4, 0xd3d3e1ab, 0xe330a81a, 0xf36e6f75,
    };

    // ignore bits[32..64] of crc (and validate arg type)
    var r = switch (@typeInfo(@TypeOf(crc)).int.bits) {
        32 => crc,
        64 => crc & 0x00000000FFFFFFFF,
        else => @compileError("invalid type of arg `crc`"),
    };

    // number of loop iterations (and validate arg type)
    const n = switch (@typeInfo(@TypeOf(v)).int.bits) {
        8, 16, 32, 64 => @typeInfo(@TypeOf(v)).int.bits / 4,
        else => @compileError("invalid type of arg `v`"),
    };

    r ^= v;
    for (0..n) |_| {
        r = (r >> 4) ^ crc32c_table[@as(u4, @truncate(r))];
    }
    return r;
}

pub inline fn _mm_crc32_u16(crc: u32, v: u16) u32 {
    if ((use_builtins) and (has_sse4_2)) {
        return struct {
            extern fn @"llvm.x86.sse42.crc32.32.16"(u32, u16) u32;
        }.@"llvm.x86.sse42.crc32.32.16"(crc, v);
    } else if ((use_asm) and (has_sse4_2) and (!bug_stage2_x86_64)) {
        var r = crc;
        asm ("crc32 %[b],%[r]"
            : [r] "+r" (r),
            : [b] "r" (v),
        );
        return r;
    } else {
        return crc32cSoft(crc, v);
    }
}

test "_mm_crc32_u16" {
    const a: u32 = 0;
    const b: u16 = 0x5A17;
    const ref: u32 = 0x7F6FCA2F;
    try std.testing.expectEqual(ref, _mm_crc32_u16(a, b));
}

pub inline fn _mm_crc32_u32(crc: u32, v: u32) u32 {
    if ((use_builtins) and (has_sse4_2)) {
        return struct {
            extern fn @"llvm.x86.sse42.crc32.32.32"(u32, u32) u32;
        }.@"llvm.x86.sse42.crc32.32.32"(crc, v);
    } else if ((use_asm) and (has_sse4_2) and (!bug_stage2_x86_64)) {
        var r = crc;
        asm ("crc32 %[b],%[r]"
            : [r] "+r" (r),
            : [b] "r" (v),
        );
        return r;
    } else {
        return crc32cSoft(crc, v);
    }
}

test "_mm_crc32_u32" {
    const a: u32 = 0x7F6FCA2F;
    const b: u32 = 0x97455E45;
    const ref: u32 = 0xFCC84559;
    try std.testing.expectEqual(ref, _mm_crc32_u32(a, b));
}

pub inline fn _mm_crc32_u64(crc: u64, v: u64) u64 {
    if ((use_builtins) and (is_x86_64) and (has_sse4_2)) {
        return struct {
            extern fn @"llvm.x86.sse42.crc32.64.64"(u64, u64) u64;
        }.@"llvm.x86.sse42.crc32.64.64"(crc, v);
    } else if ((use_asm) and (is_x86_64) and (has_sse4_2) and (!bug_stage2_x86_64)) {
        var r = crc;
        asm ("crc32 %[b],%[r]"
            : [r] "+r" (r),
            : [b] "r" (v),
        );
        return r;
    } else {
        return crc32cSoft(crc, v);
    }
}

test "_mm_crc32_u64" {
    const a: u64 = 0x0102030405060708;
    const b: u64 = 0x7F6FCA2F97455E45;
    const ref: u64 = 0x0000000010B1F424;
    try std.testing.expectEqual(ref, _mm_crc32_u64(a, b));
}

pub inline fn _mm_crc32_u8(crc: u32, v: u8) u32 {
    if ((use_builtins) and (has_sse4_2)) {
        return struct {
            extern fn @"llvm.x86.sse42.crc32.32.8"(u32, u8) u32;
        }.@"llvm.x86.sse42.crc32.32.8"(crc, v);
    } else if ((use_asm) and (has_sse4_2) and (!bug_stage2_x86_64)) {
        var r = crc;
        asm ("crc32 %[b],%[r]"
            : [r] "+r" (r),
            : [b] "r" (v),
        );
        return r;
    } else {
        return crc32cSoft(crc, v);
    }
}

test "_mm_crc32_u8" {
    const a: u32 = 0xFFFFFFFF;
    const b: u8 = 0x5A;
    const ref: u32 = 0x97455E45;
    try std.testing.expectEqual(ref, _mm_crc32_u8(a, b));
}

// CLMUL ============================================================

/// Software carryless multiplication of two 64-bit integers using native 128-bit registers.
// Modified from: https://github.com/ziglang/zig/blob/8fd15c6ca8b93fa9888e2641ebec149f6d600643/lib/std/crypto/ghash_polyval.zig#L168
// Copyright (c) Zig contributors
fn clmulSoft128(x: u64, y: u64) u128 {
    const x0 = x & 0x1111111111111110;
    const x1 = x & 0x2222222222222220;
    const x2 = x & 0x4444444444444440;
    const x3 = x & 0x8888888888888880;
    const y0 = y & 0x1111111111111111;
    const y1 = y & 0x2222222222222222;
    const y2 = y & 0x4444444444444444;
    const y3 = y & 0x8888888888888888;
    const z0 = (x0 * @as(u128, y0)) ^ (x1 * @as(u128, y3)) ^ (x2 * @as(u128, y2)) ^ (x3 * @as(u128, y1));
    const z1 = (x0 * @as(u128, y1)) ^ (x1 * @as(u128, y0)) ^ (x2 * @as(u128, y3)) ^ (x3 * @as(u128, y2));
    const z2 = (x0 * @as(u128, y2)) ^ (x1 * @as(u128, y1)) ^ (x2 * @as(u128, y0)) ^ (x3 * @as(u128, y3));
    const z3 = (x0 * @as(u128, y3)) ^ (x1 * @as(u128, y2)) ^ (x2 * @as(u128, y1)) ^ (x3 * @as(u128, y0));

    const x0_mask = @as(u64, 0) -% (x & 1);
    const x1_mask = @as(u64, 0) -% ((x >> 1) & 1);
    const x2_mask = @as(u64, 0) -% ((x >> 2) & 1);
    const x3_mask = @as(u64, 0) -% ((x >> 3) & 1);
    const extra = (x0_mask & y) ^ (@as(u128, x1_mask & y) << 1) ^
        (@as(u128, x2_mask & y) << 2) ^ (@as(u128, x3_mask & y) << 3);

    return (z0 & 0x11111111111111111111111111111111) ^
        (z1 & 0x22222222222222222222222222222222) ^
        (z2 & 0x44444444444444444444444444444444) ^
        (z3 & 0x88888888888888888888888888888888) ^ extra;
}

pub inline fn _mm_clmulepi64_si128(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    if ((use_builtins) and (has_pclmul)) {
        return @bitCast(struct {
            extern fn @"llvm.x86.pclmulqdq"(i64x2, i64x2, u8) i64x2;
        }.@"llvm.x86.pclmulqdq"(@bitCast(a), @bitCast(b), imm8));
    } else if ((use_asm) and (has_avx) and (has_pclmul)) {
        return asm ("vpclmulqdq %[c], %[b], %[a], %[out]"
            : [out] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else if ((use_asm) and (has_pclmul)) {
        var r = a;
        asm ("pclmulqdq %[c], %[b], %[r]"
            : [r] "+x" (r),
            : [b] "x" (b),
              [c] "i" (imm8),
        );
        return r;
    } else {
        const x = bitCast_u64x2(a)[imm8 & 1];
        const y = bitCast_u64x2(b)[(imm8 >> 4) & 1];
        const r = clmulSoft128(x, y);
        return _mm_set_epi64x(@bitCast(@as(u64, @truncate(r >> 64))), @bitCast(@as(u64, @truncate(r))));
    }
}

test "_mm_clmulepi64_si128" {
    const a = _mm_set_epi64x(0, 2605358807087617882);
    const b = _mm_set_epi64x(1180018182692381252, 0);
    const ref = _mm_set_epi64x(166359506906855238, 3930956087838474216);
    try std.testing.expectEqual(ref, _mm_clmulepi64_si128(a, b, 16));
}

// POPCNT ===========================================================

pub inline fn _mm_popcnt_u32(a: u32) i32 {
    return @popCount(a);
}

test "_mm_popcnt_u32" {
    const a: u32 = 2863311530;
    try std.testing.expectEqual(@as(i32, 16), _mm_popcnt_u32(a));
}

pub inline fn _mm_popcnt_u64(a: u64) i64 {
    return @popCount(a);
}

test "_mm_popcnt_u64" {
    const a: u64 = 12297829382473034410;
    try std.testing.expectEqual(@as(i64, 32), _mm_popcnt_u64(a));
}

pub inline fn _popcnt32(a: i32) i32 {
    return @popCount(a);
}

test "_popcnt32" {
    const a: i32 = -1431655766;
    try std.testing.expectEqual(@as(i32, 16), _popcnt32(a));
}

pub inline fn _popcnt64(a: i64) i32 {
    return @popCount(a);
}

test "_popcnt64" {
    const a: i64 = -6148914691236517206;
    try std.testing.expectEqual(@as(i32, 32), _popcnt64(a));
}

// LZCNT ============================================================

pub inline fn _lzcnt_u32(a: u32) u32 {
    return @clz(a);
}

test "_lzcnt_u32" {
    try std.testing.expectEqual(@as(u32, 32), _lzcnt_u32(@as(u32, 0x00000000)));
    try std.testing.expectEqual(@as(u32, 15), _lzcnt_u32(@as(u32, 0x00011000)));
    try std.testing.expectEqual(@as(u32, 0), _lzcnt_u32(@as(u32, 0x80000000)));
}

pub inline fn _lzcnt_u64(a: u64) u64 {
    return @clz(a);
}

test "_lzcnt_u64" {
    try std.testing.expectEqual(@as(u64, 64), _lzcnt_u64(@as(u64, 0x0000000000000000)));
    try std.testing.expectEqual(@as(u64, 15), _lzcnt_u64(@as(u64, 0x0001100000000200)));
    try std.testing.expectEqual(@as(u64, 0), _lzcnt_u64(@as(u64, 0x8000000000000000)));
}

// AVX ==============================================================

pub inline fn _mm256_add_pd(a: __m256d, b: __m256d) __m256d {
    return a + b;
}

pub inline fn _mm256_add_ps(a: __m256, b: __m256) __m256 {
    return a + b;
}

pub inline fn _mm256_addsub_pd(a: __m256d, b: __m256d) __m256d {
    return .{ a[0] - b[0], a[1] + b[1], a[2] - b[2], a[3] + b[3] };
}

pub inline fn _mm256_addsub_ps(a: __m256, b: __m256) __m256 {
    return .{ a[0] - b[0], a[1] + b[1], a[2] - b[2], a[3] + b[3], a[4] - b[4], a[5] + b[5], a[6] - b[6], a[7] + b[7] };
}

pub inline fn _mm256_and_pd(a: __m256d, b: __m256d) __m256d {
    return @bitCast(bitCast_u64x4(a) & bitCast_u64x4(b));
}

pub inline fn _mm256_and_ps(a: __m256, b: __m256) __m256 {
    return @bitCast(bitCast_u32x8(a) & bitCast_u32x8(b));
}

pub inline fn _mm256_andnot_pd(a: __m256d, b: __m256d) __m256d {
    return @bitCast(~bitCast_u64x4(a) & bitCast_u64x4(b));
}

pub inline fn _mm256_andnot_ps(a: __m256, b: __m256) __m256 {
    return @bitCast(~bitCast_u32x8(a) & bitCast_u32x8(b));
}

pub inline fn _mm256_blend_pd(a: __m256d, b: __m256d, comptime imm8: comptime_int) __m256d {
    return @select(f64, @as(@Vector(4, bool), @bitCast(@as(u4, imm8))), b, a);
}

pub inline fn _mm256_blend_ps(a: __m256, b: __m256, comptime imm8: comptime_int) __m256 {
    return @select(f32, @as(@Vector(8, bool), @bitCast(@as(u8, imm8))), b, a);
}

pub inline fn _mm256_blendv_pd(a: __m256d, b: __m256d, mask: __m256d) __m256d {
    const cmp = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
    return @select(f64, cmp, b, a);
}

pub inline fn _mm256_blendv_ps(a: __m256, b: __m256, mask: __m256) __m256 {
    const cmp = @as(i32x8, @splat(0)) > bitCast_i32x8(mask);
    return @select(f32, cmp, b, a);
}

pub inline fn _mm256_broadcast_pd(mem_addr: *align(1) const __m128d) __m256d {
    const x = _mm_loadu_pd(mem_addr);
    return .{ x[0], x[1], x[0], x[1] };
}

pub inline fn _mm256_broadcast_ps(mem_addr: *align(1) const __m128) __m256 {
    const x = _mm_loadu_ps(mem_addr);
    return .{ x[0], x[1], x[2], x[3], x[0], x[1], x[2], x[3] };
}

pub inline fn _mm256_broadcast_sd(mem_addr: *align(1) const f64) __m256d {
    return _mm256_set1_pd(mem_addr.*);
}

pub inline fn _mm_broadcast_ss(mem_addr: *align(1) const f32) __m128 {
    return _mm_set1_ps(mem_addr.*);
}

pub inline fn _mm256_broadcast_ss(mem_addr: *align(1) const f32) __m256 {
    return _mm256_set1_ps(mem_addr.*);
}

pub inline fn _mm256_castpd_ps(a: __m256d) __m256 {
    return @bitCast(a);
}

pub inline fn _mm256_castpd_si256(a: __m256d) __m256i {
    return @bitCast(a);
}

/// *** the upper 128-bits are undefined...
pub inline fn _mm256_castpd128_pd256(a: __m128d) __m256d {
    return .{ a[0], a[1], 0, 0 };
}

pub inline fn _mm256_castpd256_pd128(a: __m256d) __m128d {
    return .{ a[0], a[1] };
}

pub inline fn _mm256_castps_pd(a: __m256) __m256d {
    return @bitCast(a);
}

pub inline fn _mm256_castps_si256(a: __m256) __m256i {
    return @bitCast(a);
}

/// *** the upper 128-bits are undefined...
pub inline fn _mm256_castps128_ps256(a: __m128) __m256 {
    return .{ a[0], a[1], a[2], a[3], 0, 0, 0, 0 };
}

pub inline fn _mm256_castps256_ps128(a: __m256) __m128 {
    return .{ a[0], a[1], a[2], a[3] };
}

/// *** the upper 128-bits are undefined...
pub inline fn _mm256_castsi128_si256(a: __m128i) __m256i {
    return @bitCast(u64x4{ bitCast_u64x2(a)[0], bitCast_u64x2(a)[1], 0, 0 });
}

pub inline fn _mm256_castsi256_pd(a: __m256i) __m256d {
    return @bitCast(a);
}

pub inline fn _mm256_castsi256_ps(a: __m256i) __m256 {
    return @bitCast(a);
}

pub inline fn _mm256_castsi256_si128(a: __m256i) __m128i {
    return @bitCast(u64x2{ bitCast_u64x4(a)[0], bitCast_u64x4(a)[1] });
}

// ## pub inline fn _mm256_ceil_pd (a: __m256d) __m256d {}
// ## pub inline fn _mm256_ceil_ps (a: __m256) __m256 {}
// ## pub inline fn _mm_cmp_pd (a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {}
// ## pub inline fn _mm256_cmp_pd (a: __m256d, b: __m256d, comptime imm8: comptime_int) __m256d {}
// ## pub inline fn _mm_cmp_ps (a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {}
// ## pub inline fn _mm256_cmp_ps (a: __m256, b: __m256, comptime imm8: comptime_int) __m256 {}
// ## pub inline fn _mm_cmp_sd (a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {}
// ## pub inline fn _mm_cmp_ss (a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {}

pub inline fn _mm256_cvtepi32_pd(a: __m128i) __m256d {
    const x = bitCast_i32x4(a);
    return .{ @floatFromInt(x[0]), @floatFromInt(x[1]), @floatFromInt(x[2]), @floatFromInt(x[3]) };
}

pub inline fn _mm256_cvtepi32_ps(a: __m256i) __m256 {
    return @floatFromInt(bitCast_i32x8(a));
}

// ## pub inline fn _mm256_cvtpd_epi32 (a: __m256d) __m128i {}
// ## pub inline fn _mm256_cvtpd_ps (a: __m256d) __m128 {}
// ## pub inline fn _mm256_cvtps_epi32 (a: __m256) __m256i {}

pub inline fn _mm256_cvtps_pd(a: __m128) __m256d {
    return .{ @floatCast(a[0]), @floatCast(a[1]), @floatCast(a[2]), @floatCast(a[3]) };
}

pub inline fn _mm256_cvtsd_f64(a: __m256d) f64 {
    return a[0];
}

pub inline fn _mm256_cvtsi256_si32(a: __m256i) i32 {
    return bitCast_i32x8(a)[0];
}

pub inline fn _mm256_cvtss_f32(a: __m256) f32 {
    return a[0];
}

// ## pub inline fn _mm256_cvttpd_epi32 (a: __m256d) __m128i {}
// ## pub inline fn _mm256_cvttps_epi32 (a: __m256) __m256i {}
// ## pub inline fn _mm256_div_pd (a: __m256d, b: __m256d) __m256d {}
// ## pub inline fn _mm256_div_ps (a: __m256, b: __m256) __m256 {}
// ## pub inline fn _mm256_dp_ps (a: __m256, b: __m256, comptime imm8: comptime_int) __m256 {}

pub inline fn _mm256_extract_epi32(a: __m256i, comptime index: comptime_int) i32 {
    return bitCast_i32x8(a)[index];
}

pub inline fn _mm256_extract_epi64(a: __m256i, comptime index: comptime_int) i64 {
    return bitCast_i64x4(a)[index];
}

pub inline fn _mm256_extractf128_pd(a: __m256d, comptime imm8: comptime_int) __m128d {
    return .{ a[imm8 * 2], a[(imm8 * 2) + 1] };
}

pub inline fn _mm256_extractf128_ps(a: __m256, comptime imm8: comptime_int) __m128 {
    return .{ a[imm8 * 4], a[(imm8 * 4) + 1], a[(imm8 * 4) + 2], a[(imm8 * 4) + 3] };
}

pub inline fn _mm256_extractf128_si256(a: __m256i, comptime imm8: comptime_int) __m128i {
    return _mm256_extracti128_si256(a, imm8);
}

// ## pub inline fn _mm256_floor_pd(a: __m256d) __m256d {}
// ## pub inline fn _mm256_floor_ps(a: __m256) __m256 {}

pub inline fn _mm256_hadd_pd(a: __m256d, b: __m256d) __m256d {
    return .{ a[0] + a[1], b[0] + b[1], a[2] + a[3], b[2] + b[3] };
}

pub inline fn _mm256_hadd_ps(a: __m256, b: __m256) __m256 {
    return .{ a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3], a[4] + a[5], a[6] + a[7], b[4] + b[5], b[6] + b[7] };
}

pub inline fn _mm256_hsub_pd(a: __m256d, b: __m256d) __m256d {
    return .{ a[0] - a[1], b[0] - b[1], a[2] - a[3], b[2] - b[3] };
}

pub inline fn _mm256_hsub_ps(a: __m256, b: __m256) __m256 {
    return .{ a[0] - a[1], a[2] - a[3], b[0] - b[1], b[2] - b[3], a[4] - a[5], a[6] - a[7], b[4] - b[5], b[6] - b[7] };
}

pub inline fn _mm256_insert_epi16(a: __m256i, i: i16, comptime index: comptime_int) __m256i {
    var r = bitCast_i16x16(a);
    r[index] = i;
    return @bitCast(r);
}

pub inline fn _mm256_insert_epi32(a: __m256i, i: i32, comptime index: comptime_int) __m256i {
    var r = bitCast_i32x8(a);
    r[index] = i;
    return @bitCast(r);
}

pub inline fn _mm256_insert_epi64(a: __m256i, i: i64, comptime index: comptime_int) __m256i {
    var r = bitCast_i64x4(a);
    r[index] = i;
    return @bitCast(r);
}

pub inline fn _mm256_insert_epi8(a: __m256i, i: i8, comptime index: comptime_int) __m256i {
    var r = bitCast_i8x32(a);
    r[index] = i;
    return @bitCast(r);
}

pub inline fn _mm256_insertf128_pd(a: __m256d, b: __m128d, comptime imm8: comptime_int) __m256d {
    if (@as(u1, imm8) == 1) {
        return .{ a[0], a[1], b[0], b[1] };
    } else {
        return .{ b[0], b[1], a[2], a[3] };
    }
}

pub inline fn _mm256_insertf128_ps(a: __m256, b: __m128, comptime imm8: comptime_int) __m256 {
    if (@as(u1, imm8) == 1) {
        return .{ a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3] };
    } else {
        return .{ b[0], b[1], b[2], b[3], a[4], a[5], a[6], a[7] };
    }
}

pub inline fn _mm256_insertf128_si256(a: __m256i, b: __m128i, comptime imm8: comptime_int) __m256i {
    if (@as(u1, imm8) == 1) {
        return @bitCast(u64x4{ bitCast_u64x4(a)[0], bitCast_u64x4(a)[1], bitCast_u64x2(b)[0], bitCast_u64x2(b)[1] });
    } else {
        return @bitCast(u64x4{ bitCast_u64x2(b)[0], bitCast_u64x2(b)[1], bitCast_u64x4(a)[2], bitCast_u64x4(a)[3] });
    }
}

pub inline fn _mm256_lddqu_si256(mem_addr: *align(1) const __m256i) __m256i {
    return _mm256_loadu_si256(mem_addr);
}

pub inline fn _mm256_load_pd(mem_addr: *align(32) const [4]f64) __m256d {
    return .{ mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3] };
}

pub inline fn _mm256_load_ps(mem_addr: *align(32) const [8]f32) __m256 {
    return .{
        mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3],
        mem_addr[4], mem_addr[5], mem_addr[6], mem_addr[7],
    };
}

pub inline fn _mm256_load_si256(mem_addr: *align(32) const __m256i) __m256i {
    return mem_addr.*;
}

pub inline fn _mm256_loadu_pd(mem_addr: *align(1) const [4]f64) __m256d {
    return .{ mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3] };
}

pub inline fn _mm256_loadu_ps(mem_addr: *align(1) const [8]f32) __m256 {
    return .{
        mem_addr[0], mem_addr[1], mem_addr[2], mem_addr[3],
        mem_addr[4], mem_addr[5], mem_addr[6], mem_addr[7],
    };
}

pub inline fn _mm256_loadu_si256(mem_addr: *align(1) const __m256i) __m256i {
    return mem_addr.*;
}

pub inline fn _mm256_loadu2_m128(hiaddr: *align(1) const [4]f32, loaddr: *align(1) const [4]f32) __m256 {
    const hi = _mm_loadu_ps(hiaddr);
    const lo = _mm_loadu_ps(loaddr);
    return _mm256_set_m128(hi, lo);
}

pub inline fn _mm256_loadu2_m128d(hiaddr: *align(1) const [2]f64, loaddr: *align(1) const [2]f64) __m256d {
    const hi = _mm_loadu_pd(hiaddr);
    const lo = _mm_loadu_pd(loaddr);
    return _mm256_set_m128d(hi, lo);
}

pub inline fn _mm256_loadu2_m128i(hiaddr: *align(1) const __m128i, loaddr: *align(1) const __m128i) __m256i {
    const hi = _mm_loadu_si128(hiaddr);
    const lo = _mm_loadu_si128(loaddr);
    return _mm256_set_m128i(hi, lo);
}

// ## __m128d _mm_maskload_pd (double const * mem_addr, __m128i mask)
// ## __m256d _mm256_maskload_pd (double const * mem_addr, __m256i mask)
// ## __m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)
// ## __m256 _mm256_maskload_ps (float const * mem_addr, __m256i mask)
// ## void _mm_maskstore_pd (double * mem_addr, __m128i mask, __m128d a)
// ## void _mm256_maskstore_pd (double * mem_addr, __m256i mask, __m256d a)
// ## void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)
// ## void _mm256_maskstore_ps (float * mem_addr, __m256i mask, __m256 a)
// ## __m256d _mm256_max_pd (__m256d a, __m256d b)
// ## __m256 _mm256_max_ps (__m256 a, __m256 b)
// ## __m256d _mm256_min_pd (__m256d a, __m256d b)
// ## __m256 _mm256_min_ps (__m256 a, __m256 b)

pub inline fn _mm256_movedup_pd(a: __m256d) __m256d {
    return .{ a[0], a[0], a[2], a[2] };
}

pub inline fn _mm256_movehdup_ps(a: __m256) __m256 {
    return .{ a[1], a[1], a[3], a[3], a[5], a[5], a[7], a[7] };
}

pub inline fn _mm256_moveldup_ps(a: __m256) __m256 {
    return .{ a[0], a[0], a[2], a[2], a[4], a[4], a[6], a[6] };
}

pub inline fn _mm256_movemask_pd(a: __m256d) i32 {
    const cmp = @as(i64x4, @splat(0)) > bitCast_i64x4(a);
    return @intCast(@as(u4, @bitCast(cmp)));
}

pub inline fn _mm256_movemask_ps(a: __m256) i32 {
    const cmp = @as(i32x8, @splat(0)) > bitCast_i32x8(a);
    return @intCast(@as(u8, @bitCast(cmp)));
}

pub inline fn _mm256_mul_pd(a: __m256d, b: __m256d) __m256d {
    return a * b;
}

pub inline fn _mm256_mul_ps(a: __m256, b: __m256) __m256 {
    return a * b;
}

pub inline fn _mm256_or_pd(a: __m256d, b: __m256d) __m256d {
    return @bitCast(bitCast_u64x4(a) | bitCast_u64x4(b));
}

pub inline fn _mm256_or_ps(a: __m256, b: __m256) __m256 {
    return @bitCast(bitCast_u32x8(a) | bitCast_u32x8(b));
}

pub inline fn _mm_permute_pd(a: __m128d, comptime imm8: comptime_int) __m128d {
    return _mm_shuffle_pd(a, a, imm8);
}

pub inline fn _mm256_permute_pd(a: __m256d, comptime imm8: comptime_int) __m256d {
    return _mm256_shuffle_pd(a, a, imm8);
}

pub inline fn _mm_permute_ps(a: __m128, comptime imm8: comptime_int) __m128 {
    return _mm_shuffle_ps(a, a, imm8);
}

pub inline fn _mm256_permute_ps(a: __m256, comptime imm8: comptime_int) __m256 {
    return _mm256_shuffle_ps(a, a, imm8);
}

pub inline fn _mm256_permute2f128_pd(a: __m256d, b: __m256d, comptime imm8: comptime_int) __m256d {
    if ((imm8 & 0x08) == 0x08) { // optimizer hand-holding when zeroing the low 128-bits
        return switch (@as(u8, imm8) >> 4) {
            0, 4 => .{ 0, 0, a[0], a[1] },
            1, 5 => .{ 0, 0, a[2], a[3] },
            2, 6 => .{ 0, 0, b[0], b[1] },
            3, 7 => .{ 0, 0, b[2], b[3] },
            else => .{ 0, 0, 0, 0 },
        };
    }

    const lo: __m128d = switch (imm8 & 0x0F) {
        0, 4 => _mm256_extractf128_pd(a, 0),
        1, 5 => _mm256_extractf128_pd(a, 1),
        2, 6 => _mm256_extractf128_pd(b, 0),
        3, 7 => _mm256_extractf128_pd(b, 1),
        else => @splat(0),
    };

    const hi: __m128d = switch (@as(u8, imm8) >> 4) {
        0, 4 => _mm256_extractf128_pd(a, 0),
        1, 5 => _mm256_extractf128_pd(a, 1),
        2, 6 => _mm256_extractf128_pd(b, 0),
        3, 7 => _mm256_extractf128_pd(b, 1),
        else => @splat(0),
    };

    return _mm256_set_m128d(hi, lo);
}

pub inline fn _mm256_permute2f128_ps(a: __m256, b: __m256, comptime imm8: comptime_int) __m256 {
    if ((imm8 & 0x08) == 0x08) { // optimizer hand-holding when zeroing the low 128-bits
        return switch (@as(u8, imm8) >> 4) {
            0, 4 => .{ 0, 0, 0, 0, a[0], a[1], a[2], a[3] },
            1, 5 => .{ 0, 0, 0, 0, a[4], a[5], a[6], a[7] },
            2, 6 => .{ 0, 0, 0, 0, b[0], b[1], b[2], b[3] },
            3, 7 => .{ 0, 0, 0, 0, b[4], b[5], b[6], b[7] },
            else => @splat(0),
        };
    }

    const lo: __m128 = switch (imm8 & 0x0F) {
        0, 4 => _mm256_extractf128_ps(a, 0),
        1, 5 => _mm256_extractf128_ps(a, 1),
        2, 6 => _mm256_extractf128_ps(b, 0),
        3, 7 => _mm256_extractf128_ps(b, 1),
        else => @splat(0),
    };

    const hi: __m128 = switch (@as(u8, imm8) >> 4) {
        0, 4 => _mm256_extractf128_ps(a, 0),
        1, 5 => _mm256_extractf128_ps(a, 1),
        2, 6 => _mm256_extractf128_ps(b, 0),
        3, 7 => _mm256_extractf128_ps(b, 1),
        else => @splat(0),
    };

    return _mm256_set_m128(hi, lo);
}

test "_mm256_permute2f128_ps" {
    const a = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    const b = _mm256_set_ps(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
    const ref0 = _mm256_set_ps(5.5, 6.5, 7.5, 8.5, 1.0, 2.0, 3.0, 4.0);
    const ref1 = _mm256_set_ps(5.5, 6.5, 7.5, 8.5, 0.0, 0.0, 0.0, 0.0);
    const ref2 = _mm256_set_ps(5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5);
    const ref3 = _mm256_set_ps(5.0, 6.0, 7.0, 8.0, 1.5, 2.5, 3.5, 4.5);

    try std.testing.expectEqual(ref0, _mm256_permute2f128_ps(a, b, 0x21));
    try std.testing.expectEqual(ref1, _mm256_permute2f128_ps(a, b, 0x28));
    try std.testing.expectEqual(ref2, _mm256_permute2f128_ps(a, b, 0x23));
    try std.testing.expectEqual(ref3, _mm256_permute2f128_ps(a, b, 0x03));
}

pub inline fn _mm256_permute2f128_si256(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    return _mm256_permute2x128_si256(a, b, imm8);
}

pub inline fn _mm_permutevar_pd(a: __m128d, b: __m128i) __m128d {
    if ((has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpermilpd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else { // note: compiler doesn't eliminate the shift by 1
        const shuf = (bitCast_u64x2(b) & @as(u64x2, @splat(2))) >> @as(u64x2, @splat(1));
        return .{ a[shuf[0]], a[shuf[1]] };
    }
}

test "_mm_permutevar_pd" {
    const a = _mm_set_pd(1.5, 3.5);
    const b = _mm_set_epi64x(3, 2);
    const c = _mm_set_epi64x(1, 5);
    const d = _mm_set_epi64x(2, 1);
    const e = _mm_set_epi64x(1, 2);
    const ref0 = _mm_set_pd(1.5, 1.5);
    const ref1 = _mm_set_pd(3.5, 3.5);
    const ref2 = _mm_set_pd(1.5, 3.5);
    const ref3 = _mm_set_pd(3.5, 1.5);
    try std.testing.expectEqual(ref0, _mm_permutevar_pd(a, b));
    try std.testing.expectEqual(ref1, _mm_permutevar_pd(a, c));
    try std.testing.expectEqual(ref2, _mm_permutevar_pd(a, d));
    try std.testing.expectEqual(ref3, _mm_permutevar_pd(a, e));
}

pub inline fn _mm256_permutevar_pd(a: __m256d, b: __m256i) __m256d {
    if ((has_avx) and (!bug_stage2_x86_64)) {
        return asm ("vpermilpd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256d),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const lo = _mm_permutevar_pd(_mm256_extractf128_pd(a, 0), _mm256_extracti128_si256(b, 0));
        const hi = _mm_permutevar_pd(_mm256_extractf128_pd(a, 1), _mm256_extracti128_si256(b, 1));
        return _mm256_set_m128d(hi, lo);
    }
}

test "_mm256_permutevar_pd" {
    const a = _mm256_set_pd(4.5, 3.5, 2.5, 1.5);
    const b = _mm256_set_epi64x(3, 2, 3, 2);
    const c = _mm256_set_epi64x(1, 5, 1, 5);
    const d = _mm256_set_epi64x(2, 1, 2, 1);
    const e = _mm256_set_epi64x(1, 2, 1, 2);
    const ref0 = _mm256_set_pd(4.5, 4.5, 2.5, 2.5);
    const ref1 = _mm256_set_pd(3.5, 3.5, 1.5, 1.5);
    const ref2 = _mm256_set_pd(4.5, 3.5, 2.5, 1.5);
    const ref3 = _mm256_set_pd(3.5, 4.5, 1.5, 2.5);
    try std.testing.expectEqual(ref0, _mm256_permutevar_pd(a, b));
    try std.testing.expectEqual(ref1, _mm256_permutevar_pd(a, c));
    try std.testing.expectEqual(ref2, _mm256_permutevar_pd(a, d));
    try std.testing.expectEqual(ref3, _mm256_permutevar_pd(a, e));
}

pub inline fn _mm_permutevar_ps(a: __m128, b: __m128i) __m128 {
    if (has_avx) {
        return asm ("vpermilps %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const shuf = bitCast_u32x4(b) & @as(u32x4, @splat(3));
        return .{ a[shuf[0]], a[shuf[1]], a[shuf[2]], a[shuf[3]] };
    }
}

pub inline fn _mm256_permutevar_ps(a: __m256, b: __m256i) __m256 {
    if (has_avx) {
        return asm ("vpermilps %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else { // pretty bad code-gen... which is strange because _mm_permutevar_ps by itself isn't that bad...
        const lo = _mm_permutevar_ps(_mm256_extractf128_ps(a, 0), _mm256_extracti128_si256(b, 0));
        const hi = _mm_permutevar_ps(_mm256_extractf128_ps(a, 1), _mm256_extracti128_si256(b, 1));
        return _mm256_set_m128(hi, lo);
    }
}

/// Approximate reciprocal
pub inline fn _mm256_rcp_ps(a: __m256) __m256 {
    if (has_avx) {
        return asm ("vrcpps %[a], %[ret]"
            : [ret] "=x" (-> __m256),
            : [a] "x" (a),
        );
    } else {
        return @as(__m256, @splat(1.0)) / a;
    }
}

// ## pub inline fn _mm256_round_pd (a: __m256d, comptime rounding: comptime_int) __m256d {}
// ## pub inline fn _mm256_round_ps ( a: __m256, comptime rounding: comptime_int)  __m256 {}

/// Approximate reciprocal square root
pub inline fn _mm256_rsqrt_ps(a: __m256) __m256 {
    if (has_avx) {
        return asm ("vrsqrtps %[a], %[ret]"
            : [ret] "=x" (-> __m256),
            : [a] "x" (a),
        );
    } else {
        return @as(__m256, @splat(1.0)) / @sqrt(a);
    }
}

pub inline fn _mm256_set_epi16(e15: i16, e14: i16, e13: i16, e12: i16, e11: i16, e10: i16, e9: i16, e8: i16, e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m256i {
    return @bitCast(i16x16{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 });
}

test "_mm256_set_epi16" {
    const a = _mm256_set_epi16(1500, -32768, 13, 12, -11, 10, 9, 8, -7, -6, -5, -4, -3, -2, -1, 0);
    try std.testing.expectEqual(@as(i16, 0), bitCast_i16x16(a)[0]);
    try std.testing.expectEqual(@as(i16, -1), bitCast_i16x16(a)[1]);
    try std.testing.expectEqual(@as(i16, -2), bitCast_i16x16(a)[2]);
    try std.testing.expectEqual(@as(i16, -3), bitCast_i16x16(a)[3]);
    try std.testing.expectEqual(@as(i16, -4), bitCast_i16x16(a)[4]);
    try std.testing.expectEqual(@as(i16, -5), bitCast_i16x16(a)[5]);
    try std.testing.expectEqual(@as(i16, -6), bitCast_i16x16(a)[6]);
    try std.testing.expectEqual(@as(i16, -7), bitCast_i16x16(a)[7]);
    try std.testing.expectEqual(@as(i16, 8), bitCast_i16x16(a)[8]);
    try std.testing.expectEqual(@as(i16, 9), bitCast_i16x16(a)[9]);
    try std.testing.expectEqual(@as(i16, 10), bitCast_i16x16(a)[10]);
    try std.testing.expectEqual(@as(i16, -11), bitCast_i16x16(a)[11]);
    try std.testing.expectEqual(@as(i16, 12), bitCast_i16x16(a)[12]);
    try std.testing.expectEqual(@as(i16, 13), bitCast_i16x16(a)[13]);
    try std.testing.expectEqual(@as(i16, -32768), bitCast_i16x16(a)[14]);
    try std.testing.expectEqual(@as(i16, 1500), bitCast_i16x16(a)[15]);
}

pub inline fn _mm256_set_epi32(e7: i32, e6: i32, e5: i32, e4: i32, e3: i32, e2: i32, e1: i32, e0: i32) __m256i {
    return @bitCast(i32x8{ e0, e1, e2, e3, e4, e5, e6, e7 });
}

test "_mm256_set_epi32" {
    const a = _mm256_set_epi32(-2147483648, 1610612736, -5, -4, -3, -2, -1, 0);
    try std.testing.expectEqual(@as(i32, 0), bitCast_i32x8(a)[0]);
    try std.testing.expectEqual(@as(i32, -1), bitCast_i32x8(a)[1]);
    try std.testing.expectEqual(@as(i32, -2), bitCast_i32x8(a)[2]);
    try std.testing.expectEqual(@as(i32, -3), bitCast_i32x8(a)[3]);
    try std.testing.expectEqual(@as(i32, -4), bitCast_i32x8(a)[4]);
    try std.testing.expectEqual(@as(i32, -5), bitCast_i32x8(a)[5]);
    try std.testing.expectEqual(@as(i32, 1610612736), bitCast_i32x8(a)[6]);
    try std.testing.expectEqual(@as(i32, -2147483648), bitCast_i32x8(a)[7]);
}

pub inline fn _mm256_set_epi64x(e3: i64, e2: i64, e1: i64, e0: i64) __m256i {
    return @bitCast(i64x4{ e0, e1, e2, e3 });
}

test "_mm256_set_epi64x" {
    const a = _mm256_set_epi64x(6917529027641081856, -9223372036854775808, -2, -1);
    try std.testing.expectEqual(@as(i64, -1), bitCast_i64x4(a)[0]);
    try std.testing.expectEqual(@as(i64, -2), bitCast_i64x4(a)[1]);
    try std.testing.expectEqual(@as(i64, -9223372036854775808), bitCast_i64x4(a)[2]);
    try std.testing.expectEqual(@as(i64, 6917529027641081856), bitCast_i64x4(a)[3]);
}

pub inline fn _mm256_set_epi8(e31: i8, e30: i8, e29: i8, e28: i8, e27: i8, e26: i8, e25: i8, e24: i8, e23: i8, e22: i8, e21: i8, e20: i8, e19: i8, e18: i8, e17: i8, e16: i8, e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m256i {
    return @bitCast(i8x32{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31 });
}

test "_mm256_set_epi8" {
    const a = _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, -128, 127, 14, 13, 12, -11, 10, 9, 8, -7, -6, -5, -4, -3, -2, -1, 0);
    try std.testing.expectEqual(@as(i8, 0), bitCast_i8x32(a)[0]);
    try std.testing.expectEqual(@as(i8, -1), bitCast_i8x32(a)[1]);
    try std.testing.expectEqual(@as(i8, -2), bitCast_i8x32(a)[2]);
    try std.testing.expectEqual(@as(i8, -3), bitCast_i8x32(a)[3]);
    try std.testing.expectEqual(@as(i8, -4), bitCast_i8x32(a)[4]);
    try std.testing.expectEqual(@as(i8, -5), bitCast_i8x32(a)[5]);
    try std.testing.expectEqual(@as(i8, -6), bitCast_i8x32(a)[6]);
    try std.testing.expectEqual(@as(i8, -7), bitCast_i8x32(a)[7]);
    try std.testing.expectEqual(@as(i8, 8), bitCast_i8x32(a)[8]);
    try std.testing.expectEqual(@as(i8, 9), bitCast_i8x32(a)[9]);
    try std.testing.expectEqual(@as(i8, 10), bitCast_i8x32(a)[10]);
    try std.testing.expectEqual(@as(i8, -11), bitCast_i8x32(a)[11]);
    try std.testing.expectEqual(@as(i8, 12), bitCast_i8x32(a)[12]);
    try std.testing.expectEqual(@as(i8, 13), bitCast_i8x32(a)[13]);
    try std.testing.expectEqual(@as(i8, 14), bitCast_i8x32(a)[14]);
    try std.testing.expectEqual(@as(i8, 127), bitCast_i8x32(a)[15]);
    try std.testing.expectEqual(@as(i8, -128), bitCast_i8x32(a)[16]);
    try std.testing.expectEqual(@as(i8, 17), bitCast_i8x32(a)[17]);
    try std.testing.expectEqual(@as(i8, 18), bitCast_i8x32(a)[18]);
    try std.testing.expectEqual(@as(i8, 19), bitCast_i8x32(a)[19]);
    try std.testing.expectEqual(@as(i8, 20), bitCast_i8x32(a)[20]);
    try std.testing.expectEqual(@as(i8, 21), bitCast_i8x32(a)[21]);
    try std.testing.expectEqual(@as(i8, 22), bitCast_i8x32(a)[22]);
    try std.testing.expectEqual(@as(i8, 23), bitCast_i8x32(a)[23]);
    try std.testing.expectEqual(@as(i8, 24), bitCast_i8x32(a)[24]);
    try std.testing.expectEqual(@as(i8, 25), bitCast_i8x32(a)[25]);
    try std.testing.expectEqual(@as(i8, 26), bitCast_i8x32(a)[26]);
    try std.testing.expectEqual(@as(i8, 27), bitCast_i8x32(a)[27]);
    try std.testing.expectEqual(@as(i8, 28), bitCast_i8x32(a)[28]);
    try std.testing.expectEqual(@as(i8, 29), bitCast_i8x32(a)[29]);
    try std.testing.expectEqual(@as(i8, 30), bitCast_i8x32(a)[30]);
    try std.testing.expectEqual(@as(i8, 31), bitCast_i8x32(a)[31]);
}

pub inline fn _mm256_set_m128(hi: __m128, lo: __m128) __m256 {
    return .{ lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3] };
}

pub inline fn _mm256_set_m128d(hi: __m128d, lo: __m128d) __m256d {
    return .{ lo[0], lo[1], hi[0], hi[1] };
}

pub inline fn _mm256_set_m128i(hi: __m128i, lo: __m128i) __m256i {
    const l = bitCast_u64x2(lo);
    const h = bitCast_u64x2(hi);
    return @bitCast(u64x4{ l[0], l[1], h[0], h[1] });
}

pub inline fn _mm256_set_pd(e3: f64, e2: f64, e1: f64, e0: f64) __m256d {
    return .{ e0, e1, e2, e3 };
}

pub inline fn _mm256_set_ps(e7: f32, e6: f32, e5: f32, e4: f32, e3: f32, e2: f32, e1: f32, e0: f32) __m256 {
    return .{ e0, e1, e2, e3, e4, e5, e6, e7 };
}

pub inline fn _mm256_set1_epi16(a: i16) __m256i {
    return @bitCast(@as(i16x16, @splat(a)));
}

pub inline fn _mm256_set1_epi32(a: i32) __m256i {
    return @bitCast(@as(i32x8, @splat(a)));
}

pub inline fn _mm256_set1_epi64x(a: i64) __m256i {
    return @bitCast(@as(i64x4, @splat(a)));
}

pub inline fn _mm256_set1_epi8(a: i8) __m256i {
    return @bitCast(@as(i8x32, @splat(a)));
}

pub inline fn _mm256_set1_pd(a: f64) __m256d {
    return @splat(a);
}

pub inline fn _mm256_set1_ps(a: f32) __m256 {
    return @splat(a);
}

pub inline fn _mm256_setr_epi16(e15: i16, e14: i16, e13: i16, e12: i16, e11: i16, e10: i16, e9: i16, e8: i16, e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m256i {
    return @bitCast(i16x16{ e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0 });
}

pub inline fn _mm256_setr_epi32(e7: i32, e6: i32, e5: i32, e4: i32, e3: i32, e2: i32, e1: i32, e0: i32) __m256i {
    return @bitCast(i32x8{ e7, e6, e5, e4, e3, e2, e1, e0 });
}

pub inline fn _mm256_setr_epi64x(e3: i64, e2: i64, e1: i64, e0: i64) __m256i {
    return @bitCast(i64x4{ e3, e2, e1, e0 });
}

pub inline fn _mm256_setr_epi8(e31: i8, e30: i8, e29: i8, e28: i8, e27: i8, e26: i8, e25: i8, e24: i8, e23: i8, e22: i8, e21: i8, e20: i8, e19: i8, e18: i8, e17: i8, e16: i8, e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m256i {
    return @bitCast(i8x32{ e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0 });
}

pub inline fn _mm256_setr_m128(lo: __m128, hi: __m128) __m256 {
    return .{ lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3] };
}

pub inline fn _mm256_setr_m128d(lo: __m128d, hi: __m128d) __m256d {
    return .{ lo[0], lo[1], hi[0], hi[1] };
}

pub inline fn _mm256_setr_m128i(lo: __m128i, hi: __m128i) __m256i {
    const l = bitCast_u64x2(lo);
    const h = bitCast_u64x2(hi);
    return @bitCast(u64x4{ l[0], l[1], h[0], h[1] });
}

pub inline fn _mm256_setr_pd(e3: f64, e2: f64, e1: f64, e0: f64) __m256d {
    return .{ e3, e2, e1, e0 };
}

pub inline fn _mm256_setr_ps(e7: f32, e6: f32, e5: f32, e4: f32, e3: f32, e2: f32, e1: f32, e0: f32) __m256 {
    return .{ e7, e6, e5, e4, e3, e2, e1, e0 };
}

pub inline fn _mm256_setzero_pd() __m256d {
    return @splat(0);
}

pub inline fn _mm256_setzero_ps() __m256 {
    return @splat(0);
}

pub inline fn _mm256_setzero_si256() __m256i {
    return @splat(0);
}

pub inline fn _mm256_shuffle_pd(a: __m256d, b: __m256d, comptime imm8: comptime_int) __m256d {
    return .{ a[imm8 & 1], b[(imm8 >> 1) & 1], a[((imm8 >> 2) & 1) + 2], b[((imm8 >> 3) & 1) + 2] };
}

pub inline fn _mm256_shuffle_ps(a: __m256, b: __m256, comptime imm8: comptime_int) __m256 {
    return .{
        a[imm8 & 3],       a[(imm8 >> 2) & 3],       b[(imm8 >> 4) & 3],       b[(imm8 >> 6) & 3],
        a[(imm8 & 3) + 4], a[((imm8 >> 2) & 3) + 4], b[((imm8 >> 4) & 3) + 4], b[((imm8 >> 6) & 3) + 4],
    };
}

pub inline fn _mm256_sqrt_pd(a: __m256d) __m256d {
    return @sqrt(a);
}

pub inline fn _mm256_sqrt_ps(a: __m256) __m256 {
    return @sqrt(a);
}

pub inline fn _mm256_store_pd(mem_addr: *align(32) [4]f64, a: __m256d) void {
    for (0..4) |i| mem_addr[i] = a[i];
}

pub inline fn _mm256_store_ps(mem_addr: *align(32) [8]f32, a: __m256) void {
    for (0..8) |i| mem_addr[i] = a[i];
}

pub inline fn _mm256_store_si256(mem_addr: *align(32) __m256i, a: __m256i) void {
    mem_addr.* = a;
}

pub inline fn _mm256_storeu_pd(mem_addr: *align(1) [4]f64, a: __m256d) void {
    for (0..4) |i| mem_addr[i] = a[i];
}

pub inline fn _mm256_storeu_ps(mem_addr: *align(1) [8]f32, a: __m256) void {
    for (0..8) |i| mem_addr[i] = a[i];
}

pub inline fn _mm256_storeu_si256(mem_addr: *align(1) __m256i, a: __m256i) void {
    mem_addr.* = a;
}

test "_mm256_storeu_si256" {
    const a = _mm256_set_epi64x(4, 3, 2, 1);
    var arr: [8]i64 = undefined;
    @memset(&arr, 0);
    _mm256_storeu_si256(@ptrCast(&arr[1]), a);
    _mm256_storeu_si256(@ptrCast(&arr[2]), a);

    try std.testing.expectEqual(@as(i64, 0), arr[0]);
    try std.testing.expectEqual(@as(i64, 1), arr[1]);
    try std.testing.expectEqual(@as(i64, 1), arr[2]);
    try std.testing.expectEqual(@as(i64, 2), arr[3]);
    try std.testing.expectEqual(@as(i64, 3), arr[4]);
    try std.testing.expectEqual(@as(i64, 4), arr[5]);
    try std.testing.expectEqual(@as(i64, 0), arr[6]);
    try std.testing.expectEqual(@as(i64, 0), arr[7]);
}

pub inline fn _mm256_storeu2_m128(hiaddr: *align(1) [4]f32, loaddr: *align(1) [4]f32, a: __m256) void {
    _mm_storeu_ps(loaddr, _mm256_extractf128_ps(a, 0));
    _mm_storeu_ps(hiaddr, _mm256_extractf128_ps(a, 1));
}

pub inline fn _mm256_storeu2_m128d(hiaddr: *align(1) [2]f64, loaddr: *align(1) [2]f64, a: __m256d) void {
    _mm_storeu_pd(loaddr, _mm256_extractf128_pd(a, 0));
    _mm_storeu_pd(hiaddr, _mm256_extractf128_pd(a, 1));
}

test "_mm256_storeu2_m128d" {
    const a = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    var arr: [8]f64 = undefined;
    @memset(&arr, 0);
    _mm256_storeu2_m128d(arr[1..3], arr[5..7], a);

    try std.testing.expectEqual(@as(f64, 0.0), arr[0]);
    try std.testing.expectEqual(@as(f64, 3.0), arr[1]);
    try std.testing.expectEqual(@as(f64, 4.0), arr[2]);
    try std.testing.expectEqual(@as(f64, 0.0), arr[3]);
    try std.testing.expectEqual(@as(f64, 0.0), arr[4]);
    try std.testing.expectEqual(@as(f64, 1.0), arr[5]);
    try std.testing.expectEqual(@as(f64, 2.0), arr[6]);
    try std.testing.expectEqual(@as(f64, 0.0), arr[7]);
}

pub inline fn _mm256_storeu2_m128i(hiaddr: *align(1) __m128i, loaddr: *align(1) __m128i, a: __m256i) void {
    _mm_storeu_si128(loaddr, _mm256_extracti128_si256(a, 0));
    _mm_storeu_si128(hiaddr, _mm256_extracti128_si256(a, 1));
}

// ## pub inline fn _mm256_stream_pd (mem_addr: *anyopaque, a: __m256d) void {}
// ## pub inline fn _mm256_stream_ps (mem_addr: *anyopaque, a: __m256) void {}
// ## pub inline fn _mm256_stream_si256 (mem_addr: *anyopaque, a: __m256i) void {}

pub inline fn _mm256_sub_pd(a: __m256d, b: __m256d) __m256d {
    return a - b;
}

pub inline fn _mm256_sub_ps(a: __m256, b: __m256) __m256 {
    return a - b;
}

// ## pub inline fn _mm_testc_pd (a: __m128d, b: __m128d) i32 {}
// ## pub inline fn _mm256_testc_pd (a: __m256d, b: __m256d) i32 {}
// ## pub inline fn _mm_testc_ps (a: __m128, b: __m128) i32 {}
// ## pub inline fn _mm256_testc_ps (a: __m256, b: __m256) i32 {}
// ## pub inline fn _mm256_testc_si256 (a: __m256i, b: __m256i) i32 {}
// ## pub inline fn _mm_testnzc_pd (a: __m128d, b: __m128d) i32 {}
// ## pub inline fn _mm256_testnzc_pd (a: __m256d, b: __m256d) i32 {}
// ## pub inline fn _mm_testnzc_ps (a: __m128, b: __m128) i32 {}
// ## pub inline fn _mm256_testnzc_ps (a: __m256, b: __m256) i32 {}
// ## pub inline fn _mm256_testnzc_si256 (a: __m256i, b: __m256i) i32 {}
// ## pub inline fn _mm_testz_pd (a: __m128d, b: __m128d) i32 {}
// ## pub inline fn _mm256_testz_pd (a: __m256d, b: __m256d) i32 {}
// ## pub inline fn _mm_testz_ps (a: __m128, b: __m128) i32 {}
// ## pub inline fn _mm256_testz_ps (a: __m256, b: __m256) i32 {}

pub inline fn _mm256_testz_si256(a: __m256i, b: __m256i) i32 {
    return @intFromBool(@reduce(.Or, (a & b)) == 0);
}

pub inline fn _mm256_undefined_pd() __m256d {
    // zig `undefined` doesn't compare equal to itself ?
    return @splat(0);
}

pub inline fn _mm256_undefined_ps() __m256 {
    // zig `undefined` doesn't compare equal to itself ?
    return @splat(0);
}

pub inline fn _mm256_undefined_si256() __m256i {
    // zig `undefined` doesn't compare equal to itself ?
    return @splat(0);
}

pub inline fn _mm256_unpackhi_pd(a: __m256d, b: __m256d) __m256d {
    return .{ a[1], b[1], a[3], b[3] };
}

pub inline fn _mm256_unpackhi_ps(a: __m256, b: __m256) __m256 {
    return .{ a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7] };
}

pub inline fn _mm256_unpacklo_pd(a: __m256d, b: __m256d) __m256d {
    return .{ a[0], b[0], a[2], b[2] };
}

pub inline fn _mm256_unpacklo_ps(a: __m256, b: __m256) __m256 {
    return .{ a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5] };
}

pub inline fn _mm256_xor_pd(a: __m256d, b: __m256d) __m256d {
    return @bitCast(bitCast_u64x4(a) ^ bitCast_u64x4(b));
}

pub inline fn _mm256_xor_ps(a: __m256, b: __m256) __m256 {
    return @bitCast(bitCast_u32x8(a) ^ bitCast_u32x8(b));
}

// ## pub inline fn _mm256_zeroall() void {}
// ## pub inline fn _mm256_zeroupper() void {}

pub inline fn _mm256_zextpd128_pd256(a: __m128d) __m256d {
    return .{ a[0], a[1], 0, 0 };
}

pub inline fn _mm256_zextps128_ps256(a: __m128) __m256 {
    return .{ a[0], a[1], a[2], a[3], 0, 0, 0, 0 };
}

pub inline fn _mm256_zextsi128_si256(a: __m128i) __m256i {
    return @bitCast(u64x4{ bitCast_u64x2(a)[0], bitCast_u64x2(a)[1], 0, 0 });
}

// AVX2 ==============================================================

/// dst[n] = @abs(a[n]);
pub inline fn _mm256_abs_epi16(a: __m256i) __m256i {
    return @bitCast(@abs(bitCast_i16x16(a)));
}

test "_mm256_abs_epi16" {
    const a = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, -6554, 1, -2, -255, 255, -32768, 32767, 0, -1);
    const ref = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 6554, 1, 2, 255, 255, -32768, 32767, 0, 1);
    try std.testing.expectEqual(ref, _mm256_abs_epi16(a));
}

/// dst[n] = @abs(a[n]);
pub inline fn _mm256_abs_epi32(a: __m256i) __m256i {
    return @bitCast(@abs(bitCast_i32x8(a)));
}

test "_mm256_abs_epi32" {
    const a = _mm256_set_epi32(-2147483648, -2147483647, -255, 255, -32768, 32767, 0, -1);
    const ref = _mm256_set_epi32(-2147483648, 2147483647, 255, 255, 32768, 32767, 0, 1);
    try std.testing.expectEqual(ref, _mm256_abs_epi32(a));
}

/// dst[n] = @abs(a[n]);
pub inline fn _mm256_abs_epi8(a: __m256i) __m256i {
    return @bitCast(@abs(bitCast_i8x32(a)));
}

pub inline fn _mm256_add_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u16x16(a) +% bitCast_u16x16(b));
}

pub inline fn _mm256_add_epi32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u32x8(a) +% bitCast_u32x8(b));
}

pub inline fn _mm256_add_epi64(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u64x4(a) +% bitCast_u64x4(b));
}

pub inline fn _mm256_add_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u8x32(a) +% bitCast_u8x32(b));
}

pub inline fn _mm256_adds_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i16x16(a) +| bitCast_i16x16(b));
}

pub inline fn _mm256_adds_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i8x32(a) +| bitCast_i8x32(b));
}

pub inline fn _mm256_adds_epu16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u16x16(a) +| bitCast_u16x16(b));
}

pub inline fn _mm256_adds_epu8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u8x32(a) +| bitCast_u8x32(b));
}

pub inline fn _mm256_alignr_epi8(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    if (@as(u8, @intCast(imm8)) > 15) {
        return _mm256_alignr_epi8(@splat(0), a, imm8 - 16);
    }

    const shuf = comptime blk: {
        var indices: [32]i32 = undefined;
        for (0..16) |i| {
            var x: i32 = @as(i32, @intCast(i)) + imm8;
            if (x > 15) {
                x = -x + 15;
            }
            indices[i] = x;
            indices[i ^ 16] = x ^ 16;
        }
        break :blk indices;
    };

    return @bitCast(@shuffle(u8, bitCast_u8x32(b), bitCast_u8x32(a), shuf));
}

pub inline fn _mm256_and_si256(a: __m256i, b: __m256i) __m256i {
    return a & b;
}

pub inline fn _mm256_andnot_si256(a: __m256i, b: __m256i) __m256i {
    return ~a & b;
}

pub inline fn _mm256_avg_epu16(a: __m256i, b: __m256i) __m256i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to vpavgw
    const one: u32x16 = @splat(1);
    const c = intCast_u32x16(bitCast_u16x16(a));
    const d = intCast_u32x16(bitCast_u16x16(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(u16x16, @truncate(e)));
}

pub inline fn _mm256_avg_epu8(a: __m256i, b: __m256i) __m256i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to vpavgb
    const one: u16x32 = @splat(1);
    const c = intCast_u16x32(bitCast_u8x32(a));
    const d = intCast_u16x32(bitCast_u8x32(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(u8x32, @truncate(e)));
}

pub inline fn _mm256_blend_epi16(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    const mask: @Vector(16, bool) = @bitCast(@as(u16, (imm8 << 8) | imm8));
    return @bitCast(@select(i16, mask, bitCast_i16x16(b), bitCast_i16x16(a)));
}

pub inline fn _mm_blend_epi32(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    const mask: @Vector(4, bool) = @bitCast(@as(u4, imm8));
    return @bitCast(@select(i32, mask, bitCast_i32x4(b), bitCast_i32x4(a)));
}

pub inline fn _mm256_blend_epi32(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    const mask: @Vector(8, bool) = @bitCast(@as(u8, imm8));
    return @bitCast(@select(i32, mask, bitCast_i32x8(b), bitCast_i32x8(a)));
}

pub inline fn _mm256_blendv_epi8(a: __m256i, b: __m256i, mask: __m256i) __m256i {
    const cmp = @as(i8x32, @splat(0)) > bitCast_i8x32(mask);
    return @bitCast(@select(i8, cmp, bitCast_i8x32(b), bitCast_i8x32(a)));
}

pub inline fn _mm_broadcastb_epi8(a: __m128i) __m128i {
    return @bitCast(@as(i8x16, @splat(bitCast_i8x16(a)[0])));
}

pub inline fn _mm256_broadcastb_epi8(a: __m128i) __m256i {
    return @bitCast(@as(i8x32, @splat(bitCast_i8x16(a)[0])));
}

pub inline fn _mm_broadcastd_epi32(a: __m128i) __m128i {
    return @bitCast(@as(i32x4, @splat(bitCast_i32x4(a)[0])));
}

pub inline fn _mm256_broadcastd_epi32(a: __m128i) __m256i {
    return @bitCast(@as(i32x8, @splat(bitCast_i32x4(a)[0])));
}

pub inline fn _mm_broadcastq_epi64(a: __m128i) __m128i {
    return @bitCast(@as(i64x2, @splat(bitCast_i64x2(a)[0])));
}

pub inline fn _mm256_broadcastq_epi64(a: __m128i) __m256i {
    return @bitCast(@as(i64x4, @splat(bitCast_i64x2(a)[0])));
}

pub inline fn _mm_broadcastsd_pd(a: __m128d) __m128d {
    return @splat(a[0]);
}

pub inline fn _mm256_broadcastsd_pd(a: __m128d) __m256d {
    return @splat(a[0]);
}

pub inline fn _mm_broadcastsi128_si256(a: __m128i) __m256i {
    return _mm256_broadcastsi128_si256(a);
}

pub inline fn _mm256_broadcastsi128_si256(a: __m128i) __m256i {
    const e = bitCast_i32x4(a);
    return @bitCast(i32x8{ e[0], e[1], e[2], e[3], e[0], e[1], e[2], e[3] });
}

pub inline fn _mm_broadcastss_ps(a: __m128) __m128 {
    return @splat(a[0]);
}

pub inline fn _mm256_broadcastss_ps(a: __m128) __m256 {
    return @splat(a[0]);
}

pub inline fn _mm_broadcastw_epi16(a: __m128i) __m128i {
    return @bitCast(@as(i16x8, @splat(bitCast_i16x8(a)[0])));
}

pub inline fn _mm256_broadcastw_epi16(a: __m128i) __m256i {
    return @bitCast(@as(i16x16, @splat(bitCast_i16x8(a)[0])));
}

pub inline fn _mm256_bslli_epi128(a: __m256i, comptime imm8: comptime_int) __m256i {
    return _mm256_slli_si256(a, imm8);
}

pub inline fn _mm256_bsrli_epi128(a: __m256i, comptime imm8: comptime_int) __m256i {
    return _mm256_srli_si256(a, imm8);
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm256_cmpeq_epi16(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_u16x16(a) == bitCast_u16x16(b));
    return @bitCast(boolMask_u16x16(pred));
}

test "_mm256_cmpeq_epi16" {
    const a = _xx256_set_epu16(0x8002, 0x8000, 0x8000, 5, 0xFFFF, 0xFFFE, 0, 127, 0, 6, 5, 4, 3, 2, 0xFFFF, 0);
    const b = _xx256_set_epu16(0x8001, 0x8001, 0x7FFF, 5, 0xFFFF, 3, 0, 126, 0x8000, 5, 0xCCCC, 0x7F00, 6, 5, 1, 0);
    const ref = _mm256_set_epi16(0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1);
    try std.testing.expectEqual(ref, _mm256_cmpeq_epi16(a, b));
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm256_cmpeq_epi32(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_u32x8(a) == bitCast_u32x8(b));
    return @bitCast(boolMask_u32x8(pred));
}

test "_mm256_cmpeq_epi32" {
    const a = _xx256_set_epu32(0x80000001, 0x80000006, 0x80000000, 2, 3, 4, 2, 1);
    const b = _xx256_set_epu32(0x80000001, 0x80000005, 0x00000000, 1, 2, 4, 0, 0xFFFFFFFF);
    const ref = _mm256_set_epi32(-1, 0, 0, 0, 0, -1, 0, 0);
    try std.testing.expectEqual(ref, _mm256_cmpeq_epi32(a, b));
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm256_cmpeq_epi64(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_u64x4(a) == bitCast_u64x4(b));
    return @bitCast(boolMask_u64x4(pred));
}

test "_mm256_cmpeq_epi64" {
    const a = _xx256_set_epu64x(0x8000000000000001, 0x8000000000000006, 0x8000000000000000, 2);
    const b = _xx256_set_epu64x(0x8000000000000001, 0x8000000000000005, 0, 1);
    const ref = _mm256_set_epi64x(-1, 0, 0, 0);
    try std.testing.expectEqual(ref, _mm256_cmpeq_epi64(a, b));
}

/// dst[n] = if (a[n] == b[n]) -1 else 0;
pub inline fn _mm256_cmpeq_epi8(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_u8x32(a) == bitCast_u8x32(b));
    return @bitCast(boolMask_u8x32(pred));
}

test "_mm256_cmpeq_epi8" {
    const a = _mm256_set_epi8(-128, 2, -128, 0, -128, 5, -1, -2, 0, 0, 0, 0, 0, 0, 0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const b = _mm256_set_epi8(-128, 1, 0, 1, 127, 4, -2, 3, 0, 0, 0, 0, 0, 0, 0, 2, 15, 14, 13, 99, 11, 10, 99, 8, 8, 8, 8, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi8(-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1);
    try std.testing.expectEqual(ref, _mm256_cmpeq_epi8(a, b));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm256_cmpgt_epi16(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_i16x16(a) > bitCast_i16x16(b));
    return @bitCast(boolMask_u16x16(pred));
}

test "_mm256_cmpgt_epi16" {
    const a = _xx256_set_epu16(0x8002, 0x8000, 0x8000, 5, 0xFFFF, 0xFFFE, 0, 127, 7, 6, 5, 4, 3, 2, 1, 0);
    const b = _xx256_set_epu16(0x8001, 0x8001, 0x7FFF, 4, 0xFFFE, 3, 0, 126, 8, 5, 0xCCCC, 0x7F00, 6, 5, 1, 0xFFFF);
    const ref0 = _mm256_set_epi16(-1, 0, 0, -1, -1, 0, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1);
    const ref1 = _mm256_set_epi16(0, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0);
    try std.testing.expectEqual(ref0, _mm256_cmpgt_epi16(a, b));
    try std.testing.expectEqual(ref1, _mm256_cmpgt_epi16(b, a));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm256_cmpgt_epi32(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_i32x8(a) > bitCast_i32x8(b));
    return @bitCast(boolMask_u32x8(pred));
}

test "_mm256_cmpgt_epi32" {
    const a = _xx256_set_epu32(0x80002000, 0x80000000, 0x80000000, 5, 0xFFFFFFFF, 0xFFFFFFFE, 0, 127);
    const b = _xx256_set_epu32(0x80001000, 0x80000001, 0x7FFFFFFF, 4, 0xFFFFFFFE, 3, 0, 126);
    const ref0 = _mm256_set_epi32(-1, 0, 0, -1, -1, 0, 0, -1);
    const ref1 = _mm256_set_epi32(0, -1, -1, 0, 0, -1, 0, 0);
    try std.testing.expectEqual(ref0, _mm256_cmpgt_epi32(a, b));
    try std.testing.expectEqual(ref1, _mm256_cmpgt_epi32(b, a));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm256_cmpgt_epi64(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_i64x4(a) > bitCast_i64x4(b));
    return @bitCast(boolMask_u64x4(pred));
}

test "_mm256_cmpgt_epi64" {
    const a = _xx256_set_epu64x(0x8000000000000001, 0x8000000000000006, 0x8000000000000001, 2);
    const b = _xx256_set_epu64x(0x8000000000000001, 2, 0, 1);
    const ref0 = _mm256_set_epi64x(0, 0, 0, -1);
    const ref1 = _mm256_set_epi64x(0, -1, -1, 0);
    try std.testing.expectEqual(ref0, _mm256_cmpgt_epi64(a, b));
    try std.testing.expectEqual(ref1, _mm256_cmpgt_epi64(b, a));
}

/// dst[n] = if (a[n] > b[n]) -1 else 0;
pub inline fn _mm256_cmpgt_epi8(a: __m256i, b: __m256i) __m256i {
    const pred = @intFromBool(bitCast_i8x32(a) > bitCast_i8x32(b));
    return @bitCast(boolMask_u8x32(pred));
}

test "_mm256_cmpgt_epi8" {
    const a = _mm256_set_epi8(5, 4, 8, 12, 45, 54, 65, 8, 7, 6, 5, 4, 3, 2, 1, -128, -128, -127, 127, 127, -1, 10, 9, 8, 7, 6, 5, 4, 3, 9, 2, 1);
    const b = _mm256_set_epi8(65, 46, 54, 1, 3, 45, 46, 75, -8, -35, -80, 19, 0, 0, -128, 127, -128, 12, -1, -1, 7, 10, -2, 7, 6, 5, 4, 3, 10, 2, 1, -1);
    const ref0 = _mm256_set_epi8(0, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1);
    const ref1 = _mm256_set_epi8(-1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0);
    try std.testing.expectEqual(ref0, _mm256_cmpgt_epi8(a, b));
    try std.testing.expectEqual(ref1, _mm256_cmpgt_epi8(b, a));
}

/// Sign-Extend the low 8 words
pub inline fn _mm256_cvtepi16_epi32(a: __m128i) __m256i {
    const x = bitCast_i16x8(a);
    return @bitCast(i32x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm256_cvtepi16_epi32" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi32(7, 6, 5, 4, 3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi16_epi32(a));
}

/// Sign-Extend the low 4 words
pub inline fn _mm256_cvtepi16_epi64(a: __m128i) __m256i {
    const x = bitCast_i16x8(a);
    return @bitCast(i64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepi16_epi64" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi16_epi64(a));
}

/// Sign-Extend the low 4 dwords
pub inline fn _mm256_cvtepi32_epi64(a: __m128i) __m256i {
    const x = bitCast_i32x4(a);
    return @bitCast(i64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepi32_epi64" {
    const a = _mm_set_epi32(3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi32_epi64(a));
}

/// Sign-Extend the low 16 bytes
pub inline fn _mm256_cvtepi8_epi16(a: __m128i) __m256i {
    const x = bitCast_i8x16(a);
    return @bitCast(i16x16{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15] });
}

test "_mm256_cvtepi8_epi16" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi8_epi16(a));
}

/// Sign-Extend the low 8 bytes
pub inline fn _mm256_cvtepi8_epi32(a: __m128i) __m256i {
    const x = bitCast_i8x16(a);
    return @bitCast(i32x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm256_cvtepi8_epi32" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi32(7, 6, 5, 4, 3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi8_epi32(a));
}

/// Sign-Extend the low 4 bytes
// note: error in intel intrinsic guide v3.6.7
pub inline fn _mm256_cvtepi8_epi64(a: __m128i) __m256i {
    const x = bitCast_i8x16(a);
    return @bitCast(i64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepi8_epi64" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, -1, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepi8_epi64(a));
}

/// Zero-Extend the low 8 words
pub inline fn _mm256_cvtepu16_epi32(a: __m128i) __m256i {
    const x = bitCast_u16x8(a);
    return @bitCast(u32x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm256_cvtepu16_epi32" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 65535, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu16_epi32(a));
}

/// Zero-Extend the low 4 words
pub inline fn _mm256_cvtepu16_epi64(a: __m128i) __m256i {
    const x = bitCast_u16x8(a);
    return @bitCast(u64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepu16_epi64" {
    const a = _mm_set_epi16(7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, 65535, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu16_epi64(a));
}

/// Zero-Extend the low 4 dwords
pub inline fn _mm256_cvtepu32_epi64(a: __m128i) __m256i {
    const x = bitCast_u32x4(a);
    return @bitCast(u64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepu32_epi64" {
    const a = _mm_set_epi32(3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, 4294967295, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu32_epi64(a));
}

/// Zero-Extend the low 16 bytes
pub inline fn _mm256_cvtepu8_epi16(a: __m128i) __m256i {
    const x = bitCast_u8x16(a);
    return @bitCast(u16x16{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15] });
}

test "_mm256_cvtepu8_epi16" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 255, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu8_epi16(a));
}

/// Zero-Extend the low 8 bytes
pub inline fn _mm256_cvtepu8_epi32(a: __m128i) __m256i {
    const x = bitCast_u8x16(a);
    return @bitCast(u32x8{ x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] });
}

test "_mm256_cvtepu8_epi32" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 255, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu8_epi32(a));
}

/// Zero-Extend the low 4 bytes
pub inline fn _mm256_cvtepu8_epi64(a: __m128i) __m256i {
    const x = bitCast_u8x16(a);
    return @bitCast(u64x4{ x[0], x[1], x[2], x[3] });
}

test "_mm256_cvtepu8_epi64" {
    const a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, -1, 0);
    const ref = _mm256_set_epi64x(3, 2, 255, 0);
    try std.testing.expectEqual(ref, _mm256_cvtepu8_epi64(a));
}

/// extract u16 then zero-extend to i32
pub inline fn _mm256_extract_epi16(a: __m256i, comptime imm8: comptime_int) i32 {
    return bitCast_u16x16(a)[imm8];
}

test "_mm256_extract_epi16" {
    const a = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, -2, 1, 0);
    try std.testing.expectEqual(@as(i32, 65534), _mm256_extract_epi16(a, 2));
    try std.testing.expectEqual(@as(i32, 15), _mm256_extract_epi16(a, 15));
}

/// Extract u8 then zero-extend to i32.
pub inline fn _mm256_extract_epi8(a: __m256i, comptime imm8: comptime_int) i32 {
    return bitCast_u8x32(a)[imm8];
}

test "_mm256_extract_epi8" {
    const a = _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, -2, 1, 0);
    try std.testing.expectEqual(@as(i32, 254), _mm256_extract_epi8(a, 2));
    try std.testing.expectEqual(@as(i32, 31), _mm256_extract_epi8(a, 31));
}

/// Extract (high or low) __m128i from __m256i
pub inline fn _mm256_extracti128_si256(a: __m256i, comptime imm8: comptime_int) __m128i {
    const x = bitCast_i32x8(a);
    return @bitCast(i32x4{ x[imm8 * 4 + 0], x[imm8 * 4 + 1], x[imm8 * 4 + 2], x[imm8 * 4 + 3] });
}

test "_mm256_extracti128_si256" {
    const a = _mm256_set_epi32(7, 6, -5, 4, -2147483648, 2, 1, 0);
    const ref0 = _mm_set_epi32(-2147483648, 2, 1, 0);
    const ref1 = _mm_set_epi32(7, 6, -5, 4);
    try std.testing.expectEqual(ref0, _mm256_extracti128_si256(a, 0));
    try std.testing.expectEqual(ref1, _mm256_extracti128_si256(a, 1));
}

pub inline fn _mm256_hadd_epi16(a: __m256i, b: __m256i) __m256i {
    const shuf_even: [16]i32 = .{ 0, 2, 4, 6, -1, -3, -5, -7, 8, 10, 12, 14, -9, -11, -13, -15 };
    const shuf_odd: [16]i32 = .{ 1, 3, 5, 7, -2, -4, -6, -8, 9, 11, 13, 15, -10, -12, -14, -16 };
    const even = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_odd);
    return @bitCast(even +% odd);
}

pub inline fn _mm256_hadd_epi32(a: __m256i, b: __m256i) __m256 {
    const shuf_even: [8]i32 = .{ 0, 2, -1, -3, 4, 6, -5, -7 };
    const shuf_odd: [8]i32 = .{ 1, 3, -2, -4, 5, 7, -6, -8 };
    const even = @shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf_odd);
    return @bitCast(even +% odd);
}

pub inline fn _mm256_hadds_epi16(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vphaddsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const shuf_even: [16]i32 = .{ 0, 2, 4, 6, -1, -3, -5, -7, 8, 10, 12, 14, -9, -11, -13, -15 };
        const shuf_odd: [16]i32 = .{ 1, 3, 5, 7, -2, -4, -6, -8, 9, 11, 13, 15, -10, -12, -14, -16 };
        const even = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_even);
        const odd = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_odd);
        return @bitCast(even +| odd);
    }
}

pub inline fn _mm256_hsub_epi16(a: __m256i, b: __m256i) __m256i {
    const shuf_even: [16]i32 = .{ 0, 2, 4, 6, -1, -3, -5, -7, 8, 10, 12, 14, -9, -11, -13, -15 };
    const shuf_odd: [16]i32 = .{ 1, 3, 5, 7, -2, -4, -6, -8, 9, 11, 13, 15, -10, -12, -14, -16 };
    const even = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_odd);
    return @bitCast(even -% odd);
}

pub inline fn _mm256_hsub_epi32(a: __m256i, b: __m256i) __m256 {
    const shuf_even: [8]i32 = .{ 0, 2, -1, -3, 4, 6, -5, -7 };
    const shuf_odd: [8]i32 = .{ 1, 3, -2, -4, 5, 7, -6, -8 };
    const even = @shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf_odd);
    return @bitCast(even -% odd);
}

pub inline fn _mm256_hsubs_epi16(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vphsubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const shuf_even: [16]i32 = .{ 0, 2, 4, 6, -1, -3, -5, -7, 8, 10, 12, 14, -9, -11, -13, -15 };
        const shuf_odd: [16]i32 = .{ 1, 3, 5, 7, -2, -4, -6, -8, 9, 11, 13, 15, -10, -12, -14, -16 };
        const even = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_even);
        const odd = @shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf_odd);
        return @bitCast(even -| odd);
    }
}

pub inline fn _mm_i32gather_epi32(base_addr: [*]align(1) const i32, vindex: __m128i, comptime scale: comptime_int) __m128i {
    return _mm_mask_i32gather_epi32(@splat(0), base_addr, vindex, _mm_set1_epi32(-1), scale);
}

test "_mm_i32gather_epi32" {
    const vindex = _mm_set_epi32(1, 2, 0, -1);
    const arr: [4]i32 = .{ 286331153, 572662306, 858993459, 1145324612 };
    const ref = _mm_set_epi32(858993459, 1145324612, 572662306, 286331153);
    try std.testing.expectEqual(ref, _mm_i32gather_epi32(arr[1..], vindex, 4));
}

pub inline fn _mm_mask_i32gather_epi32(src: __m128i, base_addr: [*]align(1) const i32, vindex: __m128i, mask: __m128i, comptime scale: comptime_int) __m128i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherdd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i32x4(src);
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const i32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

test "_mm_mask_i32gather_epi32" {
    const vindex = _mm_set_epi32(1, 2, 0, -1);
    const arr: [4]i32 = .{ 286331153, 572662306, 858993459, 1145324612 };
    const src = _mm_set_epi32(4, 3, 2, 1);
    const mask = _mm_set_epi32(0, -1, -1, -1);
    const ref = _mm_set_epi32(4, 1145324612, 572662306, 286331153);
    try std.testing.expectEqual(ref, _mm_mask_i32gather_epi32(src, arr[1..], vindex, mask, 4));
}

pub inline fn _mm256_i32gather_epi32(base_addr: [*]align(1) const i32, vindex: __m256i, comptime scale: comptime_int) __m256i {
    return _mm256_mask_i32gather_epi32(@splat(0), base_addr, vindex, _mm256_set1_epi32(-1), scale);
}

test "_mm256_i32gather_epi32" {
    const vindex = _mm256_set_epi32(1, 2, 0, -1, 1, 2, 0, -1);
    const arr: [4]i32 = .{ 286331153, 572662306, 858993459, 1145324612 };
    const ref = _mm256_set_epi32(858993459, 1145324612, 572662306, 286331153, 858993459, 1145324612, 572662306, 286331153);
    try std.testing.expectEqual(ref, _mm256_i32gather_epi32(arr[1..], vindex, 4));
}

pub inline fn _mm256_mask_i32gather_epi32(src: __m256i, base_addr: [*]align(1) const i32, vindex: __m256i, mask: __m256i, comptime scale: comptime_int) __m256i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherdd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i32x8(src);
        const pred = @as(i32x8, @splat(0)) > bitCast_i32x8(mask);
        inline for (0..8) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x8(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const i32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

test "_mm256_mask_i32gather_epi32" {
    const vindex = _mm256_set_epi32(1, 2, 0, -1, 3, 3, 3, 0);
    const arr: [5]i32 = .{ 286331153, 572662306, 858993459, 1145324612, 555555555 };
    const src = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    const mask = _mm256_set_epi32(0, -1, -1, -1, -2147483648, 1, 0, -2);
    const ref = _mm256_set_epi32(8, 1145324612, 572662306, 286331153, 555555555, 3, 2, 572662306);
    try std.testing.expectEqual(ref, _mm256_mask_i32gather_epi32(src, arr[1..], vindex, mask, 4));
}

pub inline fn _mm_i32gather_epi64(base_addr: [*]align(1) const i64, vindex: __m128i, comptime scale: comptime_int) __m128i {
    return _mm_mask_i32gather_epi64(@splat(0), base_addr, vindex, _mm_set1_epi32(-1), scale);
}

test "_mm_i32gather_epi64" {
    const vindex = _mm_set_epi32(3, 2, -1, 0);
    const arr: [5]i32 = .{ 0, 1, 2, 3, 4 };
    const ref = _mm_set_epi32(1, 0, 3, 2);
    try std.testing.expectEqual(ref, _mm_i32gather_epi64(@ptrCast(arr[2..]), vindex, 8));
}

pub inline fn _mm_mask_i32gather_epi64(src: __m128i, base_addr: [*]align(1) const i64, vindex: __m128i, mask: __m128i, comptime scale: comptime_int) __m128i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherdq %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i64x2(src);
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const i64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

test "_mm_mask_i32gather_epi64" {
    const vindex = _mm_set_epi32(3, 2, -1, 0);
    const mask = _mm_set_epi64x(-1, 0);
    const src = _mm_set1_epi32(9);
    const arr: [5]i32 = .{ 0, 1, 2, 3, 4 };
    const ref = _mm_set_epi32(1, 0, 9, 9);
    try std.testing.expectEqual(ref, _mm_mask_i32gather_epi64(src, @ptrCast(arr[2..]), vindex, mask, 8));
}

pub inline fn _mm256_i32gather_epi64(base_addr: [*]align(1) const i64, vindex: __m128i, comptime scale: comptime_int) __m256i {
    return _mm256_mask_i32gather_epi64(@splat(0), base_addr, vindex, _mm256_set1_epi32(-1), scale);
}

pub inline fn _mm256_mask_i32gather_epi64(src: __m256i, base_addr: [*]align(1) const i64, vindex: __m128i, mask: __m256i, comptime scale: comptime_int) __m256i {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherdq %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i64x4(src);
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const i64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

pub inline fn _mm_i32gather_pd(base_addr: [*]align(1) const f64, vindex: __m128i, comptime scale: comptime_int) __m128d {
    const mask = _mm_castsi128_pd(_mm_set1_epi32(-1));
    return _mm_mask_i32gather_pd(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm_mask_i32gather_pd(src: __m128d, base_addr: [*]align(1) const f64, vindex: __m128i, mask: __m128d, comptime scale: comptime_int) __m128d {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherdpd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const f64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

pub inline fn _mm256_i32gather_pd(base_addr: [*]align(1) const f64, vindex: __m128i, comptime scale: comptime_int) __m256d {
    const mask = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
    return _mm256_mask_i32gather_pd(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm256_mask_i32gather_pd(src: __m256d, base_addr: [*]align(1) const f64, vindex: __m128i, mask: __m256d, comptime scale: comptime_int) __m256d {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherdpd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const f64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

pub inline fn _mm_i32gather_ps(base_addr: [*]align(1) const f32, vindex: __m128i, comptime scale: comptime_int) __m128 {
    const mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return _mm_mask_i32gather_ps(@splat(0), base_addr, vindex, mask, scale);
}

test "_mm_i32gather_ps" {
    const vindex = _mm_set_epi32(1, 2, 0, -1);
    const arr: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    const ref = _mm_set_ps(3.0, 4.0, 2.0, 1.0);
    try std.testing.expectEqual(ref, _mm_i32gather_ps(arr[1..], vindex, 4));
}

pub inline fn _mm_mask_i32gather_ps(src: __m128, base_addr: [*]align(1) const f32, vindex: __m128i, mask: __m128, comptime scale: comptime_int) __m128 {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherdps %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x4(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const f32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

pub inline fn _mm256_i32gather_ps(base_addr: [*]align(1) const f32, vindex: __m256i, comptime scale: comptime_int) __m256 {
    const mask = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    return _mm256_mask_i32gather_ps(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm256_mask_i32gather_ps(src: __m256, base_addr: [*]align(1) const f32, vindex: __m256i, mask: __m256, comptime scale: comptime_int) __m256 {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherdps %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i32x8, @splat(0)) > bitCast_i32x8(mask);
        inline for (0..8) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const idx = @as(i64, bitCast_i32x8(vindex)[i]);
                const offset: isize = @truncate(@as(i128, idx *% scale));
                const ptr: *align(1) const f32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

pub inline fn _mm_i64gather_epi32(base_addr: [*]align(1) const i32, vindex: __m128i, comptime scale: comptime_int) __m128i {
    return _mm_mask_i64gather_epi32(@splat(0), base_addr, vindex, _mm_set1_epi32(-1), scale);
}

pub inline fn _mm_mask_i64gather_epi32(src: __m128i, base_addr: [*]align(1) const i32, vindex: __m128i, mask: __m128i, comptime scale: comptime_int) __m128i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherqd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r: i32x4 = .{ bitCast_i32x4(src)[0], bitCast_i32x4(src)[1], 0, 0 };
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x2(vindex)[i] *% scale));
                const ptr: *align(1) const i32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

test "_mm_mask_i64gather_epi32" {
    const vindex = _mm_set_epi64x(-1, 1);
    const mask = _mm_set_epi32(-1, -1, -1, 0);
    const src = _mm_set1_epi32(9);
    const arr: [5]i32 = .{ 0, 1, 2, 3, 4 };
    const ref = _mm_set_epi32(0, 0, 1, 9);
    try std.testing.expectEqual(ref, _mm_mask_i64gather_epi32(src, arr[2..], vindex, mask, 4));
}

pub inline fn _mm256_i64gather_epi32(base_addr: [*]align(1) const i32, vindex: __m256i, comptime scale: comptime_int) __m128i {
    return _mm256_mask_i64gather_epi32(@splat(0), base_addr, vindex, _mm_set1_epi32(-1), scale);
}

pub inline fn _mm256_mask_i64gather_epi32(src: __m128i, base_addr: [*]align(1) const i32, vindex: __m256i, mask: __m128i, comptime scale: comptime_int) __m128i {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // Asm operand modifiers don't work (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherqd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i32x4(src);
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x4(vindex)[i] *% scale));
                const ptr: *align(1) const i32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

pub inline fn _mm_i64gather_epi64(base_addr: [*]align(1) const i64, vindex: __m128i, comptime scale: comptime_int) __m128i {
    return _mm_mask_i64gather_epi64(@splat(0), base_addr, vindex, _mm_set1_epi32(-1), scale);
}

pub inline fn _mm_mask_i64gather_epi64(src: __m128i, base_addr: [*]align(1) const i64, vindex: __m128i, mask: __m128i, comptime scale: comptime_int) __m128i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherqq %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i64x2(src);
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x2(vindex)[i] *% scale));
                const ptr: *align(1) const i64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

pub inline fn _mm256_i64gather_epi64(base_addr: [*]align(1) const i64, vindex: __m256i, comptime scale: comptime_int) __m256i {
    return _mm_mask_i64gather_epi64(@splat(0), base_addr, vindex, _mm256_set1_epi32(-1), scale);
}

pub inline fn _mm256_mask_i64gather_epi64(src: __m256i, base_addr: [*]align(1) const i64, vindex: __m256i, mask: __m256i, comptime scale: comptime_int) __m256i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vpgatherqq %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = bitCast_i64x4(src);
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x4(vindex)[i] *% scale));
                const ptr: *align(1) const i64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return @bitCast(r);
    }
}

test "_mm256_mask_i64gather_epi64" {
    const vindex = _mm256_set_epi64x(2, 3, -1, 1);
    const mask = _mm256_set_epi64x(-1, -1, -1, 0);
    const src = _mm256_set1_epi64x(9);
    const arr: [5]i64 = .{ 0, 1, 2, 3, 4 };
    const ref = _mm256_set_epi64x(3, 4, 0, 9);
    try std.testing.expectEqual(ref, _mm256_mask_i64gather_epi64(src, arr[1..], vindex, mask, 8));
}

pub inline fn _mm_i64gather_pd(base_addr: [*]align(1) const f64, vindex: __m128i, comptime scale: comptime_int) __m128d {
    const mask = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
    return _mm_mask_i64gather_pd(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm_mask_i64gather_pd(src: __m128d, base_addr: [*]align(1) const f64, vindex: __m128i, mask: __m128d, comptime scale: comptime_int) __m128d {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherqpd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x2(vindex)[i] *% scale));
                const ptr: *align(1) const f64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

pub inline fn _mm256_i64gather_pd(base_addr: [*]align(1) const f64, vindex: __m256i, comptime scale: comptime_int) __m256d {
    return _mm256_mask_i64gather_pd(@splat(0), base_addr, vindex, _mm256_set1_epi32(-1), scale);
}

pub inline fn _mm256_mask_i64gather_pd(src: __m256d, base_addr: [*]align(1) const f64, vindex: __m256i, mask: __m256d, comptime scale: comptime_int) __m256d {
    if (has_avx2) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherqpd %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x4(vindex)[i] *% scale));
                const ptr: *align(1) const f64 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

test "_mm256_mask_i64gather_pd" {
    if (bug_stage2_x86_64) return error.SkipZigTest; // genBinOp for cmp_gt

    const vindex = _mm256_set_epi64x(3, 3, -1, 1);
    const mask = _mm256_set_pd(-0.0, 0, -0.0, -0.0);
    const src = _mm256_set1_pd(9.0);
    const arr: [5]f64 = .{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const ref = _mm256_set_pd(4.0, 9.0, 0.0, 2.0);
    try std.testing.expectEqual(ref, _mm256_mask_i64gather_pd(src, arr[1..], vindex, mask, 8));
}

pub inline fn _mm_i64gather_ps(base_addr: [*]align(1) const f32, vindex: __m128i, comptime scale: comptime_int) __m128 {
    const mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return _mm_mask_i64gather_ps(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm_mask_i64gather_ps(src: __m128, base_addr: [*]align(1) const f32, vindex: __m128i, mask: __m128, comptime scale: comptime_int) __m128 {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherqps %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r: __m128 = .{ src[0], src[1], 0, 0 };
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..2) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x2(vindex)[i] *% scale));
                const ptr: *align(1) const f32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

test "_mm_mask_i64gather_ps" {
    const vindex = _mm_set_epi64x(-1, 1);
    const mask = _mm_set_ps(-0.0, -0.0, -0.0, 0);
    const src = _mm_set1_ps(9.0);
    const arr: [5]f32 = .{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const ref = _mm_set_ps(0.0, 0.0, 1.0, 9.0);
    try std.testing.expectEqual(ref, _mm_mask_i64gather_ps(src, arr[2..], vindex, mask, 4));
}

pub inline fn _mm256_i64gather_ps(base_addr: [*]align(1) const f32, vindex: __m256i, comptime scale: comptime_int) __m128 {
    const mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return _mm256_mask_i64gather_ps(@splat(0), base_addr, vindex, mask, scale);
}

pub inline fn _mm256_mask_i64gather_ps(src: __m128, base_addr: [*]align(1) const f32, vindex: __m256i, mask: __m128, comptime scale: comptime_int) __m128 {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        var s = src;
        var m = mask;

        // `asm` doesn't currently support operand modifiers (e.g. `%c[scale]`).
        asm volatile (std.fmt.comptimePrint("vgatherqps %[mask], (%[base_addr],%[vindex],{d}), %[src]", .{@as(u4, scale)})
            : [src] "+x" (s),
              [mask] "+x" (m),
            : [base_addr] "r" (base_addr),
              [vindex] "x" (vindex),
            : "memory"
        );
        return s;
    } else {
        switch (scale) {
            1, 2, 4, 8 => {},
            else => @compileError("Scale must be 1, 2, 4, or 8"),
        }

        var r = src;
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) {
                // needs `@intFromPtr` for pointer arithmetic with negative offsets?
                const addr: isize = @bitCast(@intFromPtr(base_addr));
                const offset: isize = @truncate(@as(i128, bitCast_i64x4(vindex)[i] *% scale));
                const ptr: *align(1) const f32 = @ptrFromInt(@as(usize, @bitCast(addr +% offset)));
                r[i] = ptr.*; // safety-checked for null, should we turn that off?
            }
        }
        return r;
    }
}

test "_mm256_mask_i64gather_ps" {
    const vindex = _mm256_set_epi64x(2, 3, -1, 1);
    const mask = _mm_set_ps(-0.0, -0.0, -0.0, 0);
    const src = _mm_set1_ps(9.0);
    const arr: [5]f32 = .{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const ref = _mm_set_ps(3.0, 4.0, 0.0, 9.0);
    try std.testing.expectEqual(ref, _mm256_mask_i64gather_ps(src, arr[1..], vindex, mask, 4));
}

pub inline fn _mm256_inserti128_si256(a: __m256i, b: __m128i, comptime imm8: comptime_int) __m256i {
    if (@as(u1, imm8) == 1) {
        return @bitCast(u64x4{ bitCast_u64x4(a)[0], bitCast_u64x4(a)[1], bitCast_u64x2(b)[0], bitCast_u64x2(b)[1] });
    } else {
        return @bitCast(u64x4{ bitCast_u64x2(b)[0], bitCast_u64x2(b)[1], bitCast_u64x4(a)[2], bitCast_u64x4(a)[3] });
    }
}

pub inline fn _mm256_madd_epi16(a: __m256i, b: __m256i) __m256i {
    const r = intCast_i32x16(bitCast_i16x16(a)) *%
        intCast_i32x16(bitCast_i16x16(b));

    const even = @shuffle(i32, r, undefined, [8]i32{ 0, 2, 4, 6, 8, 10, 12, 14 });
    const odd = @shuffle(i32, r, undefined, [8]i32{ 1, 3, 5, 7, 9, 11, 13, 15 });
    return @bitCast(even +% odd);
}

pub inline fn _mm256_maddubs_epi16(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpmaddubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const r_lo = _mm_maddubs_epi16(_mm256_extracti128_si256(a, 0), _mm256_extracti128_si256(b, 0));
        const r_hi = _mm_maddubs_epi16(_mm256_extracti128_si256(a, 1), _mm256_extracti128_si256(b, 1));
        return _mm256_set_m128i(r_hi, r_lo);
    }
}

pub inline fn _mm_maskload_epi32(mem_addr: [*]align(1) const i32, mask: __m128i) __m128i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        // mem_addr[0..4] probably covers invalid locations so
        // can't use the "m" constraint because it requires a dereference of mem_addr.
        return asm volatile ("vpmaskmovd (%[b]), %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (mask),
              [b] "r" (mem_addr),
            : "memory"
        );
    } else {
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        var r: i32x4 = @splat(0);
        inline for (0..4) |i| {
            if (pred[i]) r[i] = mem_addr[i];
        }
        return @bitCast(r);
    }
}

test "_mm_maskload_epi32" {
    const mask = _mm_set_epi32(0, -1, 0, -1);
    const arr: [4]i32 = .{ 4, 1, 2, 3 };
    const ref = _mm_set_epi32(0, 3, 0, 1);
    try std.testing.expectEqual(ref, _mm_maskload_epi32(arr[1..4].ptr, mask));
}

pub inline fn _mm256_maskload_epi32(mem_addr: [*]align(1) const i32, mask: __m256i) __m256i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        return asm volatile ("vpmaskmovd (%[b]), %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (mask),
              [b] "r" (mem_addr),
            : "memory"
        );
    } else {
        const pred = @as(i32x8, @splat(0)) > bitCast_i32x8(mask);
        var r: i32x8 = @splat(0);
        inline for (0..8) |i| {
            if (pred[i]) r[i] = mem_addr[i];
        }
        return @bitCast(r);
    }
}

pub inline fn _mm_maskload_epi64(mem_addr: [*]align(1) const i64, mask: __m128i) __m128i {
    if (has_avx2) {
        return asm volatile ("vpmaskmovq (%[b]), %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (mask),
              [b] "r" (mem_addr),
            : "memory"
        );
    } else {
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        var r: i64x2 = @splat(0);
        inline for (0..2) |i| {
            if (pred[i]) r[i] = mem_addr[i];
        }
        return @bitCast(r);
    }
}

pub inline fn _mm256_maskload_epi64(mem_addr: [*]align(1) const i64, mask: __m256i) __m256i {
    if (has_avx2) {
        return asm volatile ("vpmaskmovq (%[b]), %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (mask),
              [b] "r" (mem_addr),
            : "memory"
        );
    } else {
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        var r: i64x4 = @splat(0);
        inline for (0..4) |i| {
            if (pred[i]) r[i] = mem_addr[i];
        }
        return @bitCast(r);
    }
}

pub inline fn _mm_maskstore_epi32(mem_addr: [*]align(1) i32, mask: __m128i, a: __m128i) void {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        asm volatile ("vpmaskmovd %[a], %[mask], (%[mem_addr])"
            :
            : [mem_addr] "r" (mem_addr),
              [mask] "x" (mask),
              [a] "x" (a),
            : "memory"
        );
    } else {
        const pred = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) mem_addr[i] = bitCast_i32x4(a)[i];
        }
    }
}

test "_mm_maskstore_epi32" {
    const a = _mm_set_epi32(5, 6, 7, 8);
    const mask = _mm_set_epi32(0, -1, 0, -1);
    var arr = [_]i32{ 4, 1, 2, 3 };
    _mm_maskstore_epi32(arr[1..4].ptr, mask, a);

    try std.testing.expectEqual(@as(i32, 4), arr[0]);
    try std.testing.expectEqual(@as(i32, 8), arr[1]);
    try std.testing.expectEqual(@as(i32, 2), arr[2]);
    try std.testing.expectEqual(@as(i32, 6), arr[3]);
}

pub inline fn _mm256_maskstore_epi32(mem_addr: [*]align(1) i32, mask: __m256i, a: __m256i) void {
    if (has_avx2) {
        asm volatile ("vpmaskmovd %[a], %[mask], (%[mem_addr])"
            :
            : [mem_addr] "r" (mem_addr),
              [mask] "x" (mask),
              [a] "x" (a),
            : "memory"
        );
    } else {
        const pred = @as(i32x8, @splat(0)) > bitCast_i32x8(mask);
        inline for (0..8) |i| {
            if (pred[i]) mem_addr[i] = bitCast_i32x8(a)[i];
        }
    }
}

pub inline fn _mm_maskstore_epi64(mem_addr: [*]align(1) i64, mask: __m128i, a: __m128i) void {
    if (has_avx2) {
        asm volatile ("vpmaskmovq %[a], %[mask], (%[mem_addr])"
            :
            : [mem_addr] "r" (mem_addr),
              [mask] "x" (mask),
              [a] "x" (a),
            : "memory"
        );
    } else {
        const pred = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
        inline for (0..2) |i| {
            if (pred[i]) mem_addr[i] = bitCast_i64x2(a)[i];
        }
    }
}

pub inline fn _mm256_maskstore_epi64(mem_addr: [*]align(1) i64, mask: __m256i, a: __m256i) void {
    if (has_avx2) {
        asm volatile ("vpmaskmovq %[a], %[mask], (%[mem_addr])"
            :
            : [mem_addr] "r" (mem_addr),
              [mask] "x" (mask),
              [a] "x" (a),
            : "memory"
        );
    } else {
        const pred = @as(i64x4, @splat(0)) > bitCast_i64x4(mask);
        inline for (0..4) |i| {
            if (pred[i]) mem_addr[i] = bitCast_i64x4(a)[i];
        }
    }
}

pub inline fn _mm256_max_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_i16x16(bitCast_i16x16(a), bitCast_i16x16(b)));
}
pub inline fn _mm256_max_epi32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_i32x8(bitCast_i32x8(a), bitCast_i32x8(b)));
}
pub inline fn _mm256_max_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_i8x32(bitCast_i8x32(a), bitCast_i8x32(b)));
}
pub inline fn _mm256_max_epu16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_u16x16(bitCast_u16x16(a), bitCast_u16x16(b)));
}
pub inline fn _mm256_max_epu32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_u32x8(bitCast_u32x8(a), bitCast_u32x8(b)));
}
pub inline fn _mm256_max_epu8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(max_u8x32(bitCast_u8x32(a), bitCast_u8x32(b)));
}

pub inline fn _mm256_min_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_i16x16(bitCast_i16x16(a), bitCast_i16x16(b)));
}
pub inline fn _mm256_min_epi32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_i32x8(bitCast_i32x8(a), bitCast_i32x8(b)));
}
pub inline fn _mm256_min_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_i8x32(bitCast_i8x32(a), bitCast_i8x32(b)));
}
pub inline fn _mm256_min_epu16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_u16x16(bitCast_u16x16(a), bitCast_u16x16(b)));
}
pub inline fn _mm256_min_epu32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_u32x8(bitCast_u32x8(a), bitCast_u32x8(b)));
}
pub inline fn _mm256_min_epu8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(min_u8x32(bitCast_u8x32(a), bitCast_u8x32(b)));
}

pub inline fn _mm256_movemask_epi8(a: __m256i) i32 {
    const cmp = @as(i8x32, @splat(0)) > bitCast_i8x32(a);
    return @bitCast(@as(u32, @bitCast(cmp)));
}

pub inline fn _mm256_mpsadbw_epu8(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    if ((has_avx2) and (!bug_stage2_x86_64)) {
        return asm ("vmpsadbw %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "i" (imm8),
        );
    } else {
        const a_lo = _mm256_extracti128_si256(a, 0);
        const a_hi = _mm256_extracti128_si256(a, 1);
        const b_lo = _mm256_extracti128_si256(b, 0);
        const b_hi = _mm256_extracti128_si256(b, 1);
        const r_lo = _mm_mpsadbw_epu8(a_lo, b_lo, imm8 & 7);
        const r_hi = _mm_mpsadbw_epu8(a_hi, b_hi, imm8 >> 3);
        return _mm256_set_m128i(r_hi, r_lo);
    }
}

test "_mm256_mpsadbw_epu8" {
    const a = _xx256_set_epu32(0x07018593, 0x56312665, 0xFFFFFFFF, 0, 0x07018593, 0x56312665, 0xFFFFFFFF, 0);
    const b = _xx256_set_epu32(3, 0xFA57C0DE, 1, 0, 3, 0xFA57C0DE, 1, 0);
    const ref0 = _mm256_set_epi16(443, 649, 866, 1020, 765, 510, 255, 0, 443, 649, 866, 1020, 765, 510, 255, 0);
    const ref1 = _mm256_set_epi16(476, 456, 431, 477, 374, 322, 413, 269, 476, 456, 431, 477, 374, 322, 413, 269);
    try std.testing.expectEqual(ref0, _mm256_mpsadbw_epu8(a, b, 0));
    try std.testing.expectEqual(ref1, _mm256_mpsadbw_epu8(a, b, 54));
}

pub inline fn _mm256_mul_epi32(a: __m256i, b: __m256i) __m256i {
    const x = bitCast_i64x4(a);
    const y = bitCast_i64x4(b);
    const shift: @Vector(4, u6) = @splat(32);
    return @bitCast(((x << shift) >> shift) *% ((y << shift) >> shift));
}

pub inline fn _mm256_mul_epu32(a: __m256i, b: __m256i) __m256i {
    const x = bitCast_u64x4(a);
    const y = bitCast_u64x4(b);
    const shift: @Vector(4, u6) = @splat(32);
    return @bitCast(((x << shift) >> shift) *% ((y << shift) >> shift));
}

pub inline fn _mm256_mulhi_epi16(a: __m256i, b: __m256i) __m256i {
    const r = (intCast_i32x16(bitCast_i16x16(a)) *% intCast_i32x16(bitCast_i16x16(b)));
    return @bitCast(@as(i16x16, @truncate(r >> @splat(16))));
}

pub inline fn _mm256_mulhi_epu16(a: __m256i, b: __m256i) __m256i {
    const r = (intCast_u32x16(bitCast_u16x16(a)) *% intCast_u32x16(bitCast_u16x16(b)));
    return @bitCast(@as(u16x16, @truncate(r >> @splat(16))));
}

pub inline fn _mm256_mulhrs_epi16(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpmulhrsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        var r = intCast_i32x16(bitCast_i16x16(a));
        r *%= intCast_i32x16(bitCast_i16x16(b));
        r +%= @splat(1 << 14);
        return @bitCast(@as(i16x16, @truncate(r >> @splat(15))));
    }
}

pub inline fn _mm256_mullo_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i16x16(a) *% bitCast_i16x16(b));
}

pub inline fn _mm256_mullo_epi32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i32x8(a) *% bitCast_i32x8(b));
}

pub inline fn _mm256_or_si256(a: __m256i, b: __m256i) __m256i {
    return a | b;
}

pub inline fn _mm256_packs_epi16(a: __m256i, b: __m256i) __m256i {
    const x = bitCast_u64x2(@as(i8x16, @truncate(max_i16x16(min_i16x16(bitCast_i16x16(a), @splat(127)), @splat(-128)))));
    const y = bitCast_u64x2(@as(i8x16, @truncate(max_i16x16(min_i16x16(bitCast_i16x16(b), @splat(127)), @splat(-128)))));
    return @bitCast(u64x4{ x[0], y[0], x[1], y[1] });
}

pub inline fn _mm256_packs_epi32(a: __m256i, b: __m256i) __m256i {
    const x: i16x8 = @truncate(max_i32x8(min_i32x8(bitCast_i32x8(a), @splat(32767)), @splat(-32768)));
    const y: i16x8 = @truncate(max_i32x8(min_i32x8(bitCast_i32x8(b), @splat(32767)), @splat(-32768)));
    return @bitCast(i16x16{ x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7] });
}

pub inline fn _mm256_packus_epi16(a: __m256i, b: __m256i) __m256i {
    const x = bitCast_u64x2(@as(i8x16, @truncate(max_i16x16(min_i16x16(bitCast_i16x16(a), @splat(255)), @splat(0)))));
    const y = bitCast_u64x2(@as(i8x16, @truncate(max_i16x16(min_i16x16(bitCast_i16x16(b), @splat(255)), @splat(0)))));
    return @bitCast(u64x4{ x[0], y[0], x[1], y[1] });
}

pub inline fn _mm256_packus_epi32(a: __m256i, b: __m256i) __m256i {
    const x: i16x8 = @truncate(max_i32x8(min_i32x8(bitCast_i32x8(a), @splat(65535)), @splat(0)));
    const y: i16x8 = @truncate(max_i32x8(min_i32x8(bitCast_i32x8(b), @splat(65535)), @splat(0)));
    return @bitCast(i16x16{ x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7] });
}

pub inline fn _mm256_permute2x128_si256(a: __m256i, b: __m256i, comptime imm8: comptime_int) __m256i {
    if ((imm8 & 0x08) == 0x08) { // optimizer hand-holding when zeroing the low 128-bits
        return switch (@as(u8, imm8) >> 4) {
            0, 4 => @bitCast(u64x4{ 0, 0, bitCast_u64x4(a)[0], bitCast_u64x4(a)[1] }),
            1, 5 => @bitCast(u64x4{ 0, 0, bitCast_u64x4(a)[2], bitCast_u64x4(a)[3] }),
            2, 6 => @bitCast(u64x4{ 0, 0, bitCast_u64x4(b)[0], bitCast_u64x4(b)[1] }),
            3, 7 => @bitCast(u64x4{ 0, 0, bitCast_u64x4(b)[2], bitCast_u64x4(b)[3] }),
            else => @bitCast(u64x4{ 0, 0, 0, 0 }),
        };
    }

    const lo: __m128i = switch (imm8 & 0x0F) {
        0, 4 => _mm256_extracti128_si256(a, 0),
        1, 5 => _mm256_extracti128_si256(a, 1),
        2, 6 => _mm256_extracti128_si256(b, 0),
        3, 7 => _mm256_extracti128_si256(b, 1),
        else => @splat(0),
    };

    const hi: __m128i = switch (@as(u8, imm8) >> 4) {
        0, 4 => _mm256_extracti128_si256(a, 0),
        1, 5 => _mm256_extracti128_si256(a, 1),
        2, 6 => _mm256_extracti128_si256(b, 0),
        3, 7 => _mm256_extracti128_si256(b, 1),
        else => @splat(0),
    };

    return _mm256_set_m128i(hi, lo);
}

pub inline fn _mm256_permute4x64_epi64(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shuf = [4]i32{ imm8 & 3, (imm8 >> 2) & 3, (imm8 >> 4) & 3, (imm8 >> 6) & 3 };
    return @bitCast(@shuffle(i64, bitCast_i64x4(a), undefined, shuf));
}

pub inline fn _mm256_permute4x64_pd(a: __m256d, comptime imm8: comptime_int) __m256d {
    const shuf = [4]i32{ imm8 & 3, (imm8 >> 2) & 3, (imm8 >> 4) & 3, (imm8 >> 6) & 3 };
    return @shuffle(f64, a, undefined, shuf);
}

pub inline fn _mm256_permutevar8x32_epi32(a: __m256i, idx: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpermd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (idx),
        );
    } else {
        const i = bitCast_u32x8(idx) & @as(u32x8, @splat(0x3));
        const x = bitCast_u32x8(a);
        return @bitCast(u32x8{ x[i[0]], x[i[1]], x[i[2]], x[i[3]], x[i[4]], x[i[5]], x[i[6]], x[i[7]] });
    }
}

pub inline fn _mm256_permutevar8x32_ps(a: __m256, idx: __m256i) __m256 {
    if (has_avx2) {
        return asm ("vpermps %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256),
            : [a] "x" (a),
              [b] "x" (idx),
        );
    } else {
        const i = bitCast_u32x8(idx) & @as(u32x8, @splat(0x3));
        return .{ a[i[0]], a[i[1]], a[i[2]], a[i[3]], a[i[4]], a[i[5]], a[i[6]], a[i[7]] };
    }
}

pub inline fn _mm256_sad_epu8(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsadbw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const max = max_u8x32(bitCast_u8x32(a), bitCast_u8x32(b));
        const min = min_u8x32(bitCast_u8x32(a), bitCast_u8x32(b));
        const abd = max -% min;

        var r: u64x4 = @splat(0);
        inline for (0..32) |i| {
            r[i >> 3] +%= abd[i];
        }
        return @bitCast(r);
    }
}

pub inline fn _mm256_shuffle_epi32(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shuf = [8]i32{
        imm8 & 3,       (imm8 >> 2) & 3,       (imm8 >> 4) & 3,       (imm8 >> 6) & 3,
        4 + (imm8 & 3), 4 + ((imm8 >> 2) & 3), 4 + ((imm8 >> 4) & 3), 4 + ((imm8 >> 6) & 3),
    };
    return @bitCast(@shuffle(i32, bitCast_i32x8(a), undefined, shuf));
}

pub inline fn _mm256_shuffle_epi8(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpshufb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const r_lo = _mm_shuffle_epi8(_mm256_extracti128_si256(a, 0), _mm256_extracti128_si256(b, 0));
        const r_hi = _mm_shuffle_epi8(_mm256_extracti128_si256(a, 1), _mm256_extracti128_si256(b, 1));
        return _mm256_set_m128i(r_hi, r_lo);
    }
}

pub inline fn _mm256_shufflehi_epi16(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shuf = [16]i32{
        0, 1, 2,  3,  4 + (imm8 & 3),  4 + ((imm8 >> 2) & 3),  4 + ((imm8 >> 4) & 3),  4 + ((imm8 >> 6) & 3),
        8, 9, 10, 11, 12 + (imm8 & 3), 12 + ((imm8 >> 2) & 3), 12 + ((imm8 >> 4) & 3), 12 + ((imm8 >> 6) & 3),
    };
    return @bitCast(@shuffle(i16, bitCast_i16x16(a), undefined, shuf));
}

pub inline fn _mm256_shufflelo_epi16(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shuf = [16]i32{
        imm8 & 3,       (imm8 >> 2) & 3,       (imm8 >> 4) & 3,       (imm8 >> 6) & 3,       4,  5,  6,  7,
        8 + (imm8 & 3), 8 + ((imm8 >> 2) & 3), 8 + ((imm8 >> 4) & 3), 8 + ((imm8 >> 6) & 3), 12, 13, 14, 15,
    };
    return @bitCast(@shuffle(i16, bitCast_i16x16(a), undefined, shuf));
}

pub inline fn _mm256_sign_epi16(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsignw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const zero: i16x16 = @splat(0);
        const r = @select(i16, zero > bitCast_i16x16(b), -%bitCast_i16x16(a), bitCast_i16x16(a));
        return @bitCast(@select(i16, (zero == bitCast_i16x16(b)), zero, r));
    }
}

pub inline fn _mm256_sign_epi32(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsignd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const zero: i32x8 = @splat(0);
        const r = @select(i32, zero > bitCast_i32x8(b), -%bitCast_i32x8(a), bitCast_i32x8(a));
        return @bitCast(@select(i32, (zero == bitCast_i32x8(b)), zero, r));
    }
}

pub inline fn _mm256_sign_epi8(a: __m256i, b: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsignb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else {
        const zero: i8x32 = @splat(0);
        const r = @select(i8, zero > bitCast_i8x32(b), -%bitCast_i8x32(a), bitCast_i8x32(a));
        return @bitCast(@select(i8, (zero == bitCast_i8x32(b)), zero, r));
    }
}

pub inline fn _mm256_sll_epi16(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsllw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 15) {
            return @splat(0);
        }
        return @bitCast(bitCast_u16x16(a) << @splat(@as(u4, @truncate(shift))));
    }
}

pub inline fn _mm256_sll_epi32(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpslld %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 31) {
            return @splat(0);
        }
        return @bitCast(bitCast_u32x8(a) << @splat(@as(u5, @truncate(shift))));
    }
}

pub inline fn _mm256_sll_epi64(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsllq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 63) {
            return @splat(0);
        }
        return @bitCast(bitCast_u64x4(a) << @splat(@as(u6, @truncate(shift))));
    }
}

pub inline fn _mm256_slli_epi16(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    return @bitCast(bitCast_u16x16(a) << @splat(imm8));
}

pub inline fn _mm256_slli_epi32(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    return @bitCast(bitCast_u32x8(a) << @splat(imm8));
}

pub inline fn _mm256_slli_epi64(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 63) {
        return @splat(0);
    }
    return @bitCast(bitCast_u64x4(a) << @splat(imm8));
}

pub inline fn _mm256_slli_si256(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    return _mm256_alignr_epi8(a, @splat(0), 16 - imm8);
}

/// Shift left each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm_sllv_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx2) {
        return asm ("vpsllvd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod32 = @as(@Vector(4, u5), @truncate(bitCast_u32x4(count)));
        const pred = bitCast_u32x4(count) == count_mod32;
        const r = @select(u32, pred, bitCast_u32x4(a), @as(u32x4, @splat(0)));
        return @bitCast(r << count_mod32);
    }
}

/// Shift left each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm256_sllv_epi32(a: __m256i, count: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsllvd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod32 = @as(@Vector(8, u5), @truncate(bitCast_u32x8(count)));
        const pred = bitCast_u32x8(count) == count_mod32;
        const r = @select(u32, pred, bitCast_u32x8(a), @as(u32x8, @splat(0)));
        return @bitCast(r << count_mod32);
    }
}

/// Shift left each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm_sllv_epi64(a: __m128i, count: __m128i) __m128i {
    if (has_avx2) {
        return asm ("vpsllvq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod64 = @as(@Vector(2, u6), @truncate(bitCast_u64x2(count)));
        const pred = bitCast_u64x2(count) == count_mod64;
        const r = @select(u64, pred, bitCast_u64x2(a), @as(u64x2, @splat(0)));
        return @bitCast(r << count_mod64);
    }
}

/// Shift left each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm256_sllv_epi64(a: __m256i, count: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsllvq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod64 = @as(@Vector(4, u6), @truncate(bitCast_u64x4(count)));
        const pred = bitCast_u64x4(count) == count_mod64;
        const r = @select(u64, pred, bitCast_u64x4(a), @as(u64x4, @splat(0)));
        return @bitCast(r << count_mod64);
    }
}

pub inline fn _mm256_sra_epi16(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsraw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = @min(bitCast_u64x2(count)[0], 15);
        return @bitCast(bitCast_i16x16(a) >> @splat(shift));
    }
}

pub inline fn _mm256_sra_epi32(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsrad %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = @min(bitCast_u64x2(count)[0], 31);
        return @bitCast(bitCast_i32x8(a) >> @splat(shift));
    }
}

pub inline fn _mm256_srai_epi16(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shift = @min(@as(u8, @intCast(imm8)), 15);
    return @bitCast(bitCast_i16x16(a) >> @splat(shift));
}

pub inline fn _mm256_srai_epi32(a: __m256i, comptime imm8: comptime_int) __m256i {
    const shift = @min(@as(u8, @intCast(imm8)), 31);
    return @bitCast(bitCast_i32x8(a) >> @splat(shift));
}

pub inline fn _mm_srav_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx2) {
        return asm ("vpsravd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = @min(bitCast_u32x4(count), @as(@Vector(4, u5), @splat(31)));
        return @bitCast(bitCast_i32x4(a) >> shift);
    }
}

pub inline fn _mm256_srav_epi32(a: __m256i, count: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsravd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = @min(bitCast_u32x8(count), @as(@Vector(8, u5), @splat(31)));
        return @bitCast(bitCast_i32x8(a) >> shift);
    }
}

pub inline fn _mm256_srl_epi16(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsrlw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 15) {
            return @splat(0);
        }
        return @bitCast(bitCast_u16x16(a) >> @splat(@as(u4, @truncate(shift))));
    }
}

pub inline fn _mm256_srl_epi32(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsrld %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 31) {
            return @splat(0);
        }
        return @bitCast(bitCast_u32x8(a) >> @splat(@as(u5, @truncate(shift))));
    }
}

pub inline fn _mm256_srl_epi64(a: __m256i, count: __m128i) __m256i {
    if (has_avx2) {
        return asm ("vpsrlq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const shift = bitCast_u64x2(count)[0];
        if (shift > 63) {
            return @splat(0);
        }
        return @bitCast(bitCast_u64x4(a) >> @splat(@as(u6, @truncate(shift))));
    }
}

pub inline fn _mm256_srli_epi16(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 15) {
        return @splat(0);
    }
    return @bitCast(bitCast_u16x16(a) >> @splat(imm8));
}

pub inline fn _mm256_srli_epi32(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 31) {
        return @splat(0);
    }
    return @bitCast(bitCast_u32x8(a) >> @splat(imm8));
}

pub inline fn _mm256_srli_epi64(a: __m256i, comptime imm8: comptime_int) __m256i {
    if (@as(u8, @intCast(imm8)) > 63) {
        return @splat(0);
    }
    return @bitCast(bitCast_u64x4(a) >> @splat(imm8));
}

pub inline fn _mm256_srli_si256(a: __m256i, comptime imm8: comptime_int) __m256i {
    return _mm256_alignr_epi8(@splat(0), a, imm8);
}

/// Unsigned shift right each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm_srlv_epi32(a: __m128i, count: __m128i) __m128i {
    if (has_avx2) {
        return asm ("vpsrlvd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod32 = @as(@Vector(4, u5), @truncate(bitCast_u32x4(count)));
        const pred = bitCast_u32x4(count) == count_mod32;
        const r = @select(u32, pred, bitCast_u32x4(a), @as(u32x4, @splat(0)));
        return @bitCast(r >> count_mod32);
    }
}

/// Unsigned shift right each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm256_srlv_epi32(a: __m256i, count: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsrlvd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod32 = @as(@Vector(8, u5), @truncate(bitCast_u32x8(count)));
        const pred = bitCast_u32x8(count) == count_mod32;
        const r = @select(u32, pred, bitCast_u32x8(a), @as(u32x8, @splat(0)));
        return @bitCast(r >> count_mod32);
    }
}

/// Unsigned shift right each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm_srlv_epi64(a: __m128i, count: __m128i) __m128i {
    if (has_avx2) {
        return asm ("vpsrlvq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod64 = @as(@Vector(2, u6), @truncate(bitCast_u64x2(count)));
        const pred = bitCast_u64x2(count) == count_mod64;
        const r = @select(u64, pred, bitCast_u64x2(a), @as(u64x2, @splat(0)));
        return @bitCast(r >> count_mod64);
    }
}

/// Unsigned shift right each lane in `a` by corresponding lane in `count`.
/// The destination lane is zeroed if the shift amount is greater than lane size.
pub inline fn _mm256_srlv_epi64(a: __m256i, count: __m256i) __m256i {
    if (has_avx2) {
        return asm ("vpsrlvq %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "x" (a),
              [b] "x" (count),
        );
    } else {
        const count_mod64 = @as(@Vector(4, u6), @truncate(bitCast_u64x4(count)));
        const pred = bitCast_u64x4(count) == count_mod64;
        const r = @select(u64, pred, bitCast_u64x4(a), @as(u64x4, @splat(0)));
        return @bitCast(r >> count_mod64);
    }
}

pub inline fn _mm256_stream_load_si256(mem_addr: *align(32) const anyopaque) __m256i {
    const src: *align(32) const __m256i = @ptrCast(mem_addr);
    if (has_avx2) {
        // It seems, the "m" constraint causes the compiler to copy the data to the stack...
        return asm volatile ("vmovntdqa (%[a]),  %[ret]"
            : [ret] "=x" (-> __m256i),
            : [a] "r" (src),
            : "memory"
        );
    } else {
        // fallback: load without non-temporal hint
        return src.*;
    }
}

pub inline fn _mm256_sub_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u16x16(a) -% bitCast_u16x16(b));
}

pub inline fn _mm256_sub_epi32(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u32x8(a) -% bitCast_u32x8(b));
}

pub inline fn _mm256_sub_epi64(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u64x4(a) -% bitCast_u64x4(b));
}

pub inline fn _mm256_sub_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u8x32(a) -% bitCast_u8x32(b));
}

pub inline fn _mm256_subs_epi16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i16x16(a) -| bitCast_i16x16(b));
}

pub inline fn _mm256_subs_epi8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_i8x32(a) -| bitCast_i8x32(b));
}

pub inline fn _mm256_subs_epu16(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u16x16(a) -| bitCast_u16x16(b));
}

pub inline fn _mm256_subs_epu8(a: __m256i, b: __m256i) __m256i {
    return @bitCast(bitCast_u8x32(a) -| bitCast_u8x32(b));
}

pub inline fn _mm256_unpackhi_epi16(a: __m256i, b: __m256i) __m256i {
    const shuf: [16]i32 = .{ 4, -5, 5, -6, 6, -7, 7, -8, 12, -13, 13, -14, 14, -15, 15, -16 };
    return @bitCast(@shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf));
}

pub inline fn _mm256_unpackhi_epi32(a: __m256i, b: __m256i) __m256i {
    const shuf: [8]i32 = .{ 2, -3, 3, -4, 6, -7, 7, -8 };
    return @bitCast(@shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf));
}

pub inline fn _mm256_unpackhi_epi64(a: __m256i, b: __m256i) __m256i {
    const shuf: [4]i32 = .{ 1, -2, 3, -4 };
    return @bitCast(@shuffle(u64, bitCast_u64x4(a), bitCast_u64x4(b), shuf));
}

pub inline fn _mm256_unpackhi_epi8(a: __m256i, b: __m256i) __m256i {
    const shuf: [32]i32 = .{ 8, -9, 9, -10, 10, -11, 11, -12, 12, -13, 13, -14, 14, -15, 15, -16, 24, -25, 25, -26, 26, -27, 27, -28, 28, -29, 29, -30, 30, -31, 31, -32 };
    return @bitCast(@shuffle(u8, bitCast_u8x32(a), bitCast_u8x32(b), shuf));
}

pub inline fn _mm256_unpacklo_epi16(a: __m256i, b: __m256i) __m256i {
    const shuf: [16]i32 = .{ 0, -1, 1, -2, 2, -3, 3, -4, 8, -9, 9, -10, 10, -11, 11, -12 };
    return @bitCast(@shuffle(u16, bitCast_u16x16(a), bitCast_u16x16(b), shuf));
}

pub inline fn _mm256_unpacklo_epi32(a: __m256i, b: __m256i) __m256i {
    const shuf: [8]i32 = .{ 0, -1, 1, -2, 4, -5, 5, -6 };
    return @bitCast(@shuffle(u32, bitCast_u32x8(a), bitCast_u32x8(b), shuf));
}

pub inline fn _mm256_unpacklo_epi64(a: __m256i, b: __m256i) __m256i {
    const shuf: [4]i32 = .{ 0, -1, 2, -3 };
    return @bitCast(@shuffle(u64, bitCast_u64x4(a), bitCast_u64x4(b), shuf));
}

pub inline fn _mm256_unpacklo_epi8(a: __m256i, b: __m256i) __m256i {
    const shuf: [32]i32 = .{ 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 16, -17, 17, -18, 18, -19, 19, -20, 20, -21, 21, -22, 22, -23, 23, -24 };
    return @bitCast(@shuffle(u8, bitCast_u8x32(a), bitCast_u8x32(b), shuf));
}

pub inline fn _mm256_xor_si256(a: __m256i, b: __m256i) __m256i {
    return a ^ b;
}
