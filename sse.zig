const std = @import("std"); // for std.testing.expectEqual

pub const has_avx2 = false;
pub const has_avx = false;
pub const has_sse42 = false;
pub const has_sse41 = false;
pub const has_ssse3 = false;
pub const has_sse3 = false;
pub const has_sse2 = false;
pub const has_sse = false;

pub const __m128 = @Vector(4, f32);
pub const __m128d = @Vector(2, f64);
pub const __m128i = @Vector(4, i32);

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
const i32x16 = @Vector(16, i32);
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
/// See: Henry S. Warren, Hacker's Delight 2nd Edition, pp. 378-380
// TODO: roundps changes SNaN to QNaN ( 7F800001 -> 7FC00001 ) ?
inline fn RoundCurrentDirection_ps(a: __m128) __m128 {
    const magic: __m128 = @splat(8388608.0); // 0x4B000000
    const bits = bitCast_u32x4(a);
    const sign = bits & @as(u32x4, @splat(0x80000000));
    const abs = bits ^ sign;
    const round = (@as(__m128, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | bitCast_u32x4(round); // restore sign
    const pred = (abs >= bitCast_u32x4(magic)); // NaN or whole number
    return @select(f32, pred, a, @as(__m128, @bitCast(res)));
}

/// Round using the current rounding mode
/// See: Henry S. Warren, Hacker's Delight 2nd Edition, pp. 378-380
// TODO: roundps changes SNaN to QNaN ( 7F800001 -> 7FC00001 ) ?
inline fn RoundCurrentDirection_f32(a: f32) f32 {
    const magic: f32 = 8388608.0; // 0x4B000000
    const bits: u32 = @bitCast(a);
    const sign = bits & 0x80000000;
    const abs = bits ^ sign;
    const round = (@as(f32, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | @as(u32, @bitCast(round)); // restore sign
    const pred = (abs >= @as(u32, @bitCast(magic))); // NaN or whole number
    return if (pred) a else @as(f32, @bitCast(res));
}

/// Round using the current rounding mode
/// See: Henry S. Warren, Hacker's Delight 2nd Edition, pp. 378-380
// TODO: roundpd changes SNaN to QNaN ( 7FF0000000000001 -> 7FF8000000000001 ) ?
inline fn RoundCurrentDirection_pd(a: __m128d) __m128d {
    const magic: __m128d = @bitCast(@as(u64x2, @splat(0x4330000000000000)));
    const bits = bitCast_u64x2(a);
    const sign = bits & @as(u64x2, @splat(0x8000000000000000));
    const abs = bits ^ sign;
    const round = (@as(__m128d, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | bitCast_u64x2(round); // restore sign
    const pred = (abs >= bitCast_u64x2(magic)); // NaN or whole number
    return @select(f64, pred, a, @as(__m128d, @bitCast(res)));
}

/// Round using the current rounding mode
/// See: Henry S. Warren, Hacker's Delight 2nd Edition, pp. 378-380
// TODO: roundsd changes SNaN to QNaN ( 7FF0000000000001 -> 7FF8000000000001 ) ?
inline fn RoundCurrentDirection_f64(a: f64) f64 {
    const magic: f64 = @bitCast(@as(u64, 0x4330000000000000));
    const bits: u64 = @bitCast(a);
    const sign = bits & 0x8000000000000000;
    const abs = bits ^ sign;
    const round = (@as(f64, @bitCast(abs)) + magic) - magic; // force rounding using current mode...
    const res = sign | @as(u64, @bitCast(round)); // restore sign
    const pred = (abs >= @as(u64, @bitCast(magic))); // NaN or whole number
    return if (pred) a else @as(f64, @bitCast(res));
}

inline fn RoundEven_f32(a: f32) f32 {
    // TODO: currently, using the default rounding, which is hopefully RoundEven
    return RoundCurrentDirection_f32(a);
}

inline fn RoundEven_ps(a: __m128) __m128 {
    // TODO: currently, using the default rounding, which is hopefully RoundEven
    return RoundCurrentDirection_ps(a);
}

inline fn RoundEven_f64(a: f64) f64 {
    // TODO: currently, using the default rounding, which is hopefully RoundEven
    return RoundCurrentDirection_f64(a);
}

inline fn RoundEven_pd(a: __m128d) __m128d {
    // TODO: currently, using the default rounding, which is hopefully RoundEven
    return RoundCurrentDirection_pd(a);
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

// SSE =================================================================

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
              [c] "N" (0),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (0),
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
              [c] "N" (0),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (0),
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
              [c] "N" (2),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (2),
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
              [c] "N" (2),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (2),
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
              [c] "N" (1),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (1),
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
              [c] "N" (1),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (1),
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
              [c] "N" (4),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (4),
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
              [c] "N" (4),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (4),
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
              [c] "N" (6),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (6),
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
              [c] "N" (6),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (6),
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
              [c] "N" (5),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (5),
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
              [c] "N" (5),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (5),
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
              [c] "N" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (7),
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
              [c] "N" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (7),
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
              [c] "N" (3),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpps %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (3),
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
              [c] "N" (3),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (3),
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

// ## pub inline fn _mm_free (mem_addr: *void) void {}
// ## pub inline fn _MM_GET_EXCEPTION_MASK () u32 {}
// ## pub inline fn _MM_GET_EXCEPTION_STATE () u32 {}
// ## pub inline fn _MM_GET_FLUSH_ZERO_MODE () u32 {}
// ## pub inline fn _MM_GET_ROUNDING_MODE () u32 {}
// ## pub inline fn _mm_getcsr (void) u32 {}

// TODO: missing load/store and ?... that use mmx __m64 notation

pub inline fn _mm_load_ps(mem_addr: *align(16) const f32) __m128 {
    return @as(*const __m128, @ptrCast(mem_addr)).*;
}

pub inline fn _mm_load_ps1(mem_addr: *const f32) __m128 {
    return _mm_load1_ps(mem_addr);
}

pub inline fn _mm_load_ss(mem_addr: *align(1) const f32) __m128 {
    return .{ mem_addr.*, 0, 0, 0 };
}

pub inline fn _mm_load1_ps(mem_addr: *const f32) __m128 {
    return .{ mem_addr.*, mem_addr.*, mem_addr.*, mem_addr.* };
}

pub inline fn _mm_loadr_ps(mem_addr: *align(16) const f32) __m128 {
    const a = _mm_load_ps(mem_addr);
    return @shuffle(f32, a, undefined, [4]i32{ 3, 2, 1, 0 });
}

pub inline fn _mm_loadu_ps(mem_addr: *align(1) const f32) __m128 {
    return @as(*align(1) const __m128, @ptrCast(mem_addr)).*;
}

// ## pub inline fn _mm_malloc (size: usize, align: usize) *void {}

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

pub inline fn _mm_movehl_ps(a: __m128, b: __m128) __m128 {
    return .{ b[2], b[3], a[2], a[3] };
}

pub inline fn _mm_movelh_ps(a: __m128, b: __m128) __m128 {
    return .{ a[0], a[1], b[0], b[1] };
}

pub inline fn _mm_movemask_ps(a: __m128) i32 {
    const cmp = @intFromBool(@as(i32x4, @splat(0)) > bitCast_i32x4(a));
    return @intCast(@as(*const u4, @ptrCast(&cmp)).*);
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

// ## pub inline fn _mm_prefetch (p: *const i8, i: i32) void {}
// ## pub inline fn _mm_rcp_ps (a: __m128) __m128 {}
// ## pub inline fn _mm_rcp_ss (a: __m128) __m128 {}
// ## pub inline fn _mm_rsqrt_ps (a: __m128) __m128 {}
// ## pub inline fn _mm_rsqrt_ss (a: __m128) __m128 {}
// ## pub inline fn _MM_SET_EXCEPTION_MASK (a: u32) void {}
// ## pub inline fn _MM_SET_EXCEPTION_STATE (a: u32) void {}
// ## pub inline fn _MM_SET_FLUSH_ZERO_MODE (a: u32) void {}

pub inline fn _mm_set_ps(e3: f32, e2: f32, e1: f32, e0: f32) __m128 {
    return .{ e0, e1, e2, e3 };
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

// ## pub inline fn _mm_sfence () void {}

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

pub inline fn _mm_store_ps(mem_addr: *align(16) f32, a: __m128) void {
    @as(*__m128, @ptrCast(mem_addr)).* = a;
}

pub inline fn _mm_store_ps1(mem_addr: *align(16) f32, a: __m128) void {
    _mm_store1_ps(mem_addr, a);
}

pub inline fn _mm_store_ss(mem_addr: *align(1) f32, a: __m128) void {
    mem_addr.* = a[0];
}

pub inline fn _mm_store1_ps(mem_addr: *align(16) f32, a: __m128) void {
    @as(*__m128, @ptrCast(mem_addr)).* = __m128{ a[0], a[0], a[0], a[0] };
}

pub inline fn _mm_storer_ps(mem_addr: *align(16) f32, a: __m128) void {
    @as(*__m128, @ptrCast(mem_addr)).* = __m128{ a[3], a[2], a[1], a[0] };
}

pub inline fn _mm_storeu_ps(mem_addr: *align(1) f32, a: __m128) void {
    @as(*align(1) __m128, @ptrCast(mem_addr)).* = a;
}

// ## pub inline fn _mm_stream_ps (mem_addr: *align(16) f32, a: __m128) void {}

pub inline fn _mm_sub_ps(a: __m128, b: __m128) __m128 {
    return a - b;
}

pub inline fn _mm_sub_ss(a: __m128, b: __m128) __m128 {
    return .{ a[0] - b[0], a[1], a[2], a[3] };
}

// ## ?? void _MM_TRANSPOSE4_PS (__m128 row0, __m128 row1, __m128 row2, __m128 row3) ??

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

pub inline fn _mm_unpackhi_ps(a: __m128, b: __m128) __m128 {
    return .{ a[2], b[2], a[3], b[3] };
}

pub inline fn _mm_unpacklo_ps(a: __m128, b: __m128) __m128 {
    return .{ a[0], b[0], a[1], b[1] };
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

// ## pub inline fn _mm_clflush (p: *const void) void {}

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
              [c] "N" (0),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (0),
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
              [c] "N" (0),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (0),
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
              [c] "N" (2),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (2),
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
              [c] "N" (2),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (2),
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
              [c] "N" (1),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (1),
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
              [c] "N" (1),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (1),
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
              [c] "N" (4),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (4),
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
              [c] "N" (4),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (4),
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
              [c] "N" (6),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (6),
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
              [c] "N" (6),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (6),
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
              [c] "N" (5),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (5),
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
              [c] "N" (5),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (5),
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
              [c] "N" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (7),
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
              [c] "N" (7),
        );
    } else if (has_sse) {
        var res = a;
        asm ("cmpsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (7),
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
              [c] "N" (3),
        );
    } else if (has_sse2) {
        var res = a;
        asm ("cmppd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (3),
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
    return .{ @floatFromInt(a[0]), @floatFromInt(a[1]) };
}

pub inline fn _mm_cvtepi32_ps(a: __m128i) __m128 {
    return @floatFromInt(a);
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

// ## pub inline fn _mm_lfence () void {}

pub inline fn _mm_load_pd(mem_addr: *align(16) const f64) __m128d {
    return @as(*const __m128d, @ptrCast(mem_addr)).*;
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
    return .{ mem_addr.*, mem_addr.* };
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

pub inline fn _mm_loadr_pd(mem_addr: *align(16) const f64) __m128d {
    return @shuffle(f64, _mm_load_pd(mem_addr), undefined, [2]i32{ 1, 0 });
}

pub inline fn _mm_loadu_pd(mem_addr: *align(1) const f64) __m128d {
    return @as(*align(1) const __m128d, @ptrCast(mem_addr)).*;
}

pub inline fn _mm_loadu_si128(mem_addr: *align(1) const __m128i) __m128i {
    return mem_addr.*;
}

pub inline fn _mm_loadu_si16(mem_addr: *const void) __m128i {
    const word = @as(*align(1) const u16, @ptrCast(mem_addr)).*;
    return @bitCast(u16x8{ word, 0, 0, 0, 0, 0, 0, 0 });
}

pub inline fn _mm_loadu_si32(mem_addr: *const void) __m128i {
    const dword = @as(*align(1) const u32, @ptrCast(mem_addr)).*;
    return @bitCast(u32x4{ dword, 0, 0, 0 });
}

pub inline fn _mm_loadu_si64(mem_addr: *const void) __m128i {
    const qword = @as(*align(1) const u64, @ptrCast(mem_addr)).*;
    return @bitCast(u64x2{ qword, 0 });
}

pub inline fn _mm_madd_epi16(a: __m128i, b: __m128i) __m128i {
    const r = intCast_i32x8(bitCast_i16x8(a)) *%
        intCast_i32x8(bitCast_i16x8(b));

    const shuf_even = i32x4{ 0, 2, 4, 6 };
    const shuf_odd = i32x4{ 1, 3, 5, 7 };
    const even = @shuffle(i32, r, undefined, shuf_even);
    const odd = @shuffle(i32, r, undefined, shuf_odd);
    return @bitCast(even +% odd);
}

// ## pub inline fn _mm_maskmoveu_si128 (a: __m128i, mask: __m128i, mem_addr: *i8) void {}

pub inline fn _mm_max_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i16x8(a), bitCast_i16x8(b)));
}

pub inline fn _mm_max_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u8x16(a), bitCast_u8x16(b)));
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

// ## pub inline fn _mm_mfence () void {}

pub inline fn _mm_min_epi16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i16x8(a), bitCast_i16x8(b)));
}

pub inline fn _mm_min_epu8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u8x16(a), bitCast_u8x16(b)));
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
    const cmp = @intFromBool(@as(i8x16, @splat(0)) > bitCast_i8x16(a));
    return @intCast(@as(*const u16, @ptrCast(&cmp)).*);
}

pub inline fn _mm_movemask_pd(a: __m128d) i32 {
    const cmp = @intFromBool(@as(i64x2, @splat(0)) > bitCast_i64x2(a));
    return @intCast(@as(*const u2, @ptrCast(&cmp)).*);
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
    ab = @min(ab, @as(i16x16, @splat(127)));
    ab = @max(ab, @as(i16x16, @splat(-128)));
    return @bitCast(@as(i8x16, @truncate(ab)));
}

pub inline fn _mm_packs_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x8{ 0, 1, 2, 3, -1, -2, -3, -4 };
    var ab = @shuffle(i32, bitCast_i32x4(a), bitCast_i32x4(b), shuf);
    ab = @min(ab, @as(i32x8, @splat(32767)));
    ab = @max(ab, @as(i32x8, @splat(-32768)));
    return @bitCast(@as(i16x8, @truncate(ab)));
}

pub inline fn _mm_packus_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf = i32x16{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 };
    var ab = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf);
    ab = @min(ab, @as(i16x16, @splat(255)));
    ab = @max(ab, @as(i16x16, @splat(0)));
    return @bitCast(@as(i8x16, @truncate(ab)));
}

// ## pub inline fn _mm_pause () void {}

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

        const max = @max(bitCast_u8x16(a), bitCast_u8x16(b));
        const min = @min(bitCast_u8x16(a), bitCast_u8x16(b));
        const abd = intCast_u16x16(max - min);

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

pub inline fn _mm_store_pd(mem_addr: *align(16) f64, a: __m128d) void {
    @as(*__m128d, @ptrCast(mem_addr)).* = a;
}

pub inline fn _mm_store_pd1(mem_addr: *align(16) f64, a: __m128d) void {
    return _mm_store1_pd(mem_addr, a);
}

pub inline fn _mm_store_sd(mem_addr: *align(1) f64, a: __m128d) void {
    mem_addr.* = a[0];
}

pub inline fn _mm_store_si128(mem_addr: *__m128i, a: __m128i) void {
    mem_addr.* = a;
}

pub inline fn _mm_store1_pd(mem_addr: *align(16) f64, a: __m128d) void {
    @as(*__m128d, @ptrCast(mem_addr)).* = __m128d{ a[0], a[0] };
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

pub inline fn _mm_storer_pd(mem_addr: *align(16) f64, a: __m128d) void {
    @as(*__m128d, @ptrCast(mem_addr)).* = __m128d{ a[1], a[0] };
}

pub inline fn _mm_storeu_pd(mem_addr: *align(1) f64, a: __m128d) void {
    @as(*align(1) __m128d, @ptrCast(mem_addr)).* = a;
}

pub inline fn _mm_storeu_si128(mem_addr: *align(1) __m128i, a: __m128i) void {
    mem_addr.* = a;
}

pub inline fn _mm_storeu_si16(mem_addr: *void, a: __m128i) void {
    @as(*align(1) u16, @ptrCast(mem_addr)).* = bitCast_u16x8(a)[0];
}

pub inline fn _mm_storeu_si32(mem_addr: *void, a: __m128i) void {
    @as(*align(1) u32, @ptrCast(mem_addr)).* = bitCast_u32x4(a)[0];
}

pub inline fn _mm_storeu_si64(mem_addr: *void, a: __m128i) void {
    @as(*align(1) u64, @ptrCast(mem_addr)).* = bitCast_u64x2(a)[0];
}

// ## pub inline fn _mm_stream_pd (mem_addr: *f64, a: __m128d) void {}
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

pub inline fn _mm_abs_epi16(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i16x8(a)));
}

pub inline fn _mm_abs_epi32(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i32x4(a)));
}

pub inline fn _mm_abs_epi8(a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i8x16(a)));
}

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

pub inline fn _mm_hadd_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
    const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
    const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
    return @bitCast(even +% odd);
}

pub inline fn _mm_hadd_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf_even = i32x4{ 0, 2, -1, -3 };
    const shuf_odd = i32x4{ 1, 3, -2, -4 };
    const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
    return @bitCast(even +% odd);
}

pub inline fn _mm_hadds_epi16(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vphaddsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("phaddsw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even +| odd);
    }
}

pub inline fn _mm_hsub_epi16(a: __m128i, b: __m128i) __m128i {
    const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
    const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
    const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
    return @bitCast(even -% odd);
}

pub inline fn _mm_hsub_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf_even = i32x4{ 0, 2, -1, -3 };
    const shuf_odd = i32x4{ 1, 3, -2, -4 };
    const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
    return @bitCast(even -% odd);
}

pub inline fn _mm_hsubs_epi16(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vphsubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("phsubsw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const shuf_even = i32x8{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even -| odd);
    }
}

pub inline fn _mm_maddubs_epi16(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpmaddubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("pmaddubsw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else { // !NOT TESTED! weird saturation rules... todo
        const r = bitCast_i16x16(intCast_u16x16(bitCast_u8x16(a))) *%
            intCast_i16x16(bitCast_i8x16(b));

        const shuf_even = i32x8{ 0, 2, 4, 6, 8, 10, 12, 14 };
        const shuf_odd = i32x8{ 1, 3, 5, 7, 9, 11, 13, 15 };
        const even = @shuffle(i32, r, undefined, shuf_even);
        const odd = @shuffle(i32, r, undefined, shuf_odd);
        return @bitCast(even +| odd);
    }
}

pub inline fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpmulhrsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("pmulhrsw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        var r = intCast_i32x8(bitCast_i16x8(a));
        r *%= intCast_i32x8(bitCast_i16x8(b));
        r +%= @splat(1 << 14);
        return @bitCast(@as(i16x8, @truncate(r >> @splat(15))));
    }
}

pub inline fn _mm_shuffle_epi8(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpshufb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("pshufb %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else { // !NOT TESTED!
        var r: i8x16 = undefined;
        const shuf = bitCast_i8x16(b) & @as(i8x16, @splat(0x0F));
        const mask = bitCast_i8x16(b) >> @as(i8x16, @splat(7));
        for (0..16) |i| {
            r[i] = bitCast_i8x16(a)[@intCast(shuf[i])];
        }
        return @bitCast(~mask & r);
    }
}

pub inline fn _mm_sign_epi16(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsignw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("psignw %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const zero: i16x8 = @splat(0);
        const r = @select(i16, zero > bitCast_i16x8(b), -%bitCast_i16x8(a), bitCast_i16x8(a));
        return @bitCast(@select(i16, (zero == bitCast_i16x8(b)), zero, r));
    }
}

pub inline fn _mm_sign_epi32(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsignd %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("psignd %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const zero: i32x4 = @splat(0);
        const r = @select(i32, zero > bitCast_i32x4(b), -%bitCast_i32x4(a), bitCast_i32x4(a));
        return @bitCast(@select(i32, (zero == bitCast_i32x4(b)), zero, r));
    }
}

pub inline fn _mm_sign_epi8(a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpsignb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_ssse3) {
        var res = a;
        asm ("psignb %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
        );
        return res;
    } else {
        const zero: i8x16 = @splat(0);
        const r = @select(i8, zero > bitCast_i8x16(b), -%bitCast_i8x16(a), bitCast_i8x16(a));
        return @bitCast(@select(i8, (zero == bitCast_i8x16(b)), zero, r));
    }
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

pub inline fn _mm_blend_epi16(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    const mask = comptime blk: { // convert imm8 to vector of bools
        var m: @Vector(8, bool) = undefined;
        for (0..8) |i| {
            m[i] = (((imm8 >> i) & 1) == 1);
        }
        break :blk m;
    };
    return @bitCast(@select(i16, mask, bitCast_i16x8(b), bitCast_i16x8(a)));
}

pub inline fn _mm_blend_pd(a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {
    var r = a;
    if ((imm8 & 1) == 1) r[0] = b[0];
    if ((imm8 & 2) == 2) r[1] = b[1];
    return r;
}

pub inline fn _mm_blend_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    const mask = comptime blk: { // convert imm8 to vector of bools
        var m: @Vector(4, bool) = undefined;
        for (0..4) |i| {
            m[i] = (((imm8 >> i) & 1) == 1);
        }
        break :blk m;
    };
    return @select(f32, mask, b, a);
}

pub inline fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) __m128i {
    const cmp = @as(i8x16, @splat(0)) > bitCast_i8x16(mask);
    return @bitCast(@select(i8, cmp, bitCast_i8x16(b), bitCast_i8x16(a)));
}

pub inline fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) __m128d {
    const cmp = @as(i64x2, @splat(0)) > bitCast_i64x2(mask);
    return @select(f64, cmp, b, a);
}

pub inline fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) __m128 {
    const cmp = @as(i32x4, @splat(0)) > bitCast_i32x4(mask);
    return @select(f32, cmp, b, a);
}

pub inline fn _mm_ceil_pd(a: __m128d) __m128d {
    return _mm_round_pd(a, _MM_FROUND_CEIL);
}

pub inline fn _mm_ceil_ps(a: __m128) __m128 {
    return _mm_round_ps(a, _MM_FROUND_CEIL);
}

pub inline fn _mm_ceil_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_round_sd(a, b, _MM_FROUND_CEIL);
}

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

// ## pub inline fn _mm_dp_pd (a: __m128d, b: __m128d, comptime imm8: comptime_int) __m128d {}

// ## pub inline fn _mm_dp_ps (a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {}

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

pub inline fn _mm_floor_pd(a: __m128d) __m128d {
    return _mm_round_pd(a, _MM_FROUND_FLOOR);
}

pub inline fn _mm_floor_ps(a: __m128) __m128 {
    return _mm_round_ps(a, _MM_FROUND_FLOOR);
}

pub inline fn _mm_floor_sd(a: __m128d, b: __m128d) __m128d {
    return _mm_round_sd(a, b, _MM_FROUND_FLOOR);
}

pub inline fn _mm_floor_ss(a: __m128, b: __m128) __m128 {
    return _mm_round_ss(a, b, _MM_FROUND_FLOOR);
}

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

pub inline fn _mm_insert_ps(a: __m128, b: __m128, comptime imm8: comptime_int) __m128 {
    var r = a;
    r[(imm8 >> 4) & 3] = b[imm8 >> 6];
    if ((imm8 & 1) == 1) r[0] = 0;
    if ((imm8 & 2) == 2) r[1] = 0;
    if ((imm8 & 4) == 4) r[2] = 0;
    if ((imm8 & 8) == 8) r[3] = 0;
    return r;
}

test "_mm_insert_ps" {
    const a = _mm_set_ps(1.7, 1.5, 1.3, 1.1);
    const b = _mm_set_ps(2.7, 2.5, 2.3, 2.1);
    const ref1 = _mm_set_ps(0.0, 2.3, 1.3, 1.1);
    const ref2 = _mm_set_ps(1.7, 0.0, 1.3, 0.0);
    try std.testing.expectEqual(ref1, _mm_insert_ps(a, b, 0x68));
    try std.testing.expectEqual(ref2, _mm_insert_ps(a, b, 0x65));
}

pub inline fn _mm_max_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i32x4(a), bitCast_i32x4(b)));
}

test "_mm_max_epi32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(2147418112, 2147483647, 0, 0);
    try std.testing.expectEqual(ref, _mm_max_epi32(a, b));
}

pub inline fn _mm_max_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i8x16(a), bitCast_i8x16(b)));
}

test "_mm_max_epi8" {
    const a = _mm_set_epi8(-128, 0, 0, 0, 0, 0, 0, 0, -1, -2, -128, -128, 127, 2, 1, 1);
    const b = _mm_set_epi8(0, 0, 0, 0, -128, 0, 0, 0, -2, -1, 127, -127, -128, 2, 2, 0);
    const ref = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 127, -127, 127, 2, 2, 1);
    try std.testing.expectEqual(ref, _mm_max_epi8(a, b));
}

pub inline fn _mm_max_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u16x8(a), bitCast_u16x8(b)));
}

test "_mm_max_epu16" {
    const a = _mm_set_epi16(32767, 1, -1, -257, -32768, 0, 2, 258);
    const b = _mm_set_epi16(-32768, 0, 32767, -2, 0, -32768, 3, 257);
    const ref = _mm_set_epi16(-32768, 1, -1, -2, -32768, -32768, 3, 258);
    try std.testing.expectEqual(ref, _mm_max_epu16(a, b));
}

pub inline fn _mm_max_epu32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u32x4(a), bitCast_u32x4(b)));
}

test "_mm_max_epu32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(-2147483647, -1, -2147483648, -2147483648);
    try std.testing.expectEqual(ref, _mm_max_epu32(a, b));
}

pub inline fn _mm_min_epi32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i32x4(a), bitCast_i32x4(b)));
}

test "_mm_min_epi32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(-2147483647, -1, -2147483648, -2147483648);
    try std.testing.expectEqual(ref, _mm_min_epi32(a, b));
}

pub inline fn _mm_min_epi8(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i8x16(a), bitCast_i8x16(b)));
}

test "_mm_min_epi8" {
    const a = _mm_set_epi8(-128, 0, 0, 0, 0, 0, 0, 0, -1, -2, -128, -128, 127, 2, 1, 1);
    const b = _mm_set_epi8(0, 0, 0, 0, -128, 0, 0, 0, -2, -1, 127, -127, -128, 2, 2, 0);
    const ref = _mm_set_epi8(-128, 0, 0, 0, -128, 0, 0, 0, -2, -2, -128, -128, -128, 2, 1, 0);
    try std.testing.expectEqual(ref, _mm_min_epi8(a, b));
}

pub inline fn _mm_min_epu16(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u16x8(a), bitCast_u16x8(b)));
}

test "_mm_min_epu16" {
    const a = _mm_set_epi16(32767, 1, -1, -257, -32768, 0, 2, 258);
    const b = _mm_set_epi16(-32768, 0, 32767, -2, 0, -32768, 3, 257);
    const ref = _mm_set_epi16(32767, 0, 32767, -257, 0, 0, 2, 257);
    try std.testing.expectEqual(ref, _mm_min_epu16(a, b));
}

pub inline fn _mm_min_epu32(a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u32x4(a), bitCast_u32x4(b)));
}

test "_mm_min_epu32" {
    const a = _mm_set_epi32(2147418112, -1, -2147483648, 0);
    const b = _mm_set_epi32(-2147483647, 2147483647, 0, -2147483648);
    const ref = _mm_set_epi32(2147418112, 2147483647, 0, 0);
    try std.testing.expectEqual(ref, _mm_min_epu32(a, b));
}

pub inline fn _mm_minpos_epu16(a: __m128i) __m128i {
    if (has_avx) {
        return asm ("vphminposuw %[a], %[ret]"
            : [ret] "=x" (-> __m128i),
            : [a] "x" (a),
        );
    } else if (has_sse41) {
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

// ## pub inline fn _mm_mpsadbw_epu8 (a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {}

pub inline fn _mm_mul_epi32(a: __m128i, b: __m128i) __m128i {
    const x = intCast_i64x4(a);
    const y = intCast_i64x4(b);
    return @bitCast(i64x2{ x[0] *% y[0], x[2] *% y[2] });
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

pub inline fn _mm_packus_epi32(a: __m128i, b: __m128i) __m128i {
    const shuf: [8]i32 = .{ 0, 1, 2, 3, -1, -2, -3, -4 };
    var ab = @shuffle(i32, bitCast_i32x4(a), bitCast_i32x4(b), shuf);
    ab = @min(ab, @as(i32x8, @splat(65535)));
    ab = @max(ab, @as(i32x8, @splat(0)));
    return @bitCast(@as(i16x8, @truncate(ab)));
}

test "_mm_packus_epi32" {
    const a = _mm_set_epi32(1, 65535, 32768, 0);
    const b = _mm_set_epi32(2147483647, -2147483648, -1, 131071);
    const ref = _mm_set_epi16(-1, 0, 0, -1, 1, -1, -32768, 0);
    try std.testing.expectEqual(ref, _mm_packus_epi32(a, b));
}

pub inline fn _mm_round_pd(a: __m128d, comptime imm8: comptime_int) __m128d {
    if (has_avx) {
        return asm ("vroundpd %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [c] "N" (imm8),
        );
    } else if (has_sse41) {
        return asm ("roundpd %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [c] "N" (imm8),
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
    if (has_avx) {
        return asm ("vroundps %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [c] "N" (imm8),
        );
    } else if (has_sse41) {
        return asm ("roundps %[c], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [c] "N" (imm8),
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
    if (has_avx) {
        return asm ("vroundsd %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128d),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "N" (imm8),
        );
    } else if (has_sse41) {
        var res = a;
        asm ("roundsd %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (imm8),
        );
        return res;
    } else {
        return switch (imm8 & 0x07) {
            _MM_FROUND_TO_NEAREST_INT => .{ RoundEven_f64(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_NEG_INF => .{ @floor(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_POS_INF => .{ @ceil(b[0]), a[1], a[2], a[3] },
            _MM_FROUND_TO_ZERO => .{ @trunc(b[0]), a[1], a[2], a[3] },
            else => .{ RoundCurrentDirection_f64(b[0]), a[1], a[2], a[3] },
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
    if (has_avx) {
        return asm ("vroundss %[c], %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128),
            : [a] "x" (a),
              [b] "x" (b),
              [c] "N" (imm8),
        );
    } else if (has_sse41) {
        var res = a;
        asm ("roundss %[c], %[b], %[a]"
            : [a] "+x" (res),
            : [b] "x" (b),
              [c] "N" (imm8),
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
    return _mm_testz_si128(~a, b);
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
    if (has_avx) {
        return asm ("vptest %[b],%[a]"
            : [_] "={@cca}" (-> i32),
            : [a] "x" (a),
              [b] "x" (b),
        );
    } else if (has_sse41) {
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
    return @intFromBool(@reduce(.Or, (a & b)) == 0);
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
    const a = _mm_set_epi64x(-9223372036854775807, 1);
    const b = _mm_set_epi64x(0, 2);
    const ref0 = _mm_set_epi64x(0, 0);
    const ref1 = _mm_set_epi64x(-1, -1);
    try std.testing.expectEqual(ref0, _mm_cmpgt_epi64(a, b));
    try std.testing.expectEqual(ref1, _mm_cmpgt_epi64(b, a));
}
