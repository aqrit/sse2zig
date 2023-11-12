const std = @import("std");

pub const __m128i = @Vector(4, i32);

/// TODO
pub const has_avx2 = true;
pub const has_avx = true;
pub const has_sse42 = true;
pub const has_sse41 = true;
pub const has_ssse3 = true;
pub const has_sse3 = true;
pub const has_sse2 = true;
pub const has_sse = true;


// helpers to reduce verbosity
inline fn bitCast_u64x2 (a: anytype) @Vector(2, u64) { return @bitCast(a); }
inline fn bitCast_u32x4 (a: anytype) @Vector(4, u32) { return @bitCast(a); }
inline fn bitCast_u16x8 (a: anytype) @Vector(8, u16) { return @bitCast(a); }
inline fn bitCast_u8x16 (a: anytype) @Vector(16, u8) { return @bitCast(a); }
inline fn bitCast_i64x2 (a: anytype) @Vector(2, i64) { return @bitCast(a); }
inline fn bitCast_i32x4 (a: anytype) @Vector(4, i32) { return @bitCast(a); }
inline fn bitCast_i16x8 (a: anytype) @Vector(8, i16) { return @bitCast(a); }
inline fn bitCast_i8x16 (a: anytype) @Vector(16, i8) { return @bitCast(a); }
//
inline fn bitCast_u64x4  (a: anytype) @Vector(4, u64)  { return @bitCast(a); }
inline fn bitCast_u32x8  (a: anytype) @Vector(8, u32)  { return @bitCast(a); }
inline fn bitCast_u16x16 (a: anytype) @Vector(16, u16) { return @bitCast(a); }
inline fn bitCast_u8x32  (a: anytype) @Vector(32, u8)  { return @bitCast(a); }
inline fn bitCast_i64x4  (a: anytype) @Vector(4, i64)  { return @bitCast(a); }
inline fn bitCast_i32x8  (a: anytype) @Vector(8, i32)  { return @bitCast(a); }
inline fn bitCast_i16x16 (a: anytype) @Vector(16, i16) { return @bitCast(a); }
inline fn bitCast_i8x32  (a: anytype) @Vector(32, i8)  { return @bitCast(a); }
//
inline fn intCast_u64x2 (a: anytype) @Vector(2, u64) { return @as(@Vector(2, u64), @intCast(a)); }
inline fn intCast_u32x4 (a: anytype) @Vector(4, u32) { return @as(@Vector(4, u32), @intCast(a)); }
inline fn intCast_u16x8 (a: anytype) @Vector(8, u16) { return @as(@Vector(8, u16), @intCast(a)); }
inline fn intCast_u8x16 (a: anytype) @Vector(16, u8) { return @as(@Vector(16, u8), @intCast(a)); }
inline fn intCast_i64x2 (a: anytype) @Vector(2, i64) { return @as(@Vector(2, i64), @intCast(a)); }
inline fn intCast_i32x4 (a: anytype) @Vector(4, i32) { return @as(@Vector(4, i32), @intCast(a)); }
inline fn intCast_i16x8 (a: anytype) @Vector(8, i16) { return @as(@Vector(8, i16), @intCast(a)); }
inline fn intCast_i8x16 (a: anytype) @Vector(16, i8) { return @as(@Vector(16, i8), @intCast(a)); }
//
inline fn intCast_u64x4  (a: anytype) @Vector(4,  u64) { return @as(@Vector(4,  u64), @intCast(a)); }
inline fn intCast_u32x8  (a: anytype) @Vector(8,  u32) { return @as(@Vector(8,  u32), @intCast(a)); }
inline fn intCast_u16x16 (a: anytype) @Vector(16, u16) { return @as(@Vector(16, u16), @intCast(a)); }
inline fn intCast_u8x32  (a: anytype) @Vector(32, u8)  { return @as(@Vector(32, u8),  @intCast(a)); }
inline fn intCast_i64x4  (a: anytype) @Vector(4,  i64) { return @as(@Vector(4,  i64), @intCast(a)); }
inline fn intCast_i32x8  (a: anytype) @Vector(8,  i32) { return @as(@Vector(8,  i32), @intCast(a)); }
inline fn intCast_i16x16 (a: anytype) @Vector(16, i16) { return @as(@Vector(16, i16), @intCast(a)); }
inline fn intCast_i8x32  (a: anytype) @Vector(32, i8)  { return @as(@Vector(32, i8),  @intCast(a)); }


pub inline fn _mm_setzero_si128 () __m128i {
    return @splat(0);
}


pub inline fn _mm_undefined_si128 () __m128i {
    // zig `undefined` doesn't compare equal to itself ?
    return _mm_setzero_si128();
}


pub inline fn _mm_move_epi64 (a: __m128i) __m128i {
    const r: @Vector(2, i64) = .{ bitCast_i64x2(a)[0], 0 };
    return @bitCast(r);
}

pub inline fn _mm_set1_epi64x (a: i64) __m128i {
    return @bitCast(@as(@Vector(2, i64), @splat(a)));
}


pub inline fn _mm_set1_epi32 (a: i32) __m128i {
    return @bitCast(@as(@Vector(4, i32), @splat(a)));
}


pub inline fn _mm_set1_epi16 (a: i16) __m128i {
    return @bitCast(@as(@Vector(8, i16), @splat(a)));
}


pub inline fn _mm_set1_epi8 (a: i8) __m128i {
    return @bitCast(@as(@Vector(16, i8), @splat(a)));
}


pub inline fn _mm_set_epi64x (e1: i64, e0: i64) __m128i {
    const r: @Vector(2, i64) = .{ e0, e1 };
    return @bitCast(r);
}


pub inline fn _mm_set_epi32 (e3: i32, e2: i32, e1: i32, e0: i32) __m128i {
    const r: @Vector(4, i32) = .{ e0, e1, e2, e3 };
    return @bitCast(r);
}


pub inline fn _mm_set_epi16 (e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m128i {
    const r: @Vector(8, i16) = .{ e0, e1, e2, e3, e4, e5, e6, e7 };
    return @bitCast(r);
}


pub inline fn _mm_set_epi8 (e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m128i {
    const r: @Vector(16, i8) = .{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 };
    return @bitCast(r);
}


/// not listed in intel intrinsics guide (but may exist w/MSVC ?)
pub inline fn _mm_setr_epi64x (e1: i64, e0: i64) __m128i {
    return _mm_set_epi64x(e0, e1);
}


pub inline fn _mm_setr_epi32 (e3: i32, e2: i32, e1: i32, e0: i32) __m128i {
    const r: @Vector(4, i32) = .{ e3, e2, e1, e0 };
    return @bitCast(r);
}


pub inline fn _mm_setr_epi16 (e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16) __m128i {
    const r: @Vector(8, i16) = .{ e7, e6, e5, e4, e3, e2, e1, e0 };
    return @bitCast(r);
}


pub inline fn _mm_setr_epi8 (e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8, e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) __m128i {
    const r: @Vector(16, i8) = .{ e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0 };
    return @bitCast(r);
}


pub inline fn _mm_cvtsi128_si64 (a: __m128i) i64 {
    return bitCast_i64x2(a)[0];
}


/// this alternative name is missing from clang headers
pub inline fn _mm_cvtsi128_si64x (a: __m128i) i64 {
    return _mm_cvtsi128_si64(a);
}


pub inline fn _mm_cvtsi64_si128 (a: i64) __m128i {
    const r: @Vector(2, i64) = .{ a, 0 };
    return @bitCast(r);
}


/// this alternative name is missing from clang headers
pub inline fn _mm_cvtsi64x_si128 (a: i64) __m128i {
    return _mm_cvtsi64_si128(a);
}


pub inline fn _mm_cvtsi128_si32 (a: __m128i) i32 {
    return bitCast_i32x4(a)[0];
}


pub inline fn _mm_cvtsi32_si128 (a: i32) __m128i {
    const r: @Vector(4, i32) = .{ a, 0, 0, 0 };
    return @bitCast(r);
}


pub inline fn _mm_and_si128 (a: __m128i, b: __m128i) __m128i {
    return a & b;
}


pub inline fn _mm_andnot_si128 (a: __m128i, b: __m128i) __m128i {
    return ~a & b;
}


pub inline fn _mm_or_si128 (a: __m128i, b: __m128i) __m128i {
    return a | b;
}


pub inline fn _mm_xor_si128 (a: __m128i, b: __m128i) __m128i {
    return a ^ b;
}


pub inline fn _mm_add_epi64 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u64x2(a) +% bitCast_u64x2(b));
}


pub inline fn _mm_add_epi32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u32x4(a) +% bitCast_u32x4(b));
}


pub inline fn _mm_add_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) +% bitCast_u16x8(b));
}


pub inline fn _mm_add_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) +% bitCast_u8x16(b));
}


pub inline fn _mm_sub_epi64 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u64x2(a) -% bitCast_u64x2(b));
}


pub inline fn _mm_sub_epi32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u32x4(a) -% bitCast_u32x4(b));
}


pub inline fn _mm_sub_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) -% bitCast_u16x8(b));
}


pub inline fn _mm_sub_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) -% bitCast_u8x16(b));
}


pub inline fn _mm_adds_epu16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) +| bitCast_u16x8(b));
}


pub inline fn _mm_adds_epu8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) +| bitCast_u8x16(b));
}


pub inline fn _mm_adds_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) +| bitCast_i16x8(b));
}


pub inline fn _mm_adds_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i8x16(a) +| bitCast_i8x16(b));
}


pub inline fn _mm_subs_epu16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u16x8(a) -| bitCast_u16x8(b));
}


pub inline fn _mm_subs_epu8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_u8x16(a) -| bitCast_u8x16(b));
}


pub inline fn _mm_subs_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) -| bitCast_i16x8(b));
}


pub inline fn _mm_subs_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i8x16(a) -| bitCast_i8x16(b));
}


pub inline fn _mm_slli_epi64 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 63) { return @splat(0); }
    const shift: @Vector(2, u64) = @splat(imm8);
    return @bitCast(bitCast_u64x2(a) << shift);
}


pub inline fn _mm_slli_epi32 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) { return @splat(0); }
    const shift: @Vector(4, u32) = @splat(imm8);
    return @bitCast(bitCast_u32x4(a) << shift);
}


pub inline fn _mm_slli_epi16 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) { return @splat(0); }
    const shift: @Vector(8, u16) = @splat(imm8);
    return @bitCast(bitCast_u16x8(a) << shift);
}


pub inline fn _mm_srli_epi64 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 63) { return @splat(0); }
    const shift: @Vector(2, u64) = @splat(imm8);
    return @bitCast(bitCast_u64x2(a) >> shift);
}


pub inline fn _mm_srli_epi32 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) { return @splat(0); }
    const shift: @Vector(4, u32) = @splat(imm8);
    return @bitCast(bitCast_u32x4(a) >> shift);
}


pub inline fn _mm_srli_epi16 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) { return @splat(0); }
    const shift: @Vector(8, u16) = @splat(imm8);
    return @bitCast(bitCast_u16x8(a) >> shift);
}


pub inline fn _mm_srai_epi32 (a: __m128i, comptime imm8: comptime_int) __m128i {
    const shift: @Vector(4, i32) = @splat(@min(@as(u8, @intCast(imm8)), 31));
    return @bitCast(bitCast_i32x4(a) >> shift);
}


pub inline fn _mm_srai_epi16 (a: __m128i, comptime imm8: comptime_int) __m128i {
    const shift: @Vector(8, i16) = @splat(@min(@as(u8, @intCast(imm8)), 15));
    return @bitCast(bitCast_i16x8(a) >> shift);
}


pub inline fn _mm_abs_epi32 (a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i32x4(a)));
}


pub inline fn _mm_abs_epi16 (a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i16x8(a)));
}


pub inline fn _mm_abs_epi8 (a: __m128i) __m128i {
    return @bitCast(@abs(bitCast_i8x16(a)));
}


pub inline fn _mm_unpacklo_epi64 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, -1 };
    return @bitCast(@shuffle(u64, bitCast_u64x2(a), bitCast_u64x2(b), shuf));
}


pub inline fn _mm_unpacklo_epi32 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 0, -1, 1, -2 };
    return @bitCast(@shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf));
}


pub inline fn _mm_unpacklo_epi16 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(8, i32) = .{ 0, -1, 1, -2, 2, -3, 3, -4 };
    return @bitCast(@shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf));
}


pub inline fn _mm_unpacklo_epi8 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(16, i32) = .{0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8};
    return @bitCast(@shuffle(u8, bitCast_u8x16(a), bitCast_u8x16(b), shuf));
}


pub inline fn _mm_unpackhi_epi64 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 1, -2 };
    return @bitCast(@shuffle(u64, bitCast_u64x2(a), bitCast_u64x2(b), shuf));
}


pub inline fn _mm_unpackhi_epi32 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 2, -3, 3, -4 };
    return @bitCast(@shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf));
}


pub inline fn _mm_unpackhi_epi16 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(8, i32) = .{ 4, -5, 5, -6, 6, -7, 7, -8 };
    return @bitCast(@shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf));
}


pub inline fn _mm_unpackhi_epi8 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(16, i32) = .{8,-9,9,-10, 10,-11,11,-12, 12,-13,13,-14, 14,-15,15,-16};
    return @bitCast(@shuffle(u8, bitCast_u8x16(a), bitCast_u8x16(b), shuf));
}


pub inline fn _mm_cvtepu32_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(u32, bitCast_u32x4(a), undefined, shuf);
    return @bitCast(intCast_u64x2(lo));
}


pub inline fn _mm_cvtepu16_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(u16, bitCast_u16x8(a), undefined, shuf);
    return @bitCast(intCast_u64x2(lo));
}


pub inline fn _mm_cvtepu16_epi32 (a: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 0, 1, 2, 3 };
    const lo = @shuffle(u16, bitCast_u16x8(a), undefined, shuf);
    return @bitCast(intCast_u32x4(lo));
}


pub inline fn _mm_cvtepu8_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(u8, bitCast_u8x16(a), undefined, shuf);
    return @bitCast(intCast_u64x2(lo));
}


pub inline fn _mm_cvtepu8_epi32 (a: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 0, 1, 2, 3 };
    const lo = @shuffle(u8, bitCast_u8x16(a), undefined, shuf);
    return @bitCast(intCast_u32x4(lo));
}


pub inline fn _mm_cvtepu8_epi16 (a: __m128i) __m128i {
    const shuf: @Vector(8, i32) = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const lo = @shuffle(u8, bitCast_u8x16(a), undefined, shuf);
    return @bitCast(intCast_u16x8(lo));
}


pub inline fn _mm_cvtepi32_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(i32, bitCast_i32x4(a), undefined, shuf);
    return @bitCast(intCast_i64x2(lo));
}


pub inline fn _mm_cvtepi16_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(i16, bitCast_i16x8(a), undefined, shuf);
    return @bitCast(intCast_i64x2(lo));
}


pub inline fn _mm_cvtepi16_epi32 (a: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 0, 1, 2, 3 };
    const lo = @shuffle(i16, bitCast_i16x8(a), undefined, shuf);
    return @bitCast(intCast_i32x4(lo));
}


pub inline fn _mm_cvtepi8_epi64 (a: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 1 };
    const lo = @shuffle(i8, bitCast_i8x16(a), undefined, shuf);
    return @bitCast(intCast_i64x2(lo));
}


pub inline fn _mm_cvtepi8_epi32 (a: __m128i) __m128i {
    const shuf: @Vector(4, i32) = .{ 0, 1, 2, 3 };
    const lo = @shuffle(i8, bitCast_i8x16(a), undefined, shuf);
    return @bitCast(intCast_i32x4(lo));
}


pub inline fn _mm_cvtepi8_epi16 (a: __m128i) __m128i {
    const shuf: @Vector(8, i32) = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const lo = @shuffle(i8, bitCast_i8x16(a), undefined, shuf);
    return @bitCast(intCast_i16x8(lo));
}


pub inline fn _mm_max_epu32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u32x4(a), bitCast_u32x4(b)));
}


pub inline fn _mm_max_epu16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u16x8(a), bitCast_u16x8(b)));
}


pub inline fn _mm_max_epu8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_u8x16(a), bitCast_u8x16(b)));
}


pub inline fn _mm_max_epi32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i32x4(a), bitCast_i32x4(b)));
}


pub inline fn _mm_max_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i16x8(a), bitCast_i16x8(b)));
}


pub inline fn _mm_max_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@max(bitCast_i8x16(a), bitCast_i8x16(b)));
}


pub inline fn _mm_min_epu32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u32x4(a), bitCast_u32x4(b)));
}


pub inline fn _mm_min_epu16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u16x8(a), bitCast_u16x8(b)));
}


pub inline fn _mm_min_epu8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_u8x16(a), bitCast_u8x16(b)));
}


pub inline fn _mm_min_epi32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i32x4(a), bitCast_i32x4(b)));
}


pub inline fn _mm_min_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i16x8(a), bitCast_i16x8(b)));
}


pub inline fn _mm_min_epi8 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(@min(bitCast_i8x16(a), bitCast_i8x16(b)));
}


pub inline fn _mm_cmpeq_epi64 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i64x2(a) == bitCast_i64x2(b));
    const cmpInt : @Vector(2, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i64x2(cmpInt));
}


pub inline fn _mm_cmpeq_epi32 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i32x4(a) == bitCast_i32x4(b));
    const cmpInt : @Vector(4, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i32x4(cmpInt));
}


pub inline fn _mm_cmpeq_epi16 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i16x8(a) == bitCast_i16x8(b));
    const cmpInt : @Vector(8, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i16x8(cmpInt));
}


pub inline fn _mm_cmpeq_epi8 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i8x16(a) == bitCast_i8x16(b));
    const cmpInt : @Vector(16, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i8x16(cmpInt));
}


pub inline fn _mm_cmpgt_epi64 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i64x2(a) > bitCast_i64x2(b));
    const cmpInt : @Vector(2, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i64x2(cmpInt));
}


pub inline fn _mm_cmpgt_epi32 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i32x4(a) > bitCast_i32x4(b));
    const cmpInt : @Vector(4, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i32x4(cmpInt));
}


pub inline fn _mm_cmpgt_epi16 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i16x8(a) > bitCast_i16x8(b));
    const cmpInt : @Vector(8, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i16x8(cmpInt));
}


pub inline fn _mm_cmpgt_epi8 (a: __m128i, b: __m128i) __m128i {
    const cmpBool = (bitCast_i8x16(a) > bitCast_i8x16(b));
    const cmpInt : @Vector(16, i1) = @bitCast(@intFromBool(cmpBool));
    return @bitCast(intCast_i8x16(cmpInt));
}


pub inline fn _mm_cmplt_epi32 (a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi32(b, a);
}


pub inline fn _mm_cmplt_epi16 (a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi16(b, a);
}


pub inline fn _mm_cmplt_epi8 (a: __m128i, b: __m128i) __m128i {
    return _mm_cmpgt_epi8(b, a);
}


pub inline fn _mm_extract_epi64 (a: __m128i, comptime imm8: comptime_int) i64 {
    return bitCast_i64x2(a)[imm8];
}


pub inline fn _mm_extract_epi32 (a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_i32x4(a)[imm8];
}


/// zero-extends u16 to i32, as per C intrinsic
pub inline fn _mm_extract_epi16 (a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_u16x8(a)[imm8];
}


/// zero-extends u8 to i32, as per C intrinsic
pub inline fn _mm_extract_epi8 (a: __m128i, comptime imm8: comptime_int) i32 {
    return bitCast_u8x16(a)[imm8];
}


pub inline fn _mm_insert_epi64 (a: __m128i, i: i64, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i64x2(a);
    r[imm8] = i;
    return @bitCast(r);
}


pub inline fn _mm_insert_epi32 (a: __m128i, i: i32, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i32x4(a);
    r[imm8] = i;
    return @bitCast(r);
}


pub inline fn _mm_insert_epi16 (a: __m128i, i: i16, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i16x8(a);
    r[imm8] = i;
    return @bitCast(r);
}


pub inline fn _mm_insert_epi8 (a: __m128i, i: i8, comptime imm8: comptime_int) __m128i {
    var r = bitCast_i8x16(a);
    r[imm8] = i;
    return @bitCast(r);
}


pub inline fn _mm_hadd_epi32 (a: __m128i, b: __m128i) __m128i {
    const shuf_even: @Vector(4, i32) = .{ 0, 2, -1, -3 };
    const shuf_odd: @Vector(4, i32) = .{ 1, 3, -2, -4 };
    const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
    return @bitCast(even +% odd);
}


pub inline fn _mm_hadd_epi16 (a: __m128i, b: __m128i) __m128i {
    const shuf_even: @Vector(8, i32) = .{ 0, 2, 4, 6, -1, -3, -5, -7 };
    const shuf_odd: @Vector(8, i32) = .{ 1, 3, 5, 7, -2, -4, -6, -8 };
    const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
    return @bitCast(even +% odd);
}


pub inline fn _mm_hsub_epi32 (a: __m128i, b: __m128i) __m128i {
    const shuf_even: @Vector(4, i32) = .{ 0, 2, -1, -3 };
    const shuf_odd: @Vector(4, i32) = .{ 1, 3, -2, -4 };
    const even = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_even);
    const odd = @shuffle(u32, bitCast_u32x4(a), bitCast_u32x4(b), shuf_odd);
    return @bitCast(even -% odd);
}


pub inline fn _mm_hsub_epi16 (a: __m128i, b: __m128i) __m128i {
    const shuf_even: @Vector(8, i32) = .{ 0, 2, 4, 6, -1, -3, -5, -7 };
    const shuf_odd: @Vector(8, i32) = .{ 1, 3, 5, 7, -2, -4, -6, -8 };
    const even = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_even);
    const odd = @shuffle(u16, bitCast_u16x8(a), bitCast_u16x8(b), shuf_odd);
    return @bitCast(even -% odd);
}


pub inline fn _mm_hadds_epi16 (a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vphaddsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i)
            : [a] "x" (a), [b] "x" (b)
            : );
    } else if (has_ssse3) {
        var res = a;
        asm ("phaddsw %[b], %[a]"
            : [a] "+x" (res)
            : [b] "x" (b)
            : );
        return res;
    } else {
        const shuf_even: @Vector(8, i32) = .{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd: @Vector(8, i32) = .{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even +| odd);
    }
}


pub inline fn _mm_hsubs_epi16 (a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vphsubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i)
            : [a] "x" (a), [b] "x" (b)
            : );
    } else if (has_ssse3) {
        var res = a;
        asm ("phsubsw %[b], %[a]"
            : [a] "+x" (res)
            : [b] "x" (b)
            : );
        return res;
    } else {
        const shuf_even: @Vector(8, i32) = .{ 0, 2, 4, 6, -1, -3, -5, -7 };
        const shuf_odd: @Vector(8, i32) = .{ 1, 3, 5, 7, -2, -4, -6, -8 };
        const even = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_even);
        const odd = @shuffle(i16, bitCast_i16x8(a), bitCast_i16x8(b), shuf_odd);
        return @bitCast(even -| odd);
    }
}


pub inline fn _mm_avg_epu16 (a: __m128i, b: __m128i) __m128i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to pavgw
    const one: @Vector(8, u32) = @splat(1);
    const c = intCast_u32x8(bitCast_u16x8(a));
    const d = intCast_u32x8(bitCast_u16x8(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(@Vector(8, u16), @truncate(e)));
}


pub inline fn _mm_avg_epu8 (a: __m128i, b: __m128i) __m128i {
    // `r = (a | b) - ((a ^ b) >> 1)` isn't optimized to pavgb
    const one: @Vector(16, u16) = @splat(1);
    const c = intCast_u16x16(bitCast_u8x16(a));
    const d = intCast_u16x16(bitCast_u8x16(b));
    const e = (c +% d +% one) >> one;
    return @bitCast(@as(@Vector(16, u8), @truncate(e)));
}


pub inline fn _mm_blendv_epi8 (a: __m128i, b: __m128i, mask: __m128i) __m128i {
    const cmp = @as(@Vector(16, i8), @splat(0)) > bitCast_i8x16(mask);
    return @bitCast(@select(i8, cmp, bitCast_i8x16(b), bitCast_i8x16(a)));
}


pub inline fn _mm_blend_epi16 (a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    const mask = comptime blk: { // convert imm8 to vector of bools
        var m: @Vector(8, bool) = undefined;
        for (0..8) |i| {
            m[i] = (((imm8 >> i) & 1) == 1);
        }
        break :blk m;
    };
    return @bitCast(@select(i16, mask, bitCast_i16x8(b), bitCast_i16x8(a)));
}


pub inline fn _mm_shuffle_epi32 (a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf: @Vector(4, i32) = .{  imm8 & 3, (imm8 >> 2) & 3,
        (imm8 >> 4) & 3, (imm8 >> 6) & 3 };
    return @bitCast(@shuffle(i32, bitCast_i32x4(a), undefined, shuf));
}


pub inline fn _mm_shufflelo_epi16 (a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf: @Vector(8, i32) = .{ imm8 & 3, (imm8 >> 2) & 3,
        (imm8 >> 4) & 3, (imm8 >> 6) & 3, 4, 5, 6, 7 };
    return @bitCast(@shuffle(i16, bitCast_i16x8(a), undefined, shuf));
}


pub inline fn _mm_shufflehi_epi16 (a: __m128i, comptime imm8: comptime_int) __m128i {
    const shuf: @Vector(8, i32) = .{ 0, 1, 2, 3, 4 + (imm8 & 3),
        4 + ((imm8 >> 2) & 3), 4 + ((imm8 >> 4) & 3), 4 + ((imm8 >> 6) & 3) };
    return @bitCast(@shuffle(i16, bitCast_i16x8(a), undefined, shuf));
}


pub inline fn _mm_shuffle_epi8 (a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpshufb %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i)
            : [a] "x" (a), [b] "x" (b)
            : );
    } else if (has_ssse3) {
        var res = a;
        asm ("pshufb %[b], %[a]"
            : [a] "+x" (res)
            : [b] "x" (b)
            : );
        return res;
    } else { // !NOT TESTED!
        var r : @Vector(16, i8) = undefined;
        const shuf = bitCast_i8x16(b) & @as(@Vector(16, i8), @splat(0x0F));
        const mask = bitCast_i8x16(b) >> @as(@Vector(16, i8), @splat(7));
        for (0..16) |i| {
            r[i] = bitCast_i8x16(a)[@intCast(shuf[i])];
        }
        return @bitCast(~mask & r);
    }
}


pub inline fn _mm_alignr_epi8 (a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 31) { return @splat(0); }
    if (@as(u8, @intCast(imm8)) > 15) { return _mm_alignr_epi8(@splat(0), a, imm8 - 16); }

    const shuf = comptime blk: {
        var indices: @Vector(16, i32) = undefined;
        for (0..16) |i| {
            var x: i32 = @as(i32, @intCast(i)) + imm8;
            if (x > 15) { x = -x + 15; }
            indices[i] = x;
        }
        break :blk indices;
    };
    return @bitCast(@shuffle(u8, bitCast_u8x16(b), bitCast_u8x16(a), shuf));
}


pub inline fn _mm_slli_si128 (a: __m128i, comptime imm8: comptime_int) __m128i {
    if (@as(u8, @intCast(imm8)) > 15) { return @splat(0); }
    return _mm_alignr_epi8(a, @splat(0), 16 - imm8);
}


/// alternative name
pub inline fn _mm_bslli_si128 (a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_slli_si128(a, imm8);
}


pub inline fn _mm_srli_si128 (a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_alignr_epi8(@splat(0), a, imm8);
}


/// alternative name
pub inline fn _mm_bsrli_si128 (a: __m128i, comptime imm8: comptime_int) __m128i {
    return _mm_srli_si128(a, imm8);
}


pub inline fn _mm_mul_epi32 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 2 };
    const x = intCast_i64x2(@shuffle(i32, bitCast_i32x4(a), undefined, shuf));
    const y = intCast_i64x2(@shuffle(i32, bitCast_i32x4(b), undefined, shuf));
    return @bitCast(x *% y);
}


pub inline fn _mm_mul_epu32 (a: __m128i, b: __m128i) __m128i {
    const shuf: @Vector(2, i32) = .{ 0, 2 };
    const x = intCast_u64x2(@shuffle(u32, bitCast_u32x4(a), undefined, shuf));
    const y = intCast_u64x2(@shuffle(u32, bitCast_u32x4(b), undefined, shuf));
    return @bitCast(x *% y);
}


pub inline fn _mm_mullo_epi32 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i32x4(a) *% bitCast_i32x4(b));
}


pub inline fn _mm_mullo_epi16 (a: __m128i, b: __m128i) __m128i {
    return @bitCast(bitCast_i16x8(a) *% bitCast_i16x8(b));
}


pub inline fn _mm_madd_epi16 (a: __m128i, b: __m128i) __m128i {
    const r = intCast_i32x8(bitCast_i16x8(a)) *%
        intCast_i32x8(bitCast_i16x8(b));

    const shuf_even: @Vector(4, i32) = .{ 0, 2, 4, 6 };
    const shuf_odd: @Vector(4, i32) = .{ 1, 3, 5, 7 };
    const even = @shuffle(i32, r, undefined, shuf_even);
    const odd = @shuffle(i32, r, undefined, shuf_odd);
    return @bitCast(even +% odd);
}


pub inline fn  _mm_maddubs_epi16 (a: __m128i, b: __m128i) __m128i {
    if (has_avx) {
        return asm ("vpmaddubsw %[b], %[a], %[ret]"
            : [ret] "=x" (-> __m128i)
            : [a] "x" (a), [b] "x" (b)
            : );
    } else if (has_ssse3) {
        var res = a;
        asm ("pmaddubsw %[b], %[a]"
            : [a] "+x" (res)
            : [b] "x" (b)
            : );
        return res;
    } else { // !NOT TESTED! weird saturation rules... todo
        const r = bitCast_i16x16(intCast_u16x16(bitCast_u8x16(a))) *%
            intCast_i16x16(bitCast_i8x16(b));

        const shuf_even: @Vector(8, i32) = .{ 0, 2, 4, 6, 8, 10, 12, 14 };
        const shuf_odd: @Vector(8, i32) = .{ 1, 3, 5, 7, 9, 11, 13, 15 };
        const even = @shuffle(i32, r, undefined, shuf_even);
        const odd = @shuffle(i32, r, undefined, shuf_odd);
        return @bitCast(even +| odd);
    }
}
