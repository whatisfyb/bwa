/*
 * test_occ_aux4.c — Self-contained correctness test for __occ_aux4_neon vs cnt_table
 *
 * Compile on ARM:
 *   gcc -O3 -march=armv8.2-a -DOPT_BWT_OCC4_NEON test_occ_aux4.c -o test_occ_aux4
 *
 * Run on QEMU:
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_occ_aux4
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ---- Inline copy of __occ_aux4_neon from bwt.c ---- */
#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
static inline uint32_t __occ_aux4_neon(uint32_t b)
{
	uint32_t low  = b & 0x55555555u;
	uint32_t high = (b >> 1) & 0x55555555u;
	uint32_t mask_A = (~low) & (~high) & 0x55555555u;
	uint32_t mask_C = low & (~high) & 0x55555555u;
	uint32_t mask_G = (~low) & high & 0x55555555u;
	uint32_t mask_T = low & high & 0x55555555u;
	uint32_t cnt_A = __builtin_popcount(mask_A);
	uint32_t cnt_C = __builtin_popcount(mask_C);
	uint32_t cnt_G = __builtin_popcount(mask_G);
	uint32_t cnt_T = __builtin_popcount(mask_T);
	return cnt_A | (cnt_C << 8) | (cnt_G << 16) | (cnt_T << 24);
}
#endif

/* ---- Inline copy of cnt_table generation from bwt.c ---- */
static uint32_t cnt_table[256];
static void gen_cnt_table(void)
{
	int i, j;
	for (i = 0; i != 256; ++i) {
		uint32_t x = 0;
		for (j = 0; j != 4; ++j)
			x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);
		cnt_table[i] = x;
	}
}

/* ---- Scalar reference: cnt_table lookup (same as __occ_aux4 macro) ---- */
static inline uint32_t __occ_aux4_scalar(uint32_t b)
{
	return cnt_table[(b)&0xff] + cnt_table[(b)>>8&0xff]
	       + cnt_table[(b)>>16&0xff] + cnt_table[(b)>>24];
}

int main(void)
{
	printf("=== OPT-2 Correctness: __occ_aux4_neon vs cnt_table ===\n\n");

	gen_cnt_table();

#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
	int pass = 0, fail = 0;

	/* Test 1: all 256 byte values */
	printf("--- Test 1: all 256 byte values (repeated in 4 bytes) ---\n");
	for (int i = 0; i < 256; i++) {
		uint32_t b = i | (i << 8) | (i << 16) | (i << 24);
		uint32_t neon_r = __occ_aux4_neon(b);
		uint32_t scalar_r = __occ_aux4_scalar(b);
		if (neon_r != scalar_r) {
			printf("FAIL: byte=0x%02x neon=0x%08x scalar=0x%08x\n", i, neon_r, scalar_r);
			fail++;
		} else pass++;
	}
	printf("256 byte values: %d pass, %d fail\n\n", pass, fail);

	/* Test 2: 10,000 random 32-bit words */
	printf("--- Test 2: 10,000 random 32-bit words ---\n");
	srand(42);
	int r_pass = 0, r_fail = 0;
	for (int i = 0; i < 10000; i++) {
		uint32_t b = rand();
		uint32_t neon_r = __occ_aux4_neon(b);
		uint32_t scalar_r = __occ_aux4_scalar(b);
		if (neon_r != scalar_r) {
			if (r_fail < 20)
				printf("FAIL: word=0x%08x neon=0x%08x scalar=0x%08x\n", b, neon_r, scalar_r);
			r_fail++;
		} else r_pass++;
	}
	printf("10,000 random words: %d pass, %d fail\n\n", r_pass, r_fail);

	/* Test 3: characteristic values */
	printf("--- Test 3: characteristic values ---\n");
	uint32_t special[] = {
		0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA,
		0x33333333, 0xCCCCCCCC, 0x0F0F0F0F, 0xF0F0F0F0,
		0x01010101, 0x02020202, 0x03030303,
	};
	int s_pass = 0, s_fail = 0;
	for (int i = 0; i < 11; i++) {
		uint32_t b = special[i];
		uint32_t neon_r = __occ_aux4_neon(b);
		uint32_t scalar_r = __occ_aux4_scalar(b);
		printf("0x%08x: neon[A=%d,C=%d,G=%d,T=%d] scalar[A=%d,C=%d,G=%d,T=%d] %s\n",
		       b, neon_r&0xff, (neon_r>>8)&0xff, (neon_r>>16)&0xff, (neon_r>>24)&0xff,
		       scalar_r&0xff, (scalar_r>>8)&0xff, (scalar_r>>16)&0xff, (scalar_r>>24)&0xff,
		       neon_r == scalar_r ? "OK" : "FAIL");
		if (neon_r != scalar_r) s_fail++;
		else s_pass++;
	}
	printf("Characteristic values: %d pass, %d fail\n\n", s_pass, s_fail);

	printf("Total: %d pass, %d fail\n", pass+r_pass+s_pass, fail+r_fail+s_fail);
#else
	printf("OPT_BWT_OCC4_NEON not defined — test skipped\n");
#endif

	return 0;
}