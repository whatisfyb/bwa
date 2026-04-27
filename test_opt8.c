/*
 * test_opt8.c -- Verify bns_get_seq NEON produces identical results to scalar
 *   aarch64-linux-gnu-gcc -O2 -static -DOPT_BNS_GET_SEQ_NEON test_opt8.c -lm -o test_opt8
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_opt8
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON)
#include <arm_neon.h>
#endif

#define _set_pac(pac, l, c) ((pac)[(l)>>2] |= (c)<<((~(l)&3)<<1))
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)

static uint8_t *bns_get_seq_scalar(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len)
{
	uint8_t *seq = NULL;
	if (end < beg) { int64_t t = end; end = beg; beg = t; }
	if (end > l_pac<<1) end = l_pac<<1;
	if (beg < 0) beg = 0;
	if (beg >= l_pac || end <= l_pac) {
		int64_t k, l = 0;
		*len = end - beg;
		seq = malloc(end - beg);
		if (beg >= l_pac) {
			int64_t beg_f = (l_pac<<1) - 1 - end;
			int64_t end_f = (l_pac<<1) - 1 - beg;
			for (k = end_f; k > beg_f; --k)
				seq[l++] = 3 - _get_pac(pac, k);
		} else {
			for (k = beg; k < end; ++k)
				seq[l++] = _get_pac(pac, k);
		}
	} else {
		*len = 0;
	}
	return seq;
}

#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON)
static inline void bns_get_seq_neon_forward(const uint8_t *pac, int64_t beg, int64_t end, uint8_t *seq)
{
	int64_t k, l = 0;
	uint8x16_t mask_03 = vdupq_n_u8(0x03);
	int64_t align4 = (beg + 3) & ~(int64_t)3;
	for (k = beg; k < align4 && k < end; ++k)
		seq[l++] = _get_pac(pac, k);
	for (; k + 255 < end; k += 256) {
		uint8x16_t b0 = vld1q_u8(&pac[k >> 2]);
		uint8x16_t b1 = vld1q_u8(&pac[(k >> 2) + 16]);
		uint8x16_t b2 = vld1q_u8(&pac[(k >> 2) + 32]);
		uint8x16_t b3 = vld1q_u8(&pac[(k >> 2) + 48]);
		uint8x16x4_t out;
		out.val[0] = vandq_u8(vshrq_n_u8(b0, 6), mask_03);
		out.val[1] = vandq_u8(vshrq_n_u8(b0, 4), mask_03);
		out.val[2] = vandq_u8(vshrq_n_u8(b0, 2), mask_03);
		out.val[3] = vandq_u8(b0, mask_03);
		vst4q_u8(seq + l, out); l += 64;
		out.val[0] = vandq_u8(vshrq_n_u8(b1, 6), mask_03);
		out.val[1] = vandq_u8(vshrq_n_u8(b1, 4), mask_03);
		out.val[2] = vandq_u8(vshrq_n_u8(b1, 2), mask_03);
		out.val[3] = vandq_u8(b1, mask_03);
		vst4q_u8(seq + l, out); l += 64;
		out.val[0] = vandq_u8(vshrq_n_u8(b2, 6), mask_03);
		out.val[1] = vandq_u8(vshrq_n_u8(b2, 4), mask_03);
		out.val[2] = vandq_u8(vshrq_n_u8(b2, 2), mask_03);
		out.val[3] = vandq_u8(b2, mask_03);
		vst4q_u8(seq + l, out); l += 64;
		out.val[0] = vandq_u8(vshrq_n_u8(b3, 6), mask_03);
		out.val[1] = vandq_u8(vshrq_n_u8(b3, 4), mask_03);
		out.val[2] = vandq_u8(vshrq_n_u8(b3, 2), mask_03);
		out.val[3] = vandq_u8(b3, mask_03);
		vst4q_u8(seq + l, out); l += 64;
	}
	for (; k + 3 < end; k += 4) {
		uint8_t b = pac[k >> 2];
		seq[l++] = b >> 6 & 3;
		seq[l++] = b >> 4 & 3;
		seq[l++] = b >> 2 & 3;
		seq[l++] = b      & 3;
	}
	for (; k < end; ++k)
		seq[l++] = _get_pac(pac, k);
}

static inline void bns_get_seq_neon_reverse(const uint8_t *pac, int64_t beg_f, int64_t end_f, uint8_t *seq)
{
	int64_t len = end_f - beg_f;
	bns_get_seq_neon_forward(pac, beg_f, end_f, seq);
	for (int64_t i = 0, j = len - 1; i < j; ++i, --j) {
		uint8_t tmp = 3 - seq[i];
		seq[i] = 3 - seq[j];
		seq[j] = tmp;
	}
	if (len & 1)
		seq[len / 2] = 3 - seq[len / 2];
}

static uint8_t *bns_get_seq_neon(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len)
{
	uint8_t *seq = NULL;
	if (end < beg) { int64_t t = end; end = beg; beg = t; }
	if (end > l_pac<<1) end = l_pac<<1;
	if (beg < 0) beg = 0;
	if (beg >= l_pac || end <= l_pac) {
		*len = end - beg;
		seq = malloc(end - beg);
		if (beg >= l_pac) {
			bns_get_seq_neon_reverse(pac, (l_pac<<1) - end, (l_pac<<1) - beg, seq);
		} else {
			bns_get_seq_neon_forward(pac, beg, end, seq);
		}
	} else {
		*len = 0;
	}
	return seq;
}
#endif

int main(void) {
	printf("=== OPT-8 bns_get_seq NEON correctness test ===\n\n");
	int pass = 0, fail = 0;
	srand(42);

	for (int trial = 0; trial < 500; trial++) {
		int64_t l_pac = 100 + rand() % 9900;
		uint8_t *pac = calloc((l_pac + 3) / 4, sizeof(uint8_t));
		for (int64_t i = 0; i < l_pac; i++)
			_set_pac(pac, i, rand() % 4);
		int64_t beg = rand() % (l_pac / 2);
		int64_t end = beg + 1 + rand() % (l_pac / 2);
		if (end > l_pac) end = l_pac;
		int64_t len_s, len_n;
		uint8_t *seq_s = bns_get_seq_scalar(l_pac, pac, beg, end, &len_s);
#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON)
		uint8_t *seq_n = bns_get_seq_neon(l_pac, pac, beg, end, &len_n);
#else
		int64_t len_n2;
		uint8_t *seq_n = bns_get_seq_scalar(l_pac, pac, beg, end, &len_n2);
		len_n = len_n2;
#endif
		if (len_s != len_n) {
			printf("FAIL fwd trial=%d: len mismatch %ld vs %ld\n", trial, (long)len_s, (long)len_n);
			fail++;
		} else {
			int ok = 1;
			for (int64_t i = 0; i < len_s; i++) {
				if (seq_s[i] != seq_n[i]) {
					printf("FAIL fwd trial=%d beg=%ld end=%ld idx=%ld scalar=%d neon=%d\n",
					       trial, (long)beg, (long)end, (long)i, seq_s[i], seq_n[i]);
					ok = 0; break;
				}
			}
			if (ok) pass++; else fail++;
		}
		free(seq_s); free(seq_n); free(pac);
	}

	for (int trial = 0; trial < 500; trial++) {
		int64_t l_pac = 100 + rand() % 9900;
		uint8_t *pac = calloc((l_pac + 3) / 4, sizeof(uint8_t));
		for (int64_t i = 0; i < l_pac; i++)
			_set_pac(pac, i, rand() % 4);
		int64_t beg = l_pac + rand() % (l_pac / 2);
		int64_t end = beg + 1 + rand() % (l_pac / 2);
		if (end > l_pac * 2) end = l_pac * 2;
		int64_t len_s, len_n;
		uint8_t *seq_s = bns_get_seq_scalar(l_pac, pac, beg, end, &len_s);
#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON)
		uint8_t *seq_n = bns_get_seq_neon(l_pac, pac, beg, end, &len_n);
#else
		int64_t len_n2;
		uint8_t *seq_n = bns_get_seq_scalar(l_pac, pac, beg, end, &len_n2);
		len_n = len_n2;
#endif
		if (len_s != len_n) {
			printf("FAIL rev trial=%d: len mismatch %ld vs %ld\n", trial, (long)len_s, (long)len_n);
			fail++;
		} else {
			int ok = 1;
			for (int64_t i = 0; i < len_s; i++) {
				if (seq_s[i] != seq_n[i]) {
					printf("FAIL rev trial=%d beg=%ld end=%ld idx=%ld scalar=%d neon=%d\n",
					       trial, (long)beg, (long)end, (long)i, seq_s[i], seq_n[i]);
					ok = 0; break;
				}
			}
			if (ok) pass++; else fail++;
		}
		free(seq_s); free(seq_n); free(pac);
	}

	printf("\nResults: %d pass, %d fail\n", pass, fail);
#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON)
	printf("NEON path active\n");
#else
	printf("Scalar path only\n");
#endif
	return fail > 0 ? 1 : 0;
}