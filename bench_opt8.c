/*
 * bench_opt8.c -- Full-coverage benchmark: scalar vs NEON for bns_get_seq
 *
 * Single binary, tests both paths with identical inputs, verifies correctness.
 * Covers: forward strand, reverse strand, various offsets and lengths.
 *
 * Compile:
 *   aarch64-linux-gnu-gcc -O3 -static -DOPT_BNS_GET_SEQ_NEON_BENCH bench_opt8.c -lm -o bench_opt8
 * Or on x86 (scalar only, no NEON):
 *   gcc -O3 bench_opt8.c -lm -o bench_opt8
 *
 * Run:
 *   ./bench_opt8              # full benchmark
 *   ./bench_opt8 5000         # custom PAC size (millions)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#if defined(__ARM_NEON) && defined(OPT_BNS_GET_SEQ_NEON_BENCH)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#define _set_pac(pac, l, c) ((pac)[(l)>>2] |= (c)<<((~(l)&3)<<1))
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)

/* ----- Scalar reference (always compiled) ----- */

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

/* ----- NEON path (only with -DOPT_BNS_GET_SEQ_NEON_BENCH) ----- */

#if HAS_NEON
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
#endif /* HAS_NEON */

/* ----- Utilities ----- */

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ----- Correctness test ----- */

static int verify_correctness(int64_t l_pac, const uint8_t *pac, int ntrials)
{
	int pass = 0, fail = 0;
#if !HAS_NEON
	(void)l_pac; (void)pac; (void)ntrials;
	printf("NEON not available, skipping correctness check.\n");
	return 0;
#else
	srand(42);
	for (int t = 0; t < ntrials; t++) {
		int is_rev = rand() % 2;
		int64_t beg, end;
		if (is_rev) {
			beg = l_pac + rand() % (l_pac / 2);
			end = beg + 1 + rand() % (l_pac / 2);
			if (end > l_pac * 2) end = l_pac * 2;
		} else {
			beg = rand() % (l_pac / 2);
			end = beg + 1 + rand() % (l_pac / 2);
			if (end > l_pac) end = l_pac;
		}
		int64_t len_s, len_n;
		uint8_t *seq_s = bns_get_seq_scalar(l_pac, pac, beg, end, &len_s);
		uint8_t *seq_n = bns_get_seq_neon(l_pac, pac, beg, end, &len_n);
		if (len_s != len_n) {
			printf("  FAIL %s trial=%d beg=%ld end=%ld: len %ld vs %ld\n",
			       is_rev ? "rev" : "fwd", t, (long)beg, (long)end, (long)len_s, (long)len_n);
			fail++;
		} else {
			int ok = 1;
			for (int64_t i = 0; i < len_s; i++) {
				if (seq_s[i] != seq_n[i]) {
					printf("  FAIL %s trial=%d beg=%ld end=%ld idx=%ld: %d vs %d\n",
					       is_rev ? "rev" : "fwd", t, (long)beg, (long)end, (long)i, seq_s[i], seq_n[i]);
					ok = 0; break;
				}
			}
			if (ok) pass++; else fail++;
		}
		free(seq_s); free(seq_n);
	}
	printf("  Correctness: %d pass, %d fail (fwd+rev, %d trials)\n", pass, fail, ntrials);
	return fail;
#endif
}

/* ----- Benchmark ----- */

typedef struct {
	double t_scalar;
	double t_neon;
	int64_t len;
	int niter;
} bench_result_t;

static bench_result_t bench_strand(int64_t l_pac, const uint8_t *pac,
                                   int64_t beg, int64_t end, int niter,
                                   int is_rev)
{
	bench_result_t r = {0, 0, end - beg, niter};
	volatile uint8_t sink = 0;

	double t0 = now_sec();
	for (int i = 0; i < niter; i++) {
		int64_t len;
		uint8_t *seq = bns_get_seq_scalar(l_pac, pac, beg, end, &len);
		sink += seq[0];
		free(seq);
	}
	r.t_scalar = now_sec() - t0;

#if HAS_NEON
	t0 = now_sec();
	for (int i = 0; i < niter; i++) {
		int64_t len;
		uint8_t *seq = bns_get_seq_neon(l_pac, pac, beg, end, &len);
		sink += seq[0];
		free(seq);
	}
	r.t_neon = now_sec() - t0;
	if (is_rev) (void)is_rev;
#endif

	(void)sink;
	return r;
}

int main(int argc, char *argv[])
{
	int64_t l_pac = 5000000;
	if (argc > 1) l_pac = atol(argv[1]) * 1000000;

	printf("=== OPT-8 bns_get_seq Full-Coverage Benchmark ===\n");
	printf("PAC length: %ld bases (%.1f MB)\n", (long)l_pac, (double)l_pac / 1e6);
#if HAS_NEON
	printf("NEON path: ENABLED\n");
#else
	printf("NEON path: DISABLED (scalar only)\n");
#endif
	printf("\n");

	uint8_t *pac = calloc((l_pac + 3) / 4, sizeof(uint8_t));
	srand(12345);
	for (int64_t i = 0; i < l_pac; i++)
		_set_pac(pac, i, rand() % 4);

	printf("--- Correctness verification ---\n");
	int fails = verify_correctness(l_pac, pac, 1000);
	if (fails > 0) {
		printf("CORRECTNESS FAILED, aborting benchmark.\n");
		free(pac);
		return 1;
	}
	printf("\n");

	int slens[] = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000};
	int niters[] = {10000, 10000, 5000, 2000, 1000, 200, 100, 20};
	int nslens = sizeof(slens) / sizeof(slens[0]);

	printf("--- Forward strand benchmark ---\n");
	printf("%-10s %-12s %-12s %-10s %-10s\n", "SeqLen", "Scalar(us)", "NEON(us)", "Speedup", "nIter");
	for (int li = 0; li < nslens; li++) {
		int slen = slens[li];
		int ni = niters[li];
		int64_t beg = (l_pac / 2 - slen / 2);
		int64_t end = beg + slen;
		bench_result_t r = bench_strand(l_pac, pac, beg, end, ni, 0);
#if HAS_NEON
		printf("%-10d %-12.1f %-12.1f %-10.2fx %-10d\n",
		       slen,
		       r.t_scalar / ni * 1e6,
		       r.t_neon / ni * 1e6,
		       r.t_neon > 0 ? r.t_scalar / r.t_neon : 0,
		       ni);
#else
		printf("%-10d %-12.1f %-12s %-10s %-10d\n",
		       slen, r.t_scalar / ni * 1e6, "N/A", "N/A", ni);
#endif
	}

	printf("\n--- Reverse strand benchmark ---\n");
	printf("%-10s %-12s %-12s %-10s %-10s\n", "SeqLen", "Scalar(us)", "NEON(us)", "Speedup", "nIter");
	for (int li = 0; li < nslens; li++) {
		int slen = slens[li];
		int ni = niters[li];
		int64_t beg = l_pac + (l_pac / 2 - slen / 2);
		int64_t end = beg + slen;
		if (end > l_pac * 2) end = l_pac * 2;
		bench_result_t r = bench_strand(l_pac, pac, beg, end, ni, 1);
#if HAS_NEON
		printf("%-10d %-12.1f %-12.1f %-10.2fx %-10d\n",
		       slen,
		       r.t_scalar / ni * 1e6,
		       r.t_neon / ni * 1e6,
		       r.t_neon > 0 ? r.t_scalar / r.t_neon : 0,
		       ni);
#else
		printf("%-10d %-12.1f %-12s %-10s %-10d\n",
		       slen, r.t_scalar / ni * 1e6, "N/A", "N/A", ni);
#endif
	}

	printf("\n--- Aligned vs unaligned offsets (fwd, len=5000) ---\n");
	int64_t offsets[] = {0, 1, 2, 3, 7, 15, 63, 64, 65, 127, 128, 255, 256};
	int noff = sizeof(offsets) / sizeof(offsets[0]);
	printf("%-10s %-12s %-12s %-10s\n", "Offset", "Scalar(us)", "NEON(us)", "Speedup");
	for (int oi = 0; oi < noff; oi++) {
		int64_t beg = offsets[oi];
		int64_t end = beg + 5000;
		if (end > l_pac) end = l_pac;
		bench_result_t r = bench_strand(l_pac, pac, beg, end, 2000, 0);
#if HAS_NEON
		printf("%-10ld %-12.1f %-12.1f %-10.2fx\n",
		       (long)offsets[oi],
		       r.t_scalar / r.niter * 1e6,
		       r.t_neon / r.niter * 1e6,
		       r.t_neon > 0 ? r.t_scalar / r.t_neon : 0);
#else
		printf("%-10ld %-12.1f %-12s %-10s\n",
		       (long)offsets[oi], r.t_scalar / r.niter * 1e6, "N/A", "N/A");
#endif
	}

	free(pac);
	return 0;
}