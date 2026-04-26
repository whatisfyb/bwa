/*
 * bench_opt2.c — Benchmark bwt_occ4/bwt_2occ4: neon popcount vs cnt_table scalar
 *
 * Compile on ARM (Kunpeng):
 *   gcc -O3 -march=armv8.2-a -mtune=tsv110 -DOPT_BWT_OCC4_NEON \
 *       bench_opt2.c bwt.c ksw.c bntseq.c utils.c kstring.c malloc_wrap.c kopen.c \
 *       -lm -lz -lpthread -o bench_opt2
 *
 * Run:
 *   ./bench_opt2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "bwt.h"

static void init_bwt(bwt_t *bwt, bwtint_t seq_len) {
	bwt->primary = 0;
	bwt->seq_len = seq_len;
	bwt->bwt_size = (seq_len + 128) / 4;
	bwt->L2[0] = 0; bwt->L2[1] = seq_len/4; bwt->L2[2] = seq_len/2;
	bwt->L2[3] = seq_len*3/4; bwt->L2[4] = seq_len;
	bwt->bwt = calloc(bwt->bwt_size + 256, sizeof(uint32_t));
	bwt_gen_cnt_table(bwt);
	srand(42);
	for (bwtint_t i = 0; i < bwt->bwt_size + 256; i++)
		bwt->bwt[i] = rand();
	for (bwtint_t i = 0; i <= seq_len; i += OCC_INTERVAL) {
		uint32_t *p = bwt_occ_intv(bwt, i);
		memset(p, 0, sizeof(bwtint_t) * 4);
	}
	bwt->sa_intv = 1;
	bwt->n_sa = 0;
	bwt->sa = NULL;
}

static double bench_func_occ4(bwt_t *bwt, bwtint_t k, int n_iter,
                               void (*func)(const bwt_t*, bwtint_t, bwtint_t*)) {
	struct timespec ts1, ts2;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile bwtint_t sink;
	for (int i = 0; i < n_iter; i++) {
		bwtint_t cnt[4];
		func(bwt, k, cnt);
		sink = cnt[0];
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	return (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
}

static double bench_func_2occ4(bwt_t *bwt, bwtint_t k, bwtint_t l, int n_iter,
                                void (*func)(const bwt_t*, bwtint_t, bwtint_t, bwtint_t*, bwtint_t*)) {
	struct timespec ts1, ts2;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile bwtint_t sink;
	for (int i = 0; i < n_iter; i++) {
		bwtint_t cntk[4], cntl[4];
		func(bwt, k, l, cntk, cntl);
		sink = cntk[0];
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	return (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
}

int main(void) {
	printf("=== OPT-2 Benchmark: bwt_occ4_neon vs bwt_occ4_scalar ===\n\n");

	bwtint_t seq_len = 100000;
	bwt_t bwt;
	init_bwt(&bwt, seq_len);

	/* Correctness: compare neon vs scalar */
	printf("--- Correctness ---\n");
	int pass = 0, fail = 0;
	for (int i = 0; i < 2000; i++) {
		bwtint_t k = rand() % seq_len;
		bwtint_t cnt_s[4], cnt_n[4];
		bwt_occ4_scalar(&bwt, k, cnt_s);
#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
		bwt_occ4(&bwt, k, cnt_n);
#else
		memcpy(cnt_n, cnt_s, 16);
#endif
		if (cnt_s[0] != cnt_n[0] || cnt_s[1] != cnt_n[1] ||
		    cnt_s[2] != cnt_n[2] || cnt_s[3] != cnt_n[3]) {
			printf("FAIL: k=%lld scalar=[%lld,%lld,%lld,%lld] neon=[%lld,%lld,%lld,%lld]\n",
			       (long long)k, (long long)cnt_s[0], (long long)cnt_s[1],
			       (long long)cnt_s[2], (long long)cnt_s[3],
			       (long long)cnt_n[0], (long long)cnt_n[1],
			       (long long)cnt_n[2], (long long)cnt_n[3]);
			fail++;
			if (fail > 20) break;
		} else pass++;
	}

	/* bwt_2occ4 correctness */
	for (int i = 0; i < 1000; i++) {
		bwtint_t k = rand() % (seq_len / 2);
		bwtint_t l = k + rand() % 100;
		if (l >= seq_len) l = seq_len - 1;
		bwtint_t cntk_s[4], cntl_s[4], cntk_n[4], cntl_n[4];
		bwt_2occ4_scalar(&bwt, k, l, cntk_s, cntl_s);
#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
		bwt_2occ4(&bwt, k, l, cntk_n, cntl_n);
#else
		memcpy(cntk_n, cntk_s, 16); memcpy(cntl_n, cntl_s, 16);
#endif
		int ok = 1;
		for (int j = 0; j < 4; j++)
			if (cntk_s[j] != cntk_n[j] || cntl_s[j] != cntl_n[j]) ok = 0;
		if (!ok) {
			printf("FAIL: bwt_2occ4 k=%lld l=%lld\n", (long long)k, (long long)l);
			fail++;
		} else pass++;
	}
	printf("Correctness: %d pass, %d fail\n\n", pass, fail);

	/* Performance benchmark */
	printf("--- Performance ---\n");
	bwtint_t configs[][2] = {
		{100,    200},
		{500,    600},
		{1000,   1100},
		{5000,   5100},
		{10000,  10100},
	};
	int n_iter = 1000000;

	printf("func        k       l       iters    scalar(us)  neon(us)  speedup\n");
	printf("----------- ------- ------- -------- ----------  --------  -------\n");

	for (int c = 0; c < 5; c++) {
		bwtint_t k = configs[c][0], l = configs[c][1];

		double t_occ4_s = bench_func_occ4(&bwt, k, n_iter, bwt_occ4_scalar);
#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
		double t_occ4_n = bench_func_occ4(&bwt, k, n_iter, bwt_occ4);
#else
		double t_occ4_n = t_occ4_s;
#endif
		printf("bwt_occ4    %-7lld         %-8d  %-10.4f  %-8.4f  %.2fx\n",
		       (long long)k, n_iter,
		       t_occ4_s / n_iter * 1e6, t_occ4_n / n_iter * 1e6,
		       t_occ4_s / t_occ4_n);

		double t_2occ4_s = bench_func_2occ4(&bwt, k, l, n_iter, bwt_2occ4_scalar);
#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
		double t_2occ4_n = bench_func_2occ4(&bwt, k, l, n_iter, bwt_2occ4);
#else
		double t_2occ4_n = t_2occ4_s;
#endif
		printf("bwt_2occ4   %-7lld %-7lld %-8d  %-10.4f  %-8.4f  %.2fx\n",
		       (long long)k, (long long)l, n_iter,
		       t_2occ4_s / n_iter * 1e6, t_2occ4_n / n_iter * 1e6,
		       t_2occ4_s / t_2occ4_n);
	}

#if defined(__ARM_NEON) && defined(OPT_BWT_OCC4_NEON)
	printf("\nNEON popcount path active (OPT_BWT_OCC4_NEON defined)\n");
#else
	printf("\nScalar cnt_table fallback only\n");
#endif

	free(bwt.bwt);
	return fail > 0 ? 1 : 0;
}