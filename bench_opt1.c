/*
 * bench_opt1.c — Benchmark ksw_extend2_neon vs ksw_extend2_scalar
 *
 * Compile on ARM (Kunpeng):
 *   gcc -O3 -march=armv8.2-a -mtune=tsv110 -DOPT_KSW_EXTEND2_NEON \
 *       bench_opt1.c ksw.c -lm -lz -lpthread -o bench_opt1
 *
 * Run:
 *   ./bench_opt1
 *
 * Output: correctness check + per-call timing for scalar vs NEON
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ksw.h"

static int8_t mat[25];

static void init_mat(int match, int mismatch) {
	int i, j, k = 0;
	for (i = 0; i < 4; ++i) {
		for (j = 0; j < 4; ++j)
			mat[k++] = i == j ? match : -mismatch;
		mat[k++] = 0;
	}
	for (j = 0; j < 5; ++j) mat[k++] = 0;
}

static void rand_seq(uint8_t *s, int len) {
	for (int i = 0; i < len; i++) s[i] = rand() % 4;
}

static double bench_func(int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                         int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins,
                         int w, int h0, int n_iter,
                         int (*func)(int, const uint8_t*, int, const uint8_t*, int, const int8_t*,
                                     int, int, int, int, int, int, int, int, int*, int*, int*, int*, int*)) {
	struct timespec ts1, ts2;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile int sink;
	for (int i = 0; i < n_iter; i++) {
		int qle, tle, gtle, gscore, max_off;
		sink = func(qlen, query, tlen, target, m, mat, o_del, e_del, o_ins, e_ins, w, 0, 0, h0,
		            &qle, &tle, &gtle, &gscore, &max_off);
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	return (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
}

int main(void) {
	init_mat(1, 4);
	srand(42);

	printf("=== OPT-1 Benchmark: ksw_extend2_neon vs ksw_extend2_scalar ===\n\n");

	/* Correctness check */
	printf("--- Correctness ---\n");
	int pass = 0, fail = 0;
	for (int qlen = 20; qlen <= 200; qlen += 20) {
		for (int tlen = 20; tlen <= 200; tlen += 20) {
			uint8_t *query  = malloc(qlen);
			uint8_t *target = malloc(tlen);
			rand_seq(query, qlen);
			rand_seq(target, tlen);
			int w = qlen < tlen ? qlen : tlen;

			int qle_s, tle_s, gtle_s, gscore_s, max_off_s;
			int qle_n, tle_n, gtle_n, gscore_n, max_off_n;

			int score_s = ksw_extend2_scalar(qlen, query, tlen, target, 5, mat, 2, 1, 2, 1, w, 0, 0, 10,
			                                  &qle_s, &tle_s, &gtle_s, &gscore_s, &max_off_s);
#if defined(__ARM_NEON) && defined(OPT_KSW_EXTEND2_NEON)
			int score_n = ksw_extend2_neon(qlen, query, tlen, target, 5, mat, 2, 1, 2, 1, w, 0, 0, 10,
			                                &qle_n, &tle_n, &gtle_n, &gscore_n, &max_off_n);
#else
			int score_n = score_s; /* fallback: same function */
			qle_n = qle_s; tle_n = tle_s; gtle_n = gtle_s; gscore_n = gscore_s; max_off_n = max_off_s;
#endif

			if (score_s != score_n || qle_s != qle_n || tle_s != tle_n ||
			    gscore_s != gscore_n || max_off_s != max_off_n) {
				printf("FAIL: qlen=%d tlen=%d scalar=%d/%d neon=%d/%d\n",
				       qlen, tlen, score_s, qle_s, score_n, qle_n);
				fail++;
			} else {
				pass++;
			}
			free(query); free(target);
		}
	}
	printf("Correctness: %d pass, %d fail\n\n", pass, fail);

	/* Performance benchmark */
	printf("--- Performance ---\n");
	int configs[][3] = {
		/* qlen, tlen, w */
		{50,   50,   50},
		{100,  100,  80},
		{150,  150,  100},
		{200,  200,  150},
		{300,  300,  200},
		{500,  500,  300},
	};
	int n_iter = 50000;

	printf("qlen  tlen  w    iters   scalar(us)  neon(us)  speedup\n");
	printf("----- ----- ---- ------  ----------  --------  -------\n");

	for (int c = 0; c < 6; c++) {
		int qlen = configs[c][0], tlen = configs[c][1], w = configs[c][2];
		uint8_t *query  = malloc(qlen);
		uint8_t *target = malloc(tlen);
		rand_seq(query, qlen);
		rand_seq(target, tlen);

		double t_scalar = bench_func(qlen, query, tlen, target, 5, mat, 2, 1, 2, 1, w, 10, n_iter,
		                              ksw_extend2_scalar);
#if defined(__ARM_NEON) && defined(OPT_KSW_EXTEND2_NEON)
		double t_neon = bench_func(qlen, query, tlen, target, 5, mat, 2, 1, 2, 1, w, 10, n_iter,
		                           ksw_extend2_neon);
#else
		double t_neon = t_scalar;
#endif
		double speedup = t_scalar / t_neon;

		printf("%-5d %-5d %-4d %-6d  %-10.4f  %-8.4f  %.2fx\n",
		       qlen, tlen, w, n_iter,
		       t_scalar / n_iter * 1e6,
		       t_neon / n_iter * 1e6,
		       speedup);

		free(query); free(target);
	}

#if defined(__ARM_NEON) && defined(OPT_KSW_EXTEND2_NEON)
	printf("\nNEON path active (OPT_KSW_EXTEND2_NEON defined)\n");
#else
	printf("\nScalar fallback only (OPT_KSW_EXTEND2_NEON NOT defined)\n");
#endif

	return fail > 0 ? 1 : 0;
}