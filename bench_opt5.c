/*
 * bench_opt5.c — OPT-5: ksw_global2 NEON vs scalar benchmark
 *
 * Score-only path (n_cigar=NULL). Tests correctness by comparing
 * scalar and NEON scores, then measures per-call timing.
 *
 * Compile:
 *   gcc -O3 -march=armv8.2-a -mtune=tsv110 bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_base
 *   gcc -O3 -march=armv8.2-a -mtune=tsv110 -DOPT_KSW_GLOBAL2_NEON bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_neon
 *
 * Run:
 *   ./bench_opt5_base
 *   ./bench_opt5_neon
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "ksw.h"

static const int8_t default_mat[25] = {
    1, -4, -4, -4, 0,
    -4, 1, -4, -4, 0,
    -4, -4, 1, -4, 0,
    -4, -4, -4, 1, 0,
    0,  0,  0,  0,  0
};

static void rand_seq(uint8_t *s, int len) {
    for (int i = 0; i < len; i++) s[i] = rand() % 4;
}

int main(void)
{
#ifdef OPT_KSW_GLOBAL2_NEON
    printf("OPT5_MODE: NEON (ksw_global2_neon)\n");
#else
    printf("OPT5_MODE: BASE (ksw_global2 scalar)\n");
#endif

    int o_del = 6, e_del = 1, o_ins = 6, e_ins = 1;
    int m = 5, w;
    int n_trials = 200;
    int pass = 0, fail = 0;

    /* ---- Correctness test ---- */
    printf("--- Correctness ---\n");
    srand(42);
    for (int trial = 0; trial < n_trials; trial++) {
        int qlen = 20 + rand() % 280;
        int tlen = 20 + rand() % 280;
        w = qlen < tlen ? qlen : tlen;

        uint8_t *query  = malloc(qlen);
        uint8_t *target = malloc(tlen);
        rand_seq(query, qlen);
        rand_seq(target, tlen);

#ifdef OPT_KSW_GLOBAL2_NEON
        int s_neon = ksw_global2_neon(qlen, query, tlen, target, m, default_mat,
                                       o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        int s_base = ksw_global2(qlen, query, tlen, target, m, default_mat,
                                  o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        if (s_base == s_neon) pass++;
        else { fail++; printf("FAIL trial %d: scalar=%d neon=%d qlen=%d tlen=%d w=%d\n",
                              trial, s_base, s_neon, qlen, tlen, w); }
#else
        int s_base = ksw_global2(qlen, query, tlen, target, m, default_mat,
                                  o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        pass++;
#endif
        free(query); free(target);
    }
    printf("%d pass, %d fail\n", pass, fail);

    /* ---- Score output for cross-validation ---- */
    printf("\nSCORES:");
    for (int trial = 0; trial < 20; trial++) {
        int qlen = 30 + rand() % 270;
        int tlen = 30 + rand() % 270;
        w = qlen < tlen ? qlen : tlen;
        uint8_t *query  = malloc(qlen);
        uint8_t *target = malloc(tlen);
        rand_seq(query, qlen);
        rand_seq(target, tlen);
#ifdef OPT_KSW_GLOBAL2_NEON
        printf(" %d", ksw_global2_neon(qlen, query, tlen, target, m, default_mat,
                                        o_del, e_del, o_ins, e_ins, w, NULL, NULL));
#else
        printf(" %d", ksw_global2(qlen, query, tlen, target, m, default_mat,
                                  o_del, e_del, o_ins, e_ins, w, NULL, NULL));
#endif
        free(query); free(target);
    }
    printf("\n");

    /* ---- Performance benchmark ---- */
    printf("\n--- Performance ---\n");
    int configs[][3] = {
        /* qlen, tlen, w */
        {50,   50,   50},
        {100,  100,  80},
        {150,  150,  100},
        {200,  200,  150},
        {300,  300,  200},
        {500,  500,  300},
    };
    int n_configs = 6;
    int n_iter = 50000;

    printf("qlen  tlen  w    iters   scalar(us)  neon(us)  speedup\n");
    printf("----- ----- ---- ------  ----------  --------  -------\n");

    for (int c = 0; c < n_configs; c++) {
        int qlen = configs[c][0];
        int tlen = configs[c][1];
        w = configs[c][2];
        uint8_t *query  = malloc(qlen);
        uint8_t *target = malloc(tlen);
        rand_seq(query, qlen);
        rand_seq(target, tlen);

        struct timespec ts1, ts2;
        volatile int sink;

#ifdef OPT_KSW_GLOBAL2_NEON
        /* Run NEON */
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int it = 0; it < n_iter; it++)
            sink = ksw_global2_neon(qlen, query, tlen, target, m, default_mat,
                                     o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t_neon = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;

        /* Also run scalar for comparison */
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int it = 0; it < n_iter; it++)
            sink = ksw_global2(qlen, query, tlen, target, m, default_mat,
                               o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t_scalar = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;

        printf("%-5d %-5d %-4d %-6d  %-10.3f  %-8.3f  %.2fx\n",
               qlen, tlen, w, n_iter,
               t_scalar / n_iter * 1e6,
               t_neon / n_iter * 1e6,
               t_scalar / t_neon);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int it = 0; it < n_iter; it++)
            sink = ksw_global2(qlen, query, tlen, target, m, default_mat,
                               o_del, e_del, o_ins, e_ins, w, NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t_base = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
        printf("%-5d %-5d %-4d %-6d  %-10.3f  %-8s  %s\n",
               qlen, tlen, w, n_iter,
               t_base / n_iter * 1e6, "N/A", "run neon for ratio");
#endif
        free(query); free(target);
    }

    return fail > 0 ? 1 : 0;
}