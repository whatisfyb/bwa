/*
 * bench_opt7.c — OPT-7: no-gap score accumulation NEON optimization
 *
 * The no-gap path in bwa_gen_cigar2 computes:
 *   score = sum(mat[rseq[i]*5 + query[i]] for i in 0..l_query-1)
 *
 * This is a scalar matrix lookup + accumulation. NEON optimization uses
 * vqtbl2q_u8 (2-table lookup covering 25-byte score matrix) to lookup
 * 16 scores in one instruction, then vaddvq_u8 to horizontally sum.
 *
 * Compile:
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -static bench_opt7.c -o bench_opt7_base
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -static -DOPT7_NEON_SCORE bench_opt7.c -o bench_opt7_neon
 *
 * Run:
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench_opt7_base
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench_opt7_neon
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>

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

/* Scalar version — original BWA code path */
static int score_scalar(const uint8_t *rseq, const uint8_t *query, int l_query)
{
    int score = 0;
    for (int i = 0; i < l_query; ++i)
        score += mat[rseq[i] * 5 + query[i]];
    return score;
}

#ifdef OPT7_NEON_SCORE
/* NEON version — vqtbl2q_u8 table lookup */
static int score_neon(const uint8_t *rseq, const uint8_t *query, int l_query)
{
    /* Shift scores to non-negative range for uint8 lookup */
    int8_t min_val = 0;
    for (int i = 0; i < 25; i++)
        if (mat[i] < min_val) min_val = mat[i];
    uint8_t shift = (uint8_t)(-min_val);

    /* Build 2-table for vqtbl2q_u8: covers 25 bytes of score matrix */
    uint8_t tbl0[16], tbl1[16];
    for (int i = 0; i < 16; i++)
        tbl0[i] = (uint8_t)(mat[i] + shift);
    for (int i = 0; i < 9; i++)
        tbl1[i] = (uint8_t)(mat[16 + i] + shift);
    for (int i = 9; i < 16; i++)
        tbl1[i] = shift; /* fill gaps with shifted-zero (= original 0) */
    uint8x16x2_t tbl = {{vld1q_u8(tbl0), vld1q_u8(tbl1)}};

    int total = 0;
    int i = 0;
    /* Process 16 positions at a time */
    for (; i + 16 <= l_query; i += 16) {
        uint8x16_t r = vld1q_u8(rseq + i);
        uint8x16_t q = vld1q_u8(query + i);
        /* Index = rseq[i] * 5 + query[i], max = 4*5+4 = 24 */
        uint8x16_t idx = vaddq_u8(vmulq_u8(r, vdupq_n_u8(5)), q);
        uint8x16_t scores = vqtbl2q_u8(tbl, idx);
        /* Clamp out-of-range indices (should not happen if rseq[i]<=4 && query[i]<=4) */
        /* vqtbl2q_u8 returns 0 for indices >= 32, but indices 25-31 still read tbl1[9-15]
         * which are shift (not original 0). We need to mask: any idx > 24 → score = shift (= orig 0) */
        /* Actually: for rseq[i]<=4 and query[i]<=4, idx <= 24, so no masking needed.
         * For N (value 4): idx = 4*5+4 = 24, which reads tbl1[8] = mat[24]+shift = 0+shift.
         * That's correct since mat[24]=0. */
        total += (int)vaddvq_u8(scores) - 16 * (int)shift;
    }
    /* Handle remaining positions with scalar */
    for (; i < l_query; i++)
        total += mat[rseq[i] * 5 + query[i]];
    return total;
}
#endif

static void rand_seq(uint8_t *s, int len) {
    for (int i = 0; i < len; i++) s[i] = rand() % 5;
}

int main(void)
{
#ifdef OPT7_NEON_SCORE
    printf("OPT7_MODE: NEON\n");
#else
    printf("OPT7_MODE: BASE\n");
#endif

    init_mat(1, 4);
    srand(42);

    /* ---- Correctness test ---- */
    printf("--- Correctness ---\n");
    int pass = 0, fail = 0;
    for (int trial = 0; trial < 200; trial++) {
        int len = 10 + rand() % 290;
        uint8_t *rseq = malloc(len);
        uint8_t *query = malloc(len);
        rand_seq(rseq, len);
        rand_seq(query, len);

        int s_base = score_scalar(rseq, query, len);
#ifdef OPT7_NEON_SCORE
        int s_neon = score_neon(rseq, query, len);
        if (s_base == s_neon) pass++;
        else { fail++; printf("FAIL trial %d: scalar=%d neon=%d len=%d\n", trial, s_base, s_neon, len); }
#else
        pass++;
#endif
        free(rseq); free(query);
    }
    printf("%d pass, %d fail\n", pass, fail);

    /* ---- Score output for cross-validation ---- */
    printf("\nSCORES:");
    for (int trial = 0; trial < 20; trial++) {
        int len = 10 + rand() % 290;
        uint8_t *rseq = malloc(len);
        uint8_t *query = malloc(len);
        rand_seq(rseq, len);
        rand_seq(query, len);
#ifdef OPT7_NEON_SCORE
        printf(" %d", score_neon(rseq, query, len));
#else
        printf(" %d", score_scalar(rseq, query, len));
#endif
        free(rseq); free(query);
    }
    printf("\n");

    /* ---- Performance benchmark ---- */
    printf("\n--- Performance ---\n");
    int configs[] = {32, 64, 100, 150, 200, 300};
    int n_configs = 6;
    int n_iter = 500000;

    printf("len     iters    base(us)  neon(us)  speedup\n");
    printf("------- -------- --------- --------- --------\n");

    for (int c = 0; c < n_configs; c++) {
        int len = configs[c];
        uint8_t *rseq = malloc(len);
        uint8_t *query = malloc(len);
        rand_seq(rseq, len);
        rand_seq(query, len);

        struct timespec ts1, ts2;
        volatile int sink;

#ifdef OPT7_NEON_SCORE
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < n_iter; i++)
            sink = score_neon(rseq, query, len);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t_neon = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
        double t_base_for_ratio = 0;
        /* Also run scalar for comparison within the same binary */
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < n_iter; i++)
            sink = score_scalar(rseq, query, len);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        t_base_for_ratio = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
        printf("%-7d %-8d %-9.3f %-9.3f %.2fx\n",
               len, n_iter,
               t_base_for_ratio / n_iter * 1e6,
               t_neon / n_iter * 1e6,
               t_base_for_ratio / t_neon);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < n_iter; i++)
            sink = score_scalar(rseq, query, len);
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t_base = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
        printf("%-7d %-8d %-9.3f %-9s %s\n",
               len, n_iter, t_base / n_iter * 1e6, "N/A", "run neon for ratio");
#endif

        free(rseq); free(query);
    }

    printf("\nNote: no-gap path is currently NEVER executed in BWA-MEM\n");
    printf("      (bwa.c:169 comment says this block is never reached)\n");
    printf("      This benchmark measures the potential speedup if it were.\n");

    return fail > 0 ? 1 : 0;
}