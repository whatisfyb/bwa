# OPT-5: ksw_global2 NEON Score-Only Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NEON-vectorize the score-only path of `ksw_global2` using int32x4_t 4-lane striped DP with stack-buffer F propagation (opt-9 pattern), achieving measurable speedup on Kunpeng A5520.

**Architecture:** Create `ksw_global2_neon()` as an independent function (compilation macro `OPT_KSW_GLOBAL2_NEON`). The function uses SoA layout (separate H[], E[] arrays) for the NEON inner loop, processing 4 cells per iteration. F propagation uses vst1q_s32 stack buffers read sequentially by scalar code. When CIGAR output is requested, fall back to the original scalar `ksw_global2`. A standalone benchmark `bench_opt5.c` validates correctness and measures per-call timing.

**Tech Stack:** ARM NEON intrinsics, int32x4_t, vst1q/vld1q, vbslq/vmaxq/vaddq, GCC with `-march=armv8.2-a`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ksw.c` | Add `ksw_global2_neon()` function after existing `ksw_global2()` (after line 642) |
| `ksw.h` | Add declaration for `ksw_global2_neon()` after line 84 |
| `bench_opt5.c` | Standalone benchmark: correctness checks + per-call timing for scalar vs NEON |

---

### Task 1: Add ksw_global2_neon declaration to ksw.h

**Files:**
- Modify: `ksw.h:84` (after existing `ksw_global2` declaration)

- [ ] **Step 1: Add function declaration**

Add after line 84 in `ksw.h`:

```c
#ifdef OPT_KSW_GLOBAL2_NEON
int ksw_global2_neon(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_);
#endif
```

- [ ] **Step 2: Commit**

```bash
git add ksw.h
git commit -m "OPT-5: add ksw_global2_neon declaration to ksw.h"
```

---

### Task 2: Implement ksw_global2_neon in ksw.c

**Files:**
- Modify: `ksw.c:642` (add after ksw_global2 function ends)

- [ ] **Step 1: Add ARM NEON include guard at top of ksw.c**

At the top of ksw.c (after existing includes, before any function definitions), add:

```c
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
```

- [ ] **Step 2: Write ksw_global2_neon function**

Insert after line 642 (after the closing brace of `ksw_global2`):

```c
#ifdef OPT_KSW_GLOBAL2_NEON
int ksw_global2_neon(int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins, int w, int *n_cigar_, uint32_t **cigar_)
{
	/* Fall back to scalar for CIGAR path — only optimize score-only */
	if (n_cigar_ && cigar_)
		return ksw_global2(qlen, query, tlen, target, m, mat, o_del, e_del, o_ins, e_ins, w, n_cigar_, cigar_);

	int32_t *H, *E;   /* SoA arrays: separate H and E */
	int8_t *qp;        /* query profile */
	int i, j, k, oe_del = o_del + e_del, oe_ins = o_ins + e_ins, score;
	int32_t minus_inf = MINUS_INF;

	if (n_cigar_) *n_cigar_ = 0;
	/* allocate SoA arrays */
	qp = malloc(qlen * m);
	H  = malloc((qlen + 1) * 4);
	E  = malloc((qlen + 1) * 4);
	/* generate query profile (same as scalar) */
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	/* fill first row (scalar, same logic as original) */
	H[0] = 0;  E[0] = minus_inf;
	for (j = 1; j <= qlen && j <= w; ++j) {
		H[j] = -(o_ins + e_ins * j);
		E[j] = minus_inf;
	}
	for (; j <= qlen; ++j) {
		H[j] = minus_inf;
		E[j] = minus_inf;
	}

	/* NEON constants */
	int32x4_t v_zero     = vdupq_n_s32(0);
	int32x4_t v_minus_inf = vdupq_n_s32(MINUS_INF);
	int32x4_t v_e_del    = vdupq_n_s32(e_del);
	int32x4_t v_oe_del   = vdupq_n_s32(oe_del);
	int32x4_t v_e_ins    = vdupq_n_s32(e_ins);
	int32x4_t v_oe_ins   = vdupq_n_s32(oe_ins);

	/* DP loop */
	for (i = 0; LIKELY(i < tlen); ++i) {
		int32_t f = minus_inf, h1, beg, end, t;
		int8_t *q = &qp[target[i] * qlen];
		beg = i > w? i - w : 0;
		end = i + w + 1 < qlen? i + w + 1 : qlen;
		h1 = beg == 0? -(o_del + e_del * (i + 1)) : minus_inf;

		/* NEON inner loop: process 4 cells at a time */
		j = beg;
		for (; LIKELY(j + 3 < end); j += 4) {
			int32x4_t vH_diag = vld1q_s32(&H[j]);
			int32x4_t vE_old  = vld1q_s32(&E[j]);
			int8x8_t  q8      = vld1_s8(&q[j]);
			int16x8_t q16     = vmovl_s8(q8);
			int32x4_t vq      = vmovl_s16(vget_low_s16(q16));
			/* M = H_diag + q (no "M? M+q : 0" branch needed for global —
			 * global alignment allows negative scores; H_diag may be MINUS_INF but
			 * max(M, E, F) still works because E/F are also >= MINUS_INF) */
			int32x4_t vM      = vaddq_s32(vH_diag, vq);
			/* H = max(M, E, F) — F not yet merged */
			int32x4_t vH_noF  = vmaxq_s32(vM, vE_old);
			/* E_new = max(E_old - e_del, M - oe_del) */
			int32x4_t vE_new  = vmaxq_s32(vsubq_s32(vE_old, v_e_del), vsubq_s32(vM, v_oe_del));
			/* F_init = max(M - oe_ins, F_prev - e_ins) done per-lane in scalar below */
			int32x4_t vF_init = vsubq_s32(vM, v_oe_ins);
			/* Store E_new to array */
			vst1q_s32(&E[j], vE_new);
			/* Store H_noF and F_init to stack buffers for scalar F propagation */
			int32_t h_buf[4], f_buf[4];
			vst1q_s32(h_buf, vH_noF);
			vst1q_s32(f_buf, vF_init);
			/* Scalar F propagation from stack buffers */
			t = f_buf[0]; f = f > t? f : t;  /* f = max(f_old, M[j]-oe_ins) */
			f = f - e_ins > 0? f - e_ins : minus_inf;  /* f = max(f-e_ins, 0-like) but global uses raw values */
			t = h_buf[0]; t = t > f? t : f;  /* H = max(H_noF, f) */
			H[j]   = h1; h1 = t;
			f -= e_ins;

			t = f_buf[1]; f = f > t? f : t;
			t = h_buf[1]; t = t > f? t : f;
			H[j+1] = h1; h1 = t;
			f -= e_ins;

			t = f_buf[2]; f = f > t? f : t;
			t = h_buf[2]; t = t > f? t : f;
			H[j+2] = h1; h1 = t;
			f -= e_ins;

			t = f_buf[3]; f = f > t? f : t;
			t = h_buf[3]; t = t > f? t : f;
			H[j+3] = h1; h1 = t;
			f -= e_ins;
		}
		/* Scalar tail for remaining <4 cells */
		for (; LIKELY(j < end); ++j) {
			int32_t h, M = H[j], e = E[j];
			H[j] = h1;
			M += q[j];
			h = M >= e? M : e;
			h = h >= f?  h : f;
			h1 = h;
			t = M - oe_del;
			e -= e_del;
			e  = e > t? e : t;
			E[j] = e;
			t = M - oe_ins;
			f -= e_ins;
			f  = f > t? f : t;
		}
		H[end] = h1; E[end] = minus_inf;
	}
	score = H[qlen];
	free(H); free(E); free(qp);
	return score;
}
#endif /* OPT_KSW_GLOBAL2_NEON */
```

- [ ] **Step 3: Commit**

```bash
git add ksw.c
git commit -m "OPT-5: implement ksw_global2_neon score-only path with int32x4_t striped DP"
```

---

### Task 3: Create bench_opt5.c

**Files:**
- Create: `bench_opt5.c`

- [ ] **Step 1: Write bench_opt5.c**

```c
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
        if (w > 100) w = 100;

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
        if (w > 100) w = 100;
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
```

- [ ] **Step 2: Commit**

```bash
git add bench_opt5.c
git commit -m "OPT-5: add bench_opt5.c correctness + performance benchmark"
```

---

### Task 4: Verify compilation on QEMU (x86 cross-compile)

**Files:** No changes — verification only

- [ ] **Step 1: Cross-compile scalar version**

```bash
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -static bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_base
```

Expected: compiles without errors

- [ ] **Step 2: Cross-compile NEON version**

```bash
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -static -DOPT_KSW_GLOBAL2_NEON bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_neon
```

Expected: compiles without errors

- [ ] **Step 3: Run base version on QEMU**

```bash
qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench_opt5_base
```

Expected: 200 pass, 0 fail; SCORES output printed

- [ ] **Step 4: Run NEON version on QEMU**

```bash
qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench_opt5_neon
```

Expected: 200 pass, 0 fail; SCORES identical to base version

- [ ] **Step 5: Fix any correctness issues if scores mismatch**

Debug by adding detailed per-trial output, then fix the NEON implementation. Iterate until 200/200 pass with identical scores.

---

### Task 5: Kunpeng testing and result recording

**Files:**
- Modify: `bwa-opt-branches.md` (add opt-5 data)
- Modify: `bwa-NEON优化计划（第二阶段）.md` (update results)

- [ ] **Step 1: Package and upload to Kunpeng**

```bash
cd /workdir
tar xzf bwa_opt5.tar.gz
cd bwa
gcc -O3 -march=armv8.2-a -mtune=tsv110 bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_base
gcc -O3 -march=armv8.2-a -mtune=tsv110 -DOPT_KSW_GLOBAL2_NEON bench_opt5.c ksw.c -lm -lz -lpthread -o bench_opt5_neon
./bench_opt5_base
./bench_opt5_neon
```

- [ ] **Step 2: Record results in bwa-opt-branches.md**

Add opt-5 row to the summary table and detailed section with Kunpeng test data.

- [ ] **Step 3: Update second-stage plan document**

Update OPT-5 results section in `bwa-NEON优化计划（第二阶段）.md`.

- [ ] **Step 4: Commit documentation updates**

```bash
git add bwa-opt-branches.md bwa-NEON优化计划（第二阶段）.md
git commit -m "OPT-5: record Kunpeng benchmark results"
```

---

## Self-Review Checklist

1. **Spec coverage**: Score-only path covered (Task 2), CIGAR fallback covered (first line of ksw_global2_neon), correctness test (Task 3/4), Kunpeng test (Task 5). All requirements addressed.

2. **Placeholder scan**: No TBDs, TODOs, or "implement later" steps. All code blocks contain complete implementation code.

3. **Type consistency**: `int32_t H[], E[]` arrays used consistently in ksw_global2_neon and referenced in bench_opt5. Function signature matches ksw.h declaration. MINUS_INF constant used consistently (-0x40000000). `int32x4_t` NEON type used for all vector operations.

4. **Potential issue**: The F propagation in the NEON loop needs careful review — global alignment uses MINUS_INF (not 0) as the "no value" sentinel, unlike ksw_extend2 which uses 0-clipping. The scalar F propagation code in the NEON block must handle `f -= e_ins` correctly even when f is MINUS_INF (MINUS_INF - e_ins won't overflow for int32). This is handled by using raw max comparisons without 0-clipping.