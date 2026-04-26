/*
 * test_opt3.c — Self-contained correctness test for OPT-3 BWT software prefetch
 *
 * Compile on ARM:
 *   gcc -O3 -march=armv8.2-a -static test_opt3.c -o test_opt3
 *   gcc -O3 -march=armv8.2-a -static -DOPT_BWT_PREFETCH test_opt3.c -o test_opt3_pf
 *
 * Run on QEMU:
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_opt3
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./test_opt3_pf
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef unsigned char ubyte_t;
typedef uint64_t bwtint_t;

#define OCC_INTV_SHIFT 7
#define OCC_INTERVAL   (1LL<<OCC_INTV_SHIFT)
#define OCC_INTV_MASK  (OCC_INTERVAL - 1)

#define bwt_bwt(b, k) ((b)->bwt[((k)>>7<<4) + sizeof(bwtint_t) + (((k)&0x7f)>>4)])
#define bwt_occ_intv(b, k) ((b)->bwt + ((k)>>7<<4))
#define bwt_B0(b, k) (bwt_bwt(b, k)>>((~(k)&0xf)<<1)&3)

typedef struct {
	bwtint_t primary; bwtint_t L2[5]; bwtint_t seq_len; bwtint_t bwt_size;
	uint32_t *bwt; uint32_t cnt_table[256]; int sa_intv; bwtint_t n_sa; bwtint_t *sa;
} bwt_t;

typedef struct { bwtint_t x[3], info; } bwtintv_t;

#define bwt_set_intv(bwt, c, ik) ((ik).x[0] = (bwt)->L2[(int)(c)]+1, (ik).x[2] = (bwt)->L2[(int)(c)+1]-(bwt)->L2[(int)(c)], (ik).x[1] = (bwt)->L2[3-(c)]+1, (ik).info = 0)

/* ---- BWT core functions (from bwt.c) ---- */

static void bwt_gen_cnt_table(bwt_t *bwt)
{
	int i, j;
	for (i = 0; i != 256; ++i) {
		uint32_t x = 0;
		for (j = 0; j != 4; ++j)
			x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);
		bwt->cnt_table[i] = x;
	}
}

#define __occ_aux4(bwt, b)											\
	((bwt)->cnt_table[(b)&0xff] + (bwt)->cnt_table[(b)>>8&0xff]		\
	 + (bwt)->cnt_table[(b)>>16&0xff] + (bwt)->cnt_table[(b)>>24])

static inline int __occ_aux(uint64_t y, int c)
{
	y = ((c&2)? y : ~y) >> 1 & ((c&1)? y : ~y) & 0x5555555555555555ull;
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

bwtint_t bwt_occ(const bwt_t *bwt, bwtint_t k, ubyte_t c)
{
	bwtint_t n;
	uint32_t *p, *end;
	if (k == bwt->seq_len) return bwt->L2[c+1] - bwt->L2[c];
	if (k == (bwtint_t)(-1)) return 0;
	k -= (k >= bwt->primary);
	n = ((bwtint_t*)(p = bwt_occ_intv(bwt, k)))[c];
	p += sizeof(bwtint_t);
	end = p + (((k>>5) - ((k&~OCC_INTV_MASK)>>5))<<1);
	for (; p < end; p += 2) n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);
	n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
	if (c == 0) n -= ~k&31;
	return n;
}

void bwt_occ4(const bwt_t *bwt, bwtint_t k, bwtint_t cnt[4])
{
	bwtint_t x;
	uint32_t *p, tmp, *end;
	if (k == (bwtint_t)(-1)) { memset(cnt, 0, 4 * sizeof(bwtint_t)); return; }
	k -= (k >= bwt->primary);
	p = bwt_occ_intv(bwt, k);
	memcpy(cnt, p, 4 * sizeof(bwtint_t));
	p += sizeof(bwtint_t);
	end = p + ((k>>4) - ((k&~OCC_INTV_MASK)>>4));
	for (x = 0; p < end; ++p) x += __occ_aux4(bwt, *p);
	tmp = *p & ~((1U<<((~k&15)<<1)) - 1);
	x += __occ_aux4(bwt, tmp) - (~k&15);
	cnt[0] += x&0xff; cnt[1] += x>>8&0xff; cnt[2] += x>>16&0xff; cnt[3] += x>>24;
}

void bwt_2occ4(const bwt_t *bwt, bwtint_t k, bwtint_t l, bwtint_t cntk[4], bwtint_t cntl[4])
{
	bwtint_t _k, _l;
	_k = k - (k >= bwt->primary);
	_l = l - (l >= bwt->primary);
	if (_l>>OCC_INTV_SHIFT != _k>>OCC_INTV_SHIFT || k == (bwtint_t)(-1) || l == (bwtint_t)(-1)) {
		bwt_occ4(bwt, k, cntk);
		bwt_occ4(bwt, l, cntl);
	} else {
		bwtint_t x, y;
		uint32_t *p, tmp, *endk, *endl;
		k -= (k >= bwt->primary);
		l -= (l >= bwt->primary);
		p = bwt_occ_intv(bwt, k);
		memcpy(cntk, p, 4 * sizeof(bwtint_t));
		p += sizeof(bwtint_t);
		endk = p + ((k>>4) - ((k&~OCC_INTV_MASK)>>4));
		endl = p + ((l>>4) - ((l&~OCC_INTV_MASK)>>4));
		for (x = 0; p < endk; ++p) x += __occ_aux4(bwt, *p);
		y = x;
		tmp = *p & ~((1U<<((~k&15)<<1)) - 1);
		x += __occ_aux4(bwt, tmp) - (~k&15);
		for (; p < endl; ++p) y += __occ_aux4(bwt, *p);
		tmp = *p & ~((1U<<((~l&15)<<1)) - 1);
		y += __occ_aux4(bwt, tmp) - (~l&15);
		memcpy(cntl, cntk, 4 * sizeof(bwtint_t));
		cntk[0] += x&0xff; cntk[1] += x>>8&0xff; cntk[2] += x>>16&0xff; cntk[3] += x>>24;
		cntl[0] += y&0xff; cntl[1] += y>>8&0xff; cntl[2] += y>>16&0xff; cntl[3] += y>>24;
	}
}

static inline bwtint_t bwt_invPsi(const bwt_t *bwt, bwtint_t k)
{
	bwtint_t x = k - (k > bwt->primary);
	x = bwt_B0(bwt, x);
	x = bwt->L2[x] + bwt_occ(bwt, k, x);
	return k == bwt->primary? 0 : x;
}

bwtint_t bwt_sa_func(const bwt_t *bwt, bwtint_t k)
{
	bwtint_t sa = 0, mask = bwt->sa_intv - 1;
#ifdef OPT_BWT_PREFETCH
	{
		bwtint_t pk = k - (k >= bwt->primary);
		__builtin_prefetch(bwt_occ_intv(bwt, pk), 0, 1);
	}
#endif
	while (k & mask) {
		++sa;
		k = bwt_invPsi(bwt, k);
#ifdef OPT_BWT_PREFETCH
		if (k & mask) {
			bwtint_t pk = k - (k >= bwt->primary);
			__builtin_prefetch(bwt_occ_intv(bwt, pk), 0, 1);
		}
#endif
	}
	return sa + bwt->sa[k/bwt->sa_intv];
}

void bwt_extend_func(const bwt_t *bwt, const bwtintv_t *ik, bwtintv_t ok[4], int is_back)
{
	bwtint_t tk[4], tl[4];
	int i;
	bwt_2occ4(bwt, ik->x[!is_back] - 1, ik->x[!is_back] - 1 + ik->x[2], tk, tl);
	for (i = 0; i != 4; ++i) {
		ok[i].x[!is_back] = bwt->L2[i] + 1 + tk[i];
		ok[i].x[2] = tl[i] - tk[i];
	}
#ifdef OPT_BWT_PREFETCH
	for (i = 0; i != 4; ++i) {
		if (ok[i].x[2] > 0) {
			bwtint_t next_k = ok[i].x[!is_back] - 1;
			next_k -= (next_k >= bwt->primary);
			__builtin_prefetch(bwt_occ_intv(bwt, next_k), 0, 1);
		}
	}
#endif
	ok[3].x[is_back] = ik->x[is_back] + (ik->x[!is_back] <= bwt->primary && ik->x[!is_back] + ik->x[2] - 1 >= bwt->primary);
	ok[2].x[is_back] = ok[3].x[is_back] + ok[3].x[2];
	ok[1].x[is_back] = ok[2].x[is_back] + ok[2].x[2];
	ok[0].x[is_back] = ok[1].x[is_back] + ok[1].x[2];
}

/* ---- BWT construction ---- */

static bwt_t *build_test_bwt(const char *seq)
{
	int len = strlen(seq);
	bwt_t *bwt = (bwt_t*)calloc(1, sizeof(bwt_t));

	int cnt[5] = {0};
	for (int i = 0; i < len; i++) {
		int c;
		switch(seq[i]) {
		case 'A': c=0; break; case 'C': c=1; break;
		case 'G': c=2; break; case 'T': c=3; break;
		default:  c=4; break;
		}
		cnt[c]++;
	}
	bwt->L2[0] = 0;
	for (int i = 1; i <= 4; i++) bwt->L2[i] = bwt->L2[i-1] + cnt[i-1];
	bwt->seq_len = len;

	/* Sort suffixes */
	int *sa_idx = (int*)malloc(len * sizeof(int));
	for (int i = 0; i < len; i++) sa_idx[i] = i;
	for (int i = 0; i < len; i++) {
		for (int j = i+1; j < len; j++) {
			int ci = sa_idx[i], cj = sa_idx[j];
			int cmp = 0, k;
			for (k = 0; ci+k < len && cj+k < len; k++) {
				char a = seq[ci+k], b = seq[cj+k];
				if (a < b) { cmp = -1; break; }
				if (a > b) { cmp = 1; break; }
			}
			if (cmp == 0) {
				if (ci+k < len) cmp = 1;
				else if (cj+k < len) cmp = -1;
			}
			if (cmp < 0) { int t = sa_idx[i]; sa_idx[i] = sa_idx[j]; sa_idx[j] = t; }
		}
	}

	/* Find primary: position where suffix starts at 0 */
	bwt->primary = 0;
	for (int i = 0; i < len; i++) {
		if (sa_idx[i] == 0) { bwt->primary = i; break; }
	}

	/* Build $-removed BWT string */
	int bwt_len = len - 1;
	uint8_t *bwt_str_nd = (uint8_t*)malloc(bwt_len);
	for (int i = 0; i < len; i++) {
		if (i == bwt->primary) continue;
		int out_pos = (i < bwt->primary) ? i : i - 1;
		int prev_pos = sa_idx[i] - 1;
		if (prev_pos < 0) prev_pos += len;
		switch(seq[prev_pos]) {
		case 'A': bwt_str_nd[out_pos] = 0; break;
		case 'C': bwt_str_nd[out_pos] = 1; break;
		case 'G': bwt_str_nd[out_pos] = 2; break;
		case 'T': bwt_str_nd[out_pos] = 3; break;
		default:  bwt_str_nd[out_pos] = 0; break;
		}
	}

	/* Allocate bwt->bwt: each OCC block has 16 uint32_t slots
	 * (8 for 4 bwtint_t header + 8 for 128 packed chars) */
	bwt->bwt_size = ((bwt_len + 3) / 4);
	int n_blocks = (bwt_len + OCC_INTERVAL - 1) / OCC_INTERVAL + 1;
	int bwt_alloc = n_blocks * 16 + 8;
	if (bwt_alloc < (int)(bwt->bwt_size)) bwt_alloc = bwt->bwt_size;
	bwt->bwt = (uint32_t*)calloc(bwt_alloc, sizeof(uint32_t));

	/* Fill OCC blocks */
	for (int block = 0; block <= (bwt_len + OCC_INTERVAL - 1) / OCC_INTERVAL; block++) {
		uint32_t *p = bwt_occ_intv(bwt, block * OCC_INTERVAL);
		bwtint_t occ_cnt[4] = {0};
		int block_start = block * OCC_INTERVAL;
		for (int i = 0; i < block_start && i < bwt_len; i++) {
			if (bwt_str_nd[i] < 4) occ_cnt[bwt_str_nd[i]]++;
		}
		memcpy(p, occ_cnt, 4 * sizeof(bwtint_t));
		uint32_t *bp = p + sizeof(bwtint_t);
		for (int j = 0; block_start + j < bwt_len && j < OCC_INTERVAL; j++) {
			bp[j / 16] |= (bwt_str_nd[block_start + j] & 3) << ((j % 16) * 2);
		}
	}

	bwt_gen_cnt_table(bwt);

	/* Build SA */
	bwt->sa_intv = 4;
	bwt->n_sa = (len + 4) / 4;
	bwt->sa = (bwtint_t*)calloc(bwt->n_sa, sizeof(bwtint_t));
	bwt->sa[0] = (bwtint_t)-1;
	for (int i = 0; i < len; i++) {
		if (i % bwt->sa_intv == 0 && i / bwt->sa_intv < bwt->n_sa) {
			bwt->sa[i / bwt->sa_intv] = len - sa_idx[i];
		}
	}

	free(bwt_str_nd);
	free(sa_idx);
	return bwt;
}

/* ---- Main test ---- */

int main(void)
{
	printf("=== OPT-3 Correctness Test: BWT Software Prefetch ===\n\n");

#ifdef OPT_BWT_PREFETCH
	printf("OPT_BWT_PREFETCH enabled\n\n");
#else
	printf("No prefetch (baseline)\n\n");
#endif

	const char *seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
	                  "TGCAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"
	                  "AACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
	                  "CACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

	bwt_t *bwt = build_test_bwt(seq);
	printf("BWT: seq_len=%lu, primary=%lu\n",
	       (unsigned long)bwt->seq_len, (unsigned long)bwt->primary);

	/* Test bwt_2occ4 */
	printf("\n--- bwt_2occ4 ---\n");
	int pass = 0, fail = 0;
	for (bwtint_t k = 0; k < bwt->seq_len; k += 7) {
		for (bwtint_t l = k; l < bwt->seq_len && l < k + 40; l += 5) {
			bwtint_t cntk[4], cntl[4];
			bwt_2occ4(bwt, k, l, cntk, cntl);
			int ok = 1;
			for (int i = 0; i < 4; i++) {
				if (cntl[i] < cntk[i]) { ok = 0; break; }
			}
			if (ok) pass++; else fail++;
		}
	}
	printf("bwt_2occ4: %d pass, %d fail\n", pass, fail);

	/* Test bwt_occ4 */
	printf("\n--- bwt_occ4 ---\n");
	int pass4 = 0, fail4 = 0;
	for (bwtint_t k = 0; k < bwt->seq_len; k += 3) {
		bwtint_t cnt[4];
		bwt_occ4(bwt, k, cnt);
		if (cnt[0] >= 0) pass4++; else fail4++;
	}
	printf("bwt_occ4: %d pass, %d fail\n", pass4, fail4);

	/* Test bwt_extend */
	printf("\n--- bwt_extend ---\n");
	int qlen = strlen(seq);
	uint8_t *query = (uint8_t*)malloc(qlen);
	for (int i = 0; i < qlen; i++) {
		switch(seq[i]) {
		case 'A': query[i]=0; break; case 'C': query[i]=1; break;
		case 'G': query[i]=2; break; case 'T': query[i]=3; break;
		default:  query[i]=4; break;
		}
	}
	int pass_ext = 0, fail_ext = 0;
	for (int x = 0; x < qlen; x++) {
		if (query[x] > 3) continue;
		bwtintv_t ik;
		bwt_set_intv(bwt, query[x], ik);
		bwtintv_t ok[4];
		bwt_extend_func(bwt, &ik, ok, 0);
		int all_ok = 1;
		for (int c = 0; c < 4; c++) {
			if (ok[c].x[2] > ik.x[2] + 1) all_ok = 0;
		}
		if (all_ok) pass_ext++; else fail_ext++;
	}
	printf("bwt_extend: %d pass, %d fail\n", pass_ext, fail_ext);

	/* Test bwt_sa */
	printf("\n--- bwt_sa ---\n");
	int pass_sa = 0, fail_sa = 0;
	for (bwtint_t k = 0; k < bwt->seq_len; k += 4) {
		bwtint_t sa = bwt_sa_func(bwt, k);
		if (sa >= 0 && sa < bwt->seq_len) pass_sa++;
		else {
			fail_sa++;
			printf("FAIL: bwt_sa(k=%lu) = %lu\n", (unsigned long)k, (unsigned long)sa);
		}
	}
	printf("bwt_sa: %d pass, %d fail\n", pass_sa, fail_sa);

	int total_pass = pass + pass4 + pass_ext + pass_sa;
	int total_fail = fail + fail4 + fail_ext + fail_sa;
	printf("\nTotal: %d pass, %d fail\n", total_pass, total_fail);

	/* Performance measurement */
	printf("\n--- Performance ---\n");
	int n_iter = 500000;

	/* bwt_2occ4 timing */
	struct timespec ts1, ts2;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile bwtint_t sink1, sink2;
	for (int i = 0; i < n_iter; i++) {
		bwtint_t k = (i * 17) % bwt->seq_len;
		bwtint_t l = k + ((i * 7) % 32);
		if (l >= bwt->seq_len) l = bwt->seq_len - 1;
		bwtint_t cntk[4], cntl[4];
		bwt_2occ4(bwt, k, l, cntk, cntl);
		sink1 = cntk[0]; sink2 = cntl[3];
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	double t_2occ4 = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;

	/* bwt_extend timing */
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile bwtint_t sink3;
	for (int i = 0; i < n_iter; i++) {
		int qi = i % qlen;
		if (query[qi] > 3) qi = 0;
		bwtintv_t ik;
		bwt_set_intv(bwt, query[qi], ik);
		bwtintv_t ok[4];
		bwt_extend_func(bwt, &ik, ok, 0);
		sink3 = ok[0].x[2];
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	double t_extend = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;

	/* bwt_sa timing */
	int n_sa_iter = n_iter / 10;
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	volatile bwtint_t sink4;
	for (int i = 0; i < n_sa_iter; i++) {
		bwtint_t k = (i * 31) % bwt->seq_len;
		bwtint_t sa = bwt_sa_func(bwt, k);
		sink4 = sa;
	}
	clock_gettime(CLOCK_MONOTONIC, &ts2);
	double t_sa = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;

	printf("bwt_2occ4  %d  %.4f\n", n_iter, t_2occ4 / n_iter * 1e6);
	printf("bwt_extend %d  %.4f\n", n_iter, t_extend / n_iter * 1e6);
	printf("bwt_sa     %d  %.4f\n", n_sa_iter, t_sa / n_sa_iter * 1e6);

	free(query);
	free(bwt->bwt);
	free(bwt->sa);
	free(bwt);
	return total_fail > 0 ? 1 : 0;
}