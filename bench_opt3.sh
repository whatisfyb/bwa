#!/bin/bash
# bench_opt3.sh — Compare BWT performance: baseline vs software prefetch
# Run on Kunpeng:
#   bash bench_opt3.sh

echo "=== OPT-3: Baseline vs BWT Software Prefetch ==="
echo ""

# Compile baseline (no prefetch) — self-contained, no zlib dependency
echo "--- Compiling baseline (no OPT_BWT_PREFETCH) ---"
gcc -O3 -march=armv8.2-a -mtune=tsv110 -static test_opt3.c -o bench_opt3_base
if [ $? -ne 0 ]; then echo "Baseline compile failed"; exit 1; fi

# Compile prefetch version
echo "--- Compiling prefetch (OPT_BWT_PREFETCH) ---"
gcc -O3 -march=armv8.2-a -mtune=tsv110 -static -DOPT_BWT_PREFETCH test_opt3.c -o bench_opt3_pf
if [ $? -ne 0 ]; then echo "Prefetch compile failed"; exit 1; fi

echo ""
echo "--- Running baseline ---"
./bench_opt3_base 2>&1 | tee /tmp/opt3_base.txt

echo ""
echo "--- Running prefetch ---"
./bench_opt3_pf 2>&1 | tee /tmp/opt3_pf.txt

echo ""
echo "=== Speedup Comparison ==="

# Extract timing lines: "bwt_func  iters  time_per_iter_us"
base_2occ4=$(grep "^bwt_2occ4" /tmp/opt3_base.txt | awk '{print $3}')
pf_2occ4=$(grep "^bwt_2occ4" /tmp/opt3_pf.txt | awk '{print $3}')
base_extend=$(grep "^bwt_extend" /tmp/opt3_base.txt | awk '{print $3}')
pf_extend=$(grep "^bwt_extend" /tmp/opt3_pf.txt | awk '{print $3}')
base_sa=$(grep "^bwt_sa" /tmp/opt3_base.txt | awk '{print $3}')
pf_sa=$(grep "^bwt_sa" /tmp/opt3_pf.txt | awk '{print $3}')

echo "func         base(us)  prefetch(us)  speedup"
if [ -n "$base_2occ4" ] && [ -n "$pf_2occ4" ] && [ "$base_2occ4" != "0" ]; then
    echo "bwt_2occ4    $base_2occ4  $pf_2occ4  $(awk "BEGIN {printf \"%.2f\", $base_2occ4/$pf_2occ4}")"
fi
if [ -n "$base_extend" ] && [ -n "$pf_extend" ] && [ "$base_extend" != "0" ]; then
    echo "bwt_extend   $base_extend  $pf_extend  $(awk "BEGIN {printf \"%.2f\", $base_extend/$pf_extend}")"
fi
if [ -n "$base_sa" ] && [ -n "$pf_sa" ] && [ "$base_sa" != "0" ]; then
    echo "bwt_sa       $base_sa  $pf_sa  $(awk "BEGIN {printf \"%.2f\", $base_sa/$pf_sa}")"
fi

echo ""
echo "--- Correctness ---"
grep "Total:" /tmp/opt3_base.txt
grep "Total:" /tmp/opt3_pf.txt