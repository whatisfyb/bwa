# BWA-MEM ARM NEON 优化分支总览

基于 commit `b92993c` (bwa-0.7.19 release)，共 9 个优化分支（opt-1 ~ opt-9，无 opt-5 → 现已创建）。

## 分支一览

| 分支 | 优化目标 | 修改文件 | 编译宏 | 鲲鹏实测效果 | 状态 |
|------|---------|---------|--------|------------|------|
| opt-1 | ksw_extend2 NEON SoA向量化 | ksw.c, ksw.h | `OPT_KSW_EXTEND2_NEON` | 鲲鹏1.08~1.21x | ✅ 有效 |
| opt-2 | bwt_occ4 NEON popcount | bwt.c, bwt.h | `OPT_BWT_OCC4_NEON` | 0.55~0.84x (更慢) | ❌ 负优化 |
| opt-3 | bwt_extend/bwt_sa 预取 | bwt.c | `OPT_BWT_PREFETCH` | 需真实数据测试 | ⚠️ 待验证 |
| opt-4 | neon_sse.h 安全修复 | neon_sse.h, ksw.c | 无（总是生效） | 修bug，非性能优化 | ✅ 修复合并到opt-6 |
| opt-6 | neon_sse.h安全 + lazy-F分析 | ksw.c, neon_sse.h | 无（总是生效） | lazy-F已是SIMD，无优化空间 | ❌ 否定结论 |
| opt-7 | vqtbl2q_u8得分累积 | bwa.c | `OPT7_NEON_SCORE` | 鲲鹏1.75~3.51x，但路径不触发 | ⚠️ 优化有效但不可达 |
| opt-5 | ksw_global2 NEON score-only路径 | ksw.c, ksw.h | `OPT_KSW_GLOBAL2_NEON` | 鲲鹏1.16~1.24x | ✅ 有效（需确认调用路径） |
| opt-8 | bns_get_seq PAC解包NEON | bntseq.c | `OPT_BNS_GET_SEQ_NEON` | 正向12-16x，反向8-11x | ✅ 有效 |
| opt-9 | ksw_extend2 NEON (F传播修复) | ksw.c, ksw.h | `OPT_KSW_EXTEND2_NEON` | 同opt-1 + bug修 | ✅ opt-1的修正版 |

## 各分支详细说明

### opt-1: ksw_extend2 NEON SoA向量化
- **修改**: ksw.c 中 ksw_extend2 函数
- **方法**: SoA (Structure of Arrays) 布局，H/E分为独立数组，4-lane NEON并行计算H/E/F
- **F传播**: vgetq_lane_s32 逐lane提取（后改为vst1q_s32栈缓冲，见opt-9修复）
- **编译**: `make CFLAGS="-DOPT_KSW_EXTEND2_NEON"`（需 `__ARM_NEON`）
- **效果**: 鲲鹏 A5520 实测 1.08~1.21x 加速（本地扩展DP比对）
- **注意**: F传播顺序有bug，已由opt-9修复

### opt-1 鲲鹏实测数据（2026-04-27）

正确性: 100/100 pass。

| qlen | tlen | w | scalar(us) | neon(us) | 加速比 |
|------|------|---|-----------|---------|--------|
| 50   | 50   | 50  | 1.1055    | 0.9155  | 1.21x  |
| 100  | 100  | 80  | 1.5103    | 1.3954  | 1.08x  |
| 150  | 150  | 100 | 1.4947    | 1.3478  | 1.11x  |
| 200  | 200  | 150 | 1.9073    | 1.7670  | 1.08x  |
| 300  | 300  | 200 | 2.4689    | 2.2018  | 1.12x  |
| 500  | 500  | 300 | 3.7898    | 3.2619  | 1.16x  |

加速比随参数规模上升而提高，小参数时NEON开销摊销不足。

### opt-2: bwt_occ4 NEON popcount（负优化）
- **修改**: bwt.c 中 `__occ_aux4` 函数
- **方法**: 用 `__builtin_popcount` 位操作替代 cnt_table 查表
- **编译**: `make CFLAGS="-DOPT_BWT_OCC4_NEON"`
- **效果**: 鲲鹏实测 0.55~0.84x，**比标量更慢**
- **原因**: cnt_table (1KB, L1常驻) 4路并行查表比15+指令popcount依赖链更快
- **结论**: 不可用，已否定

### opt-3: BWT预取优化
- **修改**: bwt.c 中 `bwt_extend` 和 `bwt_sa` 函数
- **方法**: `__builtin_prefetch` 预取下一步OCC block
- **编译**: `make CFLAGS="-DOPT_BWT_PREFETCH"`
- **效果**: 微基准测试无法体现，需真实BWA-MEM端到端测试
- **状态**: 待用真实基因组数据在鲲鹏上验证

### opt-4: neon_sse.h 安全修复
- **修改**: neon_sse.h, ksw.c
- **方法**: `_mm_slli_si128` 从依赖隐式 `zero` 变量的危险宏改为安全inline函数；新增 `_mm_srli_si128`
- **编译**: 无需宏，始终生效（`__ARM_NEON` 时）
- **效果**: 修bug非性能优化
- **状态**: 已合并到opt-6

### opt-6: lazy-F NEON分析（否定结论）
- **修改**: neon_sse.h 安全修复（同opt-4）+ ksw.c 恢复旧宏
- **方法**: 分析发现BWA lazy-F循环已用 `__m128i` SIMD，SSE bridge是零成本抽象
- **编译**: 无需宏
- **效果**: 无优化空间，lazy-F路径已向量化
- **结论**: 否定（lazy-F不是优化点）

### opt-5: ksw_global2 NEON score-only路径
- **修改**: ksw.c 中新增 `ksw_global2_neon` 函数，ksw.h 增加声明
- **方法**: SoA布局（独立H[]/E[]数组），int32x4_t 4-lane striped DP，vst1q_s32栈缓冲F传播；当请求CIGAR时回退到标量ksw_global2
- **编译**: `gcc -O3 -march=armv8.2-a -DOPT_KSW_GLOBAL2_NEON bench_opt5.c ksw.c -lm -lz -lpthread`
- **效果**: 鲲鹏 A5520 实测 1.16~1.24x 加速（score-only路径）
- **注意**: 仅优化score-only路径（n_cigar=NULL），CIGAR路径回退标量；需确认BWA-MEM中ksw_global2的调用方式

### opt-5 鲲鹏实测数据（2026-04-27）

正确性: 200/200 pass，SCORES 与标量完全一致。

| qlen | tlen | w | scalar(us) | neon(us) | 加速比 |
|------|------|---|-----------|---------|--------|
| 50   | 50   | 50  | 6.124     | 5.288   | 1.16x  |
| 100  | 100  | 80  | 22.983    | 18.583  | 1.24x  |
| 150  | 150  | 100 | 47.776    | 39.972  | 1.20x  |
| 200  | 200  | 150 | 88.084    | 70.831  | 1.24x  |
| 300  | 300  | 200 | 186.179   | 151.238 | 1.23x  |
| 500  | 500  | 300 | 484.895   | 400.891 | 1.21x  |

**调试教训**: F传播在栈缓冲中的顺序必须严格匹配标量版：先合并H与old-f，再更新f（f-=e, max(f, M-oe))。
错误顺序会导致分数偏差5-14，原因是`max(f-e, M-oe) ≠ max(f, M-oe)-e`（当M-oe>f时差e_ins）。

### opt-7: vqtbl2q_u8 得分累积
- **修改**: bwa.c 中 `bwa_gen_cigar2` 的no-gap路径
- **方法**: `vqtbl2q_u8` 双表查找 + `vaddvq_u8` 水平求和，替代逐位循环
- **编译**: `make CFLAGS="-DOPT7_NEON_SCORE"`
- **效果**: 鲲鹏 A5520 实测 1.75~3.51x 加速（随query长度增长加速比上升）
- **状态**: 优化本身有效，但 no-gap 路径在 BWA-MEM 中不触发（`w_==0` 条件未满足），端到端无法生效

### opt-7 鲲鹏实测数据（2026-04-27）

正确性: 200/200 pass，SCORES 与标量完全一致。

| len | base(us) | neon(us) | 加速比 |
|-----|----------|----------|--------|
| 32  | 0.032    | 0.018    | 1.75x  |
| 64  | 0.064    | 0.027    | 2.37x  |
| 100 | 0.101    | 0.037    | 2.76x  |
| 150 | 0.156    | 0.049    | 3.17x  |
| 200 | 0.205    | 0.064    | 3.22x  |
| 300 | 0.322    | 0.091    | 3.51x  |

**核心问题**: `bwa_gen_cigar2` 中 no-gap 路径（`w_==0` 分支）在 BWA-MEM 正常流程中从未被执行。
`w_` 由 `max_gap` 和 `min_w` 计算得出，实际输入中 `w_` 总 > 0，导致优化代码永远不可达。
需要解决路径可达性问题才能产生端到端加速。

### opt-8: bns_get_seq PAC解包NEON ★
- **修改**: bntseq.c 中 `bns_get_seq` 函数
- **方法**: `vst4q_u8` 4路交错解包2-bit PAC，256 bases/iteration；反向链先解包再in-place反转+补码
- **编译**: `make CFLAGS="-DOPT_BNS_GET_SEQ_NEON"`
- **效果**: 鲲鹏 A5520 实测 **正向12-16x，反向8-11x**
- **正确性**: 1000/1000 pass（fwd+rev随机+边界case）
- **状态**: ✅ 产效显著

### opt-9: ksw_extend2 NEON F传播修复
- **修改**: ksw.c, ksw.h（基于opt-1，修复F传播顺序bug）
- **方法**: vst1q_s32栈缓冲替代vgetq_lane_s32，保证F lane按正确顺序传播
- **编译**: `make CFLAGS="-DOPT_KSW_EXTEND2_NEON"`
- **效果**: 227/227 测试通过，修复opt-1的F传播错误
- **状态**: ✅ opt-1的bug修复版

## 编译宏速查

```bash
# opt-1 / opt-9
make CFLAGS="-g -Wall -Wno-unused-function -O3 -DOPT_KSW_EXTEND2_NEON"

# opt-2（不推荐，负优化）
make CFLAGS="-g -Wall -Wno-unused-function -O3 -DOPT_BWT_OCC4_NEON"

# opt-3
make CFLAGS="-g -Wall -Wno-unused-function -O3 -DOPT_BWT_PREFETCH"

# opt-4 / opt-6（始终生效，无需宏）

# opt-5
make CFLAGS="-g -Wall -Wno-unused-function -O3 -DOPT_KSW_GLOBAL2_NEON"

# opt-7

# opt-8
make CFLAGS="-g -Wall -Wno-unused-function -O3 -DOPT_BNS_GET_SEQ_NEON"
```

## 可合并到主线的有效优化

1. **opt-8** (bns_get_seq NEON) — 产效最显著，12-16x
2. **opt-9** (ksw_extend2 NEON F修复) — 修复合并opt-1
3. **opt-5** (ksw_global2 NEON score-only) — 鲲鹏1.16~1.24x，需确认BWA-MEM调用路径
4. **opt-7** (vqtbl2q得分累积) — 鲲鹏1.75~3.51x，但需解决路径不可达问题
5. **opt-3** (BWT预取) — 待端到端验证

## 已否定的优化

- **opt-2**: cnt_table优于popcount
- **opt-6**: lazy-F已向量化，无需额外优化
- **opt-4**: 安全修复已合并到opt-6

## 关于"优化代码永远不触发"的问题

各分支的NEON优化代码均通过 `#if defined(__ARM_NEON) && defined(OPT_XXX)` 编译守卫控制：
- 在 **x86** 上编译时，`__ARM_NEON` 未定义，NEON代码不编译，走标量路径
- 在 **ARM** 上编译时，需手动添加 `-DOPT_XXX` 宏才会启用NEON路径
- **opt-4/6** 的neon_sse.h修复是唯一的例外：它仅在 `__ARM_NEON` 时生效，不需要额外宏

**如果在鲲鹏上只 `make` 不加任何 `-DOPT_XXX` 标志，所有NEON优化都不会被启用，BWA运行的是标量代码。** 这是设计意图——确保未优化路径始终可用作对照。