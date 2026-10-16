# PLoP Explorer: Full Implementation Plan (every experiment in the paper)

Paper: Hayou, Ghosh, Yu, "PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models", arXiv:2506.20629v1.
This plan supersedes the experiment sections of `plop-strategy.md`. Product decisions there still stand.

Conventions: "Target" means a number or set the paper publishes that our code must reproduce. "Unknown" means the paper does not say, so we must test or choose, and record the choice.

**Product decisions (these override `plop-strategy.md` where the two differ):** v1 is Pages 1-2 and Page 3 is v2; hosting on a Hugging Face Space (Docker, since the backend is FastAPI); FastAPI backend with a custom HTML/JS frontend and Plotly.js charts (not Streamlit or Gradio); timeline is ongoing with no deadline.

---

## 0. Status (updated 2026-09-29)

### Gates

| Gate | Status | Evidence |
|---|---|---|
| G1 parameter counter | **Passed** | `plop_explorer/lora_params.py`; `tests/test_lora_params.py` reproduces all 16 #Params values; architecture sizes checked against each model's real `config.json` |
| G2 scoring matches the authors | **Passed with a documented gap** | Recipe read from the authors' code (Section 3.2); Llama-1B math ranking and top-3 match; mean error 2.5% (Section 3.3 A) |
| G3 selection sets | **Passed: 5 of 5 models** on math (Gemma = `-it`) | `scripts/check_selection.py` (Section 3.3 B); asserted by `tests/test_verified.py` against `reference/ours/` |
| G4 E1 linear network | Not started | |
| G5 dashboard | Not started; unblocked by G2 and G3 | |
| G6-G9 | Not started | |

### What is built

| File | What it does |
|---|---|
| `plop_explorer/nfn.py` | Forward-hook NFN scoring with the exact baseline, padding mask on or off, per-type aggregation, top-k pick. Raises an error on non-finite scores (fp16 overflow). |
| `plop_explorer/data.py` | Seeded prompt sets for math, code, history and logic, matching the authors' sources and formatting. |
| `plop_explorer/lora_params.py` | Parameter counts per placement, `match_rank` for equal budgets, `PLACEMENTS` (Attn = Q, K, V). |
| `scripts/check_reference.py` | Llama-1B math against the paper's 7-number table, in the reference setup and in our masked setup. |
| `scripts/check_selection.py` | PLoP and PLoP^-1 picks for 5 models x 4 tasks against the paper's sets; retries in fp32 when fp16 overflows. |
| `tests/` | 24 tests: module naming, random inputs score about 1, aligned inputs score above 1, padding mask, heatmap grid, exact vs sampled baseline, 16 paper parameter counts, ANLI ranks. |

### Current assessment

| Area | Score | Why |
|---|---|---|
| Correctness and rigor | 8/10 | Matches the authors' code and the paper's picks; tests; gaps documented |
| What a visitor can see | 1/10 | No README, charts or demo link yet |
| Repo hygiene | 3/10 | Tests and clean commits; no README, license, topics or CI |
| Originality | 3/10 | Careful reproduction so far; X1 and X4 are where it becomes ours |
| Overall as a portfolio piece | 4/10 | Strong foundation, nothing visible yet |

### Next steps, in order
The full execution plan, with deliverables and acceptance checks per workstream, is Section 14. Summary:
1. WS0: commit, license, CI, `.gitignore` the PDF (Section 14.1).
2. WS1: finish Gemma and the per-layer check, commit small verified results (Section 14.2).
3. WS2: README v0 with figures (Section 14.3).
4. WS3: X4, X5, X6 on CPU (Section 14.4). X4 runs before the dashboard because it may change what Page 1 emphasizes.
5. WS4: E6 pilot, E6, X1 on Kaggle (Section 14.5).
6. WS5: dashboard and deploy (Section 14.6), WS6: E1 notebook (Section 14.7), WS7: write-up and interview prep (Section 14.8).

---

## 1. Experiment inventory

| ID | Paper location | What it shows | Hardware | Tier |
|---|---|---|---|---|
| I0 | (ours) | Shared infrastructure: LoRA parameter counter, data loaders, seeded sampling, result logging | CPU | Must |
| E1 | Sec 2.1, Fig 2-3, App C.1, Thm 1 | Feature norms grow during training; the random-input baseline does not | CPU, minutes | Must |
| E2 | Sec 3, Fig 4, Sec 5 table | NFN-map and per-type scores, Llama-3.2-1B on math | CPU | Must |
| E3 | Fig 5 | Per-type NFN across 4 models x 4 datasets | CPU | Must |
| E4 | Fig 6 | Specialized models (Math, Coder) score higher than the base model | CPU | Must |
| E5 | App C.7 | Extra NFN-maps: Qwen3-1.7B, Qwen2.5-3B/1.5B/Coder-1.5B, Gemma3-1B | CPU | Should |
| E6 | Sec 4.1, Fig 7 left | ANLI, Qwen2.5-0.5B, PLoP vs MLP vs Attn | Kaggle GPU | Must (Page 3) |
| E7 | Sec 4.1, Fig 7 right | ANLI, Llama-3.2-1B | Kaggle GPU | Should |
| E8 | Sec 4.2, Table 1 | MetaMathQA to GSM8K, Qwen3-0.6B | GPU, heavy, scaled down | Stretch |
| E9 | Sec 4.2, Table 2 | Same, Qwen3-1.7B | GPU, heavier | Stretch+ |
| E10 | Sec 4.3, Table 3, App C.6 | GRPO with LoRA, Qwen3-1.7B and Gemma3-1B | Paper used 2x GH200 | Not planned |
| X1 | (ours) | All 35 three-type placements vs NFN score: does NFN predict benefit? | GPU, reduced | Should |
| X2 | (ours) | Recover the authors' exact scoring recipe | CPU | **Done**: read from their code, no grid needed |
| X3 | (ours) | Statistics: seeds, binomial error, paired tests | none | Must |
| X4 | (ours) | Does NFN depend on the task, or mostly on the model? Includes a random-text control | CPU, about an hour | Must (new) |
| X5 | (ours) | How many prompts are enough for a stable pick (5, 10, 25, 100)? | CPU | Should (new) |
| X6 | (ours) | Base vs instruction-tuned: does post-training change the NFN map? | CPU | Should (new) |

E1-E5 are cheap and give the tool's whole first two pages. E6 gives Page 3. X1 and X4 are the pieces that go beyond the paper.

---

## 2. Shared infrastructure (I0)

### 2.1 LoRA parameter counter (built, G1 passed)
Implemented in `plop_explorer/lora_params.py`. Every row of the table below is a unit test, and the four architectures were confirmed against each model's `config.json`. `match_rank` reproduces the paper's ANLI ranks (Attn 36 and 27, Llama PLoP 15). The one exception is Qwen2.5-0.5B PLoP, where it returns 16 and the paper used 17.

A LoRA adapter on a Linear layer with input size d_in and output size d_out and rank r has r * (d_in + d_out) parameters. Sum over adapted matrices and layers.

I computed the paper's own "#Params" columns from published architecture sizes (recalled from public configs; confirm each against `config.json`). Every value below matches the paper, which cross-validates both the counter and the architecture numbers. Use these as unit tests.

Architecture used (hidden, layers, heads, kv heads, head dim, MLP width):
- Qwen2.5-0.5B: 896, 24, 14, 2, 64, 4864
- Llama-3.2-1B: 2048, 16, 32, 8, 64, 8192
- Qwen3-0.6B: 1024, 28, 16, 8, 128, 3072
- Qwen3-1.7B: 2048, 28, 16, 8, 128, 6144

Unit-test targets (paper vs my hand computation):

| Model | Placement, rank | Paper | Computed |
|---|---|---|---|
| Qwen3-0.6B | Attn(K,Q,V) r=64 | 12.8M | 12.85M |
| Qwen3-0.6B | MLP(D,G,U) r=64 | 22.0M | 22.02M |
| Qwen3-0.6B | PLoP(D,U,V) r=64 / r=76 | 18.4M / 21.8M | 18.35M / 21.79M |
| Qwen3-0.6B | PLoP^-1(G,K,Q) r=64 | 16.5M | 16.52M |
| Qwen3-0.6B | all r=64 | 40.4M | 40.37M |
| Qwen3-1.7B | Attn r=64 / MLP r=64 | 18.4M / 44.0M | 18.35M / 44.04M |
| Qwen3-1.7B | PLoP(D,O,V) r=64 / r=102 | 27.5M / 43.9M | 27.52M / 43.87M |
| Qwen3-1.7B | all r=64 | 69.7M | 69.73M |
| Qwen3-1.7B (GRPO) | Attn r=16 / r=25, MLP r=16, PLoP r=16 / r=25 | 4.58 / 7.17 / 11.01 / 6.88 / 10.75 M | 4.59 / 7.17 / 11.01 / 6.88 / 10.75 M |

**Consequence:** "Attn" in the paper means **Q, K, V only (no O)**. That is the only reading under which the ANLI ranks match:
- Llama-1B ANLI: MLP r=8 = 3.93M, Attn(QKV) r=27 = 3.98M, PLoP(V,O,D) r=15 = 4.06M.
- Qwen2.5-0.5B ANLI: MLP r=8 = 3.32M, Attn(QKV) r=36 = 3.32M (exactly equal), PLoP(V,O,U) r=17 = 3.50M (r=16 would be 3.29M, closer, so the authors' rank choice here is unexplained).

This corrects the earlier strategy doc, which left "Attn" open.

Also implement `match_rank(placement, budget)` that returns the rank whose parameter count is closest to a target, and log both the chosen rank and the actual count.

### 2.2 Data loaders (seeded, cached)
- NFN prompt sets: GSM8K (math), HumanEval (code), MMLU `high_school_european_history` (history), MMLU `logical_fallacies` (logic). Seeded sampling, saved sample IDs.
- ANLI (train and evaluation splits), MetaMathQA (train), GSM8K test (eval).
- Unknown: which ANLI rounds (R1-R3) the authors combined and how many examples. I recall the rounds have roughly 17k, 45k and 100k training examples; verify. We will use all three rounds combined and say so.

### 2.3 Reproducibility protocol
- One YAML config per run (model, dataset, placement, rank, LR, seed, steps, precision). Results saved as JSON with git commit hash and library versions.
- Scoring runs in fp16 on the Mac GPU (MPS), as the authors ran fp16 on CUDA. When fp16 overflows (Gemma), rerun in fp32 on CPU. Training precision is decided in E6 (see the bf16 note there).
- Machine limits: 8 GB RAM. Models up to 1.7B fit in fp16. The MPS backend has hung twice (eager attention, and Gemma in fp32), so long runs use `--device cpu` and are started in the background.
- Every random source seeded: prompt sampling, data order, LoRA init, dropout.

---

## 3. NFN experiments (CPU): E2, E3, E4, E5, plus recipe recovery X2

### 3.1 Implementation
1. Load model, find every `nn.Linear` in decoder layers whose name ends in `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Ignore embeddings and `lm_head`.
2. Register a forward hook per module capturing its input z (shape batch x tokens x d_in).
3. For every token: a = ||W z||^2 and baseline b = expected ||W z~||^2.
4. Exact baseline: if z~ is Gaussian rescaled to the same norm as z, then E||W z~||^2 = ||z||^2 * ||W||_F^2 / d_in. Unit test: this must equal a Monte Carlo average of many random draws within sampling error.
5. Accumulate, then average per module (NFN-map: 7 types x layers) and per type (mean over layers).
6. Mask padding tokens with the attention mask (our default); keep a flag to include them (authors' apparent behavior).
7. Note for understanding: q, k, v (and gate, up) receive the same input vector, so their scores differ only because their weights respond differently to that input. Also W is not square for k, v, gate, up, down, so use d_in in the baseline, not a single "n".

### 3.2 X2: the authors' recipe (resolved by reading their code)
The variant grid originally planned here is unnecessary: the authors' scoring code (`src/metrics.py`, `src/data.py`, `main.py` at commit 80597d2) answers each question directly.

| Choice | What the authors' code does | Ours |
|---|---|---|
| Ratio | **Unsquared**: mean over tokens of ‖W ẑ‖ divided by mean over tokens of ‖W u‖ | Same |
| Baseline | One random unit vector u per token (Monte Carlo) | Exact expectation ‖W‖_F / sqrt(d_in); unit test shows they agree within 1% |
| Padding | Counted. Pad token = EOS, right side | Default masks it; reference mode counts it |
| Averaging | Ratio per batch of 8, then the mean of batch ratios | Pooled over all tokens. Tested: makes no difference (Section 3.3 A) |
| Aggregation | Per-type score = plain mean of per-module NFN over layers | Same |
| Prompt text | Question only: no answer, no chat template. GSM8K samples 100 from the first 200 train questions; HumanEval gets a "# Write a Python function" prefix; MMLU gets the question plus lettered choices | Same, but seeded (theirs is unseeded) |
| Batch 200 vs 8 | The code uses batches of 8 and 100 samples | We use the code's settings |

Fig 4's 112 per-layer values are in `reference/plop_fig4_llama1b_math.json` (extracted from the PDF text; rows Key, OutProj, Query, Value, DownProj, GateProj, UpProj, 16 layers). Their layer averages equal the Section 5 table (checked: 2.62, 0.90, 2.58, 0.97, 1.05, 1.40, 1.11), so `check_reference.py` can also compare per layer. **Result (2026-10-16):** our reference-setup grid is off by 3.1% mean |relative error| per cell (masked 7.4%); by type Query 3.2%, Key 5.3%, Gate 7.1%, the rest 1.5-1.8%, the same pattern as the type-level gap.

### 3.3 Targets

**A. Llama-3.2-1B math, per type (Sec 5):** k 2.63, q 2.58, gate 1.40, up 1.11, down 1.05, v 0.97, o 0.90. Pass: same ranking, each within a few percent, and the lowest three are {v, o, down}.

**Result (A):** passed on ranking and pick; three types sit slightly outside "a few percent".

| Type | Paper | Ours, reference setup | Diff | Ours, padding masked |
|---|---|---|---|---|
| Query | 2.579 | 2.663 | +3.3% | 2.762 |
| Key | 2.627 | 2.757 | +5.0% | 2.937 |
| Value | 0.969 | 0.973 | +0.4% | 0.983 |
| OutProj | 0.900 | 0.905 | +0.6% | 0.915 |
| GateProj | 1.401 | 1.496 | +6.8% | 1.622 |
| UpProj | 1.112 | 1.126 | +1.3% | 1.146 |
| DownProj | 1.048 | 1.055 | +0.7% | 1.067 |

What the remaining gap is not:
- Prompt sampling: spread across 5 seeds is about 0.01 per type.
- Per-batch vs pooled averaging: identical to 0.005.
- Pad token choice (`<|eot_id|>` vs `<|end_of_text|>`): mean error 2.5% vs 2.3%.
- Left padding is clearly wrong (OutProj drops to 0.60, mean error 9-10%), which confirms right padding.

Precision is ruled out (2026-10-16): CPU fp32 gives the same per-type scores as MPS fp16 to 0.01 (Query 2.66, Key 2.76, Gate 1.50). Remaining suspect: the authors' older `transformers` version. The gap is largest exactly where counting padding matters most (Key, Gate, Query), so pad-position activations are the likeliest place the two environments differ. Decision: stop chasing it, record it in the README, and keep the reference-setup toggle.

**B. Selection sets reported across the paper (hard pass/fail):**

| Model | PLoP picks (lowest 3) | PLoP^-1 picks (highest 3) | Source |
|---|---|---|---|
| Llama-3.2-1B | V, O, Down | (not reported) | Fig 7, Sec 5 |
| Qwen2.5-0.5B | V, O, Up | (not reported) | Fig 7 |
| Qwen3-0.6B | Down, Up, V | Gate, K, Q | Table 1 |
| Qwen3-1.7B | Down, O, V | Gate, K, Q | Table 2, 3 |
| Gemma3-1B | K, V, Up | O, Gate, Down | App C.6 |

From Fig 5 as read by eye, Gemma3-1B's lowest three are indeed V (about 0.68), Up (about 0.9) and K (about 1.1), matching K-V-U. Pass criterion for the tool: at least 4 of 5 sets match exactly; report near-ties honestly (for example Qwen3-1.7B's O, Down and Up are all close to 0.95-0.98, so a swap is plausible).

Unknown: which dataset the authors scored for the ANLI and Qwen3 selection sets. Try ANLI prompts for E6/E7 and math for E8/E9, and record which reproduces their sets.

**Result (B), from `scripts/check_selection.py`** (100 prompts, seed 0, reference setup; the masked setup gives the same picks except where noted):

| Model | Our PLoP pick | Our PLoP^-1 pick | Matches on |
|---|---|---|---|
| Llama-3.2-1B-Instruct | Down, Out, Value | Gate, Key, Query | All 4 tasks |
| Qwen2.5-0.5B | Out, Up, Value | Gate, Key, Query | All 4 tasks |
| Qwen3-0.6B | Down, Up, Value | Gate, Key, Query ✓ | All 4 tasks, both sets |
| Qwen3-1.7B | Down, Out, Value (math) | Gate, Key, Query ✓ | Math. Code (reference setup), history and logic pick Up instead of Out |
| Gemma3-1B-it | Key, Up, Value | Down, Gate, Out ✓ | Math, history, logic, both sets (fp32 CPU). Code (reference setup) picks Query for Key: 1.28 vs 1.30, a near-tie; masked setup matches |
| Gemma3-1B-pt | Key, Up, Value (masked); Down for Key on math and logic (reference) | Never matches (Out plus Key/Query/Gate) | Not the paper's model; consistent with Fig 5b's "Gemma3-1B-Instruct" label |

Notes:
- Qwen3-1.7B's swap is a near-tie: Out, Up and Down all score 0.95-0.99. Its paper picks come from the math experiments, and math matches.
- The Value anomaly reproduces: Qwen3-0.6B Value is 0.68-0.72 and Qwen3-1.7B 0.74-0.76, clearly below 1 on every task.
- Gemma in fp16 gives NaN everywhere. The authors also report fp16, so either their older `transformers` handled Gemma's activations differently or they ran Gemma in another precision. Record whichever we find.

**C. Qualitative patterns (Fig 5, 6, 8):**
- Llama-3.2-1B and 3B: Query and Key highest (roughly 2-3), Value and Out about 0.9-1.0, ranking similar across the two sizes; history scores above math.
- Qwen3-1.7B and Gemma3-1B: Value clearly below 1 (about 0.75 and 0.7). The paper says it has no explanation. Show it, do not explain it.
- Specialized Qwen2.5-Math-1.5B and Coder-1.5B score higher than base Qwen2.5-1.5B on their own tasks.

### 3.4 Runs
- E2: Llama-3.2-1B-Instruct x GSM8K, produce map + table.
- E3: Llama-3.2-1B, Gemma3-1B, Llama-3.2-3B, Qwen3-1.7B x {math, code, history, logic}.
- E4: Qwen2.5-1.5B-Instruct vs Qwen2.5-Math-1.5B-Instruct (GSM8K) and vs Qwen2.5-Coder-1.5B-Instruct (HumanEval). The paper's text says "instruction-tuned and specialized" and its appendix maps are labelled `-Instruct`.
- E5: the additional maps in Appendix C.7.
- Paper settings: Fig 4 used a single batch of 200, max length 256; the Section 5 command uses `--batchsize 8 --nbsamples 100 --seqlen 256`. These are inconsistent, so test both and check that the result is stable.
- Note from the notes in our context file: the authors' Appendix maps for some models are reported to be on an older, incompatible scale, so treat E5 as qualitative unless E2's recipe recovery shows otherwise.

---

## 4. E1: feature-norm growth in a linear network (CPU)

**Purpose:** reproduce the core observation behind NFN, and test the theory.

**Setup (Sec 2.1, App C.1):**
- f(x) = W2 W1 W0 x with W0 in R^{n x d}, W1 in R^{n x n}, W2 in R^{1 x n}; n = d = 100.
- N = 1000 samples, x standard Gaussian, y = w^T x + noise, full batch, Adam, 300 steps.
- Track n^{-1} ||W_l z_in,l||^2 for each layer, plus a dashed baseline using a random z~ of the same norm.

**Unknowns to choose and record:** learning rate, init scale (the text says weights near 1/sqrt(n), i.e. muP-style), whether the noise value 0.025 is a variance or a standard deviation, and w's scale (main text writes d^{-1} N(0,1), the appendix writes d^{-1/2} N(0,1)).

**Outputs:** a Fig 3 look-alike (norm curves + loss curve), and the same for SignSGD. The paper says Appendix C has SignSGD results, but it does not (C.1 describes Adam only), so the SignSGD run is ours and has no target.

**Pass criteria (qualitative, from the paper):** real feature norms rise and plateau within about the first 200 steps while loss falls fastest; the baseline stays roughly flat; different layers grow by different amounts.

**Theorem 1 check (single hidden layer trained, others frozen, one data point, SignSGD with learning rate eta/n):**
- Compare the measured n^{-1} ||W_t z||^2 with the formula for Gamma_t.
- Repeat for n in {100, 400, 1600, 6400} and plot the maximum error against n on log-log axes; the theorem predicts the error shrinks like n^{-delta}.
- **Discrepancy to test:** the theorem states Gamma_t = Gamma_0 + beta^2 (1 + t(t-1)), but the proof defines the recursion Gamma_{t+1} = Gamma_t + beta^2 (1 + 2t), which sums to Gamma_0 + beta^2 t^2. These agree at t = 1 and differ at t = 0 (theorem: Gamma_0 + beta^2, but Gamma_0 is defined as the value at t = 0) and at t >= 2. One is probably a typo. Simulate and see which fits; report the finding without assuming.

---

## 5. Training experiments

### 5.1 E6 / E7: ANLI classification (Sec 4.1, Fig 7, App C.2)

**Paper configuration:**
- Models: Qwen2.5-0.5B (Fig 7 caption; the appendix says "Qwen3.5-0.5B", treated as a typo) and Llama-3.2-1B.
- Placements: PLoP, MLP, Attn(QKV). Ranks (from the appendix): Qwen2.5-0.5B: MLP 8, Attn 36, PLoP 17; Llama-1B: MLP 8, Attn 27, PLoP 15.
- Optimizer AdamW, no warmup, linear schedule, dropout 0.1, max length 256, LoRA alpha = 2r, bf16.
- Curves run to about 5000 steps and are EMA-smoothed (alpha 0.8) for display only.
- Reported behavior: Qwen2.5-0.5B, PLoP clearly best; Llama-1B, PLoP about equal to MLP, both clearly above Attn. Absolute test accuracy roughly 0.5-0.55 for Qwen and roughly 0.6-0.62 for Llama (read from the figure; three-way classification so chance is about 0.33).

**Unknown:** batch size, learning rate, evaluation frequency, ANLI round mix, whether the classification head is trained in addition to LoRA. Choose, document, and keep identical across placements.

**Our protocol:**
- Placements: PLoP (from our own NFN scores, cross-checked against the paper's set), MLP, Attn(QKV), and PLoP^-1 as a control.
- Rank via `match_rank` to the MLP r=8 budget.
- 3 seeds per placement. Report mean and standard error of final accuracy (mean of the last few evaluations), plus the curve.
- Precision: Kaggle's P100 and T4 do not support bfloat16 well, so use fp16 with loss scaling (or fp32), and state this as a deviation.
- Run a pilot (one placement, one seed, few hundred steps), time it, then set steps and seed count to fit the quota. I have not measured the runtime; do not promise numbers before the pilot.

**Pass criteria (directional):** Attn(QKV) below the others on both models; on Qwen2.5-0.5B PLoP at or above MLP; on Llama-1B PLoP within noise of MLP. Do not require matching absolute accuracy.

### 5.2 E8 / E9: MetaMathQA to GSM8K SFT (Sec 4.2, Tables 1-2, App C.3)

**Paper configuration:**
- Models: Qwen3-0.6B and Qwen3-1.7B (unknown whether base or instruct).
- Adam, 2 epochs, 10% warmup, cosine schedule, no dropout, max length 1024, alpha = 2r, bf16.
- Learning rate swept over {1,2,3,4,5}e-4 per placement, best accuracy reported.
- Evaluation: GSM8K, 8-shot, Qwen chat template, strict match, 512 generation tokens, using the `evaluate_chat_gsm8k.py` script from the official Qwen repo.

**Published targets (eval accuracy):**

| Qwen3-0.6B | Params | Acc | Qwen3-1.7B | Params | Acc |
|---|---|---|---|---|---|
| PLoP r=76 | 21.8M | 63.8% | PLoP r=102 | 43.9M | 75.4% |
| PLoP r=64 | 18.4M | 62.0% | PLoP r=64 | 27.5M | 75.2% |
| PLoP^-1 | 16.5M | 60.6% | PLoP^-1 | 27.5M | 74.6% |
| MLP | 22.0M | 63.3% | MLP | 44.0M | 75.0% |
| Attn | 12.8M | 58.6% | Attn | 18.4M | 69.5% |
| all | 40.4M | 62.4% | all | 69.7M | 73.9% |

**Reading these tables honestly (my calculation):**
- GSM8K's test set has 1,319 problems, so the binomial standard error of one accuracy is about 1.2-1.3 points. Differences of 0.2-1.5 points among PLoP, MLP and all are within noise for a single run. Only the Attn gap (5-6 points) clearly exceeds it.
- At equal rank 64, MLP beats PLoP on the 0.6B model (63.3 vs 62.0) but with more parameters; PLoP wins only at the parameter-matched r=76.
- Learning rate was chosen by best test accuracy, which is mildly optimistic. In our version choose it on a held-out validation split.

**Scaled-down plan (needs a pilot to confirm feasibility):**
- Subset of MetaMathQA, one epoch, 2-3 learning rates chosen on a validation split, parameter-matched ranks.
- Evaluate on the full GSM8K test set with generation; batch the generation.
- Do E8 only if E6 finishes cleanly. E9 (1.7B) is likely beyond free-tier limits; document it as not attempted unless the pilot says otherwise.

### 5.3 E10: GRPO (Sec 4.3, Tables 3-4, App C.4, C.6): not planned
The paper used 2x GH200 GPUs, 8 generations, batch 64, 50k MetaMathQA samples, LR 4e-6, and could not tune the learning rate. This is out of reach on a free tier. In the write-up, state that we did not reproduce it, list the paper's numbers (Qwen3-1.7B GSM8K: no RL 65.50%; PLoP r=16 74.52%, r=25 75.03%; MLP 73.61%; Attn r=16/25 71.49/72.13%; PLoP^-1 71.41%) as reported, not verified. Note two paper inconsistencies for anyone attempting it: the answer tags are `<solution>` in the main text and `<answer>` in the appendix. Also note that the Gemma3-1B GRPO shows almost no GSM8K change (29.10% to 28.05-30.52%), which the authors attribute to Gemma being weak at this task.

---

## 6. Our extension X1: does NFN predict adapter benefit?

The paper shows PLoP's pick does well but not that the score itself predicts which of the many possible placements does well. There are C(7,3) = 35 possible three-type placements. That gives a direct test:

1. Score each of the 35 combinations: s_c = mean NFN of its three types (from E2/E6 scoring).
2. Train each combination on ANLI with Qwen2.5-0.5B under an equal parameter budget (rank chosen by `match_rank`), with reduced steps and one seed first.
3. Plot accuracy against s_c (scatter), compute Spearman correlation (the prediction is negative: lower NFN, better accuracy), and report where PLoP's own pick ranks among the 35 and where Attn(QKV) and MLP rank.

This also supplies the missing control: a "random three types" baseline is just the distribution of the 35 results.

Feasibility is unknown: 35 runs is a lot. Options if it is too heavy: fewer steps, a subset of ANLI, or a subset of combinations (for example all combinations containing at least one of V, O; or a random 12 of 35). Decide after the E6 pilot timing. This is the single most valuable extension because it turns the project from "reproduction" into "independent test of the claim", including if the answer is "the score is only weakly predictive".

Optional, lower priority: repeat the paper's unsuccessful layer-level selection to see whether we can explain its inconsistency.

### 6.2 X4: does NFN measure the task or the model? (new, CPU)
**Context from the paper:** its Fig 5 `random` entry is the dashed baseline at 1.0, not a random-input dataset, so the controls below are new. The paper itself says all tasks share a task-independent "base" alignment level, so X4 measures how large that level is rather than refuting a claim.

**Why:** in the G3 results, each model's pick is nearly identical on math, code, history and logic. Llama picks Down, Out, Value on all four; so does Qwen3-0.6B (Down, Up, Value). The order of the seven types also barely moves (Qwen3 Value always lowest, Gate or Key always highest). If NFN mostly reflects the model, PLoP's placement is closer to a per-model constant than a task-aware choice. That would challenge the paper's framing.

**Runs:**
1. Add control prompt sets to `data.py`:
   - random tokens drawn uniformly from the vocabulary,
   - shuffled words from the math prompts (same words, no meaning),
   - plain English prose unrelated to any task (for example, Wikipedia sentences).
2. Score every G3 model on the 4 tasks plus the 3 controls.
3. Measure:
   - Spearman rank correlation of the 7 type scores between every pair of tasks within a model, and between models on the same task;
   - how often the top-3 pick changes between tasks vs between models;
   - how far each control moves the scores from the real tasks.

**Reading the result:**
- If controls give the same pick as the tasks, NFN is mostly a property of the model.
- If controls differ but real tasks agree, NFN measures "real text vs noise" rather than the specific task.
- If tasks differ in a meaningful way, the paper's framing holds.

Report whichever it is. **Effect on the product:** Page 1 should show how stable the pick is across tasks (for example, "this model's pick is the same on 4 of 4 tasks").

### 6.3 X5: how many prompts are enough? (new, CPU)
Scores moved about 0.01 across 5 seeds at 100 prompts. Score with 5, 10, 25, 50 and 100 prompts over 5 seeds each, and plot the spread and how often the pick changes. If about 10 prompts is stable, the "paste your own prompts" feature can be fast and credible, and the tool can say how many prompts it needs.

### 6.4 X6: base vs instruction-tuned (new, CPU)
Compare pairs: Gemma3-1B `-pt` vs `-it`, Qwen3-0.6B-Base vs Qwen3-0.6B, Llama-3.2-1B vs Llama-3.2-1B-Instruct. Does post-training change the NFN map, and does it change the pick? This fits naturally on Page 2 (Compare). The paper labels its NFN models Instruct (Gemma3-1B-Instruct, Qwen3-1.7B-Instruct), so the "base or instruct" unknown applies only to the SFT training runs; X6 can still show whether the pick would differ.

---

## 7. Statistics and reporting (X3)
- Seeds: at least 3 per configuration; show mean plus standard error, and all individual points.
- For classification and GSM8K accuracy, also report the binomial standard error of the test set, since it can exceed seed-to-seed spread.
- Compare placements with paired differences per seed; do not claim a win when the interval includes zero.
- Report null and mixed results as they are. Expected honest headline for Llama-1B: PLoP ties MLP and beats Attn.

---

## 8. Paper discrepancy and ambiguity log (keep in the README)

1. Step 1 of the boxed algorithm uses squared norms; Definition 1 uses unsquared. **Resolved:** the authors' code uses unsquared (Section 3.2).
2. Theorem 1's Gamma_t formula vs the proof's recursion (t^2 vs 1 + t(t-1)); tested in E1.
3. Linear-network data: main text says omega_i ~ d^{-1} N(0,1), appendix says d^{-1/2} N(0,1); the noise 0.025 is a variance or a standard deviation, unclear.
4. ANLI appendix names "Qwen3.5-0.5B" while Fig 7 shows Qwen2.5-0.5B.
5. NFN compute: Sec 3 says batch 200, Sec 5 command says batch 8 and 100 samples. **Resolved:** the code uses batch 8 and 100 samples, and averaging per batch or pooled gives the same result.
6. GRPO answer tags: `<solution>` (main text) vs `<answer>` (appendix).
7. Sec 3 text says Llama-1B's lowest scores are Value, Gate, Down, Up "hovering around 1", but the table gives Gate 1.40.
8. The paper's Sec 4.1 says PLoP and MLP match on Llama because MLP modules have low NFN; Gate at 1.40 makes this only partly true. Testable via an "MLP without gate" run.
9. Learning rate chosen on the test set in the SFT tables.
10. Unknowns: batch sizes, learning rates for ANLI, ANLI round mix, base vs instruct for the Qwen3 SFT runs (NFN models are labelled Instruct), which dataset produced the ANLI selection sets.
11. Qwen2.5-0.5B ANLI: the paper's PLoP rank 17 gives 3.50M parameters; rank 16 (3.29M) is closer to the 3.32M budget.
12. Our Llama-1B math scores are 3-7% higher than the paper's on Query, Key and Gate, with the same ranking and pick. Not explained by sampling, averaging or pad token (Section 3.3 A).
13. Gemma3-1B overflows to NaN in fp16 with current `transformers`, although the authors report fp16. In fp32 the paper's Gemma sets reproduce with the `-it` model; `-pt` does not match.
14. The authors' prompt sampling is unseeded, so their exact prompts cannot be reproduced; ours are seeded.
15. Picks barely change across tasks within a model (X4 tests what this means for the "task-aware" framing).
16. Sec 2.1 says Appendix C has SignSGD results for the linear network; it does not.
17. Theorem 1's statement gives probability 1 - 2n^(-1+2δ), its proof ends at 1 - 2n^(-1+δ); Theorem 2's bound on T differs between statement and proof.
18. Theorem 1's Gamma_t is also inconsistent at t = 0 (see item 2).
19. Sec 3 prints the closed-form NFN approximation with ||W||_F ||z_in|| in the denominator; the exact expectation is ||W z|| sqrt(d_in) / (||W||_F ||z||), i.e. the paper's form is off by a factor sqrt(d_in) (our `nfn.py` uses the exact one). Confirmed on the rendered page 8.

---

## 9. Repo layout

Built so far: `nfn.py` (aggregation and the pick live in `NFNResult`, so no separate `selection.py`), `data.py`, `lora_params.py`, `scripts/check_reference.py`, `scripts/check_selection.py`, `tests/`, `reference/`, `docs/`. The rest is planned.

```
plop_explorer/
  nfn.py               # hooks, exact baseline, by-type aggregation, pick (built)
  data.py              # prompt sets (built); control sets for X4 (planned)
  lora_params.py       # parameter counter, match_rank (built)
scripts/
  check_reference.py   # G2 (built)
  check_selection.py   # G3 (built)
experiments/
  e1_linear_growth.py
  e2_recipe_recovery.py
  e3_model_dataset_grid.py
  e6_anli_train.py
  e8_metamath_train.py
  x1_placement_sweep.py
configs/               # one YAML per run
results/               # JSON per run, with git hash
reference/             # paper's published numbers, targets above
tests/                 # param-count tests, baseline unit test, selection-set tests
app/                   # FastAPI backend + static HTML/JS frontend, pages 1-3
docs/                  # this plan and the strategy doc
```

---

## 10. Sequencing and gates

| Gate | Requirement before moving on |
|---|---|
| G1 ✅ | Parameter counter reproduces the paper's #Params table (Section 2.1) |
| G2 ✅ | Exact-baseline unit test passes; scoring matches the authors' recipe and the 7-number table (gap documented) |
| G3 ✅ | Selection-set table matches for at least 4 of 5 models (E3); Gemma pending |
| G3.5 | X4 (task vs model) run, so Page 1 shows the right emphasis; README v0 published |
| G4 | E1 shows the growth pattern and the Theorem 1 check is run (can run in parallel with G5) |
| G5 | Build the dashboard pages on verified numbers |
| G6 | E6 pilot timed; decide steps, seeds, X1 subset |
| G7 | E6 (and E7 if time) complete with 3 seeds; X1 run at the affordable scale |
| G8 | E8 attempted only if G7 is clean |
| G9 | README, GIF, discrepancy log, write-up |

Do not build UI on unverified scores (G2 and G3 before G5).

---

## 11. Risks

| Risk | Handling |
|---|---|
| No variant reproduces the authors' numbers | Report the closest variant and the gap; the tool ships a toggle so users see both |
| Llama access delayed | Qwen2.5-0.5B and Qwen3 models are ungated; E6 and E3 proceed without Llama, E2 waits |
| Free-GPU quota too small for 3 seeds x 3 placements | Cut steps or subset data before cutting seeds; say what was cut |
| fp16 instability vs the paper's bf16 | Loss scaling, lower LR, document; check a short run against fp32 |
| X1 too expensive | Use the subset options in Section 6 |
| A result contradicts the paper | Publish it with the exact config; verify no bug first (recompute parameter counts, reread configs) |
| fp16 overflow (Gemma) | `score_model` raises on non-finite scores; scripts retry in fp32. The API must do the same for user-entered models |
| MPS hangs on this Mac (seen twice) | Long runs on CPU in the background; precompute results offline for the deployed demo |
| 8 GB RAM | fp16 up to about 1.7B locally; larger models (Llama-3.2-3B) precomputed elsewhere or skipped |
| Flaky network during runs | Datasets and models are cached; run with `HF_HUB_OFFLINE=1` once cached |
| Hugging Face Spaces free CPU is slow for live scoring | Precompute the paper's model/task pairs; live scoring only for custom models and prompts, with a job queue |

---

## 12. What each page of the tool draws from
- Page 1 (Explorer): E2, E3 outputs, the reference-setup toggle, and X4's "pick stability across tasks" badge.
- Page 2 (Compare): E3, E4, E5 outputs, and X6's base vs instruction-tuned pairs.
- Page 3 (Does it work?): E6/E7 curves, E8 table if done, X1 scatter, and the honest-reading box from Section 5.2.
- About page: the discrepancy log (Section 8) and the "what we reproduced and what we did not" table.

### API sketch (FastAPI)
- `GET /api/models`, `GET /api/tasks` fill the pickers.
- `POST /api/nfn {model, task, mask_padding}` returns the per-layer grid, the per-type scores and the pick. Cached on disk by (model, task, number of prompts, seed, padding mode).
- Scoring takes 15-60 s on CPU, so uncached requests return a job ID to poll.
- Page 2 calls the same endpoint once per model.

---

## 13. Backlog outside the experiments

| Item | Why | Size |
|---|---|---|
| README v0: what PLoP is, what we reproduced (tables in Section 3.3), known gaps, next steps | A recruiter opening the repo today sees no explanation | About an hour |
| License (MIT), GitHub description and topics | Free and easy to forget | Minutes |
| GitHub Actions running `pytest` on each push | Visible sign of rigor; the tests are CPU-only and take seconds | Small |
| Commit results for the verified runs (small per-type JSON, not full per-module dumps) | Makes the claims in the README checkable | Small |
| Demo GIF (30-60 s) in the README | The highest-leverage item once the UI exists | After G5 |
| E1 as a notebook with the Fig 3 look-alike | A visual that explains NFN's motivation | Part of G4 |
| Write-up / blog post | What actually gets shared | After G7 |
| Update `plop-strategy.md` or mark it superseded | Its framework (Streamlit/Gradio), hosting and open questions are stale | Minutes |

---

## 14. Execution plan: from careful reproduction to an ML-engineer-worthy project

Goal: a stranger (recruiter or engineer) opens the repo and, in 90 seconds, sees (a) a working, verified method, (b) a figure and a results table, (c) one result that is ours, with statistics, and (d) honest limits. Current standing is about 5/10 (Section 0); the workstreams below are ordered by score gained per hour. Expected ratings are estimates, not measurements.

### 14.0 Definition of done (the bar)

| # | Requirement | Evidence in repo |
|---|---|---|
| 1 | One command reproduces the headline table | `make verify` (or documented 3 commands) prints Llama-1B math table, pass/fail |
| 2 | Tests and CI green | `pytest` passes in GitHub Actions on every push |
| 3 | Numbers are checkable | Small result JSONs committed under `reference/ours/`, each with git hash, library versions, seed, dtype, device |
| 4 | A visible figure | README shows heatmap + bar chart vs the paper |
| 5 | One original finding with statistics | X4 and X1 written up with correlation, intervals, controls |
| 6 | Training evidence | E6: 3 placements x 3 seeds, mean ± SE, matched parameters |
| 7 | Honest limits | README "What this does not show" section; discrepancy log linked |
| 8 | A live demo | Hugging Face Space link in the README |
| 9 | A write-up | One page: question, method, results, caveats |

### 14.1 WS0: hygiene and commit (about 1 hour, do first)
1. Commit the currently uncommitted work: `CLAUDE.md`, `docs/plop-build-spec.md`, `docs/plop-interview-guide.md`, `reference/plop_fig4_llama1b_math.json`, the plan edits. Separate commits for docs and data.
2. Add `docs/pppp.pdf` to `.gitignore` and link the arXiv abstract in the README instead (avoids redistributing the PDF).
3. `LICENSE` (MIT). GitHub description and topics: `lora`, `peft`, `llm`, `reproducibility`, `pytorch`, `interpretability`.
4. `.github/workflows/ci.yml`: Python 3.11, `pip install torch --index-url https://download.pytorch.org/whl/cpu`, `pip install -r requirements.txt`, `pytest -q`. Tests are CPU-only and finish in seconds; confirm no test downloads models or datasets, and mark any that do with `@pytest.mark.network` and deselect in CI.
5. `plop_explorer/runlog.py` (about 30 lines): `save_result(path, payload, **meta)` writes JSON with `git_hash`, `torch`/`transformers` versions, `device`, `dtype`, `seed`, ISO timestamp. All later scripts use it. Unit test: file contains the keys.
6. Replace `plop-strategy.md` header with "Superseded by plop-implementation-plan.md".
- Accept: clean `git status`, CI badge green, `runlog` test passes.

### 14.2 WS1: close verification (about half a day; CPU)
Follows `plop-build-spec.md` Steps 1-2.
1. Gemma3-1B `-it` and `-pt`: `python scripts/check_selection.py --models unsloth/gemma-3-1b-it unsloth/gemma-3-1b-pt --device cpu`. Record which matches K-V-U / O-G-D. Update Section 3.3 B and gate G3 to 5 of 5 or "4 of 5 with reason".
2. Per-layer check: extend `check_reference.py` to compare `grid()` with `reference/plop_fig4_llama1b_math.json`; print mean absolute error per type and overall.
3. Commit compact summaries (per-type scores, top-3, per-layer error) to `reference/ours/{model}_{task}.json`. Full per-module dumps stay in gitignored `results/`.
4. Add a unit test that loads the committed summaries and asserts the documented pass criteria (top-3 sets, ranking). This turns the claims in the README into tests.
- Accept: table in plan Section 3.3 B complete; per-layer error number recorded; tests green.

### 14.3 WS2: README v0 and figures (about half a day)
1. `scripts/make_figures.py` (matplotlib; add to `requirements.txt`): reads `reference/ours/*.json` and the reference JSONs, writes `docs/img/`:
   - `heatmap_llama1b_math.png`: 7 x 16 grid, one fixed color scale, 1.0 marked, ours vs paper side by side.
   - `bars_vs_paper.png`: per-type scores, ours (reference setup), ours (masked), paper.
   - `picks_table.png` or a Markdown table: 5 models x our pick vs paper pick.
2. README sections, in order: one-sentence pitch; headline figure; "What is PLoP" (3 sentences + the formula); "What I reproduced" (table with real numbers: ~2.5% mean error, selection sets, 16/16 parameter counts); "What I found that the paper gets wrong" (5 bullets from the discrepancy log); "What this does not show" (no training result yet until WS4; CPU-only; fp16 vs bf16); "Run it" (3 commands); links to plan, spec, interview guide.
3. Do not state any number that is not in a committed result file.
- Accept: a reader who knows nothing can say what the project does and what is verified; every number in the README traces to a file.

### 14.4 WS3: X4, X5, X6 on CPU (about 1 day total; this is the first original result)
Details in Sections 6.2-6.4 and spec Steps 3-4. Implementation additions:
1. `data.py`: `load_prompts` gains `random_tokens`, `shuffled_words`, `prose` (control sets). `prose`: use `wikitext` (`Salesforce/wikitext`, `wikitext-103-raw-v1` train, non-empty lines of 20-80 words); `random_tokens`: `torch.randint` over vocabulary with the math prompts' token-length distribution; `shuffled_words`: `random.Random(seed).shuffle` on each math prompt's words.
2. `scripts/x4_task_vs_model.py` writes `reference/ours/x4.json` (7 scores per model x set). Analysis in `plop_explorer/stats.py`: `spearman(a, b)` via `scipy.stats.spearmanr` (add `scipy`), `pick_change_rate(scores_by_set)`.
3. Required outputs: (a) heatmap of Spearman rho between all pairs of sets for each model; (b) bar chart of mean distance from the real tasks for each control; (c) one sentence of conclusion using the decision table in Section 6.2.
4. X5 and X6 as in spec Step 4. X5's figure: pick-change rate vs number of prompts with 5 seeds.
- Accept: figures committed; Section 6.2 gets a "Result" subsection; README gets one finding paragraph whether the answer is "mostly the model" or not. Page 1's "pick stability" badge is defined from this result.

### 14.5 WS4: training evidence on Kaggle (1-2 weeks elapsed, GPU quota bound)
1. **Pilot first** (one placement, one seed, 300 steps): measure steps/s and peak memory; set steps and seeds from the measurement. Do not promise numbers before it.
2. `experiments/e6_anli_train.py`, arguments: `--model --placement {plop,plop_inv,mlp,attn,custom:Query,Key,...} --rank --seed --steps --lr --batch --out`. Details:
   - Data: `facebook/anli`, train on `train_r1+r2+r3`, evaluate on `dev` of the three rounds combined for the curve and on `test_r1+r2+r3` once at the end (verify split names and sizes on load and log them).
   - Model: `AutoModelForSequenceClassification` (3 labels), LoRA via `peft` with `task_type="SEQ_CLS"`; the classification head is randomly initialised so it must be trained: `modules_to_save=["score"]`. Record this choice (paper silent).
   - `target_modules` from placement types via the leaf names in `nfn.MODULE_TYPES`; `r` from `lora_params.match_rank` at the MLP r=8 budget; `lora_alpha = 2r`, `lora_dropout = 0.1`; AdamW, linear schedule, no warmup, max length 256; fp16 autocast with GradScaler (T4/P100), disclosed deviation from bf16.
   - Logging: `runlog.save_result` per run plus a CSV of (step, train loss, dev accuracy) every 250 steps. Assert the trainable parameter count equals `lora_params.count(...)` plus the head, as a correctness test.
   - Seeds control data order, LoRA init and dropout.
3. Runs: Qwen2.5-0.5B x {plop, mlp, attn, plop_inv} x 3 seeds; Llama-3.2-1B only if quota allows. The PLoP placement is taken from our own X-scores and cross-checked against the paper's V,O,Up / V,O,Down.
4. `plop_explorer/stats.py`: `mean_se`, `paired_diff(a, b)` with a bootstrap 95% interval over seeds, `binomial_se(p, n)`.
5. **X1 (the original contribution)**: all 35 three-type combinations on Qwen2.5-0.5B ANLI, 1 seed, reduced steps (set from the pilot), rank via `match_rank`; then the 3 most interesting (PLoP, best, worst) re-run with 3 seeds. Figure: scatter of dev accuracy vs mean NFN of the three types, Spearman rho with a permutation p-value, PLoP/MLP/Attn highlighted. If the pilot says 35 is too slow, run a fixed random 12 and say so.
6. Page 3 and README use only committed numbers.
- Accept: results table with mean ± SE and paired differences; directional criteria from Section 5.1 reported as met or not met; X1 scatter with rho. A null or negative result is acceptable and published with the config.

### 14.6 WS5: dashboard and deploy (about 3-5 days)
Follows spec Step 6. Order of work:
1. `app/core.py`: pure functions `load_cached(model, task, n, seed, pad)`, `score_job(...)`; unit tests with the `Tiny` model from `tests/test_nfn.py` (no network).
2. `app/main.py` (FastAPI): endpoints in Section 12; jobs on a single worker thread; fp32 retry on `FloatingPointError`; request validation (model id format, n between 5 and 200, k between 1 and 7); error messages naming the expected leaf names for unsupported architectures.
3. `app/static/` (HTML + vanilla JS + Plotly.js from a pinned CDN version): Page 1 first, then Page 2. Heatmap color scale fixed, 1.0 marked, colorblind-safe diverging palette; mobile width works.
4. Precompute results for the verified model/task pairs offline and ship them in the image; live scoring only for custom input, capped at 1.7B parameters on the free CPU Space.
5. `Dockerfile` (python:3.11-slim, CPU torch), Hugging Face Space (Docker), README badge and link; add a 30-60 s GIF.
6. Smoke test in CI: start the app with the `Tiny` model fixture and hit `/api/nfn`.
- Accept: public link works cold; precomputed pairs respond instantly; a bad model id gives a readable error, not a stack trace.

### 14.7 WS6: E1 linear-network notebook (about 1 day; can run any time)
Spec Step 7. `experiments/e1_linear_growth.py` plus `notebooks/e1_feature_norm_growth.ipynb` that only plots saved JSON. Figures: Fig 3 look-alike (3 layers + dashed baselines + loss), Theorem 1 error vs width on log-log axes, both candidate formulas overlaid. Seeds: 5, mean ± band. Result goes to the discrepancy log item 2 either way.
- Accept: figure committed; the log states which formula fits and at which t the two differ.

### 14.8 WS7: write-up and interview readiness (about 1 day, last)
1. One-page write-up (`docs/writeup.md`): question, method in 5 sentences, three results (verification, X4, E6/X1), what failed or did not reproduce, limits. Numbers pulled from committed JSON.
2. Review `docs/plop-interview-guide.md` against final results: replace every "planned" with the actual number or "not done". Keep its "three honest sentences".
3. Resume bullets from Section 8 of `plop-strategy.md`, filled with real numbers only.
- Accept: no claim in any document without a file that backs it.

### 14.9 Order, dependencies and expected effect

| Order | Workstream | Depends on | Effort | Rating after (est.) |
|---|---|---|---|---|
| 1 | WS0 hygiene | none | 1 h | 5 to 5.5 |
| 2 | WS1 verification | WS0 | 0.5 d | 5.5 to 6 |
| 3 | WS2 README + figures | WS1 | 0.5 d | 6 to 6.5 |
| 4 | WS3 X4/X5/X6 | WS1 | 1 d | 6.5 to 7 |
| 5 | WS4 E6 + X1 | WS0 (runlog), pilot | 1-2 wk elapsed | 7.5 to 8 |
| 6 | WS5 dashboard + deploy | WS3 result decides emphasis | 3-5 d | +0.5 |
| 7 | WS6 E1 | none | 1 d | +0.25 |
| 8 | WS7 write-up | all | 1 d | +0.25 |

Rules while executing: no UI on unverified scores (G2, G3 first); no number in a document without a result file; if a result contradicts the paper, re-verify parameter counts and configs, then publish it; cut steps or data before cutting seeds; log every deviation in Section 8.
