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
| G3 selection sets | **Passed: 4 of 5 models** (Gemma pending) | `scripts/check_selection.py` (Section 3.3 B) |
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
1. Finish Gemma in G3 (running on CPU in fp32) and commit.
2. X4: does NFN depend on the task or mostly on the model? (Section 6.2; about an hour on CPU.) Do this before the dashboard because it may change what Page 1 emphasizes.
3. README, license and GitHub topics (Section 13).
4. G5: API, cache and Page 1, then deploy.

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

Still open: Fig 4's 112 per-layer values are not in our reference file. Copy them out of the figure (or the authors' `results/` folder) if a per-layer comparison is wanted. Useful fact: the Section 5 table is the layer-average of Fig 4's rows, so the two are consistent.

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

Remaining suspects: the authors' older `transformers` version and CUDA fp16 vs our MPS fp16. The gap is largest exactly where counting padding matters most (Key, Gate, Query), so pad-position activations are the likeliest place the two environments differ. Decision: stop chasing it, record it in the README, and keep the reference-setup toggle.

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
| Gemma3-1B | pending | pending | fp16 overflows to NaN; rerunning in fp32 on CPU, both `-pt` and `-it` |

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
- E4: Qwen2.5-1.5B vs Qwen2.5-Math-1.5B (GSM8K) and vs Qwen2.5-Coder-1.5B (HumanEval).
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

**Outputs:** a Fig 3 look-alike (norm curves + loss curve), and the same for SignSGD.

**Pass criteria (qualitative, from the paper):** real feature norms rise and plateau within about the first 200 steps while loss falls fastest; the baseline stays roughly flat; different layers grow by different amounts.

**Theorem 1 check (single hidden layer trained, others frozen, one data point, SignSGD with learning rate eta/n):**
- Compare the measured n^{-1} ||W_t z||^2 with the formula for Gamma_t.
- Repeat for n in {100, 400, 1600, 6400} and plot the maximum error against n on log-log axes; the theorem predicts the error shrinks like n^{-delta}.
- **Discrepancy to test:** the theorem states Gamma_t = Gamma_0 + beta^2 (1 + t(t-1)), but the proof defines the recursion Gamma_{t+1} = Gamma_t + beta^2 (1 + 2t), which sums to Gamma_0 + beta^2 t^2. These agree at t = 1 and differ afterwards. One is probably a typo. Simulate and see which fits; report the finding without assuming.

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
Compare pairs: Gemma3-1B `-pt` vs `-it`, Qwen3-0.6B-Base vs Qwen3-0.6B, Llama-3.2-1B vs Llama-3.2-1B-Instruct. Does post-training change the NFN map, and does it change the pick? This fits naturally on Page 2 (Compare). It also resolves the "base or instruct" unknown for the paper's Qwen3 and Gemma results, if one variant matches and the other does not.

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
10. Unknowns: batch sizes, learning rates for ANLI, ANLI round mix, base vs instruct for Qwen3, which dataset produced the ANLI selection sets.
11. Qwen2.5-0.5B ANLI: the paper's PLoP rank 17 gives 3.50M parameters; rank 16 (3.29M) is closer to the 3.32M budget.
12. Our Llama-1B math scores are 3-7% higher than the paper's on Query, Key and Gate, with the same ranking and pick. Not explained by sampling, averaging or pad token (Section 3.3 A).
13. Gemma3-1B overflows to NaN in fp16 with current `transformers`, although the authors report fp16.
14. The authors' prompt sampling is unseeded, so their exact prompts cannot be reproduced; ours are seeded.
15. Picks barely change across tasks within a model (X4 tests what this means for the "task-aware" framing).

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
