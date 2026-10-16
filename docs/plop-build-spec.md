# PLoP Explorer: Build Spec

Concise, build-ready companion to `plop-implementation-plan.md` (which holds status, rationale and the discrepancy log). Paper: arXiv:2506.20629 (`docs/pppp.pdf`). Everything here was checked against the paper's text and figures.

## 1. The method (what the code must compute)

- Module types: Query `q_proj`, Key `k_proj`, Value `v_proj`, OutProj `o_proj`, GateProj `gate_proj`, UpProj `up_proj`, DownProj `down_proj`.
- Per token: `||W z|| / ||W z~||`, where z~ is Gaussian noise rescaled to `||z||`. Exact baseline: `||W||_F ||z|| / sqrt(d_in)` (the paper's printed approximation drops the `sqrt(d_in)`).
- Per module: mean over tokens. Per type: unweighted mean over layers. Pick: the k=3 lowest types (PLoP); the 3 highest (PLoP^-1).
- Reference setup (authors' code, faithfulness mode): 100 prompts, batch 8, seq len 256, EOS pad token on the right, pad positions counted, fp16, unsquared ratio.
- Our default: padding masked, seeded prompts, exact baseline.

## 2. Existing API (do not change signatures)

| Call | Returns |
|---|---|
| `nfn.score_model(model, batches, mask_padding=True)` | `NFNResult` with `.modules`, `.by_type()`, `.grid()`, `.recommend(k=3)`; raises `FloatingPointError` on non-finite |
| `data.load_prompts(task, n=100, seed=0)` | task in `math, code, history, logic` |
| `data.make_batches(prompts, tok, batch_size=8, seq_len=256, padding_side=None)` | list of tokenizer outputs; `padding_side="right"` mimics the reference |
| `lora_params.count(config, types, rank)`, `match_rank(config, types, budget)`, `PLACEMENTS` | LoRA parameter counts; `Attn` = Q, K, V |

Targets live in `reference/plop_reference_by_type.json` (7 per-type scores) and `reference/plop_fig4_llama1b_math.json` (7 x 16 per-layer grid).

## 3. Build order

Each step lists its output and its pass check. Do not start a step before the previous check passes. Do not build UI on unverified scores.

### Step 1. Finish G3 (Gemma)
- Run `python scripts/check_selection.py --models unsloth/gemma-3-1b-it unsloth/gemma-3-1b-pt --device cpu`. It retries in fp32 on its own.
- Pass: lowest three = {Key, Value, UpProj}, highest three = {OutProj, GateProj, DownProj} (Table 4). Fig 5b by eye: V ≈ 0.68, Up ≈ 0.9, K ≈ 1.1, O highest ≈ 1.6-1.8.
- Record which of `-it`/`-pt` matches. Update plan Section 3.3 B.

### Step 2. Per-layer reference check
- Extend `scripts/check_reference.py` to also compare `result.grid()` with `plop_fig4_llama1b_math.json`.
- Pass: per-type ranking identical; report mean absolute per-cell error (expect about the 2.5% seen at type level, largest on Key, Query, Gate).

### Step 3. X4: task vs model (CPU, about 1 hour)
- Add to `data.py`: `random_tokens` (uniform over vocabulary, same length distribution as math), `shuffled_words` (words of the math prompts shuffled), `prose` (plain English sentences, for example Wikipedia). Seeded like the others.
- New `scripts/x4_task_vs_model.py`: score each of the 4 verified models on 4 tasks + 3 controls, save `results/x4.json` with the 7 type scores per (model, task).
- Report: Spearman rho of the 7 scores between task pairs within a model and between models on one task; how often the top-3 set changes across tasks vs across models; mean absolute distance of each control from the real tasks.
- Outcome decides what Page 1 emphasizes (see plan Section 6.2).

### Step 4. X5 (stability) and X6 (base vs instruct), CPU
- X5: n in {5, 10, 25, 50, 100} x seeds 0-4, same models. Report score std and top-3 change rate.
- X6: pairs Gemma3-1B `-pt`/`-it`, Qwen3-0.6B-Base/Qwen3-0.6B, Llama-3.2-1B/Instruct. Report rank correlation and pick change.

### Step 5. E3-E5 grids (precompute for the app)
- E3: Llama-3.2-1B-Instruct, Gemma3-1B-it, Llama-3.2-3B-Instruct, Qwen3-1.7B x 4 tasks. Targets (Fig 5): Llama Q, K about 2.3-3.0 with history highest; Gemma V ≈ 0.7; Qwen3 V ≈ 0.75, Gate about 1.65.
- E4: Qwen2.5-1.5B-Instruct vs Math-1.5B-Instruct (GSM8K) and vs Coder-1.5B-Instruct (HumanEval). Target (Fig 6): specialized scores higher overall.
- E5: maps for Qwen3-1.7B, Qwen2.5-3B, Qwen2.5-1.5B, Qwen2.5-Coder-1.5B, Gemma3-1B; qualitative only.
- Output: `results/nfn/{model}__{task}__{n}__{seed}__{pad}.json` with `by_type`, `grid`, `recommend`, `tokens`, git hash, library versions. Cache key is that filename.

### Step 6. G5 dashboard (FastAPI + static HTML/JS + Plotly.js)
- `app/main.py`: `GET /api/models`, `GET /api/tasks`, `POST /api/nfn {model, task, mask_padding}` returns grid, by-type scores and pick. Cached results return immediately; uncached return a job id to poll (`GET /api/jobs/{id}`), run on a single worker thread, retry in fp32 on `FloatingPointError`.
- Unsupported layer names return a clear error naming the expected leaf names.
- Page 1 (Explorer): heatmap of module type x layer on one fixed color scale with 1.0 marked, per-type bars, pick, reference-setup toggle, X4's "pick is the same on k of 4 tasks" badge, Advanced panel (samples 100, length 256, padding mode, k 1-7).
- Page 2 (Compare): same endpoint per model; E4 and X6 pairs.
- About: discrepancy log and the "reproduced vs not" table.
- Deploy: Hugging Face Space (Docker). Serve precomputed results; live scoring only for custom models and prompts.

### Step 7. E1 linear network (CPU, parallel to Step 6)
- `experiments/e1_linear_growth.py`: f(x) = W2 W1 W0 x, W0 in R^{100x100}, W1 in R^{100x100}, W2 in R^{1x100}; N=1000, x ~ N(0, I), y = w^T x + noise, full batch, 300 steps.
- Chosen and recorded (paper unspecified): init std 1/sqrt(fan_in); Adam lr 1e-3 (sweep {3e-4, 1e-3, 3e-3}); w_i ~ N(0, 1/d); noise std 0.025 (variance 0.025 as a second run).
- Track `n^-1 ||W_l z_in,l||^2` per layer and a baseline using z~ of equal norm, every step.
- Pass (qualitative): real norms rise and plateau within about 200 steps while the loss falls fastest; baseline flat; layers grow by different amounts.
- Theorem 1 check: one hidden layer trained, others frozen, one data point, SignSGD with step eta/n; n in {100, 400, 1600, 6400}; compare against both `Gamma_0 + beta^2 t^2` (proof) and `Gamma_0 + beta^2 (1 + t(t-1))` (statement); plot max error vs n on log-log axes.

### Step 8. Training (Kaggle T4/P100; time a pilot first)
- Library: `peft` LoRA, `transformers` Trainer or a plain loop.
- E6 ANLI (Qwen2.5-0.5B, then Llama-3.2-1B): placements PLoP (our scores), MLP r=8, Attn(QKV), PLoP^-1; ranks via `match_rank` to the MLP r=8 budget (paper: Qwen MLP 8 / Attn 36 / PLoP 17; Llama 8 / 27 / 15). AdamW, no warmup, linear schedule, dropout 0.1, max length 256, alpha = 2r, about 5000 steps, 3 seeds. fp16 with loss scaling (T4/P100 lack bf16), recorded as a deviation. Choose and record batch size, LR, ANLI rounds (all three combined), and whether the head is trained. Report mean and standard error plus all points.
- Pass (directional, confirmed against Fig 7): Attn lowest on both models; Qwen PLoP ≥ MLP; Llama PLoP within noise of MLP.
- X1: all 35 three-type placements on ANLI/Qwen2.5-0.5B at reduced steps, score = mean NFN of the three types; report Spearman rho and where PLoP, MLP and Attn rank. Subset (random 12 of 35) if the pilot says 35 is too slow.
- E8 (only if E6 is clean): MetaMathQA subset, 1 epoch, Qwen3-0.6B; Adam, 10% warmup, cosine, no dropout, max length 1024, alpha = 2r; 2-3 LRs from {1,...,5}e-4 chosen on a validation split; evaluate on all 1,319 GSM8K test problems, 8-shot, strict match, 512 tokens. Targets (Table 1): PLoP r=76 63.8, MLP 63.3, all 62.4, PLoP r=64 62.0, PLoP^-1 60.6, Attn 58.6. Binomial standard error is about 1.3 points, so only the Attn gap is expected to be clearly visible.
- E9, E10: not attempted. Quote the paper's numbers as reported, not verified.

## 4. Statistics and reporting
- At least 3 seeds per configuration; mean ± standard error and every point shown.
- Add the binomial standard error of the test set for accuracy numbers.
- Compare placements by paired per-seed differences; claim a win only if the interval excludes zero.
- Publish null results with the exact config; before trusting a contradiction, re-verify parameter counts and configs.

## 5. Repo work outside experiments
README v0 (what PLoP is, reproduced tables, known gaps), MIT license, GitHub topics, GitHub Actions running `pytest`, commit small per-type result JSONs, demo GIF after Step 6, mark `plop-strategy.md` superseded.

## 6. Environment constraints
8 GB RAM: models up to 1.7B in fp16 locally; larger ones precomputed elsewhere. MPS has hung twice, so long runs use `--device cpu` in the background. Run with `HF_HUB_OFFLINE=1` once models and datasets are cached.
