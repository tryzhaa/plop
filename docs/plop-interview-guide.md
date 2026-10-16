# PLoP Explorer: Plain-Language Interview Guide

Read this top to bottom once. Each idea is explained in everyday words first, then the proper term is given so you can say it with confidence.

Status key used throughout: **Built and verified** means code exists and a check passed. **Planned** means not done yet. Nothing marked planned should be described as a result.

---

## 1. The project in 30 seconds

### Elevator pitch (technical)
"I reproduced PLoP, a 2025 paper on where to put LoRA adapters when fine-tuning a language model. PLoP scores each type of linear layer by a metric called Normalized Feature Norm, or NFN, and adapts the three types with the lowest scores. I built the scoring code with forward hooks, replaced the paper's sampled baseline with an exact closed form, and checked it against the authors' published numbers. On Llama-3.2-1B it ranks all 7 module types the same way as the paper, with a mean error of about 2.5 percent, and it reproduces the paper's selection sets on 4 of 5 models. Training experiments and a dashboard are planned, not done."

### Three sentences for a non-technical person
1. Big AI models are too large to retrain fully, so people tweak only a few small parts of them.
2. A research paper says there is a cheap test that tells you which parts are best to tweak: the parts that already "match" your task the least have the most room to learn.
3. I rebuilt that test from scratch, checked it against the authors' own numbers, and wrote down where the paper was unclear or wrong.

---

## 2. The big idea

### The analogy: a class project
Imagine a school with 7 kinds of club (Query, Key, Value, and so on). Your school must put on a math show, and you can only afford to train 3 clubs.

- Some clubs are already great at math-like work. Training them gives little extra.
- Other clubs are bad at it. Training those gives big gains.

How do you tell? Give each club a quick math quiz and compare its result to what it would score on random noise. A club that already does well on real math (score well above 1) has little to learn. A club that scores about 1 or below is not "tuned in" to math yet, so it has the most room to improve.

**PLoP picks the clubs with the lowest scores.** That is the whole method.

Why low means more room: the paper's theory says training makes a layer's weights line up with the data it sees. A layer that is already lined up (high score) has less alignment left to gain. A layer that is not lined up (low score) has more.

### The one formula
```
NFN(W, x) = || W z || / || W z~ ||
```
Piece by piece:
- **W**: the layer's grid of numbers (its weight matrix). This is what the model learned.
- **x**: one input piece of text (a token).
- **z**: the numbers that arrive at this layer for that token (its input vector, `z_in`).
- **W z**: what the layer does to those numbers. The result is the layer's output.
- **|| ... ||**: "how big is this?" (the length of the vector).
- **z~**: a fake input made of random Gaussian noise (bell-curve random numbers), scaled to have the same length as z.
- **The fraction**: real output size divided by random output size. It cancels out how big W and z happen to be, so only the match between them remains.

Reading it: NFN near 1 means "this layer treats real text like noise". NFN above 1 means "this layer amplifies real text more than noise", so it is already aligned with the task.

Our code uses an exact version of the denominator:
```
NFN(W) = mean over tokens of ||W z_hat||  /  ( ||W||_F / sqrt(d_in) )
```
where `z_hat` is z scaled to length 1 and `d_in` is the layer's input width. The denominator is the average size of W's output on a random unit-length input. (See Section 7 for why the paper's printed form is off.)

Then PLoP:
1. Compute NFN for every module (a type in a layer).
2. Average over layers to get one number per type.
3. Pick the 3 lowest types (PLoP) or the 3 highest (PLoP^-1, a control).

---

## 3. Theory and glossary

Format: plain meaning, then **Say it:** the interview-ready sentence.

### 3.1 Neural network and transformer basics

**Layer.** One processing step in the model. Data goes in, changed data comes out, and the next layer takes over.
**Say it:** "A transformer is a stack of identical decoder layers; Llama-3.2-1B has 16."

**Weight matrix.** A grid of numbers the model learned. Multiplying an input by it transforms the input.
**Say it:** "Each linear module is a weight matrix W of shape d_out by d_in."

**Attention.** The part that lets each word look at the other words and decide which matter.
**Say it:** "Self-attention mixes information across token positions using query, key and value projections."

**MLP (multi-layer perceptron, the feed-forward block).** The part that thinks about each word on its own after attention has gathered context.
**Say it:** "The MLP block is gate, up and down projections applied per token."

**Projection (q, k, v, o, gate, up, down).** A projection is just a weight matrix that maps vectors into a new space.
- **q_proj (query):** what this word is looking for.
- **k_proj (key):** what this word offers to be found by.
- **v_proj (value):** the content this word passes along.
- **o_proj (output projection):** mixes the attention result back together.
- **gate_proj / up_proj:** expand the vector into a wider one; the gate decides how much passes through.
- **down_proj:** squeezes the wide vector back to normal size.
**Say it:** "There are 7 linear module types per layer: q, k, v, o in attention and gate, up, down in the MLP. Those are the 7 NFN scores."

**Token.** A chunk of text the model reads at once, often a word or part of a word.
**Say it:** "NFN is averaged over tokens; we capped sequences at 256 tokens."

**Activation.** The numbers flowing through the model while it runs. Not the weights, but what the weights produce.
**Say it:** "NFN uses activations: the input z_in and output W z_in of each module."

### 3.2 Fine-tuning

**Pretraining.** The giant first training where the model reads huge amounts of text and learns language.
**Say it:** "We never pretrain; we start from released pretrained checkpoints."

**Fine-tuning.** Extra, smaller training that makes a pretrained model good at one job.
**Say it:** "PLoP is about where to fine-tune, not how to pretrain."

**LoRA (Low-Rank Adaptation).** Instead of changing a huge weight matrix, freeze it and add two thin matrices next to it. Only the thin ones learn.
**Say it:** "LoRA learns a low-rank update B A to a frozen weight W, so far fewer parameters train."

**Rank (r).** How thick the thin matrices are. Bigger rank means more learning capacity and more parameters.
**Say it:** "A rank-r adapter on a d_in to d_out layer has r times (d_in plus d_out) trainable parameters."

**Adapter.** The small add-on (the two thin matrices) that gets trained.
**Say it:** "We place an adapter on each chosen module type in every layer."

**Alpha.** A volume knob on the adapter's output. The paper uses alpha = 2r.
**Say it:** "LoRA scales the update by alpha over r; the paper sets alpha to twice the rank."

**Parameter budget.** The cap on how many numbers you are allowed to train. Fair comparisons must use the same budget.
**Say it:** "We compare placements at equal trainable-parameter budget, choosing rank with `match_rank`."

**PEFT (parameter-efficient fine-tuning).** A family of methods that train only a small part of a model. LoRA is one.
**Say it:** "LoRA is a PEFT method; the Hugging Face `peft` library implements it."

**SFT (supervised fine-tuning).** Training on examples with the right answers shown (question and answer pairs).
**Say it:** "The paper's math experiment is SFT on MetaMathQA, evaluated on GSM8K."

**GRPO (Group Relative Policy Optimization).** A reinforcement-learning method: the model writes several answers, and answers that beat the group average get rewarded.
**Say it:** "The paper also tests LoRA placement under GRPO; we did not attempt it because it needed 2 large GPUs."

### 3.3 The method

**NFN (Normalized Feature Norm).** The score: real output size divided by random-noise output size. See Section 2.
**Say it:** "NFN is a module-data alignment score; about 1 means no alignment, above 1 means aligned."

**Feature norm.** How big the layer's output is. "Feature" means the output vector; "norm" means its length.
**Say it:** "Feature norms grow during training, while the random-input baseline stays flat."

**Alignment.** How well a layer's weights point in the same direction as the data going through it.
**Say it:** "Training aligns weights with the data they see, which shows up as growing feature norms."

**Frobenius norm.** The length of a whole matrix: square every entry, add, take the square root.
**Say it:** "||W||_F is the Frobenius norm; divided by sqrt(d_in) it is the exact random-input gain."

**Random baseline.** The "what would happen with noise" reference that cancels out size effects.
**Say it:** "The baseline is the layer's response to random Gaussian input of the same norm."

**Aggregation.** Combining many numbers into one. Here: average over tokens for a module, then average over layers for a type (unweighted).
**Say it:** "Per-type score is the unweighted mean of per-module NFN over layers, as in the paper."

**PLoP vs PLoP^-1.** PLoP takes the 3 lowest-scoring types. PLoP^-1 takes the 3 highest and is used as a "do the opposite" control.
**Say it:** "PLoP^-1 is the control: if the score means anything, the opposite pick should do worse."

**Monte Carlo vs exact expectation.** Monte Carlo means estimating an average by drawing random samples. Exact expectation means computing the average with algebra, with no sampling noise.
**Say it:** "The authors sample one random vector per token; we use the closed form ||W||_F over sqrt(d_in), and a unit test shows they agree within about 1 percent."

**Forward hook.** A small function you attach to a layer that gets called when data passes through, so you can read values without changing the model.
**Say it:** "We register a forward hook on each linear module; the module's own output is W z, so there is no extra matrix multiply."

**Padding mask.** Short prompts get filler tokens to make a batch rectangular. The mask says "ignore those fillers".
**Say it:** "Our default masks padding; the authors' code counted it, so we keep a faithfulness mode that does the same."

### 3.4 Training and evaluation

**Learning rate.** How big a step the model takes each update. Too big and training blows up; too small and it crawls.
**Say it:** "The paper swept the learning rate and chose the best on test accuracy, which is mildly optimistic."

**Adam vs SignSGD.** Adam adapts each step using recent history. SignSGD uses only the sign (up or down) of each gradient, a simplified Adam. The theory uses SignSGD because it is easier to analyze.
**Say it:** "SignSGD is momentum-free Adam; Theorem 1 is proved for it."

**muP (Maximal Update Parameterization).** A recipe for scaling initialization and learning rates so training behaves the same as the model gets wider. The paper's width-scaling theory builds on it.
**Say it:** "The theory takes width n to infinity in a muP-style setup, so n^-1 ||W z||^2 stays order 1."

**fp16 / bf16 / fp32.** Number formats of 16 or 32 bits. fp16 has small range and can overflow. bf16 has fp32's range but less precision. fp32 is slow but safe.
**Say it:** "Gemma overflowed to NaN in fp16, so our script retries in fp32 and `score_model` raises on non-finite values."

**Seed.** A starting number that makes "random" choices repeatable.
**Say it:** "Our prompt sampling is seeded; the authors' is not, so exact prompts cannot be reproduced."

**Standard error.** How much an average would wobble if you repeated the experiment. Smaller means more trust.
**Say it:** "We report mean and standard error over at least 3 seeds, plus every point." (Planned for training runs.)

**Binomial error.** For accuracy on N test questions, the unavoidable wobble is about sqrt(p(1-p)/N). On GSM8K's 1,319 problems that is about 1.2 to 1.3 points.
**Say it:** "Most gaps between PLoP, MLP and all in the paper's SFT tables are inside binomial noise; only Attn is clearly lower."

**Spearman correlation.** Do two rankings agree? 1 means same order, -1 means opposite, 0 means unrelated.
**Say it:** "X1 will use Spearman between a placement's mean NFN and its accuracy; PLoP predicts a negative value." (Planned.)

**Ablation.** Remove or change one part to see what it contributed.
**Say it:** "Masking padding on and off is an ablation of that choice."

**Control experiment.** A test where the answer is known, to prove your measurement is not fooling you.
**Say it:** "X4 uses random tokens, shuffled words and unrelated prose as controls." (Planned.)

**Baseline.** The simple thing you must beat. Two meanings here: the random-input baseline inside NFN, and the MLP / Attn placements in training comparisons.
**Say it:** "I use baseline in two senses, and I name which one."

---

## 4. What this repo does and how

### File map

| File | Plain-language job | Status |
|---|---|---|
| `plop_explorer/nfn.py` | Attaches hooks to every q/k/v/o/gate/up/down module, scores them, averages, picks the lowest 3 | Built, verified |
| `plop_explorer/data.py` | Loads and samples prompts for math (GSM8K), code (HumanEval), history and logic (MMLU), seeded | Built |
| `plop_explorer/lora_params.py` | Counts trainable LoRA parameters; finds the rank that matches a budget | Built, verified |
| `scripts/check_reference.py` | Compares our Llama-1B math scores with the paper's 7-number table | Built |
| `scripts/check_selection.py` | Checks whether our picks match the paper's selection sets for each model | Built |
| `tests/test_nfn.py` | Checks naming, random input scores near 1, aligned input above 1, padding mask, grid shape, exact vs sampled baseline | Built, passing |
| `tests/test_lora_params.py` | Checks 16 of the paper's parameter counts and the ANLI ranks | Built, passing |
| `reference/plop_reference_by_type.json` | The authors' published per-type scores (ground truth) | Data |
| `reference/plop_fig4_llama1b_math.json` | The paper's 7 by 16 per-layer grid, extracted from the PDF | Data |
| `docs/plop-implementation-plan.md` | Source of truth: status, gates, experiment IDs, discrepancy log | Doc |
| `app/` (FastAPI + Plotly.js dashboard) | Interactive explorer | **Planned, not built** |

### One scoring run, end to end
1. **Load.** `check_reference.py` loads Llama-3.2-1B-Instruct in fp16 on the Mac GPU (MPS), or fp32 on CPU.
2. **Prompts.** `load_prompts("math", 100, seed=0)` samples 100 GSM8K questions (from the first 200 train questions, like the authors).
3. **Batches.** `make_batches` tokenizes in batches of 8, max 256 tokens, with the end-of-sequence token as padding on the right (the authors' setup).
4. **Hooks.** `score_model` finds each linear module by its leaf name, computes `||W||_F / sqrt(d_in)` once, and registers a forward hook.
5. **Forward pass.** Only the transformer body runs, not the vocabulary-sized output head, under `torch.no_grad()`.
6. **Per token.** The hook computes output length divided by input length, which equals `||W z_hat||`. A mask zeroes padding tokens (or not, in faithfulness mode).
7. **Per module.** Sum over tokens, divide by token count, divide by the random gain. That is the module's NFN.
8. **Per type.** `by_type()` averages over layers. `grid()` returns type by layer for a heatmap. `recommend(3)` returns the 3 lowest.
9. **Safety.** A NaN or infinite score raises `FloatingPointError`.
10. **Report.** The script prints a table against the reference, and saves per-module JSON to `results/`.

---

## 5. Tech stack and why

| Piece | Why it is here | Alternative not picked |
|---|---|---|
| **PyTorch** | Models are PyTorch modules; forward hooks come built in | (Other frameworks exist, but the Hugging Face models and the authors' code are PyTorch) |
| **Hugging Face `transformers` (>=5)** | Loads Llama, Qwen and Gemma by name with tokenizers | Writing model code by hand would not be faithful or fast |
| **Hugging Face `datasets`** | Downloads GSM8K, HumanEval, MMLU the same way as the authors | Hand-downloading files |
| **pytest** | Fast CPU-only tests that guard the math (parameter counts, baseline) | Ad hoc scripts |
| **PEFT** (planned) | Standard LoRA implementation for the ANLI and SFT runs | Hand-rolled LoRA |
| **FastAPI** (planned) | Backend for the scoring API with polling for slow jobs | Streamlit or Gradio were rejected in the plan |
| **Plotly.js** (planned) | Interactive heatmap and bars in a custom page | Streamlit or Gradio charts |
| **Hugging Face Spaces with Docker** (planned) | Free hosting; Docker because the backend is FastAPI | Streamlit or Gradio Spaces |
| **Kaggle GPUs** (planned) | Free GPU for training runs; T4/P100 lack good bf16, so fp16 with loss scaling, recorded as a deviation | Paid cloud GPUs |

Machine limits worth knowing: 8 GB RAM, so models up to about 1.7B fit in fp16. The Mac GPU backend (MPS) hung twice, so long runs use CPU in the background.

---

## 6. What we reproduced and the honest results

### Gate summary

| Gate | Status |
|---|---|
| G1 parameter counter | Passed. All 16 of the paper's #Params values reproduced |
| G2 scoring matches authors | Passed with a documented gap (mean error about 2.5%) |
| G3 selection sets | Passed on 4 of 5 models; Gemma pending |
| G4 E1 linear network, G5 dashboard, G6-G9 | Not started |

### Llama-3.2-1B on math: the 7-number table

| Type | Paper | Ours (reference setup) | Diff | Ours (padding masked) |
|---|---|---|---|---|
| Query | 2.579 | 2.663 | +3.3% | 2.762 |
| Key | 2.627 | 2.757 | +5.0% | 2.937 |
| Value | 0.969 | 0.973 | +0.4% | 0.983 |
| OutProj | 0.900 | 0.905 | +0.6% | 0.915 |
| GateProj | 1.401 | 1.496 | +6.8% | 1.622 |
| UpProj | 1.112 | 1.126 | +1.3% | 1.146 |
| DownProj | 1.048 | 1.055 | +0.7% | 1.067 |

- Same ranking as the paper. Same lowest three: Value, OutProj, DownProj.
- Mean error about 2.5%. Three types (Query, Key, Gate) are 3 to 7% high.
- **Not** caused by prompt sampling (about 0.01 spread over 5 seeds), by per-batch vs pooled averaging, or by pad token choice. Left padding is clearly wrong (mean error 9 to 10%).
- Remaining suspects, **unverified**: the authors' older `transformers` version and their CUDA fp16 vs our MPS fp16. We chose to stop chasing it and document it.

### Selection sets (100 prompts, seed 0)

| Model | Our lowest 3 (PLoP) | Highest 3 (PLoP^-1) | Matches paper |
|---|---|---|---|
| Llama-3.2-1B-Instruct | Down, Out, Value | Gate, Key, Query | All 4 tasks (PLoP set) |
| Qwen2.5-0.5B | Out, Up, Value | Gate, Key, Query | All 4 tasks |
| Qwen3-0.6B | Down, Up, Value | Gate, Key, Query | All 4 tasks, both sets |
| Qwen3-1.7B | Down, Out, Value on math | Gate, Key, Query | Math. Other tasks pick Up instead of Out |
| Gemma3-1B | **Pending** | **Pending** | fp16 overflowed to NaN; fp32 CPU rerun in progress |

- **Qwen3-1.7B near-tie:** OutProj, UpProj and DownProj all score 0.95 to 0.99, so a swap is plausible. The paper's picks come from math, and math matches.
- The "Value below 1" oddity reproduces: Qwen3-0.6B Value is 0.68 to 0.72, Qwen3-1.7B 0.74 to 0.76. The paper says it has no explanation, and neither do we.

### Verified vs not verified

**Verified (by our code and checks):** parameter counts; scoring ranking and pick on Llama-1B math; selection sets on 4 of 5 models; exact vs sampled baseline agree (unit test); "Attn" means Q, K, V only (the only reading that gives the paper's ranks).
**Not verified / not done:** Gemma picks; feature-norm growth (E1); whether PLoP actually improves accuracy (all training); the dashboard; X1 and X4 to X6. The paper's accuracy numbers are quoted as reported, not reproduced.

---

## 7. What the paper got wrong or left unclear

Plain words, from the plan's discrepancy log (19 items). The biggest ones:

1. **Theorem 1 has a typo-level mismatch.** The statement says the feature-norm growth term is Gamma_0 + beta^2 (1 + t(t-1)). The proof's recursion adds up to Gamma_0 + beta^2 t^2. They agree at t = 1 and differ elsewhere. We do not know which is right; E1 will simulate it (planned).
2. **Missing sqrt(d_in) factor.** Section 3 prints the approximation with `||W||_F ||z||` in the denominator. The exact expectation has an extra factor of sqrt(d_in). Our code uses the exact form. We confirmed this on the rendered page.
3. **Squared vs unsquared.** The boxed algorithm uses squared norms, the definition uses unsquared. The authors' code uses unsquared. Resolved.
4. **Batch size confusion.** Text says batch 200, command says batch 8 with 100 samples. The code uses 8 and 100. Averaging per batch or pooled gives the same answer.
5. **SignSGD results missing.** The paper says Appendix C has SignSGD results for the linear network. It does not.
6. **Probability bound mismatches.** Theorem 1's statement and proof give different probabilities, and Theorem 2's bound on T differs between statement and proof.
7. **Typos and inconsistencies.** "Qwen3.5-0.5B" in the appendix vs Qwen2.5-0.5B in the figure; GRPO answer tags `<solution>` vs `<answer>`; linear-network data scale written two ways.
8. **Text vs table.** The text says Llama's Value, Gate, Down and Up hover around 1, but Gate is 1.40.
9. **Learning rate chosen on the test set** in the SFT tables, which is mildly optimistic.
10. **Rank choice unexplained.** Qwen2.5-0.5B PLoP uses rank 17 (3.50M parameters) when rank 16 (3.29M) is closer to the 3.32M budget.
11. **Gemma in fp16.** Authors report fp16, but Gemma gives NaN with current `transformers`.
12. **Unseeded prompt sampling** means their exact prompts cannot be recreated.
13. **Picks barely change across tasks** within a model. That challenges the "task-aware" framing and motivates X4.

Plus our own unexplained 3 to 7% gap on Query, Key and Gate.

---

## 8. Our own extensions and why they matter

| ID | What | Why it matters | Status |
|---|---|---|---|
| X1 | Train all 35 three-type placements (C(7,3)) on ANLI with Qwen2.5-0.5B at equal budget; plot accuracy vs mean NFN; Spearman; rank of PLoP, MLP, Attn | Tests whether the score itself predicts benefit; gives the missing "random placement" baseline | Planned |
| X2 | Recover the authors' exact scoring recipe | Needed for faithful reproduction | **Done**, read from their code |
| X3 | Statistics: seeds, standard error, binomial error, paired tests | Stops us claiming wins inside noise | Planned (protocol written) |
| X4 | Does NFN depend on the task or mostly on the model? Controls: random tokens, shuffled words, plain prose | Picks look the same across tasks; this tests whether PLoP is task-aware | Planned, not run |
| X5 | How many prompts give a stable pick (5, 10, 25, 50, 100)? | Decides if a fast "paste your own prompts" feature is credible | Planned |
| X6 | Base vs instruction-tuned: does post-training change the map or pick? | Shows what NFN reflects | Planned |

---

## 9. Interview Q&A

**Conceptual**

1. **What problem does PLoP solve?**
LoRA needs you to choose which module types get adapters. Most people guess (attention only, MLP only, or everything). PLoP gives a cheap, data-driven rule using only a forward pass over some task prompts, with no training.

2. **Explain NFN to a non-expert.**
It asks how strongly a layer reacts to real task text compared with random noise of the same size. About 1 means no special match. Higher means it is already tuned in.

3. **Why do low scores mean more room to learn?**
The theory says training aligns weights with data and so grows feature norms. A module with a high score has already grown. A low-score module has more alignment left to gain, so adapters there should help most.

4. **Why is NFN near 1, not 0?**
It is a ratio, not a distance. Noise of the same length still produces an output of a similar size, so "no alignment" gives about 1. Zero would mean the layer erases the input. Values below 1 mean the layer shrinks real text more than noise, which is what we see for Value in Qwen3.

5. **Why use the exact baseline?**
Sampling one random vector per token adds noise. The expected squared size has a closed form, `||W||_F^2 / d_in` times `||z||^2`. We use the square root of that. It removes sampling noise and avoids an extra matrix multiply. A unit test shows it agrees with sampling within about 1%. A caveat: square root of an average is not the average of square roots (Jensen's gap), but the gap is of order 1 over d_out.

6. **What is a forward hook and why use one?**
A callback fired when a layer runs. It lets us read each module's input and output without editing the model. The output is already W z, so the hook costs almost nothing.

7. **What does the LoRA parameter formula look like?**
r times (d_in + d_out) per adapted module, summed across modules and layers. We verified this against 16 of the paper's reported counts.

**Design decisions**

8. **Why mask padding when the authors did not?**
Pad tokens are not real text, so counting them blurs the score. Masking is the principled default. We kept a faithfulness switch because the authors' numbers include padding, and we needed it to compare fairly.

9. **Why average layers without weighting?**
That is what the paper does. Changing it would break comparability with its numbers.

10. **Why did you only run the transformer body and skip the output head?**
The head makes vocabulary-sized outputs that NFN never uses. Skipping it saves compute and memory on an 8 GB machine.

11. **Why use leaf names like `q_proj` to find modules?**
It works across Llama, Qwen and Gemma, which share those names. Unsupported architectures raise a clear error instead of silently scoring nothing.

12. **How did you decide what "Attn" means?**
By testing readings against the paper's parameter counts. Only Q, K, V (no O) reproduces its ranks, for example Llama-1B Attn rank 27 giving 3.98M.

**Does it work? Honest limits**

13. **Does PLoP actually win?**
Not shown by me yet. The paper reports PLoP beats Attn clearly and is about equal to MLP and "all". Its own tables show most differences within about 1.3 points of binomial noise. I have reproduced the scoring and picks. I have not run the training experiments.

14. **What is the weakest part of this project?**
The training evidence: I have no accuracy results of my own. Also a 2.5% unexplained scoring gap, Gemma still pending, and no dashboard yet.

15. **Your numbers are 3 to 7% off on some types. Why trust them?**
The ranking and the top-3 pick match. The gap is not from sampling, averaging or pad token. The gap is biggest where padding positions matter most, so I suspect environment differences, but I have not proven that. I documented it and kept a toggle.

16. **Qwen3-1.7B picked Up instead of Out on some tasks. Is that a failure?**
No. Out, Up and Down score 0.95 to 0.99, a near-tie. The paper's pick comes from math and math matches. I report it as a near-tie, not a clean match.

17. **Picks look the same across tasks. Does that undermine the paper?**
It raises a fair question: is PLoP task-aware, or a per-model constant? The paper itself says tasks share a base alignment level. X4 would measure it with controls. It is planned, not done, so I do not claim an answer.

**What went wrong / what I would do differently**

18. **What went wrong?**
Gemma overflowed to NaN in fp16, and the Mac GPU backend hung twice. I added a non-finite check, an fp32 retry, and moved long runs to CPU.

19. **What would you do differently?**
Run X4 before the dashboard, since it may change what the first page emphasizes. I would also write a README earlier so the repo explains itself.

20. **How did you validate correctness?**
Layers of checks: unit tests (random inputs near 1, aligned above 1, exact vs sampled baseline, 16 parameter counts); comparison to the authors' per-type table; comparison of selection sets across models.

**Tricky**

21. **Why not just train every placement and pick the best?**
It is expensive. PLoP is meant to need one forward pass. X1 is how I would test whether the cheap score is a good proxy.

22. **Is a higher NFN better or worse?**
For PLoP, lower is picked. Higher means already aligned. PLoP^-1 picks the highest as a control.

23. **What does Spearman tell you that accuracy alone does not?**
Whether the score ranks placements in the right order, regardless of the scale of accuracy.

24. **How would you know a win is real?**
Multiple seeds, paired per-seed differences, binomial error, and a confidence interval that excludes zero.

25. **Why did the paper's choice of learning rate matter?**
It was chosen on the test set, which can inflate results slightly. I would choose on a held-out validation split.

---

## 10. One-page cheat sheet

### 10 numbers
1. **7** module types scored (q, k, v, o, gate, up, down).
2. **3** types picked (k = 3).
3. **~2.5%** mean error vs the paper on Llama-1B math.
4. **2.58 / 2.63** Query / Key in the paper (highest); **0.90** OutProj (lowest).
5. **16 of 16** paper parameter counts reproduced.
6. **4 of 5** models match the paper's selection sets (Gemma pending).
7. **100 prompts, batch 8, 256 tokens**: the authors' scoring setup.
8. **35** = C(7,3) three-type placements for X1.
9. **1,319** GSM8K test problems, giving about 1.2 to 1.3 points of binomial error.
10. **0.95 to 0.99** the Qwen3-1.7B near-tie (Out, Up, Down).

### 10 terms
1. **NFN**: real output size over random-input output size.
2. **LoRA / rank / adapter**: thin trainable add-on to a frozen weight.
3. **PLoP**: adapt the 3 lowest-NFN module types.
4. **PLoP^-1**: the 3 highest, as a control.
5. **Frobenius norm**: whole-matrix length; the exact baseline is `||W||_F / sqrt(d_in)`.
6. **Forward hook**: callback that reads a layer's input and output.
7. **Padding mask**: ignore filler tokens (our default; authors counted them).
8. **Parameter budget**: equal trainable parameters for a fair comparison; `match_rank`.
9. **Binomial error**: accuracy noise from a finite test set.
10. **Spearman correlation**: do two rankings agree (X1's key statistic, planned).

### Three honest sentences to end on
- "I reproduced the scoring and the picks; I did not yet reproduce the accuracy gains."
- "The paper has real errors, a missing sqrt(d_in) factor and an inconsistent theorem, and I logged them."
- "My main extension, testing whether NFN predicts benefit across all 35 placements, is planned, not done."
