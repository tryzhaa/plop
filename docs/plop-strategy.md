# PLoP Explorer: Implementation Strategy

Paper: Hayou, Ghosh, Yu, "PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models", arXiv:2506.20629v1 (June 2025).

---

## 1. Decisions locked

| Topic | Decision |
|---|---|
| Core idea (checked) | Low NFN = module isn't tuned to the task = most room for an adapter to learn |
| Audience | Recruiter first, engineer second: clean default view, "Advanced" panel for settings |
| Page 3 | One small, honest training experiment (1 model, 1 task, 3 placements, 3 seeds) |
| Faithfulness | Toggle. Default is our cleaner scoring (padding excluded, exact baseline); switch reproduces the authors' setup |
| GPU | Kaggle free tier (P100 / T4) |
| Launch models | Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-1B-Instruct |
| If PLoP doesn't win | Publish honestly |
| Explainer | Analogy first, then one formula |

**Defaults I chose for questions we haven't asked (change any of them):**
- Recommendation is the 3 lowest-scoring types, as in the reference code; the user can change it to 1-7.
- Heatmap uses one fixed color scale with 1.0 marked, so colors mean the same across models.
- Tasks: the paper's four (math, code, history, logic) plus "paste your own prompts".
- "Any Hugging Face model": accept any ID; if layer names don't match `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`, show a friendly error.
- Scoring settings (samples, sequence length, padding mode) live in the Advanced panel with the paper's values as defaults (100 samples, length 256).

---

## 2. The method

For every weight matrix W in the model and every task token, take the module's input vector z_in and compare how much W amplifies it versus a random vector of the same length:

`NFN(W, x) = ||W z_in(x)|| / ||W z~_in(x)||`, where z~_in is Gaussian noise scaled to the same norm as z_in.

- NFN near 1: W treats the task like noise (not tuned to it).
- NFN above 1: W amplifies the task's inputs (already tuned).
- NFN below 1: W amplifies them less than noise.

Pipeline (paper Section 3):
1. Run ~100 prompts through the model once with forward hooks (no gradients).
2. Average NFN over the prompts for every matrix (this gives the layer x type heatmap, the "NFN-map").
3. Average over layers per type (q, k, v, o, gate, up, down).
4. Put LoRA on the lowest-scoring types.

Compute: about one batched forward pass (the paper used batch 200, max length 256).

### Things to resolve while building
1. **Squared vs. unsquared.** Definition 1 defines NFN as a ratio of norms. The boxed algorithm (Step 1) writes it with squared norms. The published table is the tiebreaker: whichever version reproduces Llama-1B's numbers is what the authors' code computes.
2. **The exact baseline.** If z~ is a random direction with the same norm as z, then E||W z~||^2 = ||z||^2 * ||W||_F^2 / n (n = input dimension). That is the closed form behind our "exact baseline" and removes sampling noise. Confirm it numerically against a Monte Carlo estimate as a unit test.
3. **Padding.** Filler tokens in a batch change the average. Our default masks them; the toggle includes them to mimic the authors.
4. **"Negative alignment."** The paper's wording for Value modules at ~0.75 (Qwen3-1.7B, Gemma3-1B) means below 1, not negative. The authors say they have no explanation. The tool should display it as "below the random baseline" and flag it as an open question.

---

## 3. Verification targets (how we know the code is right)

**Primary check: Llama-3.2-1B-Instruct, math (paper Section 5 sample output):**

| Type | NFN |
|---|---|
| k_proj | 2.63 |
| q_proj | 2.58 |
| gate_proj | 1.40 |
| up_proj | 1.11 |
| down_proj | 1.05 |
| v_proj | 0.97 |
| o_proj | 0.90 |

Pass criteria: same ranking, same top-3 pick (o, v, down), and each type within a few percent. Exact equality is not expected because the authors' prompt sampling was unseeded.

**Secondary checks (qualitative, from the paper's figures):**
- Llama-1B NFN-map (Fig. 4): Query and Key rows high (roughly 2-4), Value/Out/Down/Up near 1.
- Qwen3-1.7B (Fig. 8): Value row clearly below 1 (about 0.75 aggregated), Gate row high.
- Specialized models score higher (Fig. 6): Qwen2.5-Math-1.5B and Qwen2.5-Coder-1.5B vs. base Qwen2.5-1.5B on their own tasks.
- Ranking of types is similar between Llama-3.2-1B and 3B (Fig. 5).

Note from the context file: the authors' other published tasks use an older, incompatible scale, so only the Llama-1B math table is a hard numeric target.

---

## 4. Product plan (what the recruiter sees)

**Page 1: NFN Explorer**
- Model dropdown, task dropdown, "Run" (cached after first run).
- Hero: the NFN-map heatmap (fixed color scale, 1.0 marked), then the by-type bar chart with the baseline line at 1.0.
- Callout: "PLoP recommends adapting: V, O, Down" with a one-line reason.
- Explainer panel: microphone analogy, then the one NFN formula.
- Advanced panel: number of samples, sequence length, padding mode (ours / authors'), how many types to recommend.

**Page 2: Compare**
- Two models side by side on the same task; highlight where the type ranking differs.

**Page 3: Does it work?**
- Charts and table for the training experiment (section 5), with an honest "what this does and doesn't show" box.

**Polish:** demo GIF in the README, license, repo description and topics, short write-up.

---

## 5. Page 3: training experiment design

**Goal:** one honest data point on whether PLoP's choice trains as well as the standard habits.

**Recommended setup: mirror the paper's ANLI experiment (Section 4.1, Appendix C.2), smaller.**
- Model: Qwen2.5-0.5B (fits a free GPU; PLoP picks V, O, Up in the paper).
- Task: ANLI classification, evaluated on held-out data.
- Placements: PLoP (V, O, Up), Attn (Q, K, V, O or as defined in our code), MLP (gate, up, down). Optional: PLoP inverse (highest scores) as a control.
- Match trainable parameters across placements, as the paper does. The paper used MLP r=8, Attn r=36, PLoP r=17 for the 0.5B model (the appendix says "Qwen3.5-0.5B", but Figure 7 says Qwen2.5-0.5B; treat it as a typo and note it).
- Rank formula so you can compute it yourself: a LoRA adapter on a matrix with input size d_in and output size d_out has r * (d_in + d_out) parameters. Sum over all adapted matrices in all layers, then solve for r per placement.
- Paper's settings: AdamW, linear schedule, no warmup, dropout 0.1, max length 256, LoRA alpha = 2r; curves run to about 5000 steps.
- Seeds: 3 per placement (controls data order and LoRA init).
- Report: mean and spread of accuracy across seeds, plus the training curve. Do not cherry-pick the best seed.

**Practical constraints for Kaggle (important):**
- P100 and T4 do not support bfloat16 well. The paper trained in bf16, so we will use fp16 with loss scaling or fp32, and say so in the write-up. This is a deviation from the paper.
- The paper does not state batch size or exact step schedule; we choose our own and document them.
- Run one seed for one placement first and time it, then decide how many steps and seeds fit in the weekly GPU quota. I have not measured this; treat any time estimate as unknown until then.

**Expectations to set honestly:**
- In the paper's Llama-1B ANLI result, PLoP and MLP are about equal and both beat Attn, because Llama's MLP modules also have low NFN. Our result may show "PLoP ties MLP, beats Attn", which is a fair and publishable finding.
- With 3 seeds, gaps of about a point are probably within noise. Say so.
- A stretch second experiment is MetaMathQA to GSM8K SFT (paper Section 4.2, Tables 1-2), but it needs longer training and generation-based evaluation. Only attempt it if the ANLI run goes smoothly.
- Skip GRPO entirely: the authors used 2x GH200.

---

## 6. Phased plan with acceptance tests

| Phase | Work | Done when |
|---|---|---|
| 0 | Confirm existing `nfn.py`, `data.py` and the reference JSON run end to end | One command prints per-type NFN for a small model |
| 1 | Verify against Llama-1B math table; resolve squared vs. unsquared; unit-test exact baseline vs. Monte Carlo | Ranking and top-3 match; each type within a few percent |
| 2 | Qualitative checks on Qwen3-1.7B, Qwen2.5 base vs. specialized | Patterns from Fig. 6 and Fig. 8 reproduced |
| 3 | Page 1 (heatmap, bars, recommendation, Advanced panel), caching | Any listed model gives a result in under a minute on CPU (models up to ~1B), instant on repeat |
| 4 | Page 2 (compare) and "any HF model" input with friendly errors | Two models render side by side |
| 5 | Deploy (Hugging Face Space or Cloud Run) | Public link works cold |
| 6 | Page 3: pilot run, then 3 placements x 3 seeds | Results table and curves committed with the exact config |
| 7 | README, GIF, license, topics, write-up including what did not reproduce | A stranger can understand it in 60 seconds |

Order matters: Phase 1 is the trust foundation. Do not build UI on unverified numbers.

---

## 7. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Our numbers differ from the authors' | Expected (unseeded sampling, padding choice). Show both modes, report the gap plainly |
| Llama is gated | Request access now; Qwen models are the fallback |
| Training result is a tie or a loss | Publish it; this is the plan already agreed |
| Free-GPU limits | Pilot first, cut steps or seeds before cutting honesty |
| fp16 vs. paper's bf16 | Disclose in the write-up |
| Unexplained low-Value anomaly | Show it, label it as an open question, do not invent a reason |

---

## 8. Resume framing (draft, fill brackets with real numbers only)

**PLoP Explorer** | Python, PyTorch, Hugging Face, Streamlit/Gradio
- Reimplemented the NFN alignment score from a 2025 LoRA-placement paper and reproduced its published Llama-3.2-1B scores to within [X]% with the same top-3 module choice.
- Built an interactive dashboard (heatmaps, per-type comparison, any-model input) with cached scoring, deployed publicly.
- Trained LoRA under three placements at matched parameter counts over 3 seeds: PLoP [result vs. Attn] and [result vs. MLP].
- Documented deviations from the paper (exact baseline, padding handling, fp16 vs. bf16) and unexplained anomalies.

---

## 9. Questions still open (answer when convenient, defaults apply otherwise)

1. Hosting: Hugging Face Space or Cloud Run (matching Research-Copilot)?
2. Framework: are you comfortable with Streamlit/Gradio, or should we plan a custom frontend?
3. Timeline: weekend, two weeks, or ongoing?
4. Have you requested Llama-3.2 access on Hugging Face yet?
5. Do you want the second training experiment (MetaMathQA to GSM8K) as a stretch goal?
