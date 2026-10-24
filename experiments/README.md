# Running E6 on Kaggle

1. kaggle.com, New Notebook, Settings, Accelerator: **GPU T4 x2** (or P100), Internet **On**.
2. In the first cell:
   ```
   !git clone https://github.com/tryzhaa/plop && cd plop && pip -q install -r requirements.txt peft accelerate
   ```
3. **Pilot** (one placement, one seed, 300 steps). Note `s/step` and `peak_gpu_gib`:
   ```
   !cd plop && python experiments/e6_anli_train.py --model Qwen/Qwen2.5-0.5B --placement plop --seed 0 --steps 300
   ```
4. Send the printed lines (placement/rank/adapter_params, the `step ...` lines, `final:`) back; they decide the number of steps and seeds.
5. Full grid (after the pilot sets `STEPS` and `SEEDS`):
   ```
   !cd plop && STEPS=5000 SEEDS="0 1 2" bash experiments/run_e6_grid.sh Qwen/Qwen2.5-0.5B
   ```
6. Download `plop/results/e6/*.json` and `*.csv` (Output tab, or `!zip -r e6.zip plop/results/e6`).

Kaggle notebooks stop after about 12 hours, so run the grid in chunks (one placement per session) if the pilot says it is slow.
