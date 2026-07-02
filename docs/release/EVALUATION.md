# μ₀ — Evaluation

Two ways to evaluate a trained checkpoint:

- **(A) Batch image-only prediction** (`lerobot_predict_trace_mu0_image_only.py`)
  — runs the model over test episodes and writes trace overlays (+ metrics).
  Depth and keypoints can be **synthesized from the image** (Depth-Anything-V2 +
  Grounded-SAM-2) or taken from the dataset, via two ablation flags.
- **(B) Interactive GUI** (`lerobot_trace_gui.py`) — a browser UI to drop
  keypoints on an image and see predicted traces live.
- **(C) VLM API baselines** (`lerobot_predict_trace_gpt.py`,
  `lerobot_predict_trace_gemini.py`) — predict traces by querying a multimodal
  LLM (OpenAI / Gemini) instead of the trained policy, writing the **same
  overlay + metric outputs** as (A). No checkpoint; needs your own API key.

The checkpoint and stats are the same artifacts produced/used by training:
`final_ckpt` (the released main model) and `normalizer_stats.json`.

---

## 1. Environment

### Minimal — dataset keypoints, no depth synthesis
The headline command below (`--use_kp_from_dataset=true
--use_depth_from=none`) loads **neither** Depth-Anything nor Grounded-SAM, so the
**training env** (`mu0`, see `TRAINING.md`) is enough — no extra setup.

### Full — synthesized depth/keypoints, and the GUI
The full image-only mode and the GUI need a superset env with Depth-Anything-V2,
Grounded-SAM-2 (GroundingDINO + SAM 2.1), and viser:

```bash
# 0. Fetch the vendored inference repos (Grounded-SAM-2 + Depth-Anything-V2),
#    which are git submodules. If you cloned the repo with `--recursive` they
#    are already present; otherwise:
git submodule update --init infer_helpers/Grounded-SAM-2 infer_helpers/Depth-Anything-V2

# Clone the training env, add the inference deps
conda create -n mu0_infer --clone mu0 -y
conda activate mu0_infer
pip install matplotlib opencv-python
pip install "supervision==0.21.0" pycocotools nltk addict yapf timm
pip install viser

# Build the GroundingDINO / SAM 2 CUDA extensions. CUDA_HOME must match the
# toolkit torch was built against, and gcc must be >= 9.
module load cuda/12.8.1 gcc/11.2.0          # adjust per your system
export CUDA_HOME=/opt/common/cuda/cuda-12.8.1
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
cd infer_helpers/Grounded-SAM-2
pip install --no-build-isolation -e .
pip install --no-build-isolation -e grounding_dino
cd -

# Download the 3 checkpoints (~2.8 GB) into infer_helpers/checkpoints/
mkdir -p infer_helpers/checkpoints && cd infer_helpers/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ../..
```

> **Troubleshooting.** If GroundingDINO segmentation fails with `NameError: name
> '_C' is not defined` (usually preceded by *"Failed to load custom C++ ops.
> Running on CPU mode Only!"*), its CUDA op didn't load. It must be built **in the
> same repo clone you run from** — the GUI imports `grounding_dino` relative to the
> repo root — and **against this env's torch**. Re-run the two `pip install
> --no-build-isolation -e` commands above from this clone with `CUDA_HOME` set,
> then launch from the repo root. (The `gcc/11.2.0` module is only needed if your
> system `gcc` is < 9.)

---

## 2. Data

Test episodes use the same TraceExtract directory format as training (§2 of
`TRAINING.md`). Pass them via `--test_dirs` (globs allowed).

Download the release bundle (see also the README **Release artifacts** section):

```bash
hf download furonghuang-lab/mu0 --local-dir mu0_release
cd mu0_release && tar -xf test_set.tar
```

This gives `final_ckpt/`, `test_set/`, and `normalizer_stats.json` — pass
them to `--checkpoint`, `--test_dirs`, and `--delta_stats_path` respectively.

---

## 3. (A) Batch image-only prediction

### Headline command — dataset keypoints, no depth, no history

```bash
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_predict_trace_mu0_image_only.py \
  --checkpoint=/path/to/final_ckpt \
  --test_dirs='[/path_to/test_set/*/*]' \
  --delta_stats_path=/path/to/normalizer_stats.json \
  --first_frame_only=true --first_frame_offset=[4,8,12] \
  --n_min=16 --n_max=16 \
  --use_kp_from_dataset=true --use_depth_from=none --no_provide_history=true \
  --num_samples_for_metrics=5 \
  --diagnose_alignment=true \
  --output_dir=outputs/eval --seed=0
```

> `--save_depth_panels=true` is a no-op in `--use_depth_from=none` mode (there is
> no depth to render); it only writes a `depth/` panel dir in the depth-fed modes
> (`--use_depth_from=data|pred`).

### Ablation modes
Two flags swap a synthesized input back for its dataset original:

| `--use_kp_from_dataset` | `--use_depth_from` | what runs |
|---|---|---|
| true  | data | use query keypoint from dataset, with dataset depth |
| true  | none | use query keypoint from dataset, without depth |
| true  | pred | depth synthesized by Depth-Anything-V2 (measures depth-synthesis cost) |
| false | pred | full image-only — keypoints from Grounded-SAM-2, depth from DA-V2 (no GT, no L1) |

The last two rows require the **inference env** + the 3 checkpoints. Extra knobs:
`--num_keypoints=16` (image-only keypoint count), `--depth_align=log_percentile`,
`--gdino_box_threshold` / `--gdino_text_threshold`.

### Outputs
Under `--output_dir`:
- Trace overlays in `rollout_img/` — one PNG per sample named
  `sample_NNNNNN_<episode>_fNNNNNN.png` (predicted future trace drawn on the frame).
- `metrics.json` at the full horizon, plus `metrics_8.json` / `metrics_16.json`
  for the short-horizon slices (`--metric_horizons`, default 8/16/32). Keys
  include `l1_abs|uv|depth|delta`, `ade|fde|dtw|frechet_*`, best-of-N
  `minADE_5|minFDE_5|minDTW_5|minFrechet_5_*`, and `smooth_pred|gt_*`. Metrics
  need ground-truth keypoints, so they are written **only when
  `--use_kp_from_dataset=true`**; the full image-only mode
  (`--use_kp_from_dataset=false`) writes overlays but no metrics.
- (Depth-fed modes only) depth panels in `depth/` when `--save_depth_panels=true`
  *and* `--use_depth_from` is `data` or `pred`.
- (Image-only mode only, `--use_kp_from_dataset=false`) `hand_mask_status.json`
  — a per-sample list recording the Grounded-SAM keypoint-detection status
  (`video_dir`, `frame_idx`, and a `pass1`/`pass2` detection-stage flag).

Filenames are seeded/deterministic, so the same `--seed` yields identical names
across runs for side-by-side diffing.

---

## 4. (B) Interactive GUI

Needs the **inference env** + the 3 checkpoints (it synthesizes depth/keypoints
on the fly).

```bash
conda activate mu0_infer
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_trace_gui \
  --checkpoint=/path/to/final_ckpt \
  --delta_stats_path=/path/to/normalizer_stats.json \
  --port=8080
```

The server loads μ₀ + Depth-Anything-V2 + Grounded-SAM-2 once, then serves a
viser UI on `--port`. Open it in a browser at `http://localhost:8080`. If you run
on a remote machine, forward the port from your laptop first:

```bash
ssh -L 8080:localhost:8080 user@host   # then visit http://localhost:8080
```

Workflow in the UI (depth and keypoints are synthesized on the fly, so no
dataset is needed):

1. **Load an image** — click **Upload your image**, or click **See sample
   image** to load a bundled example with its task/classes pre-filled.
2. **Set the prompts** — type your instruction in **Task prompt** (e.g.
   `pick up the doll`) and the object words in **Classes** (e.g. `doll`,
   comma-separated for multiple).
3. **Run segmentation** — detects masks for the listed classes; tick which
   masks to track.
4. **Predict trace** — the predicted 3D traces render live in the scene.

> **Step-by-step walkthrough (with a demo video):**
> [`INTERACTIVE_GUI.md`](INTERACTIVE_GUI.md).

---

## 5. (C) VLM API baselines (GPT / Gemini)

`lerobot_predict_trace_gpt.py` and `lerobot_predict_trace_gemini.py` are
**comparison baselines**: instead of the trained μ₀, they ask a
multimodal LLM (OpenAI / Google Gemini) to predict each keypoint's future u/v
from `(RGB image, optional depth panel, text prompt that lists the current
keypoints and — optionally — their past trajectories)`. There is **no
`--checkpoint`** (no policy weights), and they always use the dataset's query
keypoints (no Grounded-SAM). Depth (z) is not asked of the model — `pred_z` is
set to `gt_z`, so the `*_uv` metrics are the meaningful numbers and `*_depth ≈ 0`.

Outputs match (A): one trace overlay per sample in `rollout_img/`, plus the same
`metrics.json` / `metrics_{8,16,32}.json`. Two extra VLM-only dirs are also
written: `raw_responses/` (the cached API JSON — enables `--offline` replay) and
`annotated_inputs/` (the exact RGB / depth / prompt sent to the API; pass
`--save_annotated_inputs=false` to skip).

### Environment
The default mode (`--use_depth_from=none`, RGB + text only) runs in the
**training env** (`mu0`); you only need the API client SDK:

```bash
pip install openai        # for lerobot_predict_trace_gpt.py
pip install google-genai  # for lerobot_predict_trace_gemini.py
```

(The `mu0_infer` env already has both.) The depth-fed modes
(`--use_depth_from=pred|data`) additionally need the **inference env** +
Depth-Anything-V2 (§1 *Full*) and a `--delta_stats_path`.

### API keys — not shipped with the repo
The repo does **not** include any `.env` or API key (`.env` and `*.key` are
gitignored). Provide your own key one of two ways — an exported variable takes
precedence over `.env`:

```bash
# 1) export it, or
export OPENAI_API_KEY=sk-...      # GPT
export GOOGLE_API_KEY=...         # Gemini (GEMINI_API_KEY is also accepted)

# 2) drop a .env at the repo root (auto-loaded; --env_file to point elsewhere):
#      OPENAI_API_KEY=sk-...
#      GOOGLE_API_KEY=...
```

### Commands

```bash
# GPT (OpenAI)
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_predict_trace_gpt.py \
  --test_dirs='[/path_to/test_set/*/*]' \
  --first_frame_only=true --first_frame_offset=[4,8,12] \
  --n_min=16 --n_max=16 \
  --use_depth_from=none --no_provide_history=true \
  --num_samples_for_metrics=5 \
  --gpt_model=gpt-5.5 --gpt_reasoning_effort=low \
  --output_dir=outputs/eval_gpt --seed=0

# Gemini (Google)
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_predict_trace_gemini.py \
  --test_dirs='[/path_to/test_set/*/*]' \
  --first_frame_only=true --first_frame_offset=[4,8,12] \
  --n_min=16 --n_max=16 \
  --use_depth_from=none --no_provide_history=true \
  --num_samples_for_metrics=5 \
  --gemini_model=gemini-3.1-pro-preview --gemini_thinking_level=low \
  --output_dir=outputs/eval_gemini --seed=0
```

> **These calls bill your API account.** Each sample issues
> `--num_samples_for_metrics` independent calls (best-of-N: `minADE_N`/`minFDE_N`
> over N samples), and `--first_frame_offset=[4,8,12]` yields up to 3 samples per
> video. Use `--max_samples=N` for a cheap smoke test, and `--offline=true` to
> re-render overlays/metrics from cached `raw_responses/` **without** calling (or
> paying) the API again.

Useful knobs: `--gpt_model` / `--gemini_model` (default `gpt-5.5` /
`gemini-3.1-pro-preview`); `--gpt_reasoning_effort` (`none`/`minimal`/`low`/
`medium`/`high`) / `--gemini_thinking_level` (`unspecified`/`minimal`/`low`/
`medium`/`high`); `--no_provide_history=false` to include each keypoint's past
trajectory in the prompt; `--use_depth_from=pred|data` (requires
`--delta_stats_path`) to also feed a depth panel.
