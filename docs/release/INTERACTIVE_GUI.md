# Interactive GUI — how to use the trace visualizer

`lerobot_trace_gui.py` is a browser UI that runs the full image → 3D trace
pipeline on a single image you provide: it synthesizes depth (Depth-Anything-V2)
and query keypoints (Grounded-SAM-2) on the fly, then predicts 3D trace with μ₀
— **no dataset needed**.

This page is the step-by-step walkthrough. For where the GUI fits among the
other evaluation modes, see [`EVALUATION.md`](EVALUATION.md) §4.

## Demo

▶️ [**Watch demo video**](../assets/interactive_gui_demo.mp4) — click to
view/download the raw recording.

## 1. Launch

Needs the **inference env** + the 3 checkpoints (μ₀, Depth-Anything-V2,
Grounded-SAM-2). See [`EVALUATION.md`](EVALUATION.md) §1 "Environment" → "Full"
for the build.

```bash
conda activate mu0_infer
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_trace_gui \
  --checkpoint=/path/to/final_ckpt \
  --delta_stats_path=/path/to/normalizer_stats.json \
  --port=8080
```

The server loads all three models once, then serves a viser UI on `--port`.
Open it in a browser at `http://localhost:8080`. If you run on a remote
machine, forward the port from your laptop first:

```bash
ssh -L 8080:localhost:8080 user@host   # then visit http://localhost:8080
```

## 2. Step-by-step

The controls live in the **Input** folder of the side panel. The **Status** line
at the bottom of that folder always tells you what to do next.

### Step 1 — Load an image

- Click **Upload your image** to pick a file from your machine, **or**
- Click **See sample image** to load a bundled example with its **Task prompt**
  (`pick up the doll`) and **Classes** (`doll`) already filled in.

After the image loads, two panels appear below the controls: **Input (resized)**
— the square image the policy actually sees — and **Depth (DA-V2)** — the
synthesized depth map. (Depth is computed once here, so re-running segmentation
or prediction does not recompute it.)

### Step 2 — Set the prompts

- **Task prompt** — the instruction fed to μ₀, e.g. `pick up the doll`.
- **Classes (comma)** — the object word(s) used by GroundingDINO + SAM to find
  things to track, e.g. `doll`. Use commas for multiple objects, e.g.
  `doll, cup`.

(If you clicked **See sample image**, both fields are already filled — you can
edit them or leave them as-is.)

### Step 3 — Run segmentation

Click **Run segmentation**. A new **Detected masks** panel appears **below** the
controls showing:

- **All visible masks (combined)** — every detected mask overlaid on the image
  in distinct colors, so you can see all candidates at a glance.
- One thumbnail per mask, labelled with its class, confidence, and area, each
  with a **track #N** checkbox.

Tick the masks you want to track. Query keypoints will be sampled from the
**union** of the checked masks. (Under **Mask selection**, the **Top-K masks**
slider filters how many candidates are shown — moving it does *not* re-run
detection — and **# keypoints** sets how many points are sampled.)

### Step 4 — Predict trace

Click **Predict trace**. The predicted 3D
trajectories render live in the main scene, starting from the sampled keypoints.
You can rotate/zoom the 3D view with the mouse.

## 3. Tips

- **Camera HFOV (deg)** — the slider just below the **Predict trace** button —
  sets the synthesized camera intrinsics.
  Adjusting it re-projects the existing prediction — no re-inference — so you can
  tune the 3D scale after predicting.
- The **Display** folder toggles each layer (point cloud, predicted trace,
  keypoints, camera frustum, world axes) and adjusts sizes/line widths live.
- To try a different prompt or object, edit **Task prompt** / **Classes** and
  re-run from Step 3. To use a new image, just load another one (Step 1).
- If segmentation finds nothing, broaden or rephrase the **Classes** words, or
  raise **Top-K masks**.
