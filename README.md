# Open X-Embodiment Datasets — EEF & Task Description Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Josh012006/OpenX-Embodiment-Datasets-Visualization/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

Based on the original DeepMind Colab: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

---

## Goal

Two complementary visualizations of the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind / Google):

1. **EEF endpoint positions** — 3D spatial distribution of where each robot's end-effector was at the last step of every training episode, expressed in the robot's world frame (base at origin). Answers: *what region of space was each robot trained to reach?*

2. **Task description embeddings** — semantic clusters of the natural-language task instructions across datasets, projected to 3D via sentence embeddings + UMAP. Answers: *what kinds of tasks do datasets overlap on?* (avoids the coordinate-frame ambiguity of comparing raw EEF coordinates across robots with different base placements)

---

## Previews

### End-effector endpoint visualization

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Josh012006/OpenX-Embodiment-Datasets-Visualization/main/public/individual-eef.png" width="600"><br>
      <em>Individual dataset</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Josh012006/OpenX-Embodiment-Datasets-Visualization/main/public/combined-eef.png" width="600"><br>
      <em>Combined visualization</em>
    </td>
  </tr>
</table>

### Task description embedding visualization

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Josh012006/OpenX-Embodiment-Datasets-Visualization/main/public/individual-task.png" width="600"><br>
      <em>Individual dataset</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Josh012006/OpenX-Embodiment-Datasets-Visualization/main/public/combined-task.png" width="600"><br>
      <em>Combined visualization</em>
    </td>
  </tr>
</table>

---

## Dataset

The [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/) aggregates demonstrations from 22 robot embodiments across 21 institutions, all following the **RLDS** format:

```
dataset -> episodes -> steps -> { observation, action, reward, is_first, is_last, ... }
```

Datasets are hosted on Google Cloud Storage at `gs://gresearch/robotics/` (~5 TB total). This project reads only metadata from GCS and streams steps for cache generation — no bulk download required.

---

## Repository Structure

```
colabs/
  Open_X_Embodiment_Datasets.ipynb   # main notebook (3 sections, 39 cells)
endpoints_cache_random/
  <dataset_name>.npy                 # EEF endpoint arrays — 1 point per episode (n, 3), randomly sampled
task_description_cache/
  <dataset_name>.json                # unique task descriptions per dataset (list of strings)
  _umap_3d_cache.json                # shared UMAP projection across all datasets
public/
  individual-eef.png                 # EEF individual viz screenshot
  combined-eef.png                   # EEF combined viz screenshot
  individual-task.png                # task embedding individual viz screenshot
  combined-task.png                  # task embedding combined viz screenshot
```

---

## Notebook Structure

All work lives in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb), organized into three sections (39 cells total).

### Section 1 — Datasets Infos (cells 6–14)

Identifies and documents the EEF position field for each of the **33 supported datasets**, and provides robot info for all 53.

| Cell | Content |
|---|---|
| cell-7 | `%pip install` all dependencies |
| cell-8 | Imports + `DATASETS` list (53 datasets) |
| cell-9 | `DATASET_ROBOT_INFO` — robot platform + gripper for all 53 datasets |
| cell-10 | Utils: `dataset2path`, `extract_endpoint`, `get_safe_split`, `normalize` |
| cell-12 | Feature exploration loop — prints full TFDS feature spec per dataset |
| cell-13 | `DATASET_EEF_CONFIG` — per-dataset EEF extraction config (33 datasets) |
| cell-14 | Markdown — lists the 20 excluded datasets and reasons |

#### `DATASET_EEF_CONFIG` structure

```python
DATASET_EEF_CONFIG = {
    "fractal20220817_data": {
        "field":   ["observation", "base_pose_tool_reached"],
        "indices": slice(0, 3),   # take [x, y, z] from a larger vector
        "reshape": False
    },
    "viola": {
        "field":   ["observation", "ee_states"],
        "indices": None,
        "reshape": True           # interpret as flattened 4x4 matrix
        # "reshape_convention": "row"  <- default, uses matrix[3, :3]
    },
    "uiuc_d3field": {
        "field":   ["observation", "state"],
        "indices": None,
        "reshape": True,
        "reshape_convention": "col"  # uses matrix[:3, 3] (standard translation column)
    },
}
```

Three extraction strategies, applied in order:

| Step | Trigger | Operation |
|---|---|---|
| 1 — Slice | `indices` is not `None` | `data = data[indices]` — always before reshape |
| 2a — Direct | `reshape=False` | return data as-is |
| 2b — Reshape (row) | `reshape=True`, no `reshape_convention` | `matrix[3, :3]` — last row |
| 2b — Reshape (col) | `reshape=True`, `reshape_convention="col"` | `matrix[:3, 3]` — translation column |

Two datasets use non-standard step-level field paths (not inside `observation`):
- `asu_table_top_converted_externally_to_rlds` — `step["ground_truth_states"]["EE"]`
- `iamlab_cmu_pickup_insert_converted_externally_to_rlds` — `step["action"]`

**Excluded datasets (20 of 53):** Some have joint-angle-only states, 2D-only states, no state at all, or are navigation/quadruped robots. Full list with explanations in cell-14.

---

### Section 2 — Visualize EEF Positions (cells 15–24)

#### Cache generation (`load_dataset`, cell-19)

For each dataset in `DATASET_EEF_CONFIG`, extracts the **final EEF position** of each episode (last step only) using **random sampling** (`shuffle_files=True`, `.shuffle(buffer_size=50, seed=42)`, `.take(n)`), and saves a `(n, 3)` array to `endpoints_cache_random/<dataset_name>.npy`. Caps at 500 episodes. Skips datasets already on disk. Releases memory with `gc.collect()` and `tf.keras.backend.clear_session()` between datasets.

#### Loading in Colab (cell-18)

Clones this repo to `/content/OpenX-Embodiment-Datasets-Visualization/` and sets `CACHE_DIR` to `endpoints_cache_random/` — no extraction at visualization time.

#### Individual visualization (cell-22)

Set `DATASET_NAME` and explore interactively:
- **View**: 3D scatter or 2D projections (XY, XZ, YZ)
- **Normalize**: scales each axis 0-1 to compare workspace shapes
- **Origin marker**: colored coordinate axes at (0, 0, 0) — X=red, Y=green, Z=blue
- **Title**: `dataset | robot | OpenVLA/OOD tag | n=<episodes>`

#### Combined visualization (cell-24)

Compare any subset of the 33 datasets on one figure:
- Checkbox panel for each dataset (first 3 checked by default)
- Each dataset gets a distinct color from a 33-color palette
- Same 3D / 2D / normalize controls
- Labels: dataset name, robot platform, OpenVLA/OOD tag, episode count
- **✅ OpenVLA** (15 datasets used in OpenVLA training) vs **🔵 OOD**

---

### Section 3 — Task Description Embeddings (cells 25–38)

This section captures *what the tasks are* rather than *where the arm moves*, which avoids the coordinate-frame ambiguity when comparing EEF positions across robots with different base placements.

#### Pipeline

1. **Extract unique descriptions** (`extract_unique_task_descriptions`, cell-33) — streams all episodes for each dataset (no episode cap — text sets are small), collects the set of unique `language_instruction` strings (or the field in `TASK_FIELD_CONFIG` if overridden), saves to `task_description_cache/<dataset_name>.json`.

2. **Embed + project** (`compute_all_embeddings_3d`, cell-34) — encodes all descriptions across all datasets with `all-mpnet-base-v2` (SentenceTransformer), then fits a **single shared UMAP** (3D, `n_neighbors=30`, `min_dist=0.05`, `random_state=42`) on the combined set so inter-dataset distances are meaningful. Result cached to `task_description_cache/_umap_3d_cache.json`.

3. **Visualize** — individual (cell-36) and combined (cell-38) interactive 3D scatter plots with hover tooltips showing the instruction text.

#### Key structures

**`TASK_FIELD_CONFIG`** (cell-29) — overrides `"language_instruction"` for datasets that store the instruction elsewhere (e.g., `fractal20220817_data` uses `observation.natural_language_instruction`).

**`TASK_SKIP_DATASETS`** (cell-29) — datasets excluded from this section:
- `language_table` — instruction encoded as int32 bytes, not a plain string
- `bridge`, `robo_net` — OOM during extraction (too many episodes with large images)
- `uiuc_d3field` — `language_instruction` field exists but is always empty

---

## Running the Notebook

### Google Colab (recommended)

Click the badge at the top. The notebook clones this repo to access both caches — no GCS credentials needed for visualization only.

> Feature exploration and cache generation cells stream from GCS.

### Local

Requires **Python 3.10** (Python 3.11 has a [known recursion bug](https://github.com/tensorflow/datasets/issues/4666) with `tfds.load`):

```bash
pip install tensorflow tensorflow-datasets plotly ipywidgets Pillow numpy gcsfs sentence-transformers umap-learn
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

---

## Key Libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Read dataset specs and stream steps from GCS |
| `tensorflow` / `gc` | Session and memory management during cache generation |
| `plotly` | Interactive 3D and 2D scatter plots |
| `ipywidgets` | Checkbox / radio widget controls in Colab |
| `numpy` | `.npy` cache storage and normalization |
| `sentence-transformers` | Sentence embeddings (`all-mpnet-base-v2`) for task descriptions |
| `umap-learn` | 3D dimensionality reduction of description embeddings |
| `subprocess` / `os` | Clone repo and locate caches inside Colab |
