# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Visualize, in two complementary ways, the robot training data from the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind):

1. **EEF endpoint positions** — 3D spatial distribution of end-effector positions at the last step of each training episode, in the robot's world frame (base at origin).
2. **Task description embeddings** — semantic clustering of natural-language task instructions via sentence embeddings (all-mpnet-base-v2) + UMAP 3D projection.

## Running the Notebook

This is a **Google Colab-first project** — all development happens in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb). There is no local build or test system.

To run locally (requires Python 3.10; 3.11 triggers a known recursion issue with `tfds.load`):

```bash
pip install tensorflow tensorflow-datasets plotly ipywidgets Pillow numpy gcsfs sentence-transformers umap-learn
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

For cloud execution, use the Colab badge in the notebook header. Both caches (`endpoints_cache_random/` and `task_description_cache/`) are loaded from the cloned repo.

## Architecture

The notebook has **39 cells** (cell-0 to cell-38), organized into three sections.

### Section 1 — Datasets Infos (cells 6–14)

| Cell | Role |
|---|---|
| cell-7 | `%pip install` all dependencies |
| cell-8 | Imports + `DATASETS` list (53 datasets) |
| cell-9 | `DATASET_ROBOT_INFO` — robot + gripper for **all 53** datasets |
| cell-10 | Utils: `dataset2path`, `extract_endpoint`, `get_safe_split`, `normalize` |
| cell-12 | Feature exploration loop — prints `b.info.features` for each dataset |
| cell-13 | `DATASET_EEF_CONFIG` — EEF extraction config for 33 datasets |
| cell-14 | Markdown listing the 20 excluded datasets and their reasons |

### Section 2 — Visualize EEF Positions (cells 15–24)

| Cell | Role |
|---|---|
| cell-17 | Extra imports (`gc`, `tf`, `subprocess`) |
| cell-18 | Clone repo + set `CACHE_DIR = ".../endpoints_cache_random"` |
| cell-19 | `load_dataset` — cache generation (random sampling, last step) |
| cell-20 | `OPENVLA_DATASETS`, `COLORS`, all visualization helpers, `build_figure` |
| cell-22 | Individual EEF visualization widget |
| cell-24 | Combined EEF visualization widget |

### Section 3 — Task Description Embeddings (cells 25–38)

| Cell | Role |
|---|---|
| cell-26 | `%pip install sentence-transformers umap-learn` |
| cell-28 | Feature inspection loop — checks which datasets have `language_instruction` |
| cell-29 | `TASK_FIELD_CONFIG`, `TASK_SKIP_DATASETS`, `get_instruction_field` |
| cell-32 | Clone repo + set `TASK_CACHE_DIR = ".../task_description_cache"` |
| cell-33 | `extract_unique_task_descriptions` — builds per-dataset JSON caches |
| cell-34 | `EMBED_MODEL`, `compute_all_embeddings_3d`, `build_task_figure`, helpers |
| cell-36 | Individual task embedding visualization |
| cell-38 | Combined task embedding visualization |

---

## Data model

All datasets follow the **RLDS** spec:

```
dataset  ->  episodes  ->  steps  ->  { observation, action, reward, ... }
```

---

## Key data structures

### `DATASET_ROBOT_INFO` (cell-9)

Robot platform and gripper type for **all 53 datasets** in `DATASETS`. Used by `get_dataset_label()` to annotate figures. Note: `DATASET_EEF_CONFIG` covers only 33 of these — the other 20 have no usable EEF position field.

### `DATASET_EEF_CONFIG` (cell-13)

Source of truth for EEF field paths and extraction strategy (33 datasets):

```python
{
    "dataset_name": {
        "field":              ["key0", "key1"],  # path: step["key0"]["key1"]
        "indices":            slice(0, 3),        # or None; always applied before reshape
        "reshape":            False,              # or True
        "reshape_convention": "col"              # optional; "row" (default) or "col"
    },
    ...
}
```

Most datasets use `step["observation"][field_name]`, but two exceptions exist:
- `asu_table_top_converted_externally_to_rlds`: `field: ["ground_truth_states", "EE"]` — top-level step key
- `iamlab_cmu_pickup_insert_converted_externally_to_rlds`: `field: ["action"]` — reads from the action vector directly

Extraction strategies:

| Strategy | Config | Use case |
|---|---|---|
| Direct | `indices=None, reshape=False` | Field is already a 3D `[x, y, z]` vector |
| Slice | `indices=slice(i,j), reshape=False` | `[x, y, z]` lives at specific offsets in a larger state vector |
| Reshape (row) | `reshape=True` (default) | 16 values -> 4x4 matrix, returns `matrix[3, :3]` (last row) |
| Reshape (col) | `reshape=True, reshape_convention="col"` | 16 values -> 4x4 matrix, returns `matrix[:3, 3]` (translation column) |

### `OPENVLA_DATASETS` (cell-20)

Set of 15 datasets used in OpenVLA training. Tagged `✅ OpenVLA` in all visualizations; all others tagged `🔵 OOD`.

### `TASK_FIELD_CONFIG` (cell-29)

Per-dataset override for the instruction field. Most datasets expose `language_instruction` directly under `steps`. Exceptions (stored under `observation.natural_language_instruction`): `fractal20220817_data`, `kuka`, `bridge`, `taco_play`, `jaco_play`, `berkeley_cable_routing`, `roboturk`, `nyu_door_opening_surprising_effectiveness`, `viola`, `berkeley_autolab_ur5`, `toto`, `columbia_cairlab_pusht_real`, `bc_z`.

### `TASK_SKIP_DATASETS` (cell-29)

Datasets excluded from the task description section:
- `language_table` — instruction encoded as int32 bytes, not a plain string
- `bridge`, `robo_net` — OOM during full-dataset extraction
- `uiuc_d3field` — `language_instruction` always empty

---

## Cache directories

| Directory | Format | Contents |
|---|---|---|
| `endpoints_cache_random/` | `<name>.npy` shape `(n, 3)` | EEF position at last step per episode; randomly sampled episodes |
| `task_description_cache/` | `<name>.json` list of strings | Unique task descriptions per dataset |
| `task_description_cache/_umap_3d_cache.json` | JSON dict with `dataset`, `description`, `x`, `y`, `z` lists | Shared 3D UMAP projection across all datasets |

---

## `extract_endpoint` logic (cell-10)

```python
def extract_endpoint(step, config):
    data = step
    for key in config["field"]:
        data = data[key]
    data = data.numpy()

    if config["indices"] is not None:
        data = data[config["indices"]]   # slice first, always before reshape

    if config["reshape"]:
        flat = data.flatten()
        if len(flat) >= 16:
            matrix = flat[:16].reshape(4, 4)
            convention = config.get("reshape_convention", "row")
            if convention == "col":
                return matrix[:3, 3]     # translation column (standard homogeneous)
            else:
                return matrix[3, :3]     # last row, first 3 elements (default)
        else:
            raise ValueError(f"Cannot reshape data of size {len(flat)} into 4x4 matrix")
    else:
        return data
```

**Important invariant**: index slicing is always applied before reshape. When adding new datasets with `reshape=True`, set `indices` to the exact sub-range containing the 16 matrix values, and set `reshape_convention="col"` if the data uses a standard homogeneous matrix layout.

---

## `load_dataset` — EEF cache generation (cell-19)

1. Skip if `endpoints_cache_random/<name>.npy` already exists
2. Load up to 500 episodes via `get_safe_split(b, max_episodes=500)`
3. Use **random sampling**: `shuffle_files=True`, `.shuffle(buffer_size=50, seed=42)`, `.take(n)` — NOT sequential slicing
4. Iterate steps to find the last one, call `extract_endpoint`
5. Save collected array as `.npy`
6. Call `gc.collect()` and `tf.keras.backend.clear_session()` after each dataset

---

## `compute_all_embeddings_3d` — UMAP cache generation (cell-34)

1. Load all per-dataset JSON caches from `task_description_cache/`
2. Concatenate all descriptions with their dataset labels
3. Encode with `SentenceTransformer('all-mpnet-base-v2')` → `(N, 768)` embeddings
4. Fit `umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, random_state=42)` on the **full combined set** — fitting per-dataset would make inter-dataset distances meaningless
5. Cache result to `task_description_cache/_umap_3d_cache.json`

---

## Visualization helpers (cell-20)

**`make_base_marker_3d(scale=0.1)`** — list of Plotly traces for the robot base:
- Three colored line segments from origin: X=red, Y=green, Z=blue
- One marker+text trace at (0, 0, 0) labelled "Base"

**`make_base_marker_2d()`** — single cross marker at origin for 2D projections.

**`build_figure(endpoints, dataset_name, view, normalize_flag, color)`** — full Plotly figure for a single EEF dataset. `view` is one of `'3D'`, `'XY'`, `'XZ'`, `'YZ'`.

**`build_task_figure(points, dataset_name, color)`** — full Plotly figure for a single dataset's UMAP task points, with hover text.

**`normalize(endpoints)`** — min-max normalization per axis, clamps range to 1 if zero variance.

**`get_dataset_label(dataset_name)`** — returns `(robot_name, tag)` where tag is `"✅ OpenVLA"` or `"🔵 OOD"`.

---

## Adding a new EEF dataset

1. Run the feature exploration cell (cell-12) to find the field that holds the EEF position.
2. Add an entry to `DATASET_EEF_CONFIG` (cell-13) with `field`, `indices`, `reshape`, and optionally `reshape_convention`.
3. Add an entry to `DATASET_ROBOT_INFO` (cell-9) with robot name and gripper type.
4. Run the cache generation cell (cell-19) — it detects the missing `.npy` and generates it automatically.
5. Commit `endpoints_cache_random/<dataset_name>.npy`.
6. If the dataset is in OpenVLA training data, add it to `OPENVLA_DATASETS` (cell-20).
