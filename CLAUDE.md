# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Visualize the **3D spatial distribution of end-effector (EEF) target positions** across all robot datasets in the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind). The goal is to show what region of space each robot was trained to operate in, relative to its base.

## Running the Notebook

This is a **Google Colab-first project** — all development happens in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb). There is no local build or test system.

To run locally (requires Python 3.10; 3.11 triggers a known recursion issue with `tfds.load`):

```bash
pip install tensorflow tensorflow-datasets plotly ipywidgets Pillow numpy gcsfs
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

For cloud execution, use the Colab badge in the notebook header. Dataset specs are streamed from `gs://gresearch/robotics/` (metadata only); endpoint caches are loaded from the cloned repo.

## Architecture

The single notebook is organized into two sections:

1. **End Effector Coordinates Fields** — identifies the EEF position field for each dataset, defines `DATASET_EEF_CONFIG` and `DATASET_ROBOT_INFO`.
2. **Visualize End Effector Positions** — interactive Plotly visualizations (individual and combined), loading from the pre-computed `endpoints_cache/`.

### Data model

All datasets follow the **RLDS** spec:

```
dataset  →  episodes  →  steps  →  { observation, action, reward, … }
```

EEF position is stored under different fields per dataset. The notebook handles three extraction strategies:

| Strategy | Config | Use case |
|---|---|---|
| Direct | `indices=None, reshape=False` | Field is already a 3D `[x, y, z]` vector |
| Slice | `indices=slice(i,j), reshape=False` | `[x, y, z]` lives at specific offsets in a larger state vector |
| Reshape | `indices=slice(i,j), reshape=True` | 16 contiguous values form a 4×4 homogeneous matrix; translation extracted from it |

### Key data structures

**`DATASET_EEF_CONFIG`** (cell-12) — source of truth for field paths and extraction strategy:
```python
{
    "dataset_name": {
        "field":   ["key0", "key1"],  # path: step["key0"]["key1"]
        "indices": slice(0, 3),       # or None; applied before reshape
        "reshape": False              # or True
    },
    ...
}
```

Most datasets use `step["observation"][field_name]`, but two exceptions exist:
- `asu_table_top_converted_externally_to_rlds`: `field: ["ground_truth_states", "EE"]` — top-level step key, not inside observation
- `iamlab_cmu_pickup_insert_converted_externally_to_rlds`: `field: ["action"]` — reads from the action vector directly

**`DATASET_ROBOT_INFO`** (cell-13) — robot platform and gripper type per dataset. Covers the same 33 datasets as `DATASET_EEF_CONFIG`.

**`OPENVLA_DATASETS`** (cell-18) — set of 15 datasets used in OpenVLA training; tagged `✅ OpenVLA` in visualizations. All others are tagged `🔵 OOD`.

### Cache workflow

Endpoint arrays are pre-computed and stored as `.npy` files in `endpoints_cache/<dataset_name>.npy`. The notebook clones the repo inside Colab at `/content/OpenX-Embodiment-Datasets-Visualization/` and reads from `endpoints_cache/` via `get_endpoints(dataset_name)`.

### Key APIs / libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Read dataset specs from GCS via `builder_from_directory` |
| `plotly` | Interactive 3D and 2D scatter plots |
| `ipywidgets` | Checkbox / radio controls in Colab |
| `numpy` | Endpoint array storage (`.npy`) and normalization |
| `subprocess` / `os` | Clone repo and locate cache inside Colab |

### `extract_endpoint` logic (cell-9)

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
            return matrix[3, :3]         # last row, first 3 columns
        else:
            raise ValueError(...)
    else:
        return data
```

> ⚠️ **Note on reshape extraction**: The current code uses `matrix[3, :3]` (last row, columns 0–2). For a standard 4×4 homogeneous transformation matrix the translation lives in the **last column** (`matrix[:3, 3]`), not the last row. Verify this is correct for your datasets before computing new endpoint caches.

**Important invariant**: index slicing is always applied before reshape. When adding new datasets with `reshape=True`, set `indices` to the exact sub-range of the state vector containing the 16 matrix values.

### Visualization helpers (cell-18)

**`make_base_marker_3d(scale=0.1)`** — returns a list of Plotly traces representing the robot base:
- Three colored line segments from origin: X axis (red), Y axis (green), Z axis (blue)
- One marker+text trace at (0, 0, 0) labelled "Base"

**`build_figure(endpoints, dataset_name, view, normalize_flag, color)`** — creates the full Plotly figure for a single dataset. `view` is one of `'3D'`, `'XY'`, `'XZ'`, `'YZ'`.

**`normalize(endpoints)`** — min-max normalization per axis, clamps range to 1 if an axis has zero variance.

### Adding a new dataset

1. Run the feature exploration cell (cell-11) to find the field that holds the EEF position.
2. Add an entry to `DATASET_EEF_CONFIG` (cell-12) with the correct `field`, `indices`, and `reshape`.
3. Add an entry to `DATASET_ROBOT_INFO` (cell-13) with the robot name and gripper type.
4. Pre-compute the endpoint array and save it as `endpoints_cache/<dataset_name>.npy`.
5. If the dataset is in OpenVLA training data, add it to `OPENVLA_DATASETS` (cell-18).
