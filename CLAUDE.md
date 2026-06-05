# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Visualize the **3D spatial distribution of end-effector (EEF) target positions** across all robot datasets in the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind). The goal is to show what region of space each robot was trained to operate in, relative to its base.

## Running the Notebook

This is a **Google Colab-first project** — all development happens in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb). There is no local build or test system.

To run locally (requires Python 3.10; 3.11 triggers a known recursion issue with `tfds.load`):

```bash
pip install tfds-nightly tensorflow plotly ipywidgets Pillow
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
| Reshape | `indices=slice(i,j), reshape=True` | 16 contiguous values form a 4×4 homogeneous matrix; translation = last column `[:3, 3]` |

### Key data structures

**`DATASET_EEF_CONFIG`** (cell-10) — source of truth for field paths and extraction strategy:
```python
{
    "dataset_name": {
        "field":   ["key0", "key1"],  # path: step["key0"]["key1"]
        "indices": slice(0, 3),       # or None
        "reshape": False              # or True
    },
    ...
}
```

**`DATASET_ROBOT_INFO`** (cell-11) — robot platform and gripper type per dataset.

**`OPENVLA_DATASETS`** (cell-16) — set of datasets used in OpenVLA training; datasets are tagged accordingly in visualizations.

### Cache workflow

Endpoint arrays are pre-computed and stored as `.npy` files in `endpoints_cache/<dataset_name>.npy`. The notebook clones the repo inside Colab at `/content/OpenX-Embodiment-Datasets-Visualization/` and reads from `endpoints_cache/` via `get_endpoints(dataset_name)`.

### Key APIs / libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Read dataset specs and stream data from GCS via `builder_from_directory` |
| `plotly` | Interactive 3D and 2D scatter plots |
| `ipywidgets` | Checkbox / radio controls in Colab |
| `numpy` | Endpoint array storage (`.npy`) and normalization |

### `extract_endpoint` logic

```python
def extract_endpoint(step, config):
    data = step
    for key in config["field"]:
        data = data[key]
    data = data.numpy()

    if config["indices"] is not None:
        data = data[config["indices"]]   # slice first

    if config["reshape"]:
        matrix = data.flatten()[:16].reshape(4, 4)
        return matrix[:3, 3]             # translation column
    else:
        return data
```

**Important**: index slicing is always applied before reshape. When adding new datasets with `reshape=True`, set `indices` to the correct sub-range of the state vector containing the 4×4 matrix.
