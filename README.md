# Open X-Embodiment Datasets — EEF Endpoint Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Josh012006/OpenX-Embodiment-Datasets-Visualization/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

Based on the original DeepMind Colab: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

---

## Goal

Visualize the **3D spatial distribution of end-effector (EEF) target positions** across robot datasets in the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind / Google).

Each dataset records robot demonstrations as sequences of steps. Every step contains the position the robot's end effector was commanded to reach, expressed in the robot's world coordinate frame with the robot base as origin. Collecting all these positions across every episode and step answers:

- What region of 3D space was each robot trained to operate in?
- How do the operational workspaces differ between robot platforms?
- Which datasets share compatible spatial distributions?

---

## Dataset

The [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/) aggregates demonstrations from 22 robot embodiments across 21 institutions, all following the **RLDS** format:

```
dataset → episodes → steps → { observation, action, reward, is_first, is_last, … }
```

Datasets are hosted on Google Cloud Storage at `gs://gresearch/robotics/` (~5 TB total). This project streams only metadata from GCS — no local download required for visualization.

---

## Notebook Structure

All work lives in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb), split into two sections.

### Section 1 — End Effector Coordinates Fields

Identifies and documents the EEF position field for each of the **33 supported datasets**.

| Cell | Content |
|---|---|
| Feature exploration | Connects to GCS and prints the full feature spec of every dataset in `DATASETS` |
| `DATASET_EEF_CONFIG` | Static dict: dataset name → extraction config (field path + strategy) |
| `DATASET_ROBOT_INFO` | Static dict: dataset name → robot platform + gripper type |

#### `DATASET_EEF_CONFIG` structure

Because each dataset uses a different schema, the EEF field and how to extract it are configured per-dataset:

```python
DATASET_EEF_CONFIG = {
    "fractal20220817_data": {
        "field":   ["observation", "base_pose_tool_reached"],  # step["observation"]["base_pose_tool_reached"]
        "indices": slice(0, 3),   # take first 3 values → [x, y, z]
        "reshape": False
    },
    "viola": {
        "field":   ["observation", "ee_states"],
        "indices": None,          # take the full field
        "reshape": True           # treat as flattened 4×4 homogeneous matrix
    },
    ...
}
```

Three extraction strategies are supported:

| Strategy | `indices` | `reshape` | Use case |
|---|---|---|---|
| **Direct** | `None` | `False` | Field is already `[x, y, z]` |
| **Slice** | `slice(i, j)` | `False` | `[x, y, z]` lives at specific offsets in a larger state vector |
| **Reshape** | `slice(i, j)` or `None` | `True` | 16 values form a 4×4 homogeneous matrix; translation extracted from it |

> **Note**: Index slicing is always applied **before** reshape.

Two datasets use non-standard field paths (not inside `observation`):
- `asu_table_top_converted_externally_to_rlds` → `step["ground_truth_states"]["EE"]`
- `iamlab_cmu_pickup_insert_converted_externally_to_rlds` → `step["action"]` directly

### Section 2 — Visualize End Effector Positions

Interactive Plotly visualizations powered by pre-computed endpoint caches.

#### Cache workflow

Endpoint arrays are pre-extracted and stored as `.npy` files in `endpoints_cache/<dataset_name>.npy`. The notebook clones this repo inside Colab and reads directly from that directory — no extraction runs at visualization time.

#### Individual visualization

Change `DATASET_NAME` to any supported dataset and explore interactively:

- **View**: 3D scatter or 2D projections (XY, XZ, YZ)
- **Normalize**: scales each axis 0→1 to compare workspace shapes across datasets
- **Origin marker**: colored coordinate axes (X=red, Y=green, Z=blue) at the robot base (0, 0, 0)
- **Title**: shows the robot platform and whether the dataset is part of OpenVLA training

#### Combined visualization

Compare multiple datasets on the same plot:

- Checkbox panel to select any combination of the 33 datasets
- Each dataset gets a distinct color from a 33-color palette
- Same 3D / 2D / normalize controls as individual mode
- Datasets tagged as **✅ OpenVLA** (15 datasets used in OpenVLA training) or **🔵 OOD**

---

## Running the Notebook

### Google Colab (recommended)

Click the badge at the top. The notebook clones this repo to access the endpoint cache — no GCS credentials needed for visualization only.

> The feature exploration cell reads dataset specs from GCS (metadata only, no data download).

### Local

Requires **Python 3.10** — Python 3.11 has a [known recursion bug](https://github.com/tensorflow/datasets/issues/4666) with `tfds.load`:

```bash
pip install tensorflow tensorflow-datasets plotly ipywidgets Pillow numpy gcsfs
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

---

## Key Libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Read dataset feature specs from GCS via `builder_from_directory` |
| `plotly` | Interactive 3D and 2D scatter plots |
| `ipywidgets` | Checkbox / radio widget controls in Colab |
| `numpy` | Endpoint `.npy` cache storage and normalization |
| `subprocess` / `os` | Clone repo and load cache inside Colab |
