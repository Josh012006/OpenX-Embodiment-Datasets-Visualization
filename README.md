# Open X-Embodiment Datasets — Endpoint Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Josh012006/OpenX-Embodiment-Datasets-Visualization/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

## Goal

Visualize the **3D spatial distribution of end-effector target positions** across all robot datasets in the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind / Google).

Each dataset records robot demonstrations as sequences of steps. Every step contains the position the robot's end effector was commanded to reach, expressed in the robot's world coordinate frame with the robot base as origin. By collecting all these positions across every episode and step, we can answer:

- What region of 3D space was each robot trained to operate in?
- How do the operational workspaces differ between datasets and robot platforms?
- Which datasets share compatible spatial distributions (good candidates for cross-dataset training)?

## Dataset

The [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/) aggregates demonstrations from 22 robot embodiments across 21 institutions, all following the **RLDS** format:

```
dataset → episodes → steps → { observation, action, reward, is_first, is_last, … }
```

All datasets are hosted on Google Cloud Storage at `gs://gresearch/robotics/` (~5 TB total). This project streams data directly from GCS — no local download required for visualization.

## Notebook Structure

All work lives in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb), organized into two sections:

### 1 — End Effector Coordinates Fields

Identifies and documents the EEF position field for each supported dataset.

- **Feature exploration cell** — connects to GCS (metadata only, no data download) and prints the full feature spec of every dataset, used to identify the correct field paths.
- **`DATASET_EEF_CONFIG`** — static dict mapping each dataset name to its extraction config:
  ```python
  "kuka": {
      "field":   ["observation", "clip_function_input/base_pose_tool_reached"],
      "indices": slice(0, 3),   # extract [x, y, z] from a larger vector
      "reshape": False
  }
  ```
  Three extraction strategies are supported:
  - **Direct** (`indices=None, reshape=False`) — field is already a 3D vector
  - **Slice** (`indices=slice(i,j), reshape=False`) — [x,y,z] lives at specific indices
  - **Reshape** (`indices=slice(i,j), reshape=True`) — 16 values form a 4×4 homogeneous matrix; translation is extracted from the last column

- **`DATASET_ROBOT_INFO`** — maps each dataset to its robot platform and gripper type, used to label visualizations.

### 2 — Visualize End Effector Positions

Interactive Plotly-based visualizations, powered by pre-computed endpoint caches.

#### Cache workflow

Endpoints for all datasets are pre-extracted and stored as `.npy` files in `endpoints_cache/`. The notebook clones this repository inside Colab and reads from that directory — no extraction computation is needed at visualization time.

#### Individual visualization

Select any dataset by name (`DATASET_NAME` variable) and explore its endpoint distribution interactively:
- **3D scatter** or **2D projections** (XY, XZ, YZ)
- **Normalize** toggle (scales each axis 0→1 for shape comparison)
- Robot base is marked at the origin (0, 0, 0)
- Title shows the robot platform and whether the dataset is part of **OpenVLA** training data

#### Combined visualization

Compare multiple datasets simultaneously on the same plot:
- Checkbox panel to select any combination of datasets
- Each dataset gets a distinct color
- Same 3D / 2D / normalize controls as individual mode
- Datasets are tagged as **✅ OpenVLA** (used in OpenVLA training) or **🔵 OOD** (out-of-distribution)

## Running the Notebook

### Google Colab (recommended)

Click the badge at the top. The notebook clones this repo inside Colab to access the endpoint cache — no GCS access needed for visualization.

> **Note**: The feature exploration cell (which reads dataset specs from GCS) requires GCS access but downloads no data.

### Local

Requires Python 3.10 (3.11 has a known recursion issue with `tfds.load`):

```bash
pip install tfds-nightly tensorflow plotly ipywidgets Pillow
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

## Key Libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Read dataset specs and stream data from GCS |
| `plotly` | Interactive 3D and 2D scatter plots |
| `ipywidgets` | Checkbox / radio controls in Colab |
| `numpy` | Endpoint array storage and normalization |
| `PIL` | Image handling |
