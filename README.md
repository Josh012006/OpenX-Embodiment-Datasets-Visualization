# Open X-Embodiment Datasets — Endpoint Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Josh012006/OpenX-Embodiment-Datasets-Visualization/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)

## Goal

Visualize the **3D spatial distribution of end-effector target positions** across all 50+ robot datasets in the [Open X-Embodiment collection](https://robotics-transformer-x.github.io/) (DeepMind / Google).

Each dataset in Open X-Embodiment records robot demonstrations as sequences of steps. Every step contains the position the robot's end effector (hand/gripper) was commanded to reach, expressed in the robot's world coordinate frame with the robot base as origin. By collecting all these positions across every episode and every step, we can answer:

- What region of 3D space was each robot trained to operate in?
- How do the operational workspaces differ between datasets and robot platforms?
- Which datasets share compatible spatial distributions, making them good candidates for cross-dataset training?

## Dataset

The [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/) aggregates demonstrations from 22 robot embodiments across 21 institutions. All datasets are hosted publicly on Google Cloud Storage at `gs://gresearch/robotics/` and follow the **RLDS** (Robotics Learning from Demonstrations) format:

```
dataset → episodes → steps → { observation, action, reward, is_first, is_last, … }
```

The full corpus is ~5 TB. This project streams data directly from GCS — no local download required.

## Notebook Structure

All work lives in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb). The notebook is organized into six sections:

| # | Section | Description |
|---|---|---|
| 1 | **Visualize Datasets** | Loads episodes via TFDS and renders them as GIF animations |
| 2 | **Endpoint Position Distribution** | *(new)* Discovers and maps the EEF position field for each dataset, then extracts positions for 3D visualization |
| 3 | **Download Datasets** | Bulk download script for local use (~5 TB); rarely needed |
| 4 | **Data Loader Example** | Minimal RLDS episode → step pipeline with batching |
| 5 | **Interleave Multiple Datasets** | Combines two datasets with weighted sampling via `tf.data` |
| 6 | **Trajectory Transformation** | Aligns heterogeneous specs and produces fixed-length overlapping trajectories via DM Reverb |

### Section 2 in detail — Endpoint Position Distribution

Because each dataset uses a different schema, the EEF position is stored under different field names. Section 2 handles this in two steps:

**Step 1 — Spec discovery** (`DATASET_EEF_FIELDS` generation)

Reads only the metadata (no data download) of all 53 datasets and finds the best field for EEF position using this priority order:

1. `action/world_vector` — explicit 3D target position, used by most RT-X datasets
2. `observation/state` — robot state vector whose first three values are conventionally `[x, y, z]`

Prints a ready-to-paste Python dict.

**Step 2 — Static field mapping**

`DATASET_EEF_FIELDS` is a static dict (analogous to the `DATASETS` list) that stores the key path per dataset:

```python
DATASET_EEF_FIELDS = {
    'fractal20220817_data': ('action', 'world_vector'),
    'kuka':                 ('action', 'world_vector'),
    'bridge':               ('observation', 'state'),
    # …
}
```

Usage: `step[key0][key1]` gives the position tensor for any dataset.

## Running the Notebook

### Google Colab (recommended)

Click the badge at the top of this file. Datasets stream from GCS — no setup required.

> **Python version**: Colab's default runtime is Python 3.11. The `tfds.load` download cells require Python 3.10 due to a [known recursion bug](https://github.com/tensorflow/datasets/issues/4666). The streaming path (`builder_from_directory`) works on 3.11.

### Local

Requires Python 3.10:

```bash
pip install tfds-nightly tensorflow rlds dm-reverb apache-beam Pillow matplotlib
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

## Key Libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Load 50+ datasets from GCS via `builder_from_directory` |
| `rlds` | Episode/step utilities, trajectory construction |
| `dm-reverb` | Structured streaming for fixed-length trajectory sampling |
| `tf.data` | Interleaving, batching, prefetching |
| `matplotlib` | 3D scatter plots of EEF position distributions |
| `PIL` / `IPython.display` | GIF rendering in Colab |
