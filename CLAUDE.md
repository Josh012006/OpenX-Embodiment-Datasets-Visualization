# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Visualize the **endpoints distribution** of robot experiments in the [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/) (DeepMind). The project extends DeepMind's provided Colab notebook with endpoint visualization capabilities.

## Running the Notebook

This is a **Google Colab-first project** — all development happens in [colabs/Open_X_Embodiment_Datasets.ipynb](colabs/Open_X_Embodiment_Datasets.ipynb). There is no local build or test system.

To run locally (requires Python 3.10; 3.11.12 triggers a known recursion issue in Colab):

```bash
pip install tfds-nightly tensorflow rlds dm-reverb apache-beam Pillow
jupyter notebook colabs/Open_X_Embodiment_Datasets.ipynb
```

For cloud execution, use the Colab badge in the notebook header — datasets stream from `gs://gresearch/robotics/` and do not need local download (~5 TB total if you do download them).

## Architecture

The single notebook is organized into six sequential sections:

1. **Visualize Datasets** — loads episodes via TFDS and renders them as GIF animations; the primary place to add new visualization logic.
2. **Endpoint Position Distribution** — *(new)* discovers the EEF position field for every dataset and collects 3D positions for visualization; see details below.
3. **Download Datasets** — bulk download script (all 50+ datasets); rarely needed.
4. **Data Loader Example** — minimal step-by-step pipeline showing the RLDS episode → step structure.
5. **Interleave Multiple Datasets** — combines two datasets with weighted sampling using `tf.data` interleave.
6. **Trajectory Transformation / Multi-Dataset Combination** — advanced: aligns heterogeneous observation specs and creates fixed-length overlapping trajectories via DM Reverb.

### Data model

All datasets follow the **RLDS** (Robotics Learning from Demonstrations) spec:

```
dataset  →  episodes  →  steps  →  { observation, action, reward, … }
```

Observations are typically `image` tensors (uint8, various resolutions). Actions are robot-specific vectors. The notebook normalizes these through spec alignment before interleaving.

### Key APIs / libraries

| Library | Role |
|---|---|
| `tensorflow_datasets` (tfds) | Load 50+ named datasets from GCS |
| `rlds` | Episode/step utilities, trajectory construction |
| `dm-reverb` | Structured streaming for trajectory pattern sampling |
| `tf.data` | Interleaving, batching, prefetching |
| `PIL` / `IPython.display` | GIF rendering in Colab |

### Dataset registry

Datasets are identified by name strings (e.g. `"fractal20220817_data"`, `"bridge"`, `"kuka"`). The full list of ~50 names is defined in the **Visualize Datasets** section near the top of the notebook. All datasets are loaded with `split='train[:N]'` to avoid downloading the full corpus during exploration.

### Endpoint Position Distribution section

The EEF position field name varies per dataset. The discovery workflow is:

1. Run the **spec discovery cell** — it reads metadata only (no data download), identifies the best position field per dataset using this priority: `action/world_vector` → `observation/state`, and prints the Python dict code.
2. Paste the output into the **`DATASET_EEF_FIELDS` cell** — a static dict mapping each dataset name to a `(key0, key1)` tuple, used as `step[key0][key1]` to retrieve the position tensor.

When extending this section (e.g. adding extraction or visualization cells), always use `DATASET_EEF_FIELDS` as the source of truth for field paths rather than hard-coding field names.
