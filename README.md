# Anonymous KRL-Flow Repository

This anonymous repository contains the implementation of the KRL-Flow module and the scripts used to reproduce the experiments on three datasets:

- simulated wolf–sheep trajectories,
- Batten disease sheep trajectories,
- Golden Shiner fish trajectories.

The repository is prepared for double-blind peer review and does not contain any author-identifying information.

---

## 1. Datasets

This project relies on three datasets. All of them are publicly available; anonymized links are provided here for review.

- **Golden Shiner fish trajectories**  
`https://cdr.lib.unc.edu/concern/dissertations/8p58pq754`

- **Batten disease sheep trajectories**  
`https://www.repository.cam.ac.uk/handle/1810/247648`

- **Wolf–sheep simulation data**  
`https://doi.org/10.5281/zenodo.17764685`

### Batten disease sheep base code

To run the Batten disease sheep experiments, we reuse part of the original authors' code.  
In this repository, the following items must be present (copied from the original Batten sheep repository):

- folder `BaseCode/`
- folder `Data/`
- files in the root:
  - `DTFile.txt`
  - `FBFile.txt`
  - `FNFile.txt`
  - `IFTFile.txt`
  - `nz1snmData.npy`
  - `ToProcess.txt`

These files are required to build the social network representation and to reproduce the same processing pipeline as in the original work.

---

## 2. Additional dependency: IDTxl

Transfer Entropy (TE) is computed using the **IDTxl** toolbox.

- IDTxl repository: `https://github.com/pwollstadt/IDTxl/tree/master`

IDTxl must be installed and importable in the same Python environment used to run the scripts.  
Please follow the installation instructions in the IDTxl repository (including any external requirements such as JIDT, if needed).

---

## 3. Golden Shiner labels

For the Golden Shiner dataset, this repository includes a file `files/golden/golden_labels.csv`, containing labels for leader–follower validation.  
These labels were generated specifically for this study to enable quantitative evaluation and follow the assumption commonly used in the literature (e.g., Daftari et al.): **fish with ID 1 tends to act as the leader for most of the experiment**.

The labels are used only for evaluation (computing metrics such as Top–1, fraction of time as leader, AUROC, AUPR); they are **not** used as input to the KRL-Flow model.

---

## 4. Installation

We recommend using a fresh virtual environment (e.g., `venv` or `conda`).

1. Clone this anonymous repository.
2. Inside the repository folder, install the Python dependencies:

```bash
   pip install -r requirements.txt
```

3. Make sure that:

   * the datasets (and, for Batten, the `BaseCode` and `Data` folders and auxiliary files) are available as described above;
   * IDTxl is correctly installed and importable.

The scripts read their inputs from the paths configured inside each file (e.g., `files/`, `Data/`, etc.). If necessary, you may adapt those paths to match your local directory structure.

---

## 5. How to run the experiments

All experiments can be executed directly from the command line. From the root of the repository:

### 5.1. Wolf–sheep simulations

```bash
python krl_flow_wolves.py
```

This script:

* loads the simulated wolf–sheep trajectories from `files/wolves/`,
* computes KRL-Flow, TSMI, TE, and GC scores across windows,
* generates evaluation tables and figures under `results_krl_out_wolves/`.

### 5.2. Batten disease sheep

```bash
python krl_flow_batten_sheep.py
```

This script:

* uses the original Batten disease base code (`BaseCode/`, `Data/`, and the auxiliary text/NPY files),
* builds the trajectory-based social network,
* runs KRL-Flow and the coupled informational metrics,
* saves CSV files and figures under `results_krl_out_batten/`.

### 5.3. Golden Shiner fish

```bash
python krl_flow_golden.py
```

This script:

* loads the Golden Shiner pairwise trajectories from the CSV file configured inside the script (by default in `files/golden/`),
* applies KRL-Flow and the informational metrics (TSMI, TE, GC),
* uses `files/golden/golden_labels.csv` to compute AUROC, AUPR, Top–1, and fraction-of-time leader measures,
* writes outputs to `results_krl_out_gshiner/`.

---

## 6. Reproducibility

The default hyperparameters and window configurations used in the paper are encoded directly in the scripts (and some can be overridden via environment variables). Running the three commands above with the provided datasets reproduces the main quantitative results and figures reported in the manuscript.
