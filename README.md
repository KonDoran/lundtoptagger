# lundtoptagger

Quark/gluon tagging on small-radius jets (SRJ) using graph neural networks on the Lund jet plane, with support for a quantum-hybrid variant (QLundNet) and a score-combination model (Combiner).

This repository was originally built for top/W tagging on large-R jets and has since been extended with a parallel set of scripts (the `*_SRJ.py` files and corresponding `*_SRJ.yaml` configs) for **quark/gluon tagging on small-radius jets**. The Combiner scripts (`weight_ONLY_TRAINS_COMBINER_SRJ.py`, `make_scores_combiner_SRJ.py`) are also designed for the SRJ q/g tagging workflow.


## Running on DIAS

This project is designed to run on UCL's DIAS HPC cluster via SLURM. All jobs are submitted using the scripts in the `submit/` directory (and a few top-level `.slurm` files). Each submit script activates the shared conda environment before running:

```bash
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
```

Jobs are submitted to either the **GPU** partition (training, scoring) or the **COMPUTE** partition (data preparation, preprocessing). For example:

```bash
# Submit data preparation as a SLURM array job (one task per event fraction slice)
sbatch submit/submit_slurm_make_data.sh

# Submit preprocessing
sbatch submit/submit_preprocess.sh

# Submit a training job
sbatch submit/submit_slurm_train.sh

# Submit scoring
sbatch submit/submit_slurm_scores.sh

# Submit combiner training / scoring
sbatch submit/submit_slurm_train_combiner.sh
sbatch submit/submit_slurm_scores_combiner.sh
```

Logs are written to the `logs/` directory. Email notifications are sent on job start/finish/failure.


## Data preparation

Run `Make_data_SRJ.py` to create graphs for training from ROOT files:

```bash
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml
```

The script applies selections defined in the configuration files, creates Lund trees (graphs) for each jet, and outputs graph files (`torch_geometric.data.Data` objects) and a ROOT file containing jet properties (truth labels, $p_T$, $\eta$, mass, constituent counts, etc.).

### Signal configuration

The jet selection cuts are controlled by `configs/config_signal_SRJ.yaml`, which defines a single `srj` block:

- **Signal**: truth labels 1--5 (light quarks + charm + bottom)
- **Background**: truth labels $-1$ and 21 (other + gluons)
- $p_T \in [20, 160]$ GeV, $|\eta| \in [3.2, 4.5]$, no mass cut, min 3 splittings
- No DSID-based splitting (SRJ files are already mixed)
- Reweighting is handled at the preprocessing stage rather than via histogram files

The signal config file to use and which block to select are specified in the main data config (`config_make_data_SRJ.yaml`) under the `signal_config_file` and `signal` keys.

You can override parameters using the `--override` command-line argument:

```bash
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml --override signal_config_file="configs/config_signal_SRJ.yaml" id="dijet" signal="srj"
```


## Models

### LundNet

The primary model. LundNet is a graph neural network that operates on the Lund jet plane representation of jets. It uses six stacked EdgeConv layers with batch normalisation, where each layer learns edge features from pairs of connected nodes in the Lund tree. The outputs of all six layers are concatenated (skip connections), passed through a fully connected layer, pooled across all nodes via global mean pooling, and then combined with the number of charged tracks ($N_\text{trk}$) before a final classification head produces a signal probability.

### QLundNet

A quantum-hybrid variant of LundNet. QLundNet replaces the first EdgeConv layer with a `QuantumEdgeConv` layer, which routes edge features through a parameterised quantum circuit before returning to classical processing. The remaining five EdgeConv layers are classical.

The quantum circuit works as follows:
1. **Classical preprocessing**: an MLP maps the concatenated edge features $[x_i \| x_j]$ down to a dimension suitable for the quantum circuit ($n_\text{qubits} \times 2$), with a Tanh activation to normalise values to $[-1, 1]$.
2. **Input encoding**: features are encoded into qubit states using $R_X$ and $R_Y$ rotation gates (two features per qubit).
3. **Parameterised layers**: each quantum layer applies IsingXX entangling gates between adjacent qubits followed by trainable $R_X$ / $R_Y$ rotations. The rotation angles are learnable parameters optimised via backpropagation through PennyLane.
4. **Measurement**: Pauli-Z expectation values are measured on each qubit, producing one classical output per qubit.
5. **Classical postprocessing**: an MLP maps the measurement outputs to the desired feature dimension, followed by batch normalisation and ReLU.

The quantum circuit uses 4 qubits and 1--3 quantum layers by default. It attempts to use GPU-accelerated simulation (`lightning.gpu`) and falls back to `lightning.qubit` or `default.qubit`. QLundNet requires PennyLane (`pip install pennylane`).

### Combiner

The Combiner is a small MLP that fuses the output scores of two independently trained taggers into a single combined score. It takes five inputs: the logit-transformed scores from tagger A and tagger B, the total number of jet constituents ($N_\text{const}$), the number of charged constituents ($N_\text{const,charged}$), and the jet mass ($m$).

The raw scores from each tagger are converted to logits via $\text{logit}(p) = \ln(p / (1 - p))$ before being fed into a two-hidden-layer MLP (hidden size configurable, default 64) with ReLU activations and a sigmoid output. The model is trained with a pairwise ranking loss that directly optimises AUC by penalising cases where a signal jet is ranked below a background jet.


## Pipelines

### LundNet / QLundNet pipeline

The SRJ quark/gluon tagging pipeline. QLundNet uses exactly the same scripts — just set `choose_model: QLundNet` in the training config. The pipeline adds a dedicated preprocessing step for feature standardisation and $p_T$/$\eta$ flattening.

| Step | Script | Config | SLURM submit |
|------|--------|--------|--------------|
| 1. Build graphs from derivations | `Make_data_SRJ.py` | `configs/config_make_data_SRJ.yaml` | `sbatch submit/submit_slurm_make_data.sh` (array job) |
| 2. Preprocess (standardise + flatten) | `preprocess_SRJ_CPU.py` | `configs/config_preprocess_SRJ.yaml` | `sbatch submit/submit_preprocess.sh` |
| 3. Train | `weight_ONLY_TRAINS_SRJ.py` | `configs/config_ONLY_TRAIN_SRJ.yaml` | `sbatch submit/submit_slurm_train.sh` |
| 4. Score | `test_make_scores_SRJ.py` | `configs/config_make_scores_SRJ.yaml` | `sbatch submit/submit_slurm_scores.sh` |

**Step 1** applies selection cuts from `config_signal_SRJ.yaml`, builds Lund tree graphs (`torch_geometric.data.Data`), and outputs graph files and a ROOT file with jet properties. Runs as a SLURM array job (one task per `event_fraction` slice).

**Step 2** loads the raw graphs from step 1, applies jet selection cuts (configurable $p_T$, $\eta$, mass ranges from `config_signal_SRJ.yaml`), standardises node features using precomputed mean/std from a JSON file, and optionally flattens the $p_T$ and $\eta$ distributions. Outputs separate train and test `.pt` files. Runs on the COMPUTE partition (CPU only, high memory).

**Step 3** trains the model. Set the model architecture (`LundNet`, `QLundNet`, etc.) via `architecture.choose_model` in the config. Checkpoints are saved every epoch. Optional CLI overrides:
```bash
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml --ln_kT_cut 0 --do_combined_training true
```

**Step 4** loads one or more trained checkpoints and writes per-jet scores to a ROOT file. The config supports evaluating multiple models in a single run (each with its own tag and checkpoint), and scores are written as separate branches. Paths support `{sample}` and `{kT_cut}` placeholders for easy switching between datasets.

### Combiner pipeline

The Combiner trains on top of existing tagger scores rather than raw graphs. It requires that you have already scored your test data with at least two taggers (e.g. LundNet and ParT) so that the ROOT file contains score branches for both.

| Step | Script | Config | SLURM submit |
|------|--------|--------|--------------|
| 1. Train combiner | `weight_ONLY_TRAINS_COMBINER_SRJ.py` | `configs/config_ONLY_TRAIN_COMBINER_SRJ.yaml` | `sbatch submit/submit_slurm_train_combiner.sh` |
| 2. Score with combiner | `make_scores_combiner_SRJ.py` | `configs/config_make_scores_combiner_SRJ.yaml` | `sbatch submit/submit_slurm_scores_combiner.sh` |

**Step 1** reads score branches (e.g. `fjet_LundNet_FTAG1_score` and `parT_score`) plus `fjet_Nconst`, `fjet_Nconst_Charged`, and `fjet_m` from a ROOT file, splits into train/validation, and trains the Combiner MLP with pairwise ranking loss. Checkpoints and a validation loss log are saved each epoch.

**Step 2** runs both the LundNet model and the trained Combiner end-to-end: it first scores jets with LundNet on the graph data, reads the existing ParT scores from the ROOT file, then passes all five features (LundNet score, ParT score, $N_\text{const}$, $N_\text{const,charged}$, $m$) through the Combiner to produce a final combined score. All scores (LundNet and Combined) are written as branches to the output ROOT file.


## Results

The `results/` directory contains ROOT files with scored jets for two kinematic regions:

- **`results/pt_160_eta_3.2_4.5/`** — forward region: $p_T > 160$ GeV, $|\eta| \in [3.2, 4.5]$. Contains LundNet, QLundNet, and Combiner scores across several train/test splits and FTAG1 samples.
- **`results/pt_160_1300_eta_0_1.2/`** — central region: $p_T \in [160, 1300]$ GeV, $|\eta| \in [0, 1.2]$. Contains LundNet and Combiner scores.

Each subdirectory corresponds to a different train/test configuration (e.g. `70%train30%test/`, `40%40%/`, `1Mtrain2Mtest/`). Combiner results are nested under their own subdirectories within these.


## Plotting

Plots are produced using the `plotting_SRJ.ipynb` Jupyter notebook. Point it at the ROOT file(s) in `results/` and run the cells to generate ROC curves, score distributions, and other performance plots.
