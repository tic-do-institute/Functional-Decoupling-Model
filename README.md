# Critical Collapse of Biological Precision: A Fold Bifurcation from Information-Metabolic Tradeoffs

**Official Simulation Code Repository**

**Manuscript Status:** Submitted to *Journal of the Royal Society Interface* (JRSI)

This repository contains the full suite of Python source codes used to generate the computational results, theoretical profiles, and empirical data analyses presented in the manuscript and its Supplementary Information (SI).

The code implements a formal non-equilibrium thermodynamic framework, modeling how biological gain control (precision) generically approaches a fold (saddle-node) catastrophe when subject to continuous metabolic constraints, leading to an irreversible "near-zero efficiency" state.

---

## 📝 Note on Terminology

*Please note: In earlier working versions of this project and some internal codebase logs, the low-precision state was occasionally referred to using the acronym **PPEN** (Proprioceptive Prediction Error Neglect). To align with the macroscopic biophysical and thermodynamic focus of the final manuscript, this terminology has been unified to **"Zero-Efficiency Regime"** or **"Collapsed Regime"**.*

---

## 📂 Repository Structure

The simulation scripts are organized according to the Figures they generate in the manuscript.

### 1. Theoretical Models & Simulations (Main Text)

* **`Theory_overview.py`**
* **Generates:** `Figure 1` (Bifurcation map, Efficiency, and Potential landscape).
* **Description:** Derives the analytical bifurcation map and thermodynamic efficiency function.


* **`Core_Simulation.py`**
* **Generates:** `Figure 2` (Phase Dynamics, Precision, and Sync Order).
* **Description:** Simulates the time-series evolution using a Gradient-based Coupled Active Inference (G-CAI) oscillator model. Note: This script incorporates natural frequency heterogeneity to correctly demonstrate emergent desynchronization (decoherence).


* **`Hysteresis_loop.py`**
* **Generates:** `Figure 3` (The Hysteresis Loop of Gain Dynamics).
* **Description:** Simulates the forward and backward stress sweeps to demonstrate the bistable region and hysteresis.


* **`Sensitivity_Analysis.py`**
* **Generates:** `Figure 4` (Topological Robustness heatmap).
* **Description:** Performs a comprehensive parameter sweep to map the robustness of the precision collapse across the parameter space.


* **`CSD_Scaling.py`**
* **Generates:** `Figure 5` (Log-log scaling of relaxation time).
* **Description:** Numerically integrates the normal form $\dot{x} = \mu - x^2$ to calculate relaxation times, proving the universal scaling law $\tau \sim |\mu|^{-1/2}$ of the fold universality class.


* **`Predictions.py`**
* **Generates:** `Figure 6` (Potential well flattening and Rescue dynamics).
* **Description:** Simulates theoretically predicted signatures of collapse and recovery, comparing steady vs. high-derivative stimulation.



### 2. Early Warning Signals & Empirical Validation (Supplementary Information)

* **`Critical_Slowing_Down.py`**
* **Generates:** `Figure S1` (EWS Theoretical Profiles).
* **Description:** Plots the predicted amplification of baseline variance and the increase in Lag-1 Autocorrelation as the system approaches criticality.


* **`Empirical_Pipeline.py`**
* **Generates:** `Figure S2`, `Figure S3`, `Figure S4`, `Figure S5` and terminal statistics.
* **Description:** The complete automated data extraction and statistical validation pipeline (**Analysis Pipeline v1.0**). This script processes the raw BIDS dataset (OpenNeuro ds003838) to confirm the presence of critical slowing down and compensatory precision drive.
* **Key Operations:** Preprocessing, extraction of dynamical proxies (Variance, AC1, Beta, Phasic), Negative Control (LMM), and Surrogate Permutation Testing (N=1000).



---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Standard scientific computing libraries:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels tqdm

```

### Installation

1. Clone this repository:

```bash
git clone https://github.com/tic-do-institute/Functional-Decoupling-Model.git
cd Functional-Decoupling-Model

```

### Running the Code

To reproduce the figures, run the corresponding scripts. Figures will be saved as high-resolution `.png` or `.jpg` files.

```bash
# Example: Generating Figure 5
python generate_fig5.py

```

---

## 📬 Contact

**Takafumi Shiga**
Director, TIC-DO Institute
Tokyo, 1070052, Japan
Email: tic.do.institute@proton.me
Web: [https://tic-do-institute.github.io](https://tic-do-institute.github.io)

---

## 📄 License

This project is licensed under the MIT License.

---

