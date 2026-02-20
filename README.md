# A Zero-Efficiency Regime Explains Irreversible Gain Collapse in the Locus Coeruleus

**Official Simulation Code Repository**


**Manuscript Status:** Submitted to *PNAS (Proceedings of the National Academy of Sciences)* 

This repository contains the source code used to generate the computational results and figures presented in the manuscript: *"A Zero-Efficiency Regime Explains Irreversible Gain Collapse in the Locus Coeruleus"*.

The code implements a **Gradient-based Coupled Active Inference (G-CAI) Kuramoto model**, simulating how chronic functional disorders emerge as metastable attractors driven by a "Zero-Efficiency" thermodynamic trap.

---

## 📝 Note on Terminology
*Please note: In earlier working versions of this project and some pre-compiled supplementary figures (e.g., Figure 4), the low-precision state was occasionally labeled using the internal working acronym **PPEN** (Proprioceptive Prediction Error Neglect). To better align with the macroscopic biophysical and thermodynamic focus of the final manuscript, this terminology has been unified to **"Zero-Efficiency Regime"** or **"Collapsed Regime"** in the latest version of this codebase.*

## 📂 Repository Structure

The simulation scripts are organized according to the Figures they generate in the manuscript:

* 
**`Theory_overview.py`** 


* **Function:** Derives the analytical bifurcation map and thermodynamic efficiency.
* 
**Output:** **Figure 1** (Thermodynamic-informational phase transition, Canonical bifurcation map, Efficiency , and Topological irreversibility).




* 
**`core_simulation.py`** 


* **Function:** Simulates the time-series evolution of the system, including phase dynamics and precision collapse under stress.
* 
**Output:** **Figure 2** (Dynamical Capture of the LC-NE System, Phase Dynamics, and Irreversibility).




* 
**`Hysteresis_loop&Efficiency.py`** 


* **Function:** Simulates the forward and backward stress sweeps to demonstrate hysteresis.
* 
**Output:** **Figure 3** (The Hysteresis Loop of Locus Coeruleus Dynamics and the Zero-Efficiency regime).




* 
**`sensitivity_analysis.py`** 


* **Function:** Performs a comprehensive parameter sweep across Error Sensitivity () and Environmental Uncertainty ().
* 
**Output:** **Figure 4** (Topological Robustness of Precision Collapse heatmap).




* 
**`Prediction.py`** 


* **Function:** Simulates the effective potential landscape and counterfactual analysis of therapeutic interventions.
* 
**Output:** **Figure 5** (Theoretically Predicted Signatures: Critical Slowing Down and High-Jerk Rescue).





---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Standard scientific computing libraries (`numpy`, `matplotlib`, `scipy`) 



### Installation

1. Clone this repository:
```bash
git clone https://github.com/tic-do-institute/Functional-Decoupling-Model.git
cd Functional-Decoupling-Model

```



[Note: Repository URL based on manuscript data availability statement ]


2. Install dependencies:
```bash
pip install -r requirements.txt

```



---

## 💻 Usage

To reproduce the figures from the manuscript, run the corresponding Python scripts. The figures will be saved as `.png` files in the same directory.

### 1. Theory & Bifurcation Map (Fig 1)

```bash
python Theory_overview.py

```

*Output:* `Figure_1.png`

### 2. Time-Series Dynamics (Fig 2)

```bash
python core_simulation.py

```

*Output:* `Fig2_Gradient_Simulation.png` 

### 3. Hysteresis Loop (Fig 3)

```bash
python Hysteresis_loop&Efficiency.py

```

*Output:* `Fig3_PhaseDiagram_Hysteresis.png` 

### 4. Robustness Analysis (Fig 4)

```bash
python sensitivity_analysis.py

```

*Output:* `Fig4_Robustness.png` 

### 5. Predictions & Rescue (Fig 5)

```bash
python Prediction.py

```

*Output:* `Fig5_Predictions.png` 

---

## 📬 Contact

**Takafumi Shiga**
Principal Investigator
TIC-DO Institute, Tokyo, Japan
Email: tic.do.institute@proton.me Web: [https://tic-do-institute.github.io]() 

---

## 📄 License

This project is licensed under the MIT License.
