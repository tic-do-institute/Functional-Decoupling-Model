import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# --- Plotting Standards ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,
    'figure.titlesize': 10,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ==============================================================================
# 1. THE PHYSICS ENGINE (Corrected for Decoherence)
# ==============================================================================
@dataclass
class ModelParams:
    n_nodes: int = 5
    dt: float = 0.05
    omega_mean: float = 1.0  # Mean natural frequency
    omega_std: float = 0.5   # ADDED: Standard deviation for heterogeneity
    g_L: float = 0.2         # UPDATED: Lower gain to ensure breakup
    g_U: float = 3.0
    g_H: float = 6.0
    k_pot: float = 2.0
    alpha_error: float = 35.0

class GCAIKuramotoChain:
    def __init__(self, params: ModelParams):
        self.p = params
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.p.n_nodes)
        self.gamma = np.ones(self.p.n_nodes) * self.p.g_H
        # FIX: Assign random natural frequencies to cause drift when uncoupled
        np.random.seed(42) # Ensure reproducibility
        self.natural_omegas = np.random.normal(self.p.omega_mean, self.p.omega_std, self.p.n_nodes)

    def _potential_force(self, gamma):
        return self.p.k_pot * (gamma - self.p.g_L) * (gamma - self.p.g_U) * (self.p.g_H - gamma)

    def step(self, noise_std_dev, sip_impulse=None):
        N = self.p.n_nodes
        theta_left = np.roll(self.theta, 1); theta_right = np.roll(self.theta, -1)
        
        # Phase differences
        diff_left = np.sin(theta_left - self.theta); diff_left[0] = 0.0
        diff_right = np.sin(theta_right - self.theta); diff_right[-1] = 0.0
        
        # Coupling term (weighted by Precision gamma)
        weighted_coupling = self.gamma * (diff_left + diff_right)

        # Error calculation for metabolic cost
        neighbor_counts = np.ones(N) * 2; neighbor_counts[0] = 1; neighbor_counts[-1] = 1
        sq_error = (diff_left**2 + diff_right**2) / neighbor_counts

        # Gamma Dynamics (Gradient Descent on Free Energy)
        F_intrinsic = self._potential_force(self.gamma)
        F_error     = - self.p.alpha_error * sq_error
        d_gamma = F_intrinsic + F_error

        if sip_impulse is not None:
            d_gamma += sip_impulse

        self.gamma += d_gamma * self.p.dt
        self.gamma = np.clip(self.gamma, 0.01, 10.0)

        # Theta Dynamics (Kuramoto with Heterogeneity)
        noise = np.random.normal(0, 1, N) * noise_std_dev * np.sqrt(self.p.dt)
        
        # FIX: Use natural_omegas instead of scalar omega
        d_theta = self.natural_omegas + weighted_coupling
        
        self.theta += d_theta * self.p.dt + noise

        z = np.mean(np.exp(1j * self.theta))
        return self.theta, self.gamma, np.abs(z)

# ==============================================================================
# 2. GENERATE FIGURE 2 (Corrected Visualization)
# ==============================================================================
def generate_figure2():
    print("Generating Figure 2 (Corrected: Decoherence)...")
    
    # Setup Model with Heterogeneity
    params = ModelParams(n_nodes=10, dt=0.01, omega_std=2.0, g_L=0.1) # Increased nodes/spread for visual effect
    model = GCAIKuramotoChain(params)

    T = 80.0
    steps = int(T / params.dt)
    time = np.linspace(0, T, steps)

    t_stress_start, t_stress_end = 20.0, 35.0
    t_sip = 60.0
    target_node = 2

    theta_hist = np.zeros((steps, params.n_nodes))
    gamma_hist = np.zeros((steps, params.n_nodes))
    order_hist = np.zeros(steps)

    np.random.seed(42)

    for i, t in enumerate(time):
        base_noise = 0.1
        noise_vec = np.ones(params.n_nodes) * base_noise
        
        # Apply Stress
        if t_stress_start <= t < t_stress_end:
            noise_vec[:] += 8.0 # Apply global stress to force break
        
        # Apply Rescue (SIP)
        sip_vec = None
        if t_sip <= t < t_sip + 0.2:
            sip_vec = np.zeros(params.n_nodes)
            sip_vec[:] = 200.0 # Strong kick to jump barrier

        theta, gamma, R = model.step(noise_vec, sip_impulse=sip_vec)
        theta_hist[i] = theta
        gamma_hist[i] = gamma
        order_hist[i] = R

    # --- Plotting ---
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.5)

    # Panel A: Phase Dynamics
    ax0 = fig.add_subplot(gs[0])
    # Plot a subset of nodes for clarity
    for n in range(min(5, params.n_nodes)):
        ax0.plot(time, np.sin(theta_hist[:, n]), lw=0.8, alpha=0.7)
    
    ax0.set_ylabel(r"State ($\sin\theta$)")
    ax0.set_title("A. Phase Dynamics: Emergent Desynchronization", loc='left', fontweight='bold')
    ax0.set_xlim(0, T); ax0.set_ylim(-1.5, 1.5)
    
    # Annotations
    y_txt = 1.3
    ax0.text(10, y_txt, "1. Homeostasis", ha='center', fontsize=7)
    ax0.text(27.5, y_txt, "2. Exposure", ha='center', color='#E41A1C', fontweight='bold', fontsize=7)
    ax0.text(47.5, y_txt, "3. Persistence", ha='center', color='#FF7F00', fontweight='bold', fontsize=7)
    ax0.text(70, y_txt, "4. Resolution", ha='center', color='#377EB8', fontweight='bold', fontsize=7)

    # Panel B: Precision Dynamics
    ax1 = fig.add_subplot(gs[1])
    ax1.axhline(params.g_H, c='#4DAF4A', ls='--', lw=1.0, alpha=0.9, label='Healthy Well')
    ax1.axhline(params.g_L, c='#E41A1C', ls='--', lw=1.0, alpha=0.9, label='Defensive Well')
    
    # Plot mean gamma for clarity
    mean_gamma = np.mean(gamma_hist, axis=1)
    ax1.plot(time, mean_gamma, c='orange', lw=1.5)
    
    ax1.set_ylabel(r"Precision ($\gamma$)")
    ax1.set_title("B. Precision Dynamics: Metabolic vs. Informational Cost", loc='left', fontweight='bold')
    ax1.set_xlim(0, T); ax1.set_ylim(0, 8.0)
    ax1.legend(loc='lower right', fontsize=6)

    # Panel C: Systemic Phase Transition (The Key Fix)
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(time, order_hist, c='#984EA3', lw=1.5)
    
    ax2.set_ylabel(r"Sync Order ($R$)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("C. Systemic Phase Transition", loc='left', fontweight='bold')
    ax2.set_xlim(0, T); ax2.set_ylim(0, 1.15)

    # Correct Labeling
    ax2.text(47.5, 0.2, "Decoherence\n(Functional Decoupling)", ha='center', va='center',
             color='#E41A1C', fontsize=7, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Shading
    for ax in [ax0, ax1, ax2]:
        ax.axvspan(t_stress_start, t_stress_end, color='#E41A1C', alpha=0.05, lw=0)
        ax.axvspan(t_stress_end, t_sip, color='#FF7F00', alpha=0.03, lw=0)
        ax.axvline(t_sip, color='#377EB8', ls='--', lw=1.0)

    plt.tight_layout()
    plt.savefig('Figure_2.png')
    print("Figure_2.png Generated.")

if __name__ == "__main__":
    generate_figure2()
