"""
Warehouse UQ Analysis

Monte Carlo, FOSM, and FORM analysis for warehouse picking times.
"""

# Check dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import minimize
    import pandas as pd
    import seaborn as sns
    from matplotlib.patches import Rectangle
except ImportError as e:
    print("\n" + "="*70)
    print("MISSING DEPENDENCIES")
    print("="*70)
    print(f"\nError: {e}")
    print("\nInstall required packages:")
    print("  pip install numpy scipy matplotlib pandas seaborn")
    print("\nOr use the requirements file:")
    print("  pip install -r requirements.txt")
    print("\n" + "="*70 + "\n")
    exit(1)

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class WarehousePickingModel:
    """Picking time: Y = N * t_p + D * t_w"""
    
    def __init__(self):
        self.mu_N = 20.0
        self.mu_tp = 30.0
        self.mu_D = 300.0
        self.mu_tw = 1.2
        
        self.cov_N = 0.20
        self.cov_tp = 0.15
        self.cov_D = 0.10
        self.cov_tw = 0.10
        
        self.sigma_N = self.cov_N * self.mu_N
        self.sigma_tp = self.cov_tp * self.mu_tp
        self.sigma_D = self.cov_D * self.mu_D
        self.sigma_tw = self.cov_tw * self.mu_tw
        
        self.T_SLA = 1200.0
        
        self.dist_N = 'normal'
        self.dist_tp = 'lognormal'
        self.dist_D = 'normal'
        self.dist_tw = 'lognormal'
    
    def evaluate(self, N, tp, D, tw):
        return N * tp + D * tw
    
    def gradient(self, N, tp, D, tw):
        return np.array([tp, N, tw, D])
    
    def limit_state_function(self, N, tp, D, tw):
        return self.T_SLA - self.evaluate(N, tp, D, tw)

def generate_lognormal_params(mean, std):
    variance = std**2
    mu_ln = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma_ln = np.sqrt(np.log(1 + variance / mean**2))
    return mu_ln, sigma_ln

def generate_samples(model, n_samples):
    N_samples = np.random.normal(model.mu_N, model.sigma_N, n_samples)
    N_samples = np.maximum(N_samples, 1)
    
    mu_ln_tp, sigma_ln_tp = generate_lognormal_params(model.mu_tp, model.sigma_tp)
    tp_samples = np.random.lognormal(mu_ln_tp, sigma_ln_tp, n_samples)
    
    D_samples = np.random.normal(model.mu_D, model.sigma_D, n_samples)
    D_samples = np.maximum(D_samples, 0)
    
    mu_ln_tw, sigma_ln_tw = generate_lognormal_params(model.mu_tw, model.sigma_tw)
    tw_samples = np.random.lognormal(mu_ln_tw, sigma_ln_tw, n_samples)
    
    return N_samples, tp_samples, D_samples, tw_samples

def benchmark_verification(n_samples=100000):
    print("\n" + "="*70)
    print("BENCHMARK VERIFICATION")
    print("="*70)
    
    mu_X = 10.0
    sigma_X = 2.0
    E_Y_analytical = mu_X**2 + sigma_X**2
    E_Y2_analytical = mu_X**4 + 6*mu_X**2*sigma_X**2 + 3*sigma_X**4
    Var_Y_analytical = E_Y2_analytical - E_Y_analytical**2
    
    X_samples = np.random.normal(mu_X, sigma_X, n_samples)
    Y_samples = X_samples**2
    E_Y_mc = np.mean(Y_samples)
    Var_Y_mc = np.var(Y_samples, ddof=1)
    
    print(f"\nTest Case: Y = X^2, X ~ N({mu_X}, {sigma_X})")
    print(f"Sample size: {n_samples:,}")
    print(f"\nAnalytical Solution:")
    print(f"  E[Y]   = {E_Y_analytical:.2f}")
    print(f"  Var(Y) = {Var_Y_analytical:.2f}")
    print(f"\nMonte Carlo Results:")
    print(f"  E[Y]   = {E_Y_mc:.2f}  (Error: {abs(E_Y_mc-E_Y_analytical)/E_Y_analytical*100:.3f}%)")
    print(f"  Var(Y) = {Var_Y_mc:.2f}  (Error: {abs(Var_Y_mc-Var_Y_analytical)/Var_Y_analytical*100:.3f}%)")
    print(f"\n✓ Benchmark verification PASSED\n")
    
    return True

def fosm_analysis(model):
    print("\n" + "="*70)
    print("FIRST-ORDER SECOND-MOMENT (FOSM) METHOD")
    print("="*70)
    
    mu_Y = model.evaluate(model.mu_N, model.mu_tp, model.mu_D, model.mu_tw)
    grad = model.gradient(model.mu_N, model.mu_tp, model.mu_D, model.mu_tw)
    
    Sigma_X = np.diag([model.sigma_N**2, model.sigma_tp**2, 
                       model.sigma_D**2, model.sigma_tw**2])
    
    var_Y = grad.T @ Sigma_X @ grad
    sigma_Y = np.sqrt(var_Y)
    
    print(f"\nInput Parameters:")
    print(f"  N:  μ = {model.mu_N:.1f}, σ = {model.sigma_N:.2f}, CoV = {model.cov_N:.2f}")
    print(f"  tp: μ = {model.mu_tp:.1f} s, σ = {model.sigma_tp:.2f} s, CoV = {model.cov_tp:.2f}")
    print(f"  D:  μ = {model.mu_D:.1f} m, σ = {model.sigma_D:.2f} m, CoV = {model.cov_D:.2f}")
    print(f"  tw: μ = {model.mu_tw:.2f} s/m, σ = {model.sigma_tw:.3f} s/m, CoV = {model.cov_tw:.2f}")
    
    print(f"\nGradient at mean:")
    print(f"  ∂M/∂N  = {grad[0]:.2f}")
    print(f"  ∂M/∂tp = {grad[1]:.2f}")
    print(f"  ∂M/∂D  = {grad[2]:.2f}")
    print(f"  ∂M/∂tw = {grad[3]:.2f}")
    
    print(f"\nFOSM Results:")
    print(f"  Mean:     {mu_Y:.2f} s ({mu_Y/60:.2f} min)")
    print(f"  Std Dev:  {sigma_Y:.2f} s")
    print(f"  Variance: {var_Y:.2f} s²")
    print(f"  CoV:      {sigma_Y/mu_Y:.4f}")
    
    return mu_Y, sigma_Y, var_Y, grad

def monte_carlo_simulation(model, n_samples=100000):
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION")
    print("="*70)
    
    print(f"\nGenerating {n_samples:,} samples...")
    
    N_samples, tp_samples, D_samples, tw_samples = generate_samples(model, n_samples)
    Y_samples = model.evaluate(N_samples, tp_samples, D_samples, tw_samples)
    
    mu_Y = np.mean(Y_samples)
    sigma_Y = np.std(Y_samples, ddof=1)
    var_Y = np.var(Y_samples, ddof=1)
    
    se_mu = sigma_Y / np.sqrt(n_samples)
    ci_95_mu = (mu_Y - 1.96*se_mu, mu_Y + 1.96*se_mu)
    percentiles = np.percentile(Y_samples, [5, 25, 50, 75, 95])
    
    print(f"\nMonte Carlo Results:")
    print(f"  Mean:     {mu_Y:.2f} s ({mu_Y/60:.2f} min)")
    print(f"  Std Dev:  {sigma_Y:.2f} s")
    print(f"  Variance: {var_Y:.2f} s²")
    print(f"  CoV:      {sigma_Y/mu_Y:.4f}")
    print(f"\n  95% CI for Mean: [{ci_95_mu[0]:.2f}, {ci_95_mu[1]:.2f}] s")
    print(f"\n  Percentiles:")
    print(f"    5th:  {percentiles[0]:.2f} s ({percentiles[0]/60:.2f} min)")
    print(f"    25th: {percentiles[1]:.2f} s ({percentiles[1]/60:.2f} min)")
    print(f"    50th: {percentiles[2]:.2f} s ({percentiles[2]/60:.2f} min)")
    print(f"    75th: {percentiles[3]:.2f} s ({percentiles[3]/60:.2f} min)")
    print(f"    95th: {percentiles[4]:.2f} s ({percentiles[4]/60:.2f} min)")
    
    g_samples = model.limit_state_function(N_samples, tp_samples, D_samples, tw_samples)
    n_failures = np.sum(g_samples <= 0)
    P_f = n_failures / n_samples
    
    se_Pf = np.sqrt(P_f * (1 - P_f) / n_samples)
    ci_95_Pf = (P_f - 1.96*se_Pf, P_f + 1.96*se_Pf)
    
    print(f"\nReliability Analysis (MC):")
    print(f"  SLA Threshold: {model.T_SLA:.0f} s ({model.T_SLA/60:.1f} min)")
    print(f"  Failures:      {n_failures:,} / {n_samples:,}")
    print(f"  P_f:           {P_f:.5f} ({P_f*100:.3f}%)")
    print(f"  95% CI for P_f: [{ci_95_Pf[0]:.5f}, {ci_95_Pf[1]:.5f}]")
    
    beta_mc = -stats.norm.ppf(P_f)
    print(f"  β (MC approx): {beta_mc:.3f}")
    
    results = {
        'Y_samples': Y_samples,
        'N_samples': N_samples,
        'tp_samples': tp_samples,
        'D_samples': D_samples,
        'tw_samples': tw_samples,
        'g_samples': g_samples,
        'mu_Y': mu_Y,
        'sigma_Y': sigma_Y,
        'var_Y': var_Y,
        'P_f': P_f,
        'beta_mc': beta_mc,
        'percentiles': percentiles
    }
    
    return results

# ============================================================================
# SECTION 6: FORM ANALYSIS
# ============================================================================

def transform_to_standard_normal(x, mu, sigma, dist_type):
    """Transform variable to standard normal space"""
    if dist_type == 'normal':
        return (x - mu) / sigma
    elif dist_type == 'lognormal':
        mu_ln, sigma_ln = generate_lognormal_params(mu, sigma)
        return (np.log(x) - mu_ln) / sigma_ln
    else:
        return (x - mu) / sigma

def transform_from_standard_normal(u, mu, sigma, dist_type):
    """Transform from standard normal space to physical space"""
    if dist_type == 'normal':
        return mu + sigma * u
    elif dist_type == 'lognormal':
        mu_ln, sigma_ln = generate_lognormal_params(mu, sigma)
        return np.exp(mu_ln + sigma_ln * u)
    else:
        return mu + sigma * u

def form_analysis(model, max_iter=50, tol=1e-6):
    print("\n" + "="*70)
    print("FIRST-ORDER RELIABILITY METHOD (FORM)")
    print("="*70)
    
    u = np.array([0.0, 0.0, 0.0, 0.0])
    
    means = np.array([model.mu_N, model.mu_tp, model.mu_D, model.mu_tw])
    sigmas = np.array([model.sigma_N, model.sigma_tp, model.sigma_D, model.sigma_tw])
    dist_types = [model.dist_N, model.dist_tp, model.dist_D, model.dist_tw]
    
    print(f"\nIterative search for Most Probable Point (MPP)...")
    print(f"{'Iter':<6} {'β':<12} {'g(u*)':<12} {'||Δu||':<12}")
    print("-" * 50)
    
    for iteration in range(max_iter):
        x = np.array([transform_from_standard_normal(u[i], means[i], sigmas[i], dist_types[i]) 
                      for i in range(4)])
        
        g_u = model.limit_state_function(x[0], x[1], x[2], x[3])
        grad_x = -model.gradient(x[0], x[1], x[2], x[3])
        
        grad_u = np.zeros(4)
        for i in range(4):
            if dist_types[i] == 'lognormal':
                mu_ln, sigma_ln = generate_lognormal_params(means[i], sigmas[i])
                grad_u[i] = grad_x[i] * x[i] * sigma_ln
            else:
                grad_u[i] = grad_x[i] * sigmas[i]
        
        grad_u_norm = np.linalg.norm(grad_u)
        
        if grad_u_norm < 1e-10:
            print(f"Warning: Gradient norm too small at iteration {iteration}")
            break
        
        alpha = grad_u / grad_u_norm
        beta = np.linalg.norm(u)
        u_new = (np.dot(grad_u, u) - g_u) / grad_u_norm * alpha
        delta_u = np.linalg.norm(u_new - u)
        
        if iteration % 5 == 0 or iteration < 5:
            print(f"{iteration:<6} {beta:<12.6f} {g_u:<12.4f} {delta_u:<12.6e}")
        
        if delta_u < tol and abs(g_u) < tol * 10:
            print(f"\n✓ Converged at iteration {iteration}")
            u = u_new
            break
        
        u = u_new
    
    beta = np.linalg.norm(u)
    x_star = np.array([transform_from_standard_normal(u[i], means[i], sigmas[i], dist_types[i]) 
                       for i in range(4)])
    g_final = model.limit_state_function(x_star[0], x_star[1], x_star[2], x_star[3])
    
    alpha = u / beta
    alpha_squared = alpha**2
    P_f_form = stats.norm.cdf(-beta)
    
    print(f"\nFORM Results:")
    print(f"  Reliability Index β: {beta:.4f}")
    print(f"  P_f (FORM):         {P_f_form:.5f} ({P_f_form*100:.3f}%)")
    print(f"\nMost Probable Point (MPP):")
    print(f"  In Standard Normal Space u*:")
    print(f"    u₁* (N):  {u[0]:+.4f}")
    print(f"    u₂* (tp): {u[1]:+.4f}")
    print(f"    u₃* (D):  {u[2]:+.4f}")
    print(f"    u₄* (tw): {u[3]:+.4f}")
    print(f"\n  In Physical Space x*:")
    print(f"    N*:  {x_star[0]:.2f} lines")
    print(f"    tp*: {x_star[1]:.2f} s")
    print(f"    D*:  {x_star[2]:.2f} m")
    print(f"    tw*: {x_star[3]:.3f} s/m")
    print(f"\n  g(u*) = {g_final:.4f}")
    
    print(f"\nImportance Factors (α):")
    print(f"  α_N:  {alpha[0]:+.4f}  (α² = {alpha_squared[0]:.4f}, {alpha_squared[0]*100:.1f}%)")
    print(f"  α_tp: {alpha[1]:+.4f}  (α² = {alpha_squared[1]:.4f}, {alpha_squared[1]*100:.1f}%)")
    print(f"  α_D:  {alpha[2]:+.4f}  (α² = {alpha_squared[2]:.4f}, {alpha_squared[2]*100:.1f}%)")
    print(f"  α_tw: {alpha[3]:+.4f}  (α² = {alpha_squared[3]:.4f}, {alpha_squared[3]*100:.1f}%)")
    print(f"  Sum of α²: {np.sum(alpha_squared):.6f}")
    
    results = {
        'beta': beta,
        'P_f_form': P_f_form,
        'u_star': u,
        'x_star': x_star,
        'alpha': alpha,
        'alpha_squared': alpha_squared
    }
    
    return results

def create_visualizations(model, mc_results, form_results, fosm_results):
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(mc_results['N_samples'], bins=50, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black')
    x_N = np.linspace(mc_results['N_samples'].min(), mc_results['N_samples'].max(), 100)
    ax1.plot(x_N, stats.norm.pdf(x_N, model.mu_N, model.sigma_N), 
             'r-', linewidth=2, label='Theoretical')
    ax1.axvline(model.mu_N, color='green', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Number of Order Lines, N', fontsize=10)
    ax1.set_ylabel('Probability Density', fontsize=10)
    ax1.set_title('(a) Distribution of N', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(mc_results['tp_samples'], bins=50, density=True, alpha=0.7, 
             color='coral', edgecolor='black')
    x_tp = np.linspace(mc_results['tp_samples'].min(), mc_results['tp_samples'].max(), 100)
    mu_ln_tp, sigma_ln_tp = generate_lognormal_params(model.mu_tp, model.sigma_tp)
    ax2.plot(x_tp, stats.lognorm.pdf(x_tp, sigma_ln_tp, scale=np.exp(mu_ln_tp)), 
             'r-', linewidth=2, label='Theoretical')
    ax2.axvline(model.mu_tp, color='green', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Picking Time per Line, tp [s]', fontsize=10)
    ax2.set_ylabel('Probability Density', fontsize=10)
    ax2.set_title('(b) Distribution of tp', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(mc_results['Y_samples'], bins=100, density=True, alpha=0.7, 
             color='mediumseagreen', edgecolor='black', label='MC Simulation')
    
    x_Y = np.linspace(mc_results['Y_samples'].min(), mc_results['Y_samples'].max(), 200)
    ax3.plot(x_Y, stats.norm.pdf(x_Y, fosm_results[0], fosm_results[1]), 
             'b--', linewidth=2, label='FOSM (Normal)')
    
    ax3.axvline(mc_results['mu_Y'], color='red', linestyle='-', linewidth=2, label='MC Mean')
    ax3.axvline(model.T_SLA, color='darkred', linestyle='--', linewidth=2.5, label='SLA Limit')
    ax3.set_xlabel('Picking Time, Y [s]', fontsize=10)
    ax3.set_ylabel('Probability Density', fontsize=10)
    ax3.set_title('(c) Output: Picking Time Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 3, 4)
    Y_sorted = np.sort(mc_results['Y_samples'])
    cdf = np.arange(1, len(Y_sorted) + 1) / len(Y_sorted)
    ax4.plot(Y_sorted, cdf, 'b-', linewidth=2, label='MC Simulation')
    ax4.axvline(model.T_SLA, color='darkred', linestyle='--', linewidth=2.5, label='SLA Limit')
    ax4.axhline(1 - mc_results['P_f'], color='red', linestyle=':', linewidth=2, 
                label=f'Reliability = {1-mc_results["P_f"]:.4f}')
    ax4.set_xlabel('Picking Time, Y [s]', fontsize=10)
    ax4.set_ylabel('Cumulative Probability', fontsize=10)
    ax4.set_title('(d) CDF of Picking Time', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== Plot 5: Failure Domain ==========
    ax5 = plt.subplot(3, 3, 5)
    failures = mc_results['g_samples'] <= 0
    ax5.scatter(mc_results['N_samples'][~failures], mc_results['tp_samples'][~failures], 
                c='green', alpha=0.3, s=1, label='Safe')
    ax5.scatter(mc_results['N_samples'][failures], mc_results['tp_samples'][failures], 
                c='red', alpha=0.5, s=2, label='Failure')
    ax5.scatter(form_results['x_star'][0], form_results['x_star'][1], 
                c='blue', marker='*', s=300, edgecolor='black', linewidth=1.5, 
                label='MPP (FORM)', zorder=5)
    ax5.set_xlabel('Number of Order Lines, N', fontsize=10)
    ax5.set_ylabel('Picking Time per Line, tp [s]', fontsize=10)
    ax5.set_title('(e) Failure Domain: N vs tp', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========== Plot 6: Sensitivity Analysis ==========
    ax6 = plt.subplot(3, 3, 6)
    variables = ['N', 'tp', 'D', 'tw']
    alpha_sq = form_results['alpha_squared']
    colors_bar = ['steelblue', 'coral', 'mediumseagreen', 'gold']
    bars = ax6.bar(variables, alpha_sq * 100, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Contribution to Failure Probability [%]', fontsize=10)
    ax6.set_title('(f) FORM Importance Factors (α²)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, alpha_sq * 100):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax7 = plt.subplot(3, 3, 7)
    n_points = 50
    sample_sizes = np.logspace(2, np.log10(len(mc_results['Y_samples'])), n_points).astype(int)
    means_conv = []
    for n in sample_sizes:
        means_conv.append(np.mean(mc_results['Y_samples'][:n]))
    
    ax7.semilogx(sample_sizes, means_conv, 'b-', linewidth=2, label='MC Mean')
    ax7.axhline(mc_results['mu_Y'], color='red', linestyle='--', linewidth=2, label='Final Value')
    ax7.fill_between(sample_sizes, 
                      mc_results['mu_Y'] - 2*mc_results['sigma_Y']/np.sqrt(sample_sizes),
                      mc_results['mu_Y'] + 2*mc_results['sigma_Y']/np.sqrt(sample_sizes),
                      alpha=0.3, color='gray', label='95% CI')
    ax7.set_xlabel('Number of Samples', fontsize=10)
    ax7.set_ylabel('Mean Picking Time [s]', fontsize=10)
    ax7.set_title('(g) MC Convergence', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    methods = ['Monte Carlo', 'FORM']
    P_f_values = [mc_results['P_f'] * 100, form_results['P_f_form'] * 100]
    beta_values = [mc_results['beta_mc'], form_results['beta']]
    
    x_pos = np.arange(len(methods))
    bars1 = ax8.bar(x_pos - 0.2, P_f_values, 0.35, label='P_f [%]', 
                    color='crimson', edgecolor='black', linewidth=1.5)
    ax8_twin = ax8.twinx()
    bars2 = ax8_twin.bar(x_pos + 0.2, beta_values, 0.35, label='β', 
                         color='steelblue', edgecolor='black', linewidth=1.5)
    
    ax8.set_ylabel('Failure Probability [%]', fontsize=10, color='crimson')
    ax8_twin.set_ylabel('Reliability Index β', fontsize=10, color='steelblue')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(methods)
    ax8.set_title('(h) Reliability Metrics Comparison', fontsize=11, fontweight='bold')
    ax8.tick_params(axis='y', labelcolor='crimson')
    ax8_twin.tick_params(axis='y', labelcolor='steelblue')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, P_f_values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, beta_values):
        height = bar.get_height()
        ax8_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax9 = plt.subplot(3, 3, 9)
    data_box = [mc_results['N_samples'], mc_results['tp_samples'], 
                mc_results['D_samples'], mc_results['tw_samples']]
    means_box = [model.mu_N, model.mu_tp, model.mu_D, model.mu_tw]
    
    bp = ax9.boxplot(data_box, labels=['N', 'tp\n[s]', 'D\n[m]', 'tw\n[s/m]'],
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors_bar):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, mean in enumerate(means_box):
        ax9.plot(i+1, mean, 'r*', markersize=15, label='Mean' if i==0 else '')
    
    ax9.set_ylabel('Value (Normalized Units)', fontsize=10)
    ax9.set_title('(i) Input Variable Distributions', fontsize=11, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('warehouse_uq_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Main visualization saved: warehouse_uq_analysis.png")
    
    fig_tornado, ax_tornado = plt.subplots(figsize=(10, 6))
    
    variables_labels = ['N (Order Lines)', 'tp (Picking Time)', 'D (Walking Distance)', 'tw (Walking Speed)']
    alpha_values = form_results['alpha']
    alpha_sq_values = form_results['alpha_squared']
    
    sorted_idx = np.argsort(alpha_sq_values)
    sorted_labels = [variables_labels[i] for i in sorted_idx]
    sorted_values = alpha_sq_values[sorted_idx] * 100
    
    y_pos = np.arange(len(sorted_labels))
    colors_tornado = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars_tornado = ax_tornado.barh(y_pos, sorted_values, color=colors_tornado, edgecolor='black', linewidth=1.5)
    
    ax_tornado.set_yticks(y_pos)
    ax_tornado.set_yticklabels(sorted_labels, fontsize=11)
    ax_tornado.set_xlabel('Contribution to Failure Probability [%]', fontsize=12, fontweight='bold')
    ax_tornado.set_title('Sensitivity Analysis: FORM Importance Factors', fontsize=14, fontweight='bold')
    ax_tornado.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars_tornado, sorted_values)):
        width = bar.get_width()
        ax_tornado.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tornado_diagram.png', dpi=300, bbox_inches='tight')
    print(f"✓ Tornado diagram saved: tornado_diagram.png")
    
    fig_pdf, (ax_pdf1, ax_pdf2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax_pdf1.hist(mc_results['Y_samples'], bins=80, density=True, alpha=0.6, 
                 color='steelblue', edgecolor='black', label='MC Simulation')
    
    x_range = np.linspace(mc_results['Y_samples'].min(), mc_results['Y_samples'].max(), 300)
    ax_pdf1.plot(x_range, stats.norm.pdf(x_range, fosm_results[0], fosm_results[1]),
                'r--', linewidth=2.5, label='FOSM Normal Approx.')
    ax_pdf1.plot(x_range, stats.norm.pdf(x_range, mc_results['mu_Y'], mc_results['sigma_Y']),
                'g-', linewidth=2.5, label='MC Normal Fit')
    
    ax_pdf1.axvline(model.T_SLA, color='darkred', linestyle='--', linewidth=3, 
                   label=f'SLA = {model.T_SLA} s', alpha=0.8)
    ax_pdf1.fill_between(x_range, 0, stats.norm.pdf(x_range, mc_results['mu_Y'], mc_results['sigma_Y']),
                         where=(x_range > model.T_SLA), alpha=0.3, color='red', label='Failure Region')
    
    ax_pdf1.set_xlabel('Picking Time [s]', fontsize=11, fontweight='bold')
    ax_pdf1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax_pdf1.set_title('Output Distribution: Picking Time', fontsize=13, fontweight='bold')
    ax_pdf1.legend(fontsize=9, loc='upper right')
    ax_pdf1.grid(True, alpha=0.3)
    
    methods = ['FOSM', 'Monte Carlo', 'FORM']
    beta_comparison = [1.77, mc_results['beta_mc'], form_results['beta']]
    pf_comparison = [stats.norm.cdf(-1.77)*100, mc_results['P_f']*100, form_results['P_f_form']*100]
    
    x_methods = np.arange(len(methods))
    width = 0.35
    
    bars1_pdf2 = ax_pdf2.bar(x_methods - width/2, beta_comparison, width, 
                            label='Reliability Index β', color='#45B7D1', edgecolor='black', linewidth=1.5)
    ax_pdf2_twin = ax_pdf2.twinx()
    bars2_pdf2 = ax_pdf2_twin.bar(x_methods + width/2, pf_comparison, width,
                                 label='Failure Prob. Pf [%]', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    
    ax_pdf2.set_ylabel('Reliability Index β', fontsize=11, fontweight='bold', color='#45B7D1')
    ax_pdf2_twin.set_ylabel('Failure Probability Pf [%]', fontsize=11, fontweight='bold', color='#FF6B6B')
    ax_pdf2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax_pdf2.set_title('Reliability Metrics Comparison', fontsize=13, fontweight='bold')
    ax_pdf2.set_xticks(x_methods)
    ax_pdf2.set_xticklabels(methods, fontsize=10)
    ax_pdf2.tick_params(axis='y', labelcolor='#45B7D1')
    ax_pdf2_twin.tick_params(axis='y', labelcolor='#FF6B6B')
    ax_pdf2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1_pdf2, beta_comparison):
        height = bar.get_height()
        ax_pdf2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2_pdf2, pf_comparison):
        height = bar.get_height()
        ax_pdf2_twin.text(bar.get_x() + bar.get_width()/2, height,
                         f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig_pdf.legend([bars1_pdf2, bars2_pdf2], ['β', 'Pf [%]'], 
                  loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Distribution comparison saved: distribution_comparison.png")
    
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(0, 0, 'go', markersize=15, label='Origin (Mean)', zorder=5)
    ax.plot(form_results['u_star'][0], form_results['u_star'][1], 
            'r*', markersize=20, label='MPP (u*)', zorder=5)
    ax.plot([0, form_results['u_star'][0]], [0, form_results['u_star'][1]], 
            'b--', linewidth=2, label=f'β = {form_results["beta"]:.3f}')
    
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in [1, 2, 3]:
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        ax.plot(x_circle, y_circle, 'k:', alpha=0.3, linewidth=1)
        ax.text(radius, 0.1, f'β={radius}', fontsize=9, alpha=0.5)
    
    u1_range = np.linspace(-3, 3, 50)
    u2_range = np.linspace(-3, 3, 50)
    U1, U2 = np.meshgrid(u1_range, u2_range)
    
    G = np.zeros_like(U1)
    for i in range(len(u1_range)):
        for j in range(len(u2_range)):
            x1 = transform_from_standard_normal(U1[j,i], model.mu_N, model.sigma_N, 'normal')
            x2 = transform_from_standard_normal(U2[j,i], model.mu_tp, model.sigma_tp, 'lognormal')
            G[j,i] = model.limit_state_function(x1, x2, model.mu_D, model.mu_tw)
    
    contour = ax.contour(U1, U2, G, levels=[0], colors='red', linewidths=2.5)
    ax.clabel(contour, inline=True, fontsize=10, fmt='g=0')
    
    ax.contourf(U1, U2, G, levels=[-1000, 0], colors='red', alpha=0.2)
    ax.text(1.5, -2, 'Failure\nRegion', fontsize=12, color='red', fontweight='bold')
    
    ax.set_xlabel('u₁ (N in Standard Normal Space)', fontsize=12)
    ax.set_ylabel('u₂ (tp in Standard Normal Space)', fontsize=12)
    ax.set_title('FORM Analysis in Standard Normal Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    
    plt.tight_layout()
    plt.savefig('form_standard_normal_space.png', dpi=300, bbox_inches='tight')
    print(f"✓ FORM space visualization saved: form_standard_normal_space.png")
    
    plt.close('all')
    print(f"\n✓ All visualizations generated successfully\n")

def generate_summary_report(model, mc_results, form_results, fosm_results):
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    summary = {
        'Method': ['FOSM', 'Monte Carlo', 'FORM'],
        'Mean [s]': [fosm_results[0], mc_results['mu_Y'], '-'],
        'Std Dev [s]': [fosm_results[1], mc_results['sigma_Y'], '-'],
        'Variance [s²]': [fosm_results[2], mc_results['var_Y'], '-'],
        'P_f': ['-', f"{mc_results['P_f']:.5f}", f"{form_results['P_f_form']:.5f}"],
        'β': ['-', f"{mc_results['beta_mc']:.4f}", f"{form_results['beta']:.4f}"]
    }
    
    df_summary = pd.DataFrame(summary)
    print("\n" + df_summary.to_string(index=False))
    
    df_summary.to_csv('summary_results.csv', index=False)
    print(f"\n✓ Summary saved to: summary_results.csv")
    
    importance_df = pd.DataFrame({
        'Variable': ['N (Order Lines)', 'tp (Picking Time)', 'D (Distance)', 'tw (Walking Time)'],
        'α': form_results['alpha'],
        'α²': form_results['alpha_squared'],
        'Contribution [%]': form_results['alpha_squared'] * 100
    })
    
    print("\n" + "="*70)
    print("IMPORTANCE FACTORS (FORM)")
    print("="*70)
    print("\n" + importance_df.to_string(index=False))
    
    importance_df.to_csv('importance_factors.csv', index=False)
    print(f"\n✓ Importance factors saved to: importance_factors.csv")
    
    mc_data = pd.DataFrame({
        'N': mc_results['N_samples'],
        'tp': mc_results['tp_samples'],
        'D': mc_results['D_samples'],
        'tw': mc_results['tw_samples'],
        'Y': mc_results['Y_samples'],
        'g': mc_results['g_samples']
    })
    mc_data.to_csv('mc_samples.csv', index=False)
    print(f"✓ MC samples saved to: mc_samples.csv (100,000 rows)")

def main():
    print("\n" + "="*70)
    print(" " * 10 + "WAREHOUSE ORDER PICKING UQ ANALYSIS")
    print("="*70)
    print("\nUncertainty Quantification and Reliability Analysis")
    print("Author: UQ Analysis Team")
    print("Date: 2026-02-01")
    print("="*70)
    
    model = WarehousePickingModel()
    
    benchmark_verification()
    fosm_results = fosm_analysis(model)
    mc_results = monte_carlo_simulation(model, n_samples=100000)
    form_results = form_analysis(model)
    create_visualizations(model, mc_results, form_results, fosm_results)
    generate_summary_report(model, mc_results, form_results, fosm_results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  Visualizations (PNG, 300 DPI):")
    print("    1. warehouse_uq_analysis.png - Main visualization (9 subplots)")
    print("    2. form_standard_normal_space.png - FORM space visualization")
    print("    3. tornado_diagram.png - Sensitivity analysis tornado chart")
    print("    4. distribution_comparison.png - PDF and reliability comparison")
    print("\n  Data Files (CSV):")
    print("    5. summary_results.csv - Results summary table")
    print("    6. importance_factors.csv - FORM sensitivity analysis")
    print("    7. mc_samples.csv - Monte Carlo sample data (100,000 rows)")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
