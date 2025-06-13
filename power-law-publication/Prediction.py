import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar

plt.rc('figure', dpi=133)
plt.rc('lines', linewidth=1)
plt.rc('font', size=10, family='sans-serif')
plt.rc('axes', titlesize=9, labelsize=8)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=10)

# Electrolyte model parameters
sM_neg, sM_pos = -40, 60
N = 201
tol, max_iter = 1e-8, 10000

# Range of surface charge densities
s_pos = np.linspace(0, 100, N)
s_neg = -s_pos

# Quantum parameters
a = 2 # Quantum capacitance minima at uq = 0 (PZC)
b = 21 # Slope of quantum capacitance dependence on uq
uq_neg = -np.sqrt((np.sqrt((2*s_neg)**2 + a**4) - a**2) / b**2)
uq_pos =  np.sqrt((np.sqrt((2*s_pos)**2 + a**4) - a**2) / b**2)
Cq_neg = ((a**2+b**2)*uq_neg**2 + a**2)/(2*np.sqrt(b**2*uq_neg**2 + a**2))
Cq_pos = ((a**2+b**2)*uq_pos**2 + a**2)/(2*np.sqrt(b**2*uq_pos**2 + a**2))
Cq_neg_inv = 1 / ((a**2+b**2)*uq_neg**2 + a**2)/(2*np.sqrt(b**2*uq_neg**2 + a**2))
Cq_pos_inv = 1 / ((a**2+b**2)*uq_pos**2 + a**2)/(2*np.sqrt(b**2*uq_pos**2 + a**2))
# Parameter lists
uM_neg_list = np.linspace(-10, -4, 10)
uM_pos_list = np.linspace(6, 15, 10)

aM_pos_list = np.linspace(0.2, 1, 20)
aM_neg_list = np.linspace(0.2, 1, 20)
k_neg_list  = np.linspace(3, 12, 5)
k_pos_list  = np.linspace(3, 12, 5)
param_combos = list(product(uM_neg_list, uM_pos_list,
                            aM_neg_list, aM_pos_list,
                             k_neg_list,  k_pos_list))

def solve_a(s, sM, aM, k):
    s = np.float64(s)
    sM = np.float64(sM)
    aM = np.float64(aM)
    k = np.float64(k)
    def f(a):
        if a <= 1e-4:
            return np.inf  # avoid division by tiny a
        try:
            exponent = (s / sM)**(1 / a)
            if np.isinf(exponent) or np.isnan(exponent):
                return np.inf
        except FloatingPointError:
            return np.inf
        return a - (aM + (1 - aM) * np.exp(-k * exponent))
    result = root_scalar(f, method='brentq', bracket=[aM, 1], xtol=1e-10)
    if result.converged:
        return result.root
    else:
        raise RuntimeError("Root finding did not converge")

def compute_branch(params):
    uM_neg, uM_pos, aM_neg,  aM_pos, k_neg, k_pos = params

    s_pos = np.linspace(0, 100, N)
    s_neg = -s_pos

    a_neg = np.array([solve_a(si, sM_neg, aM_neg, k_neg) for si in s_neg])
    a_pos = np.array([solve_a(si, sM_pos, aM_pos, k_pos) for si in s_pos])

    u_neg = uM_neg * (s_neg / sM_neg)**(1 / a_neg)
    u_pos = uM_pos * (s_pos / sM_pos)**(1 / a_pos)

    Ce_neg_inv = np.gradient(u_neg, s_neg) # inverse capacitance of electrolyte
    Ce_pos_inv = np.gradient(u_pos, s_pos) # inverse capacitance of electrolyte

    voltage = u_pos - u_neg + uq_pos - uq_neg

    C_total = 1/(Ce_neg_inv + Ce_pos_inv + Cq_neg_inv + Cq_pos_inv)
    s_tot = s_pos - s_neg
    ds = np.diff(s_tot)
    v_mid = 0.5 * (voltage[:-1] + voltage[1:])
    E_density = np.zeros_like(voltage)
    E_density[1:] = np.cumsum(v_mid * ds)

    idx_4V = np.argmin(np.abs(voltage - 4.0))
    E_at_4V = E_density[idx_4V]
    return voltage, C_total, E_density, s_tot, E_at_4V, params
if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_branch, param_combos)

    all_voltage, all_C_total, all_E_total, all_s_tot, E_at_4V_list, all_params = zip(*results)
    best_idx = np.argmax(E_at_4V_list)
    best_params = all_params[best_idx]
    # 3-panel figure
    bw, gw = 1.6, 0.55
    fw, fh = 3*bw + 2*gw + 0.8, 3.0
    fig = plt.figure(figsize=(fw, fh), dpi=200)
    axs = []
    for i in range(3):
        left = (0.574 + i*(bw+gw))/fw if i == 1 else (0.6 + i*(bw+gw))/fw
        bottom = ((fh-bw)/2)/fh
        width = bw/fw
        height = bw/fh
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_aspect('auto')
        ax.tick_params(labelsize=8)
        axs.append(ax)
        ax.set_xlim(0,6); ax.grid(False)

    for volts, C, E, s in zip(all_voltage, all_C_total, all_E_total, all_s_tot):
        axs[0].plot(volts, s, color='blue', alpha=0.0025)
        axs[1].plot(volts, C, color='blue', alpha=0.0025)
        axs[2].plot(volts, E, color='blue', alpha=0.0025)

    axs[0].plot(all_voltage[best_idx], all_s_tot[best_idx], color='red', lw=1.2, zorder=10)
    axs[1].plot(all_voltage[best_idx], all_C_total[best_idx], color='red', lw=1.2, zorder=10)
    axs[2].plot(all_voltage[best_idx], all_E_total[best_idx], color='red', lw=1.2, zorder=10)
    axs[0].set_xlabel('Applied voltage / V'); axs[0].set_ylim(0,125); axs[0].set_ylabel('Surf. charge density / μC cm⁻²', labelpad=0)
    axs[1].set_xlabel('Applied voltage / V'); axs[1].set_ylim(0, 25); axs[1].set_ylabel('Diff. capacitance / μF cm⁻²')
    axs[2].set_xlabel('Applied voltage / V'); axs[2].set_ylim(0,300); axs[2].set_ylabel('Energy density / μJ cm⁻²')
    axs[0].set_yticks([0, 25, 50, 75, 100, 125])
    axs[1].set_yticks([0, 5, 10, 15, 20, 25])
    axs[2].set_yticks([0, 50, 100, 150, 200, 250, 300])

    plt.savefig('predictionfinal.png', dpi=600)
    plt.show()
    # Ce_neg_inv = np.gradient(u_neg, s_neg) # inverse capacitance of electrolyte
    # Ce_pov_inv = np.gradient(u_pos, s_pos) # inverse capacitance of electrolyte
    with open("bestcomb2.txt", "w") as f:
        f.write(f"uM_pos={best_params[1]}, aM_pos={best_params[3]}, k_pos={best_params[5]}, "
                f"uM_neg={best_params[0]}, aM_neg={best_params[2]}, k_neg={best_params[4]}\n")


    #standalone energy plot
    fig2, ax2 = plt.subplots(figsize=(bw,bw), dpi=200)
    ax2.set_xlim(0,6); ax2.set_ylim(0,300)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    for volts, E in zip(all_voltage, all_E_total):
       ax2.plot(volts, E, color='blue', alpha=0.0025)
    ax2.plot(all_voltage[best_idx], all_E_total[best_idx], color='red', lw=1.2, zorder=10)
    fig2.tight_layout(pad=0)
    fig2.savefig('energydensity.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()

