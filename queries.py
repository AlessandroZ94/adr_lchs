import numpy as np
import matplotlib.pyplot as plt

def computeQ(t, R, alpha_h, alpha_l, eps, alpha, x=1.0):
    return 6 * t * (R * alpha_l + alpha_h) + 9 * np.log((36 * alpha) / (eps * (2 - x)))

def computeTotalGateCost(n, t, eps, R, alpha_h, alpha_l, alpha, x=1.0):
    Q = computeQ(t, R, alpha_h, alpha_l, eps, alpha, x)
    
    return n * Q * np.log(Q)

plt.rcParams.update({""
    "text.usetex": True,
    "font.size": 18,
    "figure.figsize": (5, 4), # SMALLER figure = LARGER relative text
    "axes.labelsize": 18,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# 1. Define global algorithm & physical constants OUTSIDE the loops
d = 1
mu = 0.1
a_am = 0.1
a_mid = 0.2
c_max = 0.4
L = 2 * np.pi               

c_L = 0.0
c_H = c_max

alpha = np.exp(1)

eps_vector = [1e-3, 1e-4, 1e-5]
times = np.linspace(1, 100, 100)
n_qubits = 6
costs_times = []

for eps in eps_vector:
    queries = []
    total_costs = []
    for t in times:
        R = 2 * (1 + np.log((3 + 3 / (2 * np.pi)) / eps))

        # 2. Only grid-dependent variables go inside the loop
        N = 2**n_qubits
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)

        # Continuous functions sampled at current resolution
        c = -4 * (c_H - c_L) / L**2 * x**2 + 4 * (c_H - c_L) / L * x
        cx = -8 * (c_H - c_L) / L**2 * x + 4 * (c_H - c_L) / L
        a = (a_am * np.sin(x) + a_mid)

        # Maxima re-evaluated based on discrete sampling
        gamma_c = np.max(np.abs(c))
        gamma_r = np.max(a + 0.5 * cx) - np.min(a + 0.5 * cx)

        # Normalization constants (Eq 42, 64, 67 bounds)
        alpha_h = 8 * np.pi * d * gamma_c / dx
        alpha_l = 4 * mu * d * np.pi**2 / (dx**2) + gamma_r

        q = computeQ(t, R, alpha_h, alpha_l, eps, alpha)
        queries.append(q)
        total_costs.append(computeTotalGateCost(n_qubits, t, eps, R, alpha_h, alpha_l, alpha))

    plt.plot(times, queries, label=rf'$\epsilon = {eps:.0e}$')
    

plt.xlabel(r'time ($t$)')
plt.ylabel('oracle queries')

plt.legend(loc='best', fontsize=14)
plt.tight_layout()
stringname="./figures/queries.pdf"
plt.savefig(stringname, bbox_inches='tight')
plt.show()