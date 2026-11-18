import numpy as np
import matplotlib.pyplot as plt


def computeQ(t, R, alpha_h, alpha_l, eps, alpha, x=1.0):
    return 6 * t * (R * alpha_l + alpha_h) + 9 * np.log((36 * alpha) / (eps * (2 - x)))


def computeTotalGateCost(n, t, eps, R, alpha_h, alpha_l, alpha, x=1.0):
    Q = computeQ(t, R, alpha_h, alpha_l, eps, alpha, x)
    return n * Q * np.log(Q)


def computeGateCost(n, t, eps, R, alpha_h, alpha_l):
    return n * np.log(n * t / eps * (R * alpha_l + alpha_h))


plt.rcParams.update({""
    "text.usetex": True,
    "font.size": 18,
    "figure.figsize": (5, 4), # SMALLER figure = LARGER relative text
    "axes.labelsize": 18,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

mu = 0.1
a_am = 0.1
a_mid = 0.2
c_max = 0.4
L = 2 * np.pi
c_L = 0.0
c_H = c_max
alpha = np.exp(1)
eps = 1e-3
R = 2 * (1 + np.log((3 + 3 / (2 * np.pi)) / eps))
t = 1
n_qubits = np.arange(4,11, 1)

Tau_c = L / c_H
Tau_r = 1 / (a_mid - a_am) if (a_mid - a_am) != 0 else np.inf
Tau_D = L**2 / mu
print(f"Tau_c = {Tau_c:.2f}, Tau_r = {Tau_r:.2f}, Tau_D = {Tau_D:.2f}")

for d in [1, 2, 3]:
    costs = []
    for n in n_qubits:
        N = 2**n
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)

        c = -4 * (c_H - c_L) / L**2 * x**2 + 4 * (c_H - c_L) / L * x
        cx = -8 * (c_H - c_L) / L**2 * x + 4 * (c_H - c_L) / L
        a = a_am * np.sin(x) + a_mid

        gamma_c = np.max(np.abs(c))
        gamma_r = np.max(a + 0.5 * cx) - np.min(a + 0.5 * cx)

        alpha_h = 8 * np.pi * d * gamma_c / dx
        alpha_l = 4 * mu * d * np.pi**2 / (dx**2) + gamma_r

        costs.append(computeTotalGateCost(n, t, eps, R, alpha_h, alpha_l, alpha))

    plt.plot(n_qubits, costs, ".:", label=rf'$d = {d}$', markersize=8)

plt.xlabel(r'number of qubits ($n$)')
plt.ylabel('total gate cost')
plt.legend(loc='best', fontsize=14)

plt.tight_layout()
stringname = "./figures/totalcost_d.pdf"
plt.savefig(stringname, bbox_inches='tight')
plt.show()
