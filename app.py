
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# Numericke pomocne funkce
# ============================================================

def normalize(psi, dx):
    """Normalizace 1D vlnove funkce / obalky."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        return psi
    return psi / norm


def normalize_eigenvectors(vecs, dx):
    """Normalizuje sloupce matice vlastnich vektoru."""
    out = vecs.astype(complex).copy()
    for i in range(out.shape[1]):
        out[:, i] = normalize(out[:, i], dx)
    return out


def hamiltonian_dirichlet(x, V):
    """
    Diskretni 1D Hamiltonian
        H = -(1/2) d^2/dx^2 + V(x)
    s Dirichletovymi okrajovymi podminkami.
    """
    dx = x[1] - x[0]
    N = len(x)
    main = np.full(N, 1.0 / dx**2) + V
    off = np.full(N - 1, -0.5 / dx**2)
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    return H, dx


def project_state(psi0, basis, dx, n_use=None):
    """Projekce stavu psi0 do baze vlastnich funkci."""
    if n_use is None:
        n_use = basis.shape[1]
    return basis[:, :n_use].conj().T @ (psi0 * dx)


def evolve_from_basis(coeffs, eigvals, basis, times):
    """
    Casovy / propagacni vyvoj
        psi(x,t) = sum_n c_n phi_n(x) exp(-i E_n t)
    nebo opticky
        A(x,z)   = sum_n c_n phi_n(x) exp(-i beta_n z)
    """
    phases = np.exp(-1j * np.outer(eigvals, times))
    return basis @ (coeffs[:, None] * phases)


def expectation_x(psi_xt, x, dx):
    dens = np.abs(psi_xt) ** 2
    return np.sum(dens * x[:, None], axis=0) * dx


def peak_position(psi_xt, x):
    dens = np.abs(psi_xt) ** 2
    return x[np.argmax(dens, axis=0)]


def classical_box_trajectory(times, x0, v, xmin, xmax):
    """
    Klasicka trajektorie mezi dvema stenami s dokonale pruznym odrazem.
    """
    L = xmax - xmin
    if L <= 0:
        return np.full_like(times, x0)
    y = (x0 - xmin + v * times) % (2 * L)
    return xmin + np.where(y <= L, y, 2 * L - y)


def infinite_well_analytic(x, L, n_values):
    """
    Analyticke vlastni funkce nekonecne hluboke 1D jamy
    na intervalu (0, L).
    """
    phi = np.sqrt(2.0 / L) * np.sin(np.pi * np.outer(x / L, n_values))
    E = (n_values**2) * np.pi**2 / (2 * L**2)
    return phi, E


# ============================================================
# Fyzikalni modely
# ============================================================

@st.cache_data(show_spinner=False)
def compute_single_well(L=1.0, N=180, n_show=4):
    x = np.linspace(L / (N + 1), L - L / (N + 1), N)
    V = np.zeros_like(x)
    H, dx = hamiltonian_dirichlet(x, V)
    E_num, psi_num = np.linalg.eigh(H)
    psi_num = normalize_eigenvectors(psi_num, dx)

    n_vals = np.arange(1, n_show + 1)
    phi_an, E_an = infinite_well_analytic(x, L, n_vals)
    return x, dx, V, E_num, psi_num, E_an, phi_an


@st.cache_data(show_spinner=False)
def compute_single_time_evolution(L=1.0, N=180, x0=0.22, sigma=0.06, k0=24.0,
                                  n_basis=100, t_max=0.16, n_times=220):
    x = np.linspace(L / (N + 1), L - L / (N + 1), N)
    V = np.zeros_like(x)
    H, dx = hamiltonian_dirichlet(x, V)
    E, psi = np.linalg.eigh(H)
    psi = normalize_eigenvectors(psi, dx)

    psi0 = np.exp(-(x - x0) ** 2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi0 = normalize(psi0, dx)

    n_basis = min(n_basis, len(E))
    coeffs = project_state(psi0, psi, dx, n_use=n_basis)
    basis = psi[:, :n_basis]
    energies = E[:n_basis]
    times = np.linspace(0.0, t_max, n_times)
    psi_xt = evolve_from_basis(coeffs, energies, basis, times)
    dens = np.abs(psi_xt) ** 2

    x_mean = expectation_x(psi_xt, x, dx)
    x_peak = peak_position(psi_xt, x)
    x_class = classical_box_trajectory(times, x0=x0, v=k0, xmin=0.0, xmax=L)
    return x, dx, times, psi0, psi_xt, dens, x_mean, x_peak, x_class


@st.cache_data(show_spinner=False)
def compute_double_well(L2=1.8, N2=260, barrier_width=0.12, V0=80.0, n_times=240):
    x2 = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    center = L2 / 2
    V = np.zeros_like(x2)
    mask = np.abs(x2 - center) < barrier_width / 2
    V[mask] = V0

    H, dx2 = hamiltonian_dirichlet(x2, V)
    E, psi = np.linalg.eigh(H)
    psi = normalize_eigenvectors(psi, dx2)

    psi_g = psi[:, 0]
    psi_e = psi[:, 1]
    dE = E[1] - E[0]
    T_tunnel = np.pi / dE

    psi_left = normalize(psi_g + psi_e, dx2)
    coeffs = project_state(psi_left, psi, dx2, n_use=2)
    basis = psi[:, :2]
    times = np.linspace(0.0, 2.1 * T_tunnel, n_times)
    psi_xt = evolve_from_basis(coeffs, E[:2], basis, times)
    dens = np.abs(psi_xt) ** 2

    x_mean = expectation_x(psi_xt, x2, dx2)
    left_mask = x2 < center
    right_mask = x2 > center
    P_left = np.sum(dens[left_mask, :], axis=0) * dx2
    P_right = np.sum(dens[right_mask, :], axis=0) * dx2

    x_class = classical_box_trajectory(
        times, x0=0.38, v=0.35,
        xmin=0.0, xmax=center - barrier_width / 2
    )

    return {
        "x": x2, "dx": dx2, "V": V, "E": E, "psi": psi,
        "psi_g": psi_g, "psi_e": psi_e, "psi_left": psi_left,
        "times": times, "psi_xt": psi_xt, "dens": dens,
        "x_mean": x_mean, "P_left": P_left, "P_right": P_right,
        "dE": dE, "T_tunnel": T_tunnel, "center": center,
        "barrier_width": barrier_width, "V0": V0, "x_class": x_class
    }


@st.cache_data(show_spinner=False)
def compute_optical_analogy(L2=1.8, N2=260, dn_core=0.012, n_clad=1.450):
    """
    Didakticky paraxialni model dvou vlnovodu.
    Vyssi index lomu -> efektivne 'hlubsi jama'.
    """
    x = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    dx = x[1] - x[0]
    center = L2 / 2

    n_profile = np.full_like(x, n_clad)
    left_core = (x > 0.33) & (x < 0.74)
    right_core = (x > 1.06) & (x < 1.47)
    n_profile[left_core] = n_clad + dn_core
    n_profile[right_core] = n_clad + dn_core

    V_opt = -35.0 * (n_profile - n_clad) / dn_core
    H_opt, _ = hamiltonian_dirichlet(x, V_opt)
    beta, phi = np.linalg.eigh(H_opt)
    phi = normalize_eigenvectors(phi, dx)

    phi_s = phi[:, 0]
    phi_a = phi[:, 1]
    d_beta = beta[1] - beta[0]
    L_couple = np.pi / d_beta

    A_left = normalize(phi_s + phi_a, dx)
    coeffs = project_state(A_left, phi, dx, n_use=2)
    basis = phi[:, :2]
    z_vals = np.linspace(0.0, 2.1 * L_couple, 240)
    A_xz = evolve_from_basis(coeffs, beta[:2], basis, z_vals)
    I_opt = np.abs(A_xz) ** 2

    x_mean = expectation_x(A_xz, x, dx)
    left_half = x < center
    right_half = x > center
    P_left = np.sum(I_opt[left_half, :], axis=0) * dx
    P_right = np.sum(I_opt[right_half, :], axis=0) * dx

    return {
        "x": x, "dx": dx, "n_profile": n_profile, "V_opt": V_opt,
        "beta": beta, "phi": phi, "I_opt": I_opt, "A_xz": A_xz,
        "z_vals": z_vals, "x_mean": x_mean, "center": center,
        "P_left": P_left, "P_right": P_right, "L_couple": L_couple,
        "d_beta": d_beta, "n_clad": n_clad
    }


@st.cache_data(show_spinner=False)
def compute_barrier_scan(L2=1.8, N2=220, V0=80.0, widths=None):
    if widths is None:
        widths = np.linspace(0.04, 0.30, 12)

    x = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    center = L2 / 2

    E0, E1, dE, Tt = [], [], [], []

    for bw in widths:
        V = np.zeros_like(x)
        V[np.abs(x - center) < bw / 2] = V0
        H, _ = hamiltonian_dirichlet(x, V)
        E, _ = np.linalg.eigh(H)
        E0.append(E[0])
        E1.append(E[1])
        dE.append(E[1] - E[0])
        Tt.append(np.pi / (E[1] - E[0]))

    return widths, np.array(E0), np.array(E1), np.array(dE), np.array(Tt)


@st.cache_data(show_spinner=False)
def compute_finite_well(L_f=1.8, N_f=360, well_width=0.60, V_barrier=120.0):
    x = np.linspace(L_f / (N_f + 1), L_f - L_f / (N_f + 1), N_f)
    center = L_f / 2
    V = np.full_like(x, V_barrier)
    inside = np.abs(x - center) < well_width / 2
    V[inside] = 0.0

    H, dx = hamiltonian_dirichlet(x, V)
    E, psi = np.linalg.eigh(H)
    psi = normalize_eigenvectors(psi, dx)
    bound_idx = np.where(E < V_barrier)[0]

    n_plot = min(3, len(bound_idx))
    n_inf = np.arange(1, n_plot + 1)
    E_inf = (n_inf**2) * np.pi**2 / (2 * well_width**2)
    return x, dx, V, E, psi, bound_idx, E_inf


# ============================================================
# Graficke funkce
# ============================================================

def plot_single_stationary(x, V, E_num, psi_num, E_an, n_show=4):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    ax.plot(x, V, lw=2, label="V(x)")
    for n in range(n_show):
        y = 0.18 * np.real(psi_num[:, n]) + E_num[n]
        ax.plot(x, y, lw=2, label=f"n={n+1}")
        ax.axhline(E_num[n], lw=0.8, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("energie / tvar funkce")
    ax.set_title("Nekonečně hluboká jáma: první stavy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    idx = np.arange(1, n_show + 1)
    ax.plot(idx, E_num[:n_show], "o-", lw=2, label="numericky")
    ax.plot(idx, E_an[:n_show], "s--", lw=2, label="analyticky")
    ax.set_xlabel("číslo stavu n")
    ax.set_ylabel("energie")
    ax.set_title("Srovnání energií")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_single_probabilities(x, psi_num, n=1, L=1.0):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    idx = n - 1
    P_quant = np.abs(psi_num[:, idx])**2
    P_class = np.ones_like(x) / L
    ax.plot(x, P_quant, lw=2, label=rf"kvantově $|\psi_{n}|^2$")
    ax.plot(x, P_class, "--", lw=2, label="klasicky: rovnoměrně")
    ax.set_xlabel("x")
    ax.set_ylabel("pravděpodobnostní hustota")
    ax.set_title(f"Kvantová vs. klasická hustota pro stav n={n}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_single_time_snapshot(x, dens, x_mean, x_class, times, idx):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, dens[:, idx], lw=2, label=r"kvantově $|\psi(x,t)|^2$")
    ax.axvline(x_mean[idx], ls="--", lw=1.8, label=r"$\langle x \rangle$")
    ax.plot([x_class[idx]], [0.05 * np.max(dens)], "o", ms=8, label="klasická částice")
    ax.set_xlabel("x")
    ax.set_ylabel("hustota")
    ax.set_title(f"Jedna jáma: t = {times[idx]:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_single_time_history(times, x_mean, x_class):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(times, x_mean, lw=2, label=r"kvantově $\langle x \rangle$")
    ax.plot(times, x_class, "--", lw=2, label="klasická trajektorie")
    ax.set_xlabel("čas t")
    ax.set_ylabel("poloha")
    ax.set_title("Vývoj střední polohy v jedné jámě")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_double_stationary(data):
    x = data["x"]
    V = data["V"]
    E = data["E"]
    psi_g = data["psi_g"]
    psi_e = data["psi_e"]

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    ax.plot(x, V / np.max(V) * max(E[:2]) * 1.3, lw=2, label="bariéra (škálováno)")
    ax.plot(x, 0.18 * np.real(psi_g) + E[0], lw=2, label="základní stav")
    ax.plot(x, 0.18 * np.real(psi_e) + E[1], lw=2, label="1. excitovaný stav")
    ax.axhline(E[0], lw=0.8, alpha=0.35)
    ax.axhline(E[1], lw=0.8, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("energie / tvar funkce")
    ax.set_title("Dvojitá jáma: symetrický a antisymetrický stav")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_double_snapshot(data, idx):
    x = data["x"]
    dens = data["dens"]
    V = data["V"]
    x_mean = data["x_mean"]
    x_class = data["x_class"]
    times = data["times"]
    T_tunnel = data["T_tunnel"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    scale = max(np.max(dens[:, idx]), 1e-9)
    ax.plot(x, dens[:, idx], lw=2, label=r"kvantově $|\psi(x,t)|^2$")
    ax.plot(x, (V / np.max(V)) * scale * 0.8 if np.max(V) > 0 else V, "k--", lw=1.5, label="bariéra")
    ax.axvline(x_mean[idx], ls="--", lw=1.6, label=r"$\langle x \rangle$")
    ax.plot([x_class[idx]], [0.08 * scale], "o", ms=8, label="klasicky")
    ax.set_xlabel("x")
    ax.set_ylabel("hustota")
    ax.set_title(f"Snapshot pro t/T = {times[idx]/T_tunnel:.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(times / T_tunnel, data["P_left"], lw=2, label="pravděpodobnost vlevo")
    ax.plot(times / T_tunnel, data["P_right"], lw=2, label="pravděpodobnost vpravo")
    ax.axvline(times[idx] / T_tunnel, color="k", ls="--", lw=1.5)
    ax.set_xlabel(r"normalizovaný čas $t/T_{\rm tunnel}$")
    ax.set_ylabel("pravděpodobnost")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tunelování mezi jamami")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_optical_snapshot(opt, idx):
    x = opt["x"]
    I = opt["I_opt"]
    z_vals = opt["z_vals"]
    x_mean = opt["x_mean"]
    n_profile = opt["n_profile"]
    n_clad = opt["n_clad"]
    Lc = opt["L_couple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(x, I[:, idx], lw=2, label=r"optická intenzita $I(x,z)$")
    ax.axvline(x_mean[idx], ls="--", lw=1.6, label=r"střed intenzity")
    ax.plot(x, 150 * (n_profile - n_clad), "k--", lw=1.5, label="profil indexu (škál.)")
    ax.set_xlabel("x")
    ax.set_ylabel("intenzita")
    ax.set_title(f"Optický řez pro z/Lc = {z_vals[idx]/Lc:.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax.text(
        0.03, 0.92, r"Směr $z$ je kolmý k řezu $I(x)$",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.75"),
        fontsize=9
    )

    ax = axes[1]
    extent = [z_vals[0] / Lc, z_vals[-1] / Lc, x[0], x[-1]]
    im = ax.imshow(I, origin="lower", aspect="auto", extent=extent)
    ax.axvline(z_vals[idx] / Lc, color="w", ls="--", lw=1.5)
    ax.set_xlabel(r"$z/L_c$")
    ax.set_ylabel("x")
    ax.set_title(r"2D mapa $I(x,z)$")
    fig.colorbar(im, ax=ax, label="intenzita")

    fig.tight_layout()
    return fig


def plot_barrier_scan(widths, E0, E1, dE, Tt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    ax.plot(widths, E0, "o-", lw=2, label=r"$E_0$")
    ax.plot(widths, E1, "s-", lw=2, label=r"$E_1$")
    ax.set_xlabel("šířka bariéry")
    ax.set_ylabel("energie")
    ax.set_title("Štěpení prvních dvou hladin")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(widths, dE, "o-", lw=2, label=r"$\Delta E$")
    ax2 = ax.twinx()
    ax2.plot(widths, Tt, "s--", lw=2, label=r"$T_{\rm tunnel}$")
    ax.set_xlabel("šířka bariéry")
    ax.set_ylabel(r"$\Delta E$")
    ax2.set_ylabel(r"$T_{\rm tunnel}$")
    ax.set_title("Širší bariéra -> menší vazba a pomalejší tunelování")
    ax.grid(True, alpha=0.25)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig


def plot_finite_well(x, V, E, psi, bound_idx, E_inf):
    n_plot = min(3, len(bound_idx))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(x, V, lw=2, label="V(x)")
    for j in range(n_plot):
        idx = bound_idx[j]
        y = 0.22 * np.real(psi[:, idx]) + E[idx]
        ax.plot(x, y, lw=2, label=f"vázaný stav {j+1}")
        ax.axhline(E[idx], lw=0.8, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("energie / tvar funkce")
    ax.set_title("Konečně hluboká jáma")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    if n_plot > 0:
        ax.plot(np.arange(1, n_plot + 1), E[bound_idx[:n_plot]], "o-", lw=2, label="konečná jáma")
        ax.plot(np.arange(1, n_plot + 1), E_inf[:n_plot], "s--", lw=2, label="nekonečná jáma stejné šířky")
    ax.set_xlabel("číslo stavu")
    ax.set_ylabel("energie")
    ax.set_title("Srovnání energií")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_3d_surface_quantum_or_optical(x, yvals, Z, y_label="t", z_label="hustota", title="3D vývoj"):
    if not HAS_PLOTLY:
        return None

    X, Y = np.meshgrid(x, yvals)
    surface = go.Surface(x=X, y=Y, z=Z.T, showscale=True)
    fig = go.Figure(surface)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Kvantová jáma: Streamlit demo", layout="wide")
st.title("Kvantová a klasická částice v jámě + optická analogie")
st.caption("Didaktická Streamlit aplikace odvozená z výukového notebooku.")

with st.sidebar:
    st.header("Nastavení aplikace")
    section_info = st.radio(
        "Vyber sekci",
        [
            "Teorie",
            "Jedna jáma",
            "Dvojitá jáma",
            "Optická analogie",
            "Konečně hluboká jáma",
        ],
    )
    st.markdown("---")
    st.markdown(
        """
        **Poznámka:**  
        Do Streamlit verze jsou převedeny hlavně ty části,
        které se dobře ovládají interaktivně pomocí sliderů:
        snapshoty, parametrické závislosti a 2D/3D mapy.
        """
    )

if section_info == "Teorie":
    st.subheader("Krátká teoretická sekce")
    st.markdown(
        r"""
        Používáme tyto základní rovnice:

        Stacionární Schrödingerova rovnice:
        $$
        \hat H \psi_n(x) = E_n \psi_n(x),
        \qquad
        \hat H = -\frac{1}{2}\frac{d^2}{dx^2} + V(x).
        $$

        Časový vývoj kvantového stavu:
        $$
        \psi(x,t) = \sum_n c_n \psi_n(x) e^{-i E_n t}.
        $$

        Klasická částice v jámě se mezi stěnami pohybuje přímočaře
        a na hranách se pružně odráží.

        Pro optickou analogii používáme paraxiální rovnici pro obálku:
        $$
        i \frac{\partial A}{\partial z}
        =
        -\frac{1}{2k}\frac{\partial^2 A}{\partial x^2}
        + V_{\mathrm{opt}}(x) A,
        $$
        kde $z$ hraje roli evoluční proměnné analogické času $t$.

        V dvojité jámě / dvojvlnovodu jsou zvlášť důležité první dva módy.
        Rozdíl jejich vlastních hodnot určuje tunelovací nebo koplingovou délku:
        $$
        T_{\rm tunnel} \approx \frac{\pi}{E_1-E_0},
        \qquad
        L_c \approx \frac{\pi}{\beta_1-\beta_0}.
        $$
        """
    )

    st.subheader("Vybrané reference")
    st.markdown(
        """
        - D. J. Griffiths, D. F. Schroeter: *Introduction to Quantum Mechanics*  
        - E. Hecht: *Optics*  
        - B. E. A. Saleh, M. C. Teich: *Fundamentals of Photonics*  
        - D. N. Christodoulides, F. Lederer, Y. Silberberg, *Nature* **424**, 817 (2003)  
        """
    )

elif section_info == "Jedna jáma":
    st.subheader("1. Jedna nekonečně hluboká jáma")

    colL, colR = st.columns([1, 1])
    with colL:
        L = st.slider("Šířka jámy L", 0.6, 2.0, 1.0, 0.05)
        N = st.slider("Počet bodů mřížky", 120, 280, 180, 20)
        n_show = st.slider("Kolik stavů ukázat", 2, 6, 4, 1)
    with colR:
        state_n = st.slider("Vybraný stacionární stav n", 1, max(2, n_show), 1, 1)
        x0 = st.slider("Počáteční poloha balíku x0", 0.05, float(L - 0.05), min(0.22, float(L - 0.1)), 0.01)
        sigma = st.slider("Šířka balíku sigma", 0.02, 0.18, 0.06, 0.01)
        k0 = st.slider("Počáteční impuls k0", 2.0, 35.0, 24.0, 1.0)

    x, dx, V, E_num, psi_num, E_an, phi_an = compute_single_well(L=L, N=N, n_show=n_show)
    st.pyplot(plot_single_stationary(x, V, E_num, psi_num, E_an, n_show=n_show), use_container_width=True)
    st.pyplot(plot_single_probabilities(x, psi_num, n=state_n, L=L), use_container_width=True)

    st.markdown("### Časový vývoj vlnového balíku")
    t_max = st.slider("Maximální čas", 0.05, 0.35, 0.16, 0.01)
    n_basis = st.slider("Počet stavů v rozvoji", 20, min(140, N - 2), min(100, N - 2), 10)
    data = compute_single_time_evolution(L=L, N=N, x0=x0, sigma=sigma, k0=k0, n_basis=n_basis, t_max=t_max)
    times = data[2]
    idx = st.slider("Vyber časový snímek", 0, len(times) - 1, len(times) // 3, 1)
    st.pyplot(plot_single_time_snapshot(data[0], data[5], data[6], data[8], times, idx), use_container_width=True)
    st.pyplot(plot_single_time_history(times, data[6], data[8]), use_container_width=True)

    if st.checkbox("Ukázat 3D povrch kvantového vývoje", value=False):
        dens = data[5]
        fig3d = plot_3d_surface_quantum_or_optical(
            data[0], times, dens,
            y_label="čas t",
            z_label=r"|psi|^2",
            title="3D povrch: kvantový vývoj v jedné jámě"
        )
        if fig3d is not None:
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Plotly zde není k dispozici.")

elif section_info == "Dvojitá jáma":
    st.subheader("2. Dvojitá kvantová jáma a tunelování")

    col1, col2 = st.columns([1, 1])
    with col1:
        barrier_width = st.slider("Šířka centrální bariéry", 0.04, 0.30, 0.12, 0.01)
        V0 = st.slider("Výška bariéry", 20.0, 140.0, 80.0, 5.0)
    with col2:
        N2 = st.slider("Počet bodů mřížky", 160, 320, 260, 20)

    data = compute_double_well(N2=N2, barrier_width=barrier_width, V0=V0)
    st.metric(r"$\Delta E = E_1 - E_0$", f"{data['dE']:.5f}")
    st.metric(r"$T_{tunnel} \approx \pi/\Delta E$", f"{data['T_tunnel']:.5f}")

    st.pyplot(plot_double_stationary(data), use_container_width=True)

    idx = st.slider(
        r"Vyber časový snímek (index přes interval 0 až 2.1 T_tunnel)",
        0, len(data["times"]) - 1, len(data["times"]) // 4, 1
    )
    st.pyplot(plot_double_snapshot(data, idx), use_container_width=True)

    st.markdown("### Jak bariéra řídí tunelování")
    widths, E0, E1, dE, Tt = compute_barrier_scan(V0=V0)
    st.pyplot(plot_barrier_scan(widths, E0, E1, dE, Tt), use_container_width=True)

    if st.checkbox("Ukázat 3D povrch kvantového tunelování", value=False):
        fig3d = plot_3d_surface_quantum_or_optical(
            data["x"], data["times"] / data["T_tunnel"], data["dens"],
            y_label=r"t / T_tunnel",
            z_label=r"|psi|^2",
            title="3D povrch: tunelování v dvojité jámě"
        )
        if fig3d is not None:
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Plotly zde není k dispozici.")

elif section_info == "Optická analogie":
    st.subheader("3. Optický analog: dva vlnovody")

    col1, col2 = st.columns([1, 1])
    with col1:
        N2 = st.slider("Počet bodů mřížky", 160, 320, 260, 20, key="opt_N2")
        dn_core = st.slider("Přídavek indexu jádra dn", 0.004, 0.030, 0.012, 0.001)
    with col2:
        n_clad = st.slider("Index pláště n_clad", 1.30, 1.60, 1.45, 0.01)

    opt = compute_optical_analogy(N2=N2, dn_core=dn_core, n_clad=n_clad)
    st.metric(r"$\Delta \beta = \beta_1 - \beta_0$", f"{opt['d_beta']:.5f}")
    st.metric(r"$L_c \approx \pi/\Delta \beta$", f"{opt['L_couple']:.5f}")

    idx = st.slider("Vyber propagaci z/Lc", 0, len(opt["z_vals"]) - 1, len(opt["z_vals"]) // 4, 1)
    st.pyplot(plot_optical_snapshot(opt, idx), use_container_width=True)

    st.markdown(
        r"""
        **Interpretace:**  
        V řezu $I(x)$ je směr $z$ kolmý k rovině obrázku.
        Proto je užitečné doplnit i 2D mapu $I(x,z)$, kde je $z$ explicitně jednou z os.
        """
    )

    if st.checkbox("Ukázat 3D povrch optického vývoje", value=False):
        fig3d = plot_3d_surface_quantum_or_optical(
            opt["x"], opt["z_vals"] / opt["L_couple"], opt["I_opt"],
            y_label=r"z / L_c",
            z_label="I(x,z)",
            title="3D povrch: optická intenzita v dvojvlnovodu"
        )
        if fig3d is not None:
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Plotly zde není k dispozici.")

elif section_info == "Konečně hluboká jáma":
    st.subheader("4. Konečně hluboká kvantová jáma")

    col1, col2 = st.columns([1, 1])
    with col1:
        well_width = st.slider("Šířka jamy", 0.30, 1.00, 0.60, 0.02)
        V_barrier = st.slider("Výška bariéry", 20.0, 180.0, 120.0, 5.0)
    with col2:
        L_f = st.slider("Celková šířka výpočetní oblasti", 1.0, 2.5, 1.8, 0.1)
        N_f = st.slider("Počet bodů mřížky", 180, 420, 360, 20)

    x, dx, V, E, psi, bound_idx, E_inf = compute_finite_well(
        L_f=L_f, N_f=N_f, well_width=well_width, V_barrier=V_barrier
    )
    st.write(f"Počet vázaných stavů: **{len(bound_idx)}**")
    st.pyplot(plot_finite_well(x, V, E, psi, bound_idx, E_inf), use_container_width=True)

    st.markdown(
        """
        Tato sekce je užitečná didakticky, protože studenti uvidí:
        - že počet vázaných stavů je konečný,
        - že vlnová funkce lehce proniká do bariéry,
        - že energie se liší od nekonečně hluboké jámy stejné šířky.
        """
    )
