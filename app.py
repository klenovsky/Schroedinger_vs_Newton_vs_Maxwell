from __future__ import annotations

import os
import tempfile
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import eigh_tridiagonal

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# Page setup and style
# ============================================================

st.set_page_config(
    page_title="Quantum well classroom app",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(59,130,246,0.13), rgba(16,185,129,0.10));
        border: 1px solid rgba(120,120,120,0.22);
        margin-bottom: 1rem;
    }
    .card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(120,120,120,0.18);
        background: rgba(250,250,250,0.03);
        margin-bottom: 0.8rem;
    }
    .small-note {
        font-size: 0.94rem;
        opacity: 0.88;
    }
    .metric-card {
        padding: 0.5rem 0.8rem;
        border-radius: 12px;
        border: 1px solid rgba(120,120,120,0.18);
        background: rgba(255,255,255,0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Language strings
# ============================================================

TXT: Dict[str, Dict[str, str]] = {
    "English": {
        "app_title": "Quantum, classical, and optical dynamics in wells",
        "app_subtitle": "Interactive teaching app derived from the classroom notebook. English is the default language; Czech can be selected from the sidebar.",
        "language": "Language",
        "section": "Section",
        "theory": "Theory",
        "single": "Single infinite well",
        "double": "Double well and tunnelling",
        "optical": "Optical analogy",
        "finite": "Finite well",
        "about": "App notes",
        "notes_text": "The numerics are vectorized, cached, and based on tridiagonal eigensolvers to keep the app responsive on GitHub + Streamlit Community Cloud.",
        "how_to_title": "How to use the app",
        "how_to_text": "1. Choose a section in the sidebar. 2. Set physical parameters with the sliders. 3. Open Static plots for stationary states and energies. 4. Use Snapshot for one selected time or propagation distance. 5. Use Animation and press Play below the graph to run the evolution. 6. Increase Animation speed (ms per frame) if you want a slower and clearer motion.",
        "theory_title": "Short theory overview",
        "theory_text": r"""
We use the following equations throughout the app.

Stationary Schrödinger equation:
$$
\hat H \psi_n(x)=E_n\psi_n(x),
\qquad
\hat H=-\frac{1}{2}\frac{d^2}{dx^2}+V(x).
$$

Time-dependent quantum evolution:
$$
\psi(x,t)=\sum_n c_n\,\psi_n(x)\,e^{-iE_n t}.
$$

In the classical picture, a particle moves between hard walls and reflects elastically.

For the optical analogy we use the paraxial envelope equation:
$$
i\frac{\partial A}{\partial z}
=-\frac{1}{2k_0n_0}\frac{\partial^2A}{\partial x^2}+V_{\mathrm{opt}}(x)A,
$$
where $z$ plays the role of an evolution variable analogous to time $t$.

In a double well or double waveguide, the splitting of the two lowest modes controls the oscillation period:
$$
T_{\mathrm{tunnel}}\approx \frac{\pi}{E_1-E_0},
\qquad
L_c\approx \frac{\pi}{\beta_1-\beta_0}.
$$
""",
        "references": "Selected references",
        "refs_text": """
- D. J. Griffiths, D. F. Schroeter, *Introduction to Quantum Mechanics*, 3rd ed. (Cambridge University Press, 2018).
- B. E. A. Saleh, M. C. Teich, *Fundamentals of Photonics*, 3rd ed. (Wiley, 2019).
- E. Hecht, *Optics*, 5th ed. (Pearson, 2016).
- D. N. Christodoulides, F. Lederer, Y. Silberberg, *Nature* **424**, 817–823 (2003).
- R. W. Robinett, *Physics Reports* **392**, 1–119 (2004) — wave-packet revivals.
""",
        "single_intro": "Static eigenstates, probability density, wave-packet motion, and a video of the time evolution.",
        "double_intro": "Comparison of three related pictures: classical motion below the barrier, quantum tunnelling, and the optical coupled-waveguide analogue.",
        "optical_intro": "Paraxial propagation of the optical envelope in two coupled waveguides.",
        "finite_intro": "A finite well shows evanescent tails and only a finite number of bound states.",
        "grid": "Grid points",
        "show_states": "Number of eigenstates shown",
        "chosen_state": "Selected stationary state n",
        "well_width": "Well width",
        "total_box": "Total computational width",
        "packet_center": "Initial packet center",
        "packet_sigma": "Packet width sigma",
        "packet_k0": "Initial momentum k0",
        "max_time": "Maximum time",
        "basis_count": "Number of basis states in expansion",
        "snapshot": "Snapshot index",
        "barrier_width": "Barrier width",
        "barrier_height": "Barrier height",
        "dn_core": "Core index contrast Δn",
        "n_clad": "Cladding index n_clad",
        "generate_video": "Generate GIF animation",
        "download_video": "Download GIF",
        "video_help": "The GIF is generated on demand and cached, so the second run is much faster.",
        "tab_static": "Static plots",
        "tab_snapshot": "Snapshot",
        "tab_video": "Animation",
        "single_states_title": "Single infinite well: first eigenstates",
        "single_prob_title": "Quantum vs classical probability density",
        "single_snapshot_title": "Time snapshot of the wave packet",
        "single_history_title": "Mean position versus time",
        "double_states_title": "Double well: symmetric and antisymmetric states",
        "double_snapshot_title": "Instantaneous comparison",
        "double_scan_title": "Barrier controls the splitting and the tunnelling time",
        "optical_snapshot_title": "Optical intensity slice and I(x,z) map",
        "finite_title": "Finite well and bound states",
        "bound_states": "Number of bound states",
        "interpretation": "Interpretation",
        "optical_note": "For a 1D slice $I(x)$ the propagation direction $z$ is perpendicular to the drawing plane; the 2D map makes this explicit.",
        "classical": "Classical",
        "quantum": "Quantum",
        "optical_label": "Optical",
        "left_prob": "Left probability",
        "right_prob": "Right probability",
        "mean_position": "Mean position",
        "classical_particle": "Classical particle",
        "quantum_density": "Quantum density",
        "optical_intensity": "Optical intensity",
        "theory_card": "The app is intentionally didactic: simple 1D models, explicit equations, and visual comparison between classical, quantum, and optical pictures.",
        "revival_note": "The spreading of the packet after reflection is not decoherence. It is coherent dispersion and interference of different stationary components; for suitable times, revivals can appear.",
        "scan_note": r"A wider barrier reduces the splitting $\Delta E$, which makes the tunnelling oscillation slower.",
        "footer": "Prepared for teaching use. The app structure is repository-ready for GitHub and Streamlit Community Cloud.",
        "no_plotly": "Plotly is not available in this environment.",
        "video_section_single": "Animated wave-packet evolution",
        "video_section_double": "Animated classical / quantum / optical comparison",
        "video_section_optical": "Animated optical propagation",
        "left_right": "Left / right population",
        "expander_numerics": "Numerical implementation",
        "numerics_text": "The Hamiltonians are tridiagonal and diagonalized with `scipy.linalg.eigh_tridiagonal`; time evolution is evaluated in a vectorized basis-expansion form `basis @ phases`.",
        "play_note": "Use the Play button below the plot to start the evolution.",
        "gif_note": "Optional: you can also export the same evolution as a GIF.",
        "anim_speed": "Animation speed (ms per frame)",
        "anim_smoother": "Slower values make the motion easier to follow and reduce visual flicker.",
        "all_states_note": "An infinite well has infinitely many exact eigenstates. The app therefore shows all states resolved by the current numerical grid/basis and lets you inspect any selected one.",
        "all_states_title": "All numerically resolved states",
        "revival_title": "Autocorrelation and revival",
        "revival_strength": "Revival strength max |<ψ(0)|ψ(t)>|²",
        "revival_time": "Revival time T_rev",
        "time_factor": "Maximum time in units of T_rev",
    },
    "Czech": {
        "app_title": "Kvantová, klasická a optická dynamika v jamách",
        "app_subtitle": "Interaktivní výuková aplikace odvozená z notebooku. Výchozí jazyk je angličtina; češtinu lze přepnout v levém panelu.",
        "language": "Jazyk",
        "section": "Sekce",
        "theory": "Teorie",
        "single": "Jedna nekonečně hluboká jáma",
        "double": "Dvojitá jáma a tunelování",
        "optical": "Optická analogie",
        "finite": "Konečně hluboká jáma",
        "about": "Poznámky k aplikaci",
        "notes_text": "Numerika je vektorizovaná, cachovaná a postavená na tridiagonálních eigensolverech, aby aplikace běžela svižně i na GitHubu a Streamlit Community Cloud.",
        "how_to_title": "Jak aplikaci používat",
        "how_to_text": "1. V levém panelu vyber sekci. 2. Pomocí sliderů nastav fyzikální parametry. 3. Ve Statických grafech sleduj stacionární stavy a energie. 4. Ve Snímku zobraz jeden vybraný čas nebo propagační vzdálenost. 5. V Animaci spusť vývoj tlačítkem Play pod grafem. 6. Pro pomalejší a přehlednější pohyb zvyš hodnotu Rychlost animace (ms na snímek).",
        "theory_title": "Krátký teoretický přehled",
        "theory_text": r"""
V celé aplikaci používáme tyto základní rovnice.

Stacionární Schrödingerova rovnice:
$$
\hat H \psi_n(x)=E_n\psi_n(x),
\qquad
\hat H=-\frac{1}{2}\frac{d^2}{dx^2}+V(x).
$$

Časový vývoj kvantového stavu:
$$
\psi(x,t)=\sum_n c_n\,\psi_n(x)\,e^{-iE_n t}.
$$

V klasickém obrazu se částice pohybuje mezi tvrdými stěnami a na hranách se pružně odráží.

Pro optickou analogii používáme paraxiální rovnici pro obálku:
$$
i\frac{\partial A}{\partial z}
=-\frac{1}{2k_0n_0}\frac{\partial^2A}{\partial x^2}+V_{\mathrm{opt}}(x)A,
$$
kde $z$ hraje roli evoluční proměnné analogické času $t$.

V dvojité jámě nebo dvojvlnovodu řídí periodu oscilace rozštěpení dvou nejnižších módů:
$$
T_{\mathrm{tunnel}}\approx \frac{\pi}{E_1-E_0},
\qquad
L_c\approx \frac{\pi}{\beta_1-\beta_0}.
$$
""",
        "references": "Vybrané reference",
        "refs_text": """
- D. J. Griffiths, D. F. Schroeter, *Introduction to Quantum Mechanics*, 3. vyd. (Cambridge University Press, 2018).
- B. E. A. Saleh, M. C. Teich, *Fundamentals of Photonics*, 3. vyd. (Wiley, 2019).
- E. Hecht, *Optics*, 5. vyd. (Pearson, 2016).
- D. N. Christodoulides, F. Lederer, Y. Silberberg, *Nature* **424**, 817–823 (2003).
- R. W. Robinett, *Physics Reports* **392**, 1–119 (2004) — revivaly vlnových balíků.
""",
        "single_intro": "Stacionární stavy, pravděpodobnostní hustota, pohyb vlnového balíku a video časového vývoje.",
        "double_intro": "Srovnání tří příbuzných obrazů: klasický pohyb pod bariérou, kvantové tunelování a optický analog ve dvojici vlnovodů.",
        "optical_intro": "Paraxiální šíření optické obálky ve dvou vazebně spojených vlnovodech.",
        "finite_intro": "Konečná jáma ukazuje evanescentní ocasy a jen konečný počet vázaných stavů.",
        "grid": "Počet bodů mřížky",
        "show_states": "Počet zobrazených vlastních stavů",
        "chosen_state": "Vybraný stacionární stav n",
        "well_width": "Šířka jámy",
        "total_box": "Celková šířka výpočetní oblasti",
        "packet_center": "Počáteční střed balíku",
        "packet_sigma": "Šířka balíku sigma",
        "packet_k0": "Počáteční impuls k0",
        "max_time": "Maximální čas",
        "basis_count": "Počet stavů v rozvoji",
        "snapshot": "Index snímku",
        "barrier_width": "Šířka bariéry",
        "barrier_height": "Výška bariéry",
        "dn_core": "Kontrast indexu jádra Δn",
        "n_clad": "Index pláště n_clad",
        "generate_video": "Vygenerovat GIF animaci",
        "download_video": "Stáhnout GIF",
        "video_help": "GIF se generuje na vyžádání a ukládá se do cache, takže druhé spuštění je výrazně rychlejší.",
        "tab_static": "Statické grafy",
        "tab_snapshot": "Snímek",
        "tab_video": "Animace",
        "single_states_title": "Jedna nekonečně hluboká jáma: první stavy",
        "single_prob_title": "Kvantová vs. klasická pravděpodobnostní hustota",
        "single_snapshot_title": "Časový snímek vlnového balíku",
        "single_history_title": "Střední poloha v závislosti na čase",
        "double_states_title": "Dvojitá jáma: symetrický a antisymetrický stav",
        "double_snapshot_title": "Okamžité srovnání",
        "double_scan_title": "Bariéra řídí rozštěpení i dobu tunelování",
        "optical_snapshot_title": "Řez optickou intenzitou a mapa I(x,z)",
        "finite_title": "Konečně hluboká jáma a vázané stavy",
        "bound_states": "Počet vázaných stavů",
        "interpretation": "Interpretace",
        "optical_note": "U 1D řezu $I(x)$ je směr šíření $z$ kolmý k rovině obrázku; 2D mapa to ukazuje explicitně.",
        "classical": "Klasika",
        "quantum": "Kvantově",
        "optical_label": "Optika",
        "left_prob": "Pravděpodobnost vlevo",
        "right_prob": "Pravděpodobnost vpravo",
        "mean_position": "Střední poloha",
        "classical_particle": "Klasická částice",
        "quantum_density": "Kvantová hustota",
        "optical_intensity": "Optická intenzita",
        "theory_card": "Aplikace je záměrně didaktická: jednoduché 1D modely, explicitní rovnice a přímé srovnání klasického, kvantového a optického obrazu.",
        "revival_note": "Rozpad balíku po odrazu není dekoherence. Jde o koherentní disperzi a interferenci různých stacionárních složek; pro vhodné časy se mohou objevit revivaly.",
        "scan_note": r"Širší bariéra zmenšuje rozštěpení $\Delta E$, a tím zpomaluje tunelovací oscilaci.",
        "footer": "Připraveno pro výukové použití. Struktura aplikace je připravená pro GitHub i Streamlit Community Cloud.",
        "no_plotly": "V tomto prostředí není k dispozici Plotly.",
        "video_section_single": "Animovaný vývoj vlnového balíku",
        "video_section_double": "Animované srovnání klasiky / kvantového případu / optiky",
        "video_section_optical": "Animované optické šíření",
        "left_right": "Levá / pravá populace",
        "expander_numerics": "Numerická implementace",
        "numerics_text": "Hamiltoniány jsou tridiagonální a diagonalizují se pomocí `scipy.linalg.eigh_tridiagonal`; časový vývoj je vyhodnocen vektorizovaně ve tvaru `basis @ phases`.",
        "play_note": "Pro spuštění vývoje použij tlačítko Play pod grafem.",
        "gif_note": "Volitelně lze stejný vývoj exportovat i jako GIF.",
        "anim_speed": "Rychlost animace (ms na snímek)",
        "anim_smoother": "Vyšší hodnota animaci zpomalí a omezí vizuální blikání.",
        "all_states_note": "Nekonečně hluboká jáma má přesně vzato nekonečně mnoho vlastních stavů. Aplikace proto ukazuje všechny stavy zachycené zvolenou numerickou mřížkou/bází a dovoluje vybrat libovolný z nich.",
        "all_states_title": "Všechny numericky zachycené stavy",
        "revival_title": "Autokorelace a revival",
        "revival_strength": "Síla revivalu max |<ψ(0)|ψ(t)>|²",
        "revival_time": "Revival time T_rev",
        "time_factor": "Maximální čas v jednotkách T_rev",
    },
}


def tr(lang: str, key: str) -> str:
    return TXT[lang][key]


# ============================================================
# Numerical helpers
# ============================================================


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi if norm == 0 else psi / norm



def normalize_columns(vecs: np.ndarray, dx: float) -> np.ndarray:
    norms = np.sqrt(np.sum(np.abs(vecs) ** 2, axis=0) * dx)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms[None, :]



def build_tridiagonal(x: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    dx = float(x[1] - x[0])
    diag = 1.0 / dx**2 + V
    off = np.full(x.size - 1, -0.5 / dx**2, dtype=float)
    return diag, off, dx



def solve_tridiagonal(diag: np.ndarray, off: np.ndarray, n_eigs: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if n_eigs is None or n_eigs >= diag.size:
        vals, vecs = eigh_tridiagonal(diag, off)
    else:
        vals, vecs = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_eigs - 1))
    return vals, vecs



def project_state(psi0: np.ndarray, basis: np.ndarray, dx: float) -> np.ndarray:
    return basis.conj().T @ (psi0 * dx)



def evolve_basis(coeffs: np.ndarray, eigvals: np.ndarray, basis: np.ndarray, times: np.ndarray) -> np.ndarray:
    phases = np.exp(-1j * eigvals[:, None] * times[None, :])
    return basis @ (coeffs[:, None] * phases)



def expectation_x(psi_xt: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    dens = np.abs(psi_xt) ** 2
    return np.sum(dens * x[:, None], axis=0) * dx



def classical_box_trajectory(times: np.ndarray, x0: float, v: float, xmin: float, xmax: float) -> np.ndarray:
    L = xmax - xmin
    if L <= 0:
        return np.full_like(times, x0)
    y = (x0 - xmin + v * times) % (2.0 * L)
    return xmin + np.where(y <= L, y, 2.0 * L - y)



def infinite_well_analytic(x: np.ndarray, L: float, n_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.sqrt(2.0 / L) * np.sin(np.pi * np.outer(x / L, n_values))
    E = (np.pi**2 / (2.0 * L**2)) * n_values**2
    return phi, E


# ============================================================
# Physics models
# ============================================================


@st.cache_data(show_spinner=False)
def compute_single_well(L: float = 1.0, N: int = 220, n_show: int = 8):
    x = np.linspace(L / (N + 1), L - L / (N + 1), N)
    V = np.zeros_like(x)
    diag, off, dx = build_tridiagonal(x, V)
    n_all = min(N, max(180, n_show))
    E_num, psi_num = solve_tridiagonal(diag, off, n_eigs=n_all)
    psi_num = normalize_columns(psi_num.astype(complex), dx)
    n_vals = np.arange(1, n_all + 1)
    phi_an, E_an = infinite_well_analytic(x, L, n_vals)
    return x, dx, V, E_num, psi_num, E_an, phi_an


@st.cache_data(show_spinner=False)
def compute_single_dynamics(L: float = 1.0, N: int = 220, x0: float = 0.22, sigma: float = 0.06,
                            k0: float = 24.0, n_basis: int = 120, t_factor: float = 1.10,
                            n_times: int = 560):
    x = np.linspace(L / (N + 1), L - L / (N + 1), N)
    V = np.zeros_like(x)
    diag, off, dx = build_tridiagonal(x, V)
    E, psi = solve_tridiagonal(diag, off, n_eigs=n_basis)
    psi = normalize_columns(psi.astype(complex), dx)

    psi0 = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    psi0 = normalize(psi0, dx)

    coeffs = project_state(psi0, psi, dx)
    T_rev = 4.0 * L**2 / np.pi
    t_max = float(t_factor) * T_rev
    times = np.linspace(0.0, t_max, n_times)
    psi_xt = evolve_basis(coeffs, E, psi, times)
    dens = np.abs(psi_xt) ** 2
    x_mean = expectation_x(psi_xt, x, dx)
    x_class = classical_box_trajectory(times, x0=x0, v=k0, xmin=0.0, xmax=L)
    autocorr = np.abs(np.sum(np.conj(psi0)[:, None] * psi_xt, axis=0) * dx) ** 2
    return {
        "x": x, "dx": dx, "times": times, "psi0": psi0, "psi_xt": psi_xt, "dens": dens,
        "x_mean": x_mean, "x_class": x_class, "E": E, "basis": psi,
        "autocorr": autocorr, "T_rev": T_rev, "t_max": t_max,
    }


@st.cache_data(show_spinner=False)
def compute_double_well(barrier_width: float = 0.12, V0: float = 80.0, L2: float = 1.8,
                        N2: int = 260, n_basis: int = 50, n_times: int = 260,
                        sigma: float = 0.10, k0: float = 0.0):
    x = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    center = L2 / 2.0
    V = np.zeros_like(x)
    mask = np.abs(x - center) < barrier_width / 2.0
    V[mask] = V0

    diag, off, dx = build_tridiagonal(x, V)
    E, psi = solve_tridiagonal(diag, off, n_eigs=max(n_basis, 12))
    psi = normalize_columns(psi.astype(complex), dx)

    psi_g = psi[:, 0]
    psi_e = psi[:, 1]
    dE = E[1] - E[0]
    T_tunnel = np.pi / dE

    x0 = 0.33 * center
    psi0 = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    psi0 = normalize(psi0, dx)
    coeffs = project_state(psi0, psi[:, :n_basis], dx)
    times = np.linspace(0.0, 2.2 * T_tunnel, n_times)
    psi_xt = evolve_basis(coeffs, E[:n_basis], psi[:, :n_basis], times)
    dens = np.abs(psi_xt) ** 2
    x_mean = expectation_x(psi_xt, x, dx)

    left_mask = x < center
    right_mask = x > center
    P_left = np.sum(dens[left_mask, :], axis=0) * dx
    P_right = np.sum(dens[right_mask, :], axis=0) * dx

    x_class = classical_box_trajectory(
        times,
        x0=x0,
        v=0.45,
        xmin=0.0,
        xmax=center - barrier_width / 2.0,
    )
    return {
        "x": x, "dx": dx, "V": V, "E": E, "psi": psi,
        "psi_g": psi_g, "psi_e": psi_e, "times": times,
        "dens": dens, "x_mean": x_mean, "P_left": P_left, "P_right": P_right,
        "dE": dE, "T_tunnel": T_tunnel, "center": center,
        "barrier_width": barrier_width, "V0": V0, "x_class": x_class,
    }


@st.cache_data(show_spinner=False)
def compute_optical_analogy(dn_core: float = 0.012, n_clad: float = 1.45, L2: float = 1.8,
                            N2: int = 260, n_basis: int = 24, n_steps: int = 260):
    x = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    dx = float(x[1] - x[0])
    center = L2 / 2.0

    n_profile = np.full_like(x, n_clad)
    left_core = (x > 0.33) & (x < 0.74)
    right_core = (x > 1.06) & (x < 1.47)
    n_profile[left_core] = n_clad + dn_core
    n_profile[right_core] = n_clad + dn_core

    V_opt = -35.0 * (n_profile - n_clad) / max(dn_core, 1e-9)
    diag, off, _ = build_tridiagonal(x, V_opt)
    beta, phi = solve_tridiagonal(diag, off, n_eigs=n_basis)
    phi = normalize_columns(phi.astype(complex), dx)

    d_beta = beta[1] - beta[0]
    L_couple = np.pi / d_beta

    x0 = 0.53
    A0 = np.exp(-((x - x0) ** 2) / (2.0 * 0.085**2))
    A0 = normalize(A0.astype(complex), dx)
    coeffs = project_state(A0, phi, dx)
    z_vals = np.linspace(0.0, 2.2 * L_couple, n_steps)
    A_xz = evolve_basis(coeffs, beta, phi, z_vals)
    I_opt = np.abs(A_xz) ** 2
    x_mean = expectation_x(A_xz, x, dx)

    left_half = x < center
    right_half = x > center
    P_left = np.sum(I_opt[left_half, :], axis=0) * dx
    P_right = np.sum(I_opt[right_half, :], axis=0) * dx
    return {
        "x": x, "dx": dx, "center": center, "n_profile": n_profile,
        "V_opt": V_opt, "beta": beta, "phi": phi, "z_vals": z_vals,
        "I_opt": I_opt, "x_mean": x_mean, "P_left": P_left, "P_right": P_right,
        "L_couple": L_couple, "d_beta": d_beta, "n_clad": n_clad,
    }


@st.cache_data(show_spinner=False)
def compute_barrier_scan(widths: np.ndarray, V0: float = 80.0, L2: float = 1.8, N2: int = 220):
    x = np.linspace(L2 / (N2 + 1), L2 - L2 / (N2 + 1), N2)
    center = L2 / 2.0
    E0, E1, dE, Tt = [], [], [], []
    for bw in widths:
        V = np.zeros_like(x)
        V[np.abs(x - center) < bw / 2.0] = V0
        diag, off, _ = build_tridiagonal(x, V)
        E, _ = solve_tridiagonal(diag, off, n_eigs=2)
        E0.append(E[0])
        E1.append(E[1])
        split = E[1] - E[0]
        dE.append(split)
        Tt.append(np.pi / split)
    return np.array(E0), np.array(E1), np.array(dE), np.array(Tt)


@st.cache_data(show_spinner=False)
def compute_finite_well(well_width: float = 0.60, V_barrier: float = 120.0, L_f: float = 1.8,
                        N_f: int = 360, n_eigs: int | None = None):
    x = np.linspace(L_f / (N_f + 1), L_f - L_f / (N_f + 1), N_f)
    center = L_f / 2.0
    V = np.full_like(x, V_barrier)
    inside = np.abs(x - center) < well_width / 2.0
    V[inside] = 0.0

    diag, off, dx = build_tridiagonal(x, V)
    # Solve the full tridiagonal problem so that all numerically resolved bound states are available.
    E, psi = solve_tridiagonal(diag, off, n_eigs=n_eigs)
    psi = normalize_columns(psi.astype(complex), dx)
    bound_idx = np.where(E < V_barrier)[0]

    n_bound = len(bound_idx)
    n_inf = np.arange(1, n_bound + 1)
    E_inf = (np.pi**2 / (2.0 * well_width**2)) * n_inf**2 if n_bound > 0 else np.array([])
    return {"x": x, "dx": dx, "V": V, "E": E, "psi": psi, "bound_idx": bound_idx, "E_inf": E_inf}


# ============================================================
# Plot helpers
# ============================================================


FIG_FACE = (0.0, 0.0, 0.0, 0.0)


def _base_fig(nrows=1, ncols=1, figsize=(8, 4.5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(FIG_FACE)
    return fig, ax



def plot_single_stationary(data, L: float, n_show: int, lang: str):
    x, V, E_num, psi_num = data[0], data[2], data[3], data[4]
    n_all = len(E_num)
    n_show = min(n_show, n_all)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    fig.patch.set_facecolor(FIG_FACE)

    ax = axes[0]
    ax.plot(x, V, lw=2, label="V(x)")
    scale = 0.16 * (E_num[min(n_show, n_all - 1)] - E_num[0] + 1e-9)
    scale = max(scale, 0.16)
    for n in range(n_show):
        y = scale * np.real(psi_num[:, n]) + E_num[n]
        ax.plot(x, y, lw=1.8, label=f"n={n+1}")
        ax.axhline(E_num[n], lw=0.8, alpha=0.22)
    ax.set_xlabel("x")
    ax.set_ylabel("energy / state shape" if lang == "English" else "energie / tvar stavu")
    ax.set_title(tr(lang, "single_states_title"))
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    idx = np.arange(1, n_all + 1)
    ax.plot(idx, E_num, "o", ms=3.2, alpha=0.8, label=tr(lang, "all_states_title"))
    ax.set_xlabel("state index n" if lang == "English" else "číslo stavu n")
    ax.set_ylabel("energy" if lang == "English" else "energie")
    ax.set_title(tr(lang, "all_states_title"))
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_single_probability(data, L: float, state_n: int, lang: str):
    x, psi_num = data[0], data[4]
    idx = state_n - 1
    fig, ax = _base_fig(figsize=(8.3, 4.3))
    P_quant = np.abs(psi_num[:, idx]) ** 2
    P_class = np.ones_like(x) / L
    ax.plot(x, P_quant, lw=2.2, label=r"$|\psi_n|^2$")
    ax.plot(x, P_class, "--", lw=2, label="uniform classical" if lang == "English" else "rovnoměrně klasicky")
    ax.set_xlabel("x")
    ax.set_ylabel("probability density" if lang == "English" else "pravděpodobnostní hustota")
    ax.set_title(tr(lang, "single_prob_title"))
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    return fig



def plot_single_snapshot(sim, idx: int, lang: str):
    x, dens, x_mean, x_class, times = sim["x"], sim["dens"], sim["x_mean"], sim["x_class"], sim["times"]
    autocorr, T_rev = sim["autocorr"], sim["T_rev"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    fig.patch.set_facecolor(FIG_FACE)

    ax = axes[0]
    ax.plot(x, dens[:, idx], lw=2.2, label=tr(lang, "quantum_density"))
    ax.axvline(x_mean[idx], ls="--", lw=1.8, label=tr(lang, "mean_position"))
    ax.plot([x_class[idx]], [0.08 * max(np.max(dens[:, idx]), 1e-12)], "o", ms=8, label=tr(lang, "classical_particle"))
    ax.set_xlabel("x")
    ax.set_ylabel("density" if lang == "English" else "hustota")
    ax.set_title(f"{tr(lang, 'single_snapshot_title')}  (t/T_rev = {times[idx]/T_rev:.3f})")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(times / T_rev, x_mean, lw=2.2, label=tr(lang, "mean_position"))
    ax.plot(times / T_rev, x_class, "--", lw=2, label=tr(lang, "classical_particle"))
    ax.axvline(times[idx] / T_rev, color="k", ls=":", lw=1.5)
    ax.set_xlabel(r"t / T$_{rev}$")
    ax.set_ylabel("position" if lang == "English" else "poloha")
    ax.set_title(tr(lang, "single_history_title"))
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(times / T_rev, autocorr, lw=2.2)
    ax.axvline(times[idx] / T_rev, color="k", ls=":", lw=1.5)
    ax.set_xlabel(r"t / T$_{rev}$")
    ax.set_ylabel("overlap" if lang == "English" else "překryv")
    ax.set_ylim(0, 1.05)
    ax.set_title(tr(lang, "revival_title"))
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def plot_double_stationary(dw, lang: str):
    fig, ax = _base_fig(figsize=(8.6, 4.8))
    x, V, E, psi_g, psi_e = dw["x"], dw["V"], dw["E"], dw["psi_g"], dw["psi_e"]
    ax.plot(x, V / max(np.max(V), 1e-9) * max(E[1] * 1.25, 1e-9), lw=2, label="barrier" if lang == "English" else "bariéra")
    ax.plot(x, 0.18 * np.real(psi_g) + E[0], lw=2.2, label="ground" if lang == "English" else "základní stav")
    ax.plot(x, 0.18 * np.real(psi_e) + E[1], lw=2.2, label="1st excited" if lang == "English" else "1. excitovaný stav")
    ax.axhline(E[0], lw=0.8, alpha=0.25)
    ax.axhline(E[1], lw=0.8, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("energy / state shape" if lang == "English" else "energie / tvar stavu")
    ax.set_title(tr(lang, "double_states_title"))
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    return fig



def plot_double_snapshot(dw, opt, idx_q: int, idx_o: int, lang: str):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    fig.patch.set_facecolor(FIG_FACE)

    # Classical / quantum density
    ax = axes[0, 0]
    x = dw["x"]
    dens = dw["dens"][:, idx_q]
    scale = max(np.max(dens), 1e-12)
    ax.plot(x, dens, lw=2.1, label=tr(lang, "quantum_density"))
    ax.plot(x, (dw["V"] / max(np.max(dw["V"]), 1e-9)) * 0.8 * scale, "k--", lw=1.5, label="barrier" if lang == "English" else "bariéra")
    ax.axvline(dw["x_mean"][idx_q], ls="--", lw=1.6, label=tr(lang, "mean_position"))
    ax.plot([dw["x_class"][idx_q]], [0.08 * scale], "o", ms=8, label=tr(lang, "classical_particle"))
    ax.set_xlabel("x")
    ax.set_ylabel("density" if lang == "English" else "hustota")
    ax.set_title(tr(lang, "double_snapshot_title"))
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9)

    # Left/right population quantum
    ax = axes[0, 1]
    norm_t = dw["times"] / dw["T_tunnel"]
    ax.plot(norm_t, dw["P_left"], lw=2.1, label=tr(lang, "left_prob"))
    ax.plot(norm_t, dw["P_right"], lw=2.1, label=tr(lang, "right_prob"))
    ax.axvline(norm_t[idx_q], color="k", ls=":", lw=1.5)
    ax.set_xlabel(r"t / T$_{tunnel}$")
    ax.set_ylabel("population" if lang == "English" else "populace")
    ax.set_ylim(0, 1.05)
    ax.set_title(tr(lang, "left_right"))
    ax.grid(True, alpha=0.22)
    ax.legend()

    # Optical slice
    ax = axes[1, 0]
    x_opt = opt["x"]
    I_now = opt["I_opt"][:, idx_o]
    ax.plot(x_opt, I_now, lw=2.1, label=tr(lang, "optical_intensity"))
    ax.plot(x_opt, 120 * (opt["n_profile"] - opt["n_clad"]), "k--", lw=1.4, label="index profile" if lang == "English" else "profil indexu")
    ax.axvline(opt["x_mean"][idx_o], ls="--", lw=1.6, label=tr(lang, "mean_position"))
    ax.set_xlabel("x")
    ax.set_ylabel("intensity" if lang == "English" else "intenzita")
    ax.set_title(tr(lang, "optical_snapshot_title"))
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(0.03, 0.92, r"$z$ ⟂ plane", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"), fontsize=9)

    # Optical map
    ax = axes[1, 1]
    extent = [opt["z_vals"][0] / opt["L_couple"], opt["z_vals"][-1] / opt["L_couple"], x_opt[0], x_opt[-1]]
    im = ax.imshow(opt["I_opt"], origin="lower", aspect="auto", extent=extent)
    ax.axvline(opt["z_vals"][idx_o] / opt["L_couple"], color="w", ls=":", lw=1.6)
    ax.set_xlabel(r"z / L$_c$")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax, label="intensity" if lang == "English" else "intenzita")

    fig.tight_layout()
    return fig



def plot_barrier_scan(widths: np.ndarray, E0: np.ndarray, E1: np.ndarray, dE: np.ndarray, Tt: np.ndarray, lang: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    fig.patch.set_facecolor(FIG_FACE)

    ax = axes[0]
    ax.plot(widths, E0, "o-", lw=2.2, label=r"$E_0$")
    ax.plot(widths, E1, "s-", lw=2.2, label=r"$E_1$")
    ax.set_xlabel("barrier width" if lang == "English" else "šířka bariéry")
    ax.set_ylabel("energy" if lang == "English" else "energie")
    ax.set_title(tr(lang, "double_scan_title"))
    ax.grid(True, alpha=0.22)
    ax.legend()

    ax = axes[1]
    ax.plot(widths, dE, "o-", lw=2.2, label=r"$\Delta E$")
    ax2 = ax.twinx()
    ax2.plot(widths, Tt, "s--", lw=2.1, label=r"$T_{tunnel}$")
    ax.set_xlabel("barrier width" if lang == "English" else "šířka bariéry")
    ax.set_ylabel(r"$\Delta E$")
    ax2.set_ylabel(r"$T_{tunnel}$")
    ax.grid(True, alpha=0.22)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    return fig



def plot_finite_well(fw, lang: str):
    x, V, E, psi, bound_idx, E_inf = fw["x"], fw["V"], fw["E"], fw["psi"], fw["bound_idx"], fw["E_inf"]
    n_plot = len(bound_idx)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    fig.patch.set_facecolor(FIG_FACE)

    ax = axes[0]
    ax.plot(x, V, lw=2.2, label="V(x)")
    if n_plot > 0:
        energy_span = max(E[bound_idx[-1]] - E[bound_idx[0]], 1e-9)
        state_scale = max(0.12 * energy_span, 0.12)
        cmap = plt.get_cmap("tab10", max(n_plot, 1))
        for j, idx in enumerate(bound_idx):
            ax.plot(x, state_scale * np.real(psi[:, idx]) + E[idx], lw=1.8, color=cmap(j), label=f"n={j+1}")
            ax.axhline(E[idx], lw=0.8, alpha=0.25, color=cmap(j))
    ax.set_xlabel("x")
    ax.set_ylabel("energy / state shape" if lang == "English" else "energie / tvar stavu")
    ax.set_title(tr(lang, "finite_title"))
    ax.grid(True, alpha=0.22)
    if n_plot > 0:
        ax.legend(title="bound" if lang == "English" else "vázané", fontsize=8, ncol=2)

    ax = axes[1]
    if n_plot > 0:
        idx = np.arange(1, n_plot + 1)
        ax.plot(idx, E[bound_idx], "o-", lw=2.2, label="finite" if lang == "English" else "konečná")
        ax.plot(idx, E_inf, "s--", lw=2.0, label="infinite" if lang == "English" else "nekonečná")
    ax.set_xlabel("state index" if lang == "English" else "číslo stavu")
    ax.set_ylabel("energy" if lang == "English" else "energie")
    ax.grid(True, alpha=0.22)
    if n_plot > 0:
        ax.legend()
    fig.tight_layout()
    return fig



def make_surface(x: np.ndarray, y: np.ndarray, Z: np.ndarray, title: str, y_label: str, z_label: str):
    if not HAS_PLOTLY:
        return None
    X, Y = np.meshgrid(x, y)
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z.T, colorscale="Viridis", showscale=True, opacity=0.95))
    fig.update_layout(
        title=title,
        height=560,
        scene=dict(xaxis_title="x", yaxis_title=y_label, zaxis_title=z_label),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _animation_controls(frame_duration_ms: int = 180, redraw: bool = False):
    return [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.02,
            "y": 1.15,
            "direction": "left",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": redraw}, "fromcurrent": True, "transition": {"duration": 0}}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                },
            ],
        }
    ]


def _animation_slider(steps):
    return [{
        "currentvalue": {"prefix": "frame: "},
        "pad": {"t": 40},
        "steps": steps,
    }]


def make_single_animation(sim, lang: str, frame_duration_ms: int = 220):
    if not HAS_PLOTLY:
        return None
    x = sim["x"]
    dens = sim["dens"]
    x_mean = sim["x_mean"]
    x_class = sim["x_class"]
    times = sim["times"]
    autocorr = sim["autocorr"]
    T_rev = sim["T_rev"]
    step = max(1, len(times) // 90)
    idxs = np.arange(0, len(times), step, dtype=int)
    if idxs[-1] != len(times) - 1:
        idxs = np.append(idxs, len(times) - 1)
    ymax = float(1.05 * np.max(dens))
    ymin_h = float(min(np.min(x_mean), np.min(x_class)) - 0.05)
    ymax_h = float(max(np.max(x_mean), np.max(x_class)) + 0.05)
    tnorm = times / T_rev

    fig = make_subplots(rows=1, cols=3, subplot_titles=(tr(lang, "single_snapshot_title"), tr(lang, "single_history_title"), tr(lang, "revival_title")))
    fig.add_trace(go.Scatter(x=x, y=dens[:, idxs[0]], mode="lines", name=tr(lang, "quantum_density"), line=dict(width=3, color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_mean[idxs[0]], x_mean[idxs[0]]], y=[0, ymax], mode="lines", name=tr(lang, "mean_position"), line=dict(dash="dash", color="#d62728")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_class[idxs[0]]], y=[0.08 * ymax], mode="markers", name=tr(lang, "classical_particle"), marker=dict(size=9, color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=tnorm, y=x_mean, mode="lines", name=tr(lang, "mean_position"), line=dict(width=3, color="#1f77b4"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=tnorm, y=x_class, mode="lines", name=tr(lang, "classical_particle"), line=dict(dash="dash", width=3, color="#2ca02c"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[tnorm[idxs[0]]], y=[x_mean[idxs[0]]], mode="markers", marker=dict(size=9, color="#1f77b4"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[tnorm[idxs[0]]], y=[x_class[idxs[0]]], mode="markers", marker=dict(size=9, symbol="diamond", color="#2ca02c"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=tnorm, y=autocorr, mode="lines", line=dict(width=3, color="#9467bd"), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[tnorm[idxs[0]]], y=[autocorr[idxs[0]]], mode="markers", marker=dict(size=9, color="#9467bd"), showlegend=False), row=1, col=3)

    frames = []
    slider_steps = []
    for i in idxs:
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=x, y=dens[:, i], mode="lines", line=dict(width=3, color="#1f77b4")),
                go.Scatter(x=[x_mean[i], x_mean[i]], y=[0, ymax], mode="lines", line=dict(dash="dash", color="#d62728")),
                go.Scatter(x=[x_class[i]], y=[0.08 * ymax], mode="markers", marker=dict(size=9, color="#2ca02c")),
                go.Scatter(x=[tnorm[i]], y=[x_mean[i]], mode="markers", marker=dict(size=9, color="#1f77b4")),
                go.Scatter(x=[tnorm[i]], y=[x_class[i]], mode="markers", marker=dict(size=9, symbol="diamond", color="#2ca02c")),
                go.Scatter(x=[tnorm[i]], y=[autocorr[i]], mode="markers", marker=dict(size=9, color="#9467bd")),
            ],
            traces=[0, 1, 2, 5, 6, 8],
            layout=go.Layout(title_text=f"{tr(lang, 'video_section_single')} — t/T_rev = {tnorm[i]:.3f}"),
        ))
        slider_steps.append({
            "label": f"{tnorm[i]:.2f}",
            "method": "animate",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        })

    fig.frames = frames
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="density" if lang == "English" else "hustota", range=[0, ymax], row=1, col=1)
    fig.update_xaxes(title_text=r"t / T$_{rev}$", row=1, col=2)
    fig.update_yaxes(title_text="position" if lang == "English" else "poloha", range=[ymin_h, ymax_h], row=1, col=2)
    fig.update_xaxes(title_text=r"t / T$_{rev}$", row=1, col=3)
    fig.update_yaxes(title_text="overlap" if lang == "English" else "překryv", range=[0, 1.05], row=1, col=3)
    fig.update_layout(height=500, template="plotly_white", title_text=f"{tr(lang, 'video_section_single')} — t/T_rev = {tnorm[idxs[0]]:.3f}", updatemenus=_animation_controls(frame_duration_ms=frame_duration_ms, redraw=False), sliders=_animation_slider(slider_steps))
    return fig


def make_double_animation(dw, opt, lang: str, frame_duration_ms: int = 220):
    if not HAS_PLOTLY:
        return None
    q_step = max(1, len(dw["times"]) // 50)
    q_idx = np.arange(0, len(dw["times"]), q_step, dtype=int)
    o_idx = np.linspace(0, len(opt["z_vals"]) - 1, len(q_idx)).astype(int)

    qmax = float(1.05 * np.max(dw["dens"]))
    omax = float(1.05 * np.max(opt["I_opt"]))
    barrier_edge = dw["center"] - dw["barrier_width"] / 2.0

    fig = make_subplots(rows=1, cols=3, subplot_titles=(tr(lang, "classical"), tr(lang, "quantum"), tr(lang, "optical_label")))
    fig.add_trace(go.Scatter(x=[dw["x_class"][q_idx[0]]], y=[0.45], mode="markers", name=tr(lang, "classical_particle"), marker=dict(size=12)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dw["x"], y=dw["dens"][:, q_idx[0]], mode="lines", name=tr(lang, "quantum_density"), line=dict(width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=dw["x"], y=(dw["V"] / max(np.max(dw["V"]), 1e-9)) * (0.8 * qmax), mode="lines", name="barrier", line=dict(dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[dw["x_mean"][q_idx[0]], dw["x_mean"][q_idx[0]]], y=[0, qmax], mode="lines", name=tr(lang, "mean_position"), line=dict(dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=opt["x"], y=opt["I_opt"][:, o_idx[0]], mode="lines", name=tr(lang, "optical_intensity"), line=dict(width=3)), row=1, col=3)
    fig.add_trace(go.Scatter(x=opt["x"], y=120 * (opt["n_profile"] - opt["n_clad"]), mode="lines", name="index", line=dict(dash="dash")), row=1, col=3)
    fig.add_trace(go.Scatter(x=[opt["x_mean"][o_idx[0]], opt["x_mean"][o_idx[0]]], y=[0, omax], mode="lines", name=tr(lang, "mean_position"), line=dict(dash="dot")), row=1, col=3)

    frames = []
    slider_steps = []
    for iq, io in zip(q_idx, o_idx):
        frames.append(go.Frame(
            name=str(iq),
            data=[
                go.Scatter(x=[dw["x_class"][iq]], y=[0.45]),
                go.Scatter(x=dw["x"], y=dw["dens"][:, iq]),
                go.Scatter(x=dw["x"], y=(dw["V"] / max(np.max(dw["V"]), 1e-9)) * (0.8 * qmax)),
                go.Scatter(x=[dw["x_mean"][iq], dw["x_mean"][iq]], y=[0, qmax]),
                go.Scatter(x=opt["x"], y=opt["I_opt"][:, io]),
                go.Scatter(x=opt["x"], y=120 * (opt["n_profile"] - opt["n_clad"])),
                go.Scatter(x=[opt["x_mean"][io], opt["x_mean"][io]], y=[0, omax]),
            ],
            traces=[0, 1, 2, 3, 4, 5, 6],
            layout=go.Layout(title_text=f"{tr(lang, 'video_section_double')} — t/T = {dw['times'][iq] / dw['T_tunnel']:.2f}, z/Lc = {opt['z_vals'][io] / opt['L_couple']:.2f}"),
        ))
        slider_steps.append({
            "label": f"{dw['times'][iq] / dw['T_tunnel']:.2f}",
            "method": "animate",
            "args": [[str(iq)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        })

    fig.frames = frames
    fig.add_vline(x=0.0, line_width=1.5, line_color="black", row=1, col=1)
    fig.add_vline(x=barrier_edge, line_width=1.5, line_color="black", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.05, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="density" if lang == "English" else "hustota", range=[0, qmax], row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)
    fig.update_yaxes(title_text="intensity" if lang == "English" else "intenzita", range=[0, omax], row=1, col=3)
    fig.update_layout(height=460, title_text=f"{tr(lang, 'video_section_double')} — t/T = {dw['times'][q_idx[0]] / dw['T_tunnel']:.2f}, z/Lc = {opt['z_vals'][o_idx[0]] / opt['L_couple']:.2f}", updatemenus=_animation_controls(frame_duration_ms=frame_duration_ms, redraw=False), sliders=_animation_slider(slider_steps), showlegend=False)
    return fig


def make_optical_animation(opt, lang: str, frame_duration_ms: int = 260):
    if not HAS_PLOTLY:
        return None
    idxs = np.arange(0, len(opt["z_vals"]), max(1, len(opt["z_vals"]) // 70), dtype=int)
    if idxs[-1] != len(opt["z_vals"]) - 1:
        idxs = np.append(idxs, len(opt["z_vals"]) - 1)
    omax = float(1.05 * np.max(opt["I_opt"]))
    z_norm = opt["z_vals"] / opt["L_couple"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(tr(lang, "optical_intensity"), tr(lang, "optical_snapshot_title")))
    fig.add_trace(go.Scatter(x=opt["x"], y=opt["I_opt"][:, idxs[0]], mode="lines", name=tr(lang, "optical_intensity"), line=dict(width=3, color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=opt["x"], y=120 * (opt["n_profile"] - opt["n_clad"]), mode="lines", name="index", line=dict(dash="dash", color="#555555")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[opt["x_mean"][idxs[0]], opt["x_mean"][idxs[0]]], y=[0, omax], mode="lines", name=tr(lang, "mean_position"), line=dict(dash="dot", color="#d62728")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=opt["I_opt"], x=z_norm, y=opt["x"], colorscale="Viridis", showscale=True, colorbar=dict(title="I"), zmin=float(np.min(opt["I_opt"])), zmax=float(np.max(opt["I_opt"]))), row=1, col=2)
    fig.add_trace(go.Scatter(x=[z_norm[idxs[0]], z_norm[idxs[0]]], y=[opt["x"][0], opt["x"][-1]], mode="lines", name="cursor", line=dict(color="#ffdddd", dash="dot", width=3)), row=1, col=2)

    frames = []
    slider_steps = []
    for i in idxs:
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=opt["x"], y=opt["I_opt"][:, i], mode="lines", line=dict(width=3, color="#1f77b4")),
                go.Scatter(x=[opt["x_mean"][i], opt["x_mean"][i]], y=[0, omax], mode="lines", line=dict(dash="dot", color="#d62728")),
                go.Scatter(x=[z_norm[i], z_norm[i]], y=[opt["x"][0], opt["x"][-1]], mode="lines", line=dict(color="#ffdddd", dash="dot", width=3)),
            ],
            traces=[0, 2, 4],
            layout=go.Layout(title_text=f"{tr(lang, 'video_section_optical')} — z/Lc = {z_norm[i]:.2f}"),
        ))
        slider_steps.append({
            "label": f"{z_norm[i]:.2f}",
            "method": "animate",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        })

    fig.frames = frames
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="intensity" if lang == "English" else "intenzita", range=[0, omax], row=1, col=1)
    fig.update_xaxes(title_text=r"z / L$_c$", row=1, col=2)
    fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_layout(height=500, template="plotly_white", title_text=f"{tr(lang, 'video_section_optical')} — z/Lc = {z_norm[idxs[0]]:.2f}", updatemenus=_animation_controls(frame_duration_ms=frame_duration_ms, redraw=False), sliders=_animation_slider(slider_steps), showlegend=False)
    return fig


# ============================================================
# GIF generation
# ============================================================


def _save_animation(anim: FuncAnimation) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
        path = tmp.name
    try:
        anim.save(path, writer=PillowWriter(fps=12))
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    plt.close("all")
    return data


@st.cache_data(show_spinner=False)
def make_single_gif(x, dens, x_mean, x_class, times, lang):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor("white")
    line_dens, = axes[0].plot([], [], lw=2.2)
    line_mean = axes[0].axvline(x_mean[0], ls="--", lw=1.8)
    point_class, = axes[0].plot([], [], "o", ms=7)
    axes[0].set_xlim(x[0], x[-1])
    axes[0].set_ylim(0, 1.05 * np.max(dens))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density" if lang == "English" else "hustota")

    line_qhist, = axes[1].plot(times, x_mean, lw=2.2, label=tr(lang, "mean_position"))
    line_chist, = axes[1].plot(times, x_class, "--", lw=2.0, label=tr(lang, "classical_particle"))
    cursor = axes[1].axvline(times[0], color="k", ls=":", lw=1.6)
    axes[1].set_xlim(times[0], times[-1])
    axes[1].set_ylim(min(np.min(x_mean), np.min(x_class)) - 0.05, max(np.max(x_mean), np.max(x_class)) + 0.05)
    axes[1].set_xlabel("time t" if lang == "English" else "čas t")
    axes[1].set_ylabel("position" if lang == "English" else "poloha")
    axes[1].legend(fontsize=9)

    def update(i):
        line_dens.set_data(x, dens[:, i])
        line_mean.set_xdata([x_mean[i], x_mean[i]])
        point_class.set_data([x_class[i]], [0.08 * np.max(dens)])
        axes[0].set_title(f"t = {times[i]:.4f}")
        cursor.set_xdata([times[i], times[i]])
        return line_dens, line_mean, point_class, cursor

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(times), max(1, len(times)//90)), interval=80)
    return _save_animation(anim)


@st.cache_data(show_spinner=False)
def make_double_compare_gif(dw, opt, lang):
    q_idx = np.arange(0, len(dw["times"]), max(1, len(dw["times"]) // 90))
    o_idx = np.linspace(0, len(opt["z_vals"]) - 1, len(q_idx)).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    fig.patch.set_facecolor("white")

    # Classical
    axes[0].set_xlim(dw["x"][0], dw["x"][-1])
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].axvline(0.0, color="k", lw=1.5)
    axes[0].axvline(dw["center"] - dw["barrier_width"] / 2.0, color="k", lw=1.5)
    point_c, = axes[0].plot([], [], "o", ms=8)
    axes[0].set_title(tr(lang, "classical"))
    axes[0].set_xlabel("x")
    axes[0].set_yticks([])

    # Quantum
    dens_line, = axes[1].plot([], [], lw=2.1)
    barrier_line, = axes[1].plot(dw["x"], (dw["V"] / max(np.max(dw["V"]), 1e-9)) * (0.8 * np.max(dw["dens"])), "k--", lw=1.5)
    axes[1].set_xlim(dw["x"][0], dw["x"][-1])
    axes[1].set_ylim(0, 1.05 * np.max(dw["dens"]))
    axes[1].set_title(tr(lang, "quantum"))
    axes[1].set_xlabel("x")

    # Optical
    opt_line, = axes[2].plot([], [], lw=2.1)
    idx_line, = axes[2].plot(opt["x"], 120 * (opt["n_profile"] - opt["n_clad"]), "k--", lw=1.5)
    axes[2].set_xlim(opt["x"][0], opt["x"][-1])
    axes[2].set_ylim(0, 1.05 * np.max(opt["I_opt"]))
    axes[2].set_title(tr(lang, "optical_label"))
    axes[2].set_xlabel("x")

    def update(j):
        iq = q_idx[j]
        io = o_idx[j]
        point_c.set_data([dw["x_class"][iq]], [0.45])
        dens_line.set_data(dw["x"], dw["dens"][:, iq])
        opt_line.set_data(opt["x"], opt["I_opt"][:, io])
        fig.suptitle(
            f"t/T = {dw['times'][iq] / dw['T_tunnel']:.2f}    |    z/Lc = {opt['z_vals'][io] / opt['L_couple']:.2f}",
            fontsize=12,
        )
        return point_c, dens_line, opt_line, barrier_line, idx_line

    anim = FuncAnimation(fig, update, frames=len(q_idx), interval=85)
    return _save_animation(anim)


@st.cache_data(show_spinner=False)
def make_optical_gif(opt, lang):
    idxs = np.arange(0, len(opt["z_vals"]), max(1, len(opt["z_vals"]) // 90))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    fig.patch.set_facecolor("white")

    line_slice, = axes[0].plot([], [], lw=2.1)
    line_mean = axes[0].axvline(opt["x_mean"][0], ls="--", lw=1.6)
    axes[0].plot(opt["x"], 120 * (opt["n_profile"] - opt["n_clad"]), "k--", lw=1.4)
    axes[0].set_xlim(opt["x"][0], opt["x"][-1])
    axes[0].set_ylim(0, 1.05 * np.max(opt["I_opt"]))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("intensity" if lang == "English" else "intenzita")

    extent = [opt["z_vals"][0] / opt["L_couple"], opt["z_vals"][-1] / opt["L_couple"], opt["x"][0], opt["x"][-1]]
    axes[1].imshow(opt["I_opt"], origin="lower", aspect="auto", extent=extent)
    cursor = axes[1].axvline(0.0, color="w", ls=":", lw=1.6)
    axes[1].set_xlabel(r"z / L$_c$")
    axes[1].set_ylabel("x")

    def update(j):
        i = idxs[j]
        line_slice.set_data(opt["x"], opt["I_opt"][:, i])
        line_mean.set_xdata([opt["x_mean"][i], opt["x_mean"][i]])
        axes[0].set_title(f"z/Lc = {opt['z_vals'][i] / opt['L_couple']:.2f}")
        cursor.set_xdata([opt["z_vals"][i] / opt["L_couple"], opt["z_vals"][i] / opt["L_couple"]])
        return line_slice, line_mean, cursor

    anim = FuncAnimation(fig, update, frames=len(idxs), interval=85)
    return _save_animation(anim)


# ============================================================
# UI
# ============================================================

with st.sidebar:
    lang = st.selectbox("Language / Jazyk", ["English", "Czech"], index=0)
    st.markdown("---")
    section = st.radio(
        tr(lang, "section"),
        [tr(lang, "theory"), tr(lang, "single"), tr(lang, "double"), tr(lang, "optical"), tr(lang, "finite")],
    )
    st.markdown("---")
    st.markdown(f"**{tr(lang, 'about')}**")
    st.caption(tr(lang, "notes_text"))

st.markdown(
    f"""
    <div class="hero">
        <h1 style="margin:0;">{tr(lang, 'app_title')}</h1>
        <div class="small-note">{tr(lang, 'app_subtitle')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="card">
        <b>{tr(lang, 'how_to_title')}</b><br>
        <span class="small-note">{tr(lang, 'how_to_text')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if section == tr(lang, "theory"):
    st.markdown(f"<div class='card'>{tr(lang, 'theory_card')}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.subheader(tr(lang, "theory_title"))
        st.markdown(tr(lang, "theory_text"))
    with c2:
        st.subheader(tr(lang, "references"))
        st.markdown(tr(lang, "refs_text"))

elif section == tr(lang, "single"):
    st.write(tr(lang, "single_intro"))
    c1, c2, c3 = st.columns(3)
    with c1:
        L = st.slider(tr(lang, "well_width"), 0.6, 2.0, 1.0, 0.05)
        N = st.slider(tr(lang, "grid"), 140, 320, 220, 20)
        n_show = st.slider(tr(lang, "show_states"), 4, min(24, N), min(10, N), 1)
    with c2:
        state_n = st.slider(tr(lang, "chosen_state"), 1, min(180, N), 1, 1)
        x0 = st.slider(tr(lang, "packet_center"), 0.05, float(L - 0.05), min(0.22, float(L - 0.08)), 0.01)
        sigma = st.slider(tr(lang, "packet_sigma"), 0.02, 0.18, 0.06, 0.01)
    with c3:
        k0 = st.slider(tr(lang, "packet_k0"), 2.0, 35.0, 24.0, 1.0)
        t_factor = st.slider(tr(lang, "time_factor"), 0.20, 2.20, 1.10, 0.01)
        n_basis = st.slider(tr(lang, "basis_count"), 40, min(180, N), min(120, N), 10)

    static_data = compute_single_well(L=L, N=N, n_show=n_show)
    sim = compute_single_dynamics(L=L, N=N, x0=x0, sigma=sigma, k0=k0, n_basis=n_basis, t_factor=t_factor)
    idx = st.slider(tr(lang, "snapshot"), 0, len(sim["times"]) - 1, len(sim["times"]) // 3, 1)

    tabs = st.tabs([tr(lang, "tab_static"), tr(lang, "tab_snapshot"), tr(lang, "tab_video")])
    with tabs[0]:
        st.pyplot(plot_single_stationary(static_data, L, n_show, lang), use_container_width=True)
        st.caption(tr(lang, "all_states_note"))
        st.pyplot(plot_single_probability(static_data, L, state_n, lang), use_container_width=True)
    with tabs[1]:
        st.pyplot(plot_single_snapshot(sim, idx, lang), use_container_width=True)
        cmet1, cmet2 = st.columns(2)
        cmet1.metric(tr(lang, "revival_time"), f"{sim['T_rev']:.4f}")
        cmet2.metric(tr(lang, "revival_strength"), f"{np.max(sim['autocorr']):.4f}")
        st.info(tr(lang, "revival_note"))
    with tabs[2]:
        st.markdown(f"**{tr(lang, 'video_section_single')}**")
        speed_single = st.slider(tr(lang, "anim_speed"), 100, 520, 240, 10, key="speed_single")
        st.caption(tr(lang, "anim_smoother"))
        st.caption(tr(lang, "play_note"))
        anim_fig = make_single_animation(sim, lang, frame_duration_ms=speed_single)
        if anim_fig is not None:
            st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info(tr(lang, "no_plotly"))
        st.caption(tr(lang, "gif_note"))
        if st.button(tr(lang, "generate_video"), key="single_gif"):
            gif = make_single_gif(sim["x"], sim["dens"], sim["x_mean"], sim["x_class"], sim["times"], lang)
            st.markdown(f'<img src="data:image/gif;base64,{__import__("base64").b64encode(gif).decode()}" width="100%" />', unsafe_allow_html=True)
            st.download_button(tr(lang, "download_video"), gif, file_name="single_well_wavepacket.gif", mime="image/gif")
elif section == tr(lang, "double"):
    st.write(tr(lang, "double_intro"))
    c1, c2, c3 = st.columns(3)
    with c1:
        barrier_width = st.slider(tr(lang, "barrier_width"), 0.04, 0.30, 0.12, 0.01)
        V0 = st.slider(tr(lang, "barrier_height"), 20.0, 140.0, 80.0, 5.0)
    with c2:
        N2 = st.slider(tr(lang, "grid"), 160, 340, 260, 20)
        n_basis = st.slider(tr(lang, "basis_count"), 10, 80, 50, 5, key="dw_basis")
    with c3:
        dn_core = st.slider(tr(lang, "dn_core"), 0.004, 0.030, 0.012, 0.001)
        n_clad = st.slider(tr(lang, "n_clad"), 1.30, 1.60, 1.45, 0.01)

    dw = compute_double_well(barrier_width=barrier_width, V0=V0, N2=N2, n_basis=n_basis)
    opt = compute_optical_analogy(dn_core=dn_core, n_clad=n_clad, N2=N2)
    idx_q = st.slider(tr(lang, "snapshot") + " (quantum)", 0, len(dw["times"]) - 1, len(dw["times"]) // 4, 1)
    idx_o = st.slider(tr(lang, "snapshot") + " (optical)", 0, len(opt["z_vals"]) - 1, len(opt["z_vals"]) // 4, 1)

    m1, m2, m3 = st.columns(3)
    m1.metric(r"$\Delta E = E_1 - E_0$", f"{dw['dE']:.5f}")
    m2.metric(r"$T_{tunnel}$", f"{dw['T_tunnel']:.5f}")
    m3.metric(r"$L_c$", f"{opt['L_couple']:.5f}")

    tabs = st.tabs([tr(lang, "tab_static"), tr(lang, "tab_snapshot"), tr(lang, "tab_video")])
    with tabs[0]:
        st.pyplot(plot_double_stationary(dw, lang), use_container_width=True)
        widths = np.linspace(0.04, 0.30, 12)
        E0, E1, dE, Tt = compute_barrier_scan(widths, V0=V0)
        st.pyplot(plot_barrier_scan(widths, E0, E1, dE, Tt, lang), use_container_width=True)
        st.info(tr(lang, "scan_note"))
    with tabs[1]:
        st.pyplot(plot_double_snapshot(dw, opt, idx_q, idx_o, lang), use_container_width=True)
    with tabs[2]:
        st.markdown(f"**{tr(lang, 'video_section_double')}**")
        speed_double = st.slider(tr(lang, "anim_speed"), 80, 420, 220, 10, key="speed_double")
        st.caption(tr(lang, "anim_smoother"))
        st.caption(tr(lang, "play_note"))
        anim_fig = make_double_animation(dw, opt, lang, frame_duration_ms=speed_double)
        if anim_fig is not None:
            st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info(tr(lang, "no_plotly"))
        st.caption(tr(lang, "gif_note"))
        if st.button(tr(lang, "generate_video"), key="double_gif"):
            gif = make_double_compare_gif(dw, opt, lang)
            st.markdown(f'<img src="data:image/gif;base64,{__import__("base64").b64encode(gif).decode()}" width="100%" />', unsafe_allow_html=True)
            st.download_button(tr(lang, "download_video"), gif, file_name="double_well_compare.gif", mime="image/gif")
elif section == tr(lang, "optical"):
    st.write(tr(lang, "optical_intro"))
    c1, c2 = st.columns(2)
    with c1:
        N2 = st.slider(tr(lang, "grid"), 160, 340, 260, 20, key="opt_grid")
        dn_core = st.slider(tr(lang, "dn_core"), 0.004, 0.030, 0.012, 0.001, key="opt_dn")
    with c2:
        n_clad = st.slider(tr(lang, "n_clad"), 1.30, 1.60, 1.45, 0.01, key="opt_clad")
        n_basis = st.slider(tr(lang, "basis_count"), 8, 40, 24, 2, key="opt_basis")

    opt = compute_optical_analogy(dn_core=dn_core, n_clad=n_clad, N2=N2, n_basis=n_basis)
    idx_o = st.slider(tr(lang, "snapshot"), 0, len(opt["z_vals"]) - 1, len(opt["z_vals"]) // 4, 1, key="opt_snap")
    st.metric(r"$\Delta \beta = \beta_1-\beta_0$", f"{opt['d_beta']:.5f}")
    st.metric(r"$L_c$", f"{opt['L_couple']:.5f}")

    tabs = st.tabs([tr(lang, "tab_snapshot"), tr(lang, "tab_video")])
    with tabs[0]:
        fake_dw = {
            "x": opt["x"], "dens": opt["I_opt"], "x_mean": opt["x_mean"],
            "x_class": np.full_like(opt["x_mean"], np.nan), "times": opt["z_vals"],
            "T_tunnel": opt["L_couple"], "P_left": opt["P_left"], "P_right": opt["P_right"],
            "V": np.zeros_like(opt["x"]), "center": opt["center"], "barrier_width": 0.0,
        }
        st.pyplot(plot_double_snapshot(fake_dw, opt, idx_o, idx_o, lang), use_container_width=True)
        st.info(tr(lang, "optical_note"))
    with tabs[1]:
        st.markdown(f"**{tr(lang, 'video_section_optical')}**")
        speed_opt = st.slider(tr(lang, "anim_speed"), 100, 520, 280, 10, key="speed_opt")
        st.caption(tr(lang, "anim_smoother"))
        st.caption(tr(lang, "play_note"))
        anim_fig = make_optical_animation(opt, lang, frame_duration_ms=speed_opt)
        if anim_fig is not None:
            st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info(tr(lang, "no_plotly"))
        st.caption(tr(lang, "gif_note"))
        if st.button(tr(lang, "generate_video"), key="opt_gif"):
            gif = make_optical_gif(opt, lang)
            st.markdown(f'<img src="data:image/gif;base64,{__import__("base64").b64encode(gif).decode()}" width="100%" />', unsafe_allow_html=True)
            st.download_button(tr(lang, "download_video"), gif, file_name="optical_propagation.gif", mime="image/gif")
elif section == tr(lang, "finite"):
    st.write(tr(lang, "finite_intro"))
    c1, c2 = st.columns(2)
    with c1:
        well_width = st.slider(tr(lang, "well_width"), 0.30, 1.00, 0.60, 0.02)
        V_barrier = st.slider(tr(lang, "barrier_height"), 20.0, 180.0, 120.0, 5.0, key="finite_barrier")
    with c2:
        L_f = st.slider(tr(lang, "total_box"), 1.0, 2.5, 1.8, 0.1)
        N_f = st.slider(tr(lang, "grid"), 180, 460, 360, 20, key="finite_grid")

    fw = compute_finite_well(well_width=well_width, V_barrier=V_barrier, L_f=L_f, N_f=N_f)
    st.metric(tr(lang, "bound_states"), str(len(fw["bound_idx"])))
    st.pyplot(plot_finite_well(fw, lang), use_container_width=True)

st.markdown("---")
st.caption(tr(lang, "footer"))
