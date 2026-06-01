# Quantum wells, tunnelling, and optical analogy — Streamlit app

A bilingual Streamlit teaching app for introductory university physics.

Default language: **English**  
Alternative language: **Czech**

## Included topics

- single infinite quantum well,
- time evolution of a quantum wave packet and comparison with a classical particle,
- double well and tunnelling,
- optical analogy with two coupled waveguides,
- finite well with evanescent tails,
- optional animated GIFs for the dynamical sections,
- optional 3D surfaces for the `x-t` and `x-z` evolution.

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files from this folder.
3. Open Streamlit Community Cloud.
4. Select the repository.
5. Set **Main file path** to `app.py`.
6. Deploy.

## Notes

- The app is written so that it can be uploaded to GitHub directly.
- Numerics are cached and vectorized where possible.
- Tridiagonal Hamiltonians are diagonalized efficiently with `scipy.linalg.eigh_tridiagonal`.
- GIF animations are generated on demand and cached by Streamlit.

## Česká poznámka

Aplikace je připravena přímo pro GitHub a následné nasazení do Streamlit Community Cloud. Výchozí jazyk je angličtina, ale v levém panelu lze přepnout do češtiny.


## Update in this version

- real interactive Plotly animations with Play/Pause controls inside the app
- GIF export kept as an optional fallback/download
