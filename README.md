# Quantum Well Streamlit App

Interactive educational Streamlit app comparing:
- classical motion in a 1D box,
- quantum wave-packet dynamics in a single well,
- tunnelling in a double well together with its optical analogue,
- finite-well bound states and evanescent tails.

## App structure
- **Single well**: stationary states, probability densities, wave-packet motion, and revivals.
- **Double well**: classical picture, quantum tunnelling, and optical coupled-waveguide analogue in one place.
- **Finite well**: all numerically resolved bound states below the barrier.
- **Theory and references**: available in a collapsible panel above the simulations.

## New in this version
- more compact teaching-oriented layout,
- no standalone optical section, because the optical analogue is integrated into the double-well part,
- collapsible theory block above the simulations,
- slower default animation playback,
- in-app animation speed sliders,
- reduced optical flicker by keeping the 2D intensity map static and animating only the cursor and 1D slice,
- vectorized numerical core based on NumPy and `scipy.linalg.eigh_tridiagonal`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
Push this repository to GitHub and choose `app.py` as the main file.
