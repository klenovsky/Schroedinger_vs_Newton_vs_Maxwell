# Quantum Well Streamlit App

Interactive educational Streamlit app comparing:
- classical motion in a 1D box,
- quantum wave-packet dynamics in single and double wells,
- optical paraxial propagation as an analogue of tunnelling,
- finite well bound states.

## New in this version
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
