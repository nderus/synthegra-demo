# Synthegra demo

Interactive, precomputed results from [Synthegra](https://github.com/nderus/synthegra), a synthetic
benchmark for multimodal survival integration with known information structure.

Live: https://nderus.github.io/synthegra-demo/

Three stories, one question each:

1. **When does fusion help?** Sliders for planted synergy, shared signal and cohort size; held-out
   C-index for the best single layer, late fusion (vote), early linear and early non-linear fusion,
   with the oracle ceiling and the exact planted structure.
2. **Can you measure synergy at your cohort size?** The same planted synergy at every cohort size, read
   by two instruments: the PID estimate (paper's resolution sweep) and the C-index gain of early linear,
   early non-linear and the best neural intermediate fusion over the best single layer.
3. **Does synthetic survival look real?** Kaplan-Meier calibration of the generator against a real
   MDS/AML cohort.

## Design

- Static site: one `index.html`, Plotly from a pinned CDN, three small JSON files in `data/`. No Python
  in the browser, loads in well under a second on a phone.
- Every number is produced offline from the Synthegra generator and model fits:
  `tools/export_demo_data.py` (tab 1 grid, tab 2 PID read-out) and `tools/export_resolution_models.py`
  (tab 2 model read-out). The JSON carries the Synthegra commit, config and seeds; the page shows them
  under "How these numbers were computed".

## Regenerating the data

```bash
conda activate synthegra            # env from ../synthegra/env.yml
OMP_NUM_THREADS=1 python tools/export_demo_data.py --procs 4          # data/grid.json, data/resolution.json (~35 min)
OMP_NUM_THREADS=1 python tools/export_resolution_models.py --procs 4  # data/resolution_models.json (~45 min)
```

## Local preview

```bash
python3 -m http.server 8787         # then open http://localhost:8787/
```

`demo.py` is the previous Streamlit version, kept for reference; it is no longer what the site serves.
