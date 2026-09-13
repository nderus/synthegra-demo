# Synthegra demo

Interactive, precomputed results from [Synthegra](https://github.com/nderus/synthegra), a synthetic
benchmark for multimodal survival integration with known information structure.

Live: https://nderus.github.io/synthegra-demo/

Three stories, one question each:

1. **When does fusion help?** Sliders for planted synergy, shared signal and cohort size; held-out
   C-index for the best single layer and early / intermediate / late fusion, with a live
   redundancy / uniqueness / synergy meter.
2. **Can you measure synergy at your cohort size?** Measured synergy across seeds versus cohort size,
   with a marker for your own n.
3. **Does synthetic survival look real?** Kaplan-Meier calibration of the generator against a real
   MDS/AML cohort.

## Design

- Static site: one `index.html`, Plotly from a pinned CDN, two small JSON files in `data/`. No Python
  in the browser, loads in well under a second on a phone.
- Every number is produced offline by `tools/export_demo_data.py` from the Synthegra generator and
  model fits. The JSON carries the Synthegra commit, config and seeds; the page shows them under
  "How these numbers were computed".

## Regenerating the data

```bash
conda activate synthegra            # env from ../synthegra/env.yml
python tools/export_demo_data.py    # writes data/grid.json and data/resolution.json
```

## Local preview

```bash
python3 -m http.server 8787         # then open http://localhost:8787/
```

`demo.py` is the previous Streamlit version, kept for reference; it is no longer what the site serves.
