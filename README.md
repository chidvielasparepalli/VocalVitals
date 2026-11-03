# VocalVitals

VocalVitals is a research prototype for detecting annotated swallow events from cervical-auscultation recordings.

## Judge dashboard

The browser demo is in **`vocalvitals/vercel-demo/`**. It displays selected, real processed signal windows, their source recording metadata, the 4-channel waveform, a swallow / non-swallow prediction, and the saved subject-wise validation metrics.

The dashboard runs the exported 1D CNN in the browser through ONNX Runtime Web. It does not need the Python training environment to show the selected examples.

### Run locally

From `vocalvitals/vercel-demo`, serve the static files with any local static server, for example:

```powershell
py -m http.server 8080
```

Then open `http://localhost:8080`.

The model is served by the current public demo deployment. To make a standalone copy, download the ONNX file from the model URL configured in `app.js` and place it beside `index.html`.

## Data and model

The original source recordings are four-channel `.npy` files accompanied by CSV annotation files. The preprocessing pipeline extracts 200 ms windows at 16 kHz:

- `swallow`: windows centered on annotated swallow intervals.
- `non_swallow`: windows sampled outside annotated swallow intervals.

The saved binary model classifies these two labels only. The supplied dataset does not contain clinical aspiration labels. **This project does not diagnose aspiration or establish clinical safety.**

The dashboard contains six selected examples for demonstration, not the full source dataset.

## Reported experimental metrics

The saved validation report records a subject-wise holdout of 508 windows:

- Accuracy: 99.41%
- Swallow recall: 100.00%
- Non-swallow specificity: 98.55%
- F1 (swallow): 99.50%

These are internal experimental results. The best checkpoint was selected using the same validation set, so the figures are not an independent final test estimate.

## Repository scope

This initial GitHub publication restores the public-facing judge dashboard and its demo data. The original complete raw dataset is not included because it is large. The full training/preprocessing source and reproducibility materials should be added from the local project checkout.

## Live demo

https://vercel-demo-gqf7mx8p9-chidvigamer-5480s-projects.vercel.app/
