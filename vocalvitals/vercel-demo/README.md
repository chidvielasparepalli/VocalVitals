# VocalVitals Browser Judge Demo

Static browser dashboard for the VocalVitals swallow-event prototype.

## Run locally

From this folder:

```powershell
.\download-model.ps1
py -m http.server 8080
```

Open http://localhost:8080. The page runs the exported 4-channel 1D CNN through ONNX Runtime Web. Internet access is required to load ONNX Runtime Web from jsDelivr.

## Included data

`demo-data.json` contains six selected examples from the supplied processed cervical-auscultation dataset, with annotation metadata and waveform values. It is not the full raw dataset.

## Model scope

The CNN predicts `Swallow` versus `NonSwallow`. It does not identify aspiration, delayed swallowing, or clinical risk. The reported validation metrics are internal subject-wise validation estimates, not independent clinical validation.

The ONNX file is downloaded from the current public Vercel demo deployment via `download-model.ps1` rather than checked into this initial dashboard-only publication.
