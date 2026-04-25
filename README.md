# Project Oreon — Covert Activity Detection in Camouflaged Terrain

**Team:** Sahasra · Laranya · Sudha  
**Model:** TerraMind-small TiM (Thinking-in-Modalities) — SAR + Optical Fusion

---

## 1. The Customer

Intelligence analysts and border-security operators who need to detect camouflaged activity (vehicles, installations, personnel) in forested or desert terrain where optical imagery alone fails. The mission constraint: **downlink bandwidth is scarce** — a satellite cannot stream 15 bands of raw raster data to ground. The system must reason on the edge and transmit only the answer.

---

## 2. The Model

We fine-tune **TerraMind-small** (IBM's geospatial foundation model, pre-trained on 9 EO modalities) using its **TiM (Thinking-in-Modalities)** capability:

- **Inputs:** Sentinel-2 optical (13 bands, S2L1C) + Sentinel-1 SAR (2 bands, S1GRD)
- **Architecture:** EncoderDecoder with TerraMind backbone → UNetDecoder (256→128→64→32 channels) → 2-class segmentation head
- **Key idea:** TiM lets the model *synthesize* what SAR *should* look like given the optical view. We compare synthesized SAR vs. actual SAR — the inconsistency reveals camouflaged objects that absorb or scatter radar differently from their optical appearance.
- **Training:** Sen1Floods11 dataset, AdamW + ReduceLROnPlateau, 15 epochs max, early stopping on val/mIoU, mixed precision (fp16)

**Anomaly pipeline:**
```
S2 optical → TerraMind TiM → Synthesized S1 SAR
                                      ↓
            Actual S1 SAR → Delta map (Gaussian smoothed) → Binary mask + Confidence score
```

Dataset: Sen1Floods11 v1.1 — https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS
---

## 3. The Evidence
Run the Gradio demo (entry point: `oreon.ipynb`, Section 15):
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute oreon.ipynb
```
Or run the last cell directly — it launches a Gradio UI where you upload a Sentinel-2 tile and its paired Sentinel-1 tile, and get back: synthesized SAR, anomaly heatmap, and an orbital insight report.

**What the demo shows:** Upload any S2+S1 tile pair from `sample_input/`. The model synthesizes SAR from the optical view, computes the delta, and outputs a per-pixel anomaly heatmap with confidence score and bandwidth savings report.

---

## 4. The Numbers

| Metric | Baseline (S2 only) | Multimodal (S2 + S1) |
|---|---|---|
| Test mIoU | 0.4218 | **0.8645** |
| Relative improvement | — | **+104.96%** |
| Model size | — | 1,376 MB *(hosted externally — see link below)* |
| Inference latency (GPU) | — | 34.75 ms/batch |
| Bandwidth saved | — | **98.3%** vs raw downlink |
| Jetson estimate | — | ~56 ms/batch |

> ⚠️ **Model weights (1,376 MB) exceed the 200 MB repo limit.** https://drive.google.com/file/d/1N6WjKRP5ObLADJohPbQ_NF3lLQnr4ONz/view?usp=share_link

**Bandwidth math:** Raw downlink = 15 bands × 4 bytes/pixel × H × W. Edge answer = binary mask (1 byte/pixel) + confidence scalar (4 bytes). For a 256×256 tile: raw ≈ 3,840 KB vs answer ≈ 65 KB → **98.3% reduction**.

---

## 5. Limits & What's Next

**Current limits:**
- Trained on flood-domain data (Sen1Floods11); anomaly score is a proxy from delta maps, not a true camouflage-specific label
- Threshold (0.25) is manually tuned — requires calibration per terrain type
- No temporal fusion (single-pass, not change-detection across revisits)
- SAR synthesis is approximate for camouflage scenarios without labeled camouflage ground truth

**What's next:**
- Collect or synthesize camouflage-specific labels and fine-tune the anomaly head
- Add temporal delta (compare T and T-Δt SAR frames) for moving-target indication
- Quantize to INT8 for real Jetson deployment
- Integrate with actual downlink scheduler to demonstrate end-to-end bandwidth budget

---

*Weights hosted externally if >200 MB — link TBD after training. Dataset not included in repo — see Google Drive link above.*
