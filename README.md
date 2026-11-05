# Automated Lung Disease Classification from Chest X-rays
DenseNet121 (transfer learning) + Squeeze-and-Excitation attention, CLAHE preprocessing,
focal loss + class weights, two-stage fine-tuning, held-out test evaluation and Grad-CAM.

**Student:** Syed Afeezu Raheman A (2403310914821045) | JAL2511 Machine Learning Laboratory

## What is in this folder
| File | Purpose |
|---|---|
| `lung_common.py` | Data pipeline, split, model, focal loss, Grad-CAM (shared code) |
| `train.py` | Two-stage training -> saves `outputs/lung_disease_densenet_se.keras` + `history.json` |
| `evaluate.py` | Test-set metrics, confusion matrix, training curves |
| `gradcam.py` | Grad-CAM heatmaps, one per class |
| `predict.py` | Predict a single X-ray (the "For Validation" step) |
| `lung_disease_colab.ipynb` | Ready-to-run Google Colab notebook (recommended) |
| `Lung_Disease_Classification_Report.docx` | Report in the ML8 format; fill the yellow fields |
| `requirements.txt` | Python packages |

## 1. Get the dataset
Download the **COVID-19 Radiography Database** from Kaggle and unzip it so you have:
```
COVID-19_Radiography_Dataset/
    COVID/images/*.png
    Lung_Opacity/images/*.png
    Normal/images/*.png
    Viral Pneumonia/images/*.png
```
(The `masks` folders are not used.)

## 2. Run (Google Colab, recommended: Runtime > Change runtime type > GPU)
Open `lung_disease_colab.ipynb` and run the cells top to bottom. It installs nothing heavy,
downloads the dataset, trains, evaluates, makes Grad-CAMs and zips the results.

## 2b. Run locally instead
```bash
pip install -r requirements.txt
python train.py    --data_dir COVID-19_Radiography_Dataset
python evaluate.py --data_dir COVID-19_Radiography_Dataset
python gradcam.py  --data_dir COVID-19_Radiography_Dataset
python predict.py  some_xray.png --gradcam
```
A GPU is strongly recommended. On CPU only, try `--head_epochs 3 --ft_epochs 5` first.

## 3. Fill in the report
After the run, `outputs/` contains everything the yellow fields need:
- `metrics.json` and `classification_report.txt` -> Table 2 (accuracy, macro-F1, ROC-AUC) and Table 3 (per-class precision/recall/F1/support)
- `history.json` and the last lines printed by `train.py` -> training / validation accuracy and loss
- `training_curves.png`, `confusion_matrix.png` -> insert as figures
- `gradcam_*.png` -> insert as Grad-CAM figures
- `python predict.py <image>` on 4 test images (one per class) -> "Sample predictions" table

Report your **own** numbers only.

## Troubleshooting
- **Out of memory:** `--batch_size 16`
- **`Expected folder not found`:** check `--data_dir`; class folders must be directly inside it
- **Slow epochs:** you are on CPU; switch Colab to a GPU runtime
- **`AdamW` not found:** upgrade TensorFlow (`pip install -U tensorflow`), needs 2.11+
- Run `evaluate.py` / `gradcam.py` with the same `--data_dir` as training (the split is deterministic via a fixed seed, so the test set is identical)

## Disclaimer
Educational/research project. Not a medical device and not a substitute for professional diagnosis.
