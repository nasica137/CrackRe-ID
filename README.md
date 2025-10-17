# Crack Re-ID

A compact, deployable pipeline for UAV-based crack detection and single-shot re-identification (Re-ID). A single Ultralytics YOLO11n-seg checkpoint is reused for both detection and instance-mask extraction. Per-detection crops are encoded with DINOv2 visual embeddings and fused with an 11-D geometry descriptor derived from masks. A gallery-aware matcher (AttentionSimilarity) is trained with a staged Triplet Schedule. The project is part of my master’s thesis at the Albert-Ludwigs-Universität Freiburg and Fraunhofer IPM.

Goal: robust, fair, and efficient single-shot crack Re-ID under UAV constraints, evaluated with OSR, mAP_fair, and CMC.

## Installation

- Python 3.8+
- GPU with CUDA recommended

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: Weights & Biases for sweep tracking.
```bash
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=yolo_segmentation_hpo_baseline_yolov11
```

## Data

- Use YOLO-style segmentation data and update paths in crack-seg.yaml.
- Recommended: Roboflow Universe Crack Dataset (bphdr), split into train/val/test.

Directory sketch (example):
- datasets/crack/
  - images/{train,val,test}/*.jpg
  - labels/{train,val,test}/*.txt (YOLO-seg polygons)

## Usage

1) Train instance segmentation (YOLO11n-seg)

```bash
python train_instance_segmentation_sweep_baseline_yolov11.py
```

- Best checkpoint will be stored under runs/ (Ultralytics default). You can also point downstream steps to a custom weights path.

2) Build cropped galleries and augmented queries

```bash
python reidentification_dataset_creation.py
```

- Produces:
  - Cropped galleries per split with per-object metadata (object_mapping.json)
  - Augmented query images for Re-ID training/evaluation

3) Evaluate Re-ID (cosine, visual-only)

```bash
python reidentification_evaluation.py \
```

4) Plots and CMC aggregation

```bash
python reidentification_plots.py 

python get_cmc_ablation.py 
```

## Metrics

- Detection/segmentation: mAP@50:95, Precision, Recall, mIoU (Ultralytics)
- Re-ID: OSR (%), mAP_fair, CMC@k (k=1,5,10)

## Docker

```bash
docker build -t crack-reid -f DockerFile .
docker run --gpus all -it --rm \
  -v $PWD:/workspace -w /workspace \
  crack-reid /bin/bash
```
