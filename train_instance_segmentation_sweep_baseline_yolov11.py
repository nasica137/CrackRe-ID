# python
import wandb
from ultralytics import YOLO

sweep_config = {
    "name": "segmentation_baseline",
    "method": "grid",
    "metric": {"name": "metrics/mAP50-95(M)", "goal": "maximize"},
    "parameters": {
        "model_variant": {"values": ["yolo11n-seg.pt"]},
        "epochs": {"value": 500},
        "batch_size": {"values": [32, 64]},
        "img_size": {"values": [416, 640]},
        "learning_rate": {"values": [0.01, 0.001]},
        "seed": {"values": [0, 47, 1337]},
    }
}

def baseline_run():
    def train(config=None):
        with wandb.init(config=config, project="yolo_segmentation_hpo_baseline_yolov11") as run:
            config = wandb.config
            model = YOLO(config.model_variant)
            run_name = (
                f"seg_model-{config.model_variant}_batch-{config.batch_size}"
                f"_imgsz-{config.img_size}_lr-{config.learning_rate}_seed-{config.seed}"
            )
            run.name = run_name
            model.train(
                data="/workspace/crack-seg.yaml",  # YAML above
                epochs=config.epochs,
                imgsz=config.img_size,
                batch=config.batch_size,
                lr0=config.learning_rate,
                device=0,
                workers=16,
                project="/workspace/runs/crack_segmentation/baseline_project_yolov11",
                name=run_name,
                seed=config.seed,
                mosaic=0.0,
                verbose=False,
                optimizer="AdamW",
                patience=50,
                cos_lr=True
            )

    sweep_id = wandb.sweep(sweep_config, project="yolo_segmentation_hpo_baseline_yolov11")
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    baseline_run()