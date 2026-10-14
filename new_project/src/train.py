"""Train: python new_project/src/train.py --model simple_cnn."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("simple_cnn", "complex_cnn", "efficientnetb2"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("float32", "mixed_float16"))
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data/processed/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="EfficientNetB2 initialization; default: imagenet")
    args = parser.parse_args(argv)
    complex_run = args.model == "complex_cnn"
    args.epochs = args.epochs if args.epochs is not None else (90 if complex_run else 50)
    args.batch_size = args.batch_size if args.batch_size is not None else (8 if complex_run else 32)
    args.backend = "torch" if complex_run else "tensorflow"
    args.device = args.device or ("cuda" if complex_run else "auto")
    args.precision = args.precision or ("mixed_float16" if complex_run else "float32")
    args.image_size = 240 if args.model == "efficientnetb2" else 224
    if args.epochs <= 0 or args.batch_size <= 0 or args.seed < 0:
        parser.error("epochs and batch size must be positive; seed must be non-negative")
    return args


def configure_runtime(args):
    os.environ["KERAS_BACKEND"] = args.backend
    if args.backend == "torch":
        os.environ["KERAS_TORCH_DEVICE"] = args.device
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable. Use --device cpu --precision float32, "
                               "or install a CUDA-enabled PyTorch environment.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(8)
    import tensorflow as tf

    if args.backend == "tensorflow":
        if args.device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif args.device == "cuda" and not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("TensorFlow GPU is unavailable. Use --device cpu or a GPU-enabled "
                               "Linux/WSL2 environment.")
        args.device = "cuda" if tf.config.get_visible_devices("GPU") else "cpu"
    else:
        # TensorFlow supplies data only,  PyTorch own the GPU
        tf.config.set_visible_devices([], "GPU")
    import keras

    keras.mixed_precision.set_global_policy(args.precision)
    if args.backend == "torch":
        keras.utils.set_random_seed(args.seed)
        tf.random.set_seed(args.seed)
    else:
        from src.reproducibility import set_global_seed

        set_global_seed(args.seed)


def run(args):
    configure_runtime(args)
    import keras
    from src.callbacks import complex_callbacks, simple_callbacks, transfer_callbacks
    from src.data_loader import build_tf_dataset
    from src.evaluation import (
        CLASS_NAMES, evaluate_model, save_json,
    )
    from src.preprocessing import load_split_manifest
    from src.training import train_model

    manifest = load_split_manifest(args.manifest)
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {args.raw_dir}")
    missing = [name for name in manifest.file if not (args.raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing raw files, e.g. {missing[0]}")
    if (manifest.groupby("patient_id")["split"].nunique() > 1).any():
        raise ValueError("Manifest contains patients in multiple splits")
    datasets = {
        split: build_tf_dataset(
            manifest, args.raw_dir, split=split, image_size=args.image_size,
            batch_size=args.batch_size, training=(split == "train"),
            augment=(split == "train" and not args.no_augmentation), seed=args.seed,
        ) for split in ("train", "validation", "test")
    }
    run_dir = args.output_dir or PROJECT_ROOT / "runs" / (
        f"{args.model}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    reports, models = run_dir / "reports", run_dir / "models"
    reports.mkdir()
    models.mkdir()
    checkpoint = models / (
        "best.keras" if args.model == "efficientnetb2" else
        "simple_cnn_6conv_patience10_best.keras" if args.model == "simple_cnn" else
        "complex_cnn_best.keras"
    )
    if args.model == "simple_cnn":
        from src.models.simple_cnn import build_simple_cnn

        model = build_simple_cnn(image_size=args.image_size)
        callbacks = simple_callbacks(datasets["validation"], CLASS_NAMES, checkpoint)
    elif args.model == "complex_cnn":
        from src.models.complex_cnn import build_complex_cnn

        batches = (int(manifest.split.eq("train").sum()) + args.batch_size - 1) // args.batch_size
        model = build_complex_cnn(decay_steps=batches * 60, image_size=args.image_size)
        callbacks = complex_callbacks(checkpoint, reports)
    else:
        from src.models.transfer_learning import build_transfer_efficientnetb2

        model = build_transfer_efficientnetb2(
            image_size=args.image_size, weights=None if args.weights == "none" else "imagenet",
        )
        callbacks = transfer_callbacks(checkpoint)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(output_dir=str(run_dir.resolve()), orientation="matlab_rows_columns")
    save_json(config, reports / "run_config.json")
    manifest.to_csv(reports / "split_manifest_used.csv", index=False)
    summary = []
    model.summary(print_fn=summary.append)
    (reports / f"{args.model}_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"{args.model}: {args.backend}, {args.device}, {args.precision}, "
          f"{args.image_size}px, batch {args.batch_size}, {args.epochs} epochs", flush=True)
    print(f"Results: {run_dir.resolve()}", flush=True)
    history = train_model(
        model, datasets["train"], datasets["validation"], checkpoint, reports,
        args.model, callbacks=callbacks, epochs=args.epochs,
    )
    if args.model == "efficientnetb2":
        model.save(models / "last.keras")
    best = keras.models.load_model(checkpoint)
    for split in ("validation", "test"):
        metrics = evaluate_model(
            best, datasets[split], args.model, reports, split=split,
            history=history if split == "test" else None,
        )
        print(f"{split}: accuracy={metrics['accuracy']:.4f}, "
              f"macro_f1={metrics['macro_f1']:.4f}")
    if args.model == "simple_cnn":
        clean_train = build_tf_dataset(
            manifest, args.raw_dir, split="train", training=True, augment=False,
            image_size=args.image_size, batch_size=args.batch_size, seed=args.seed,
        )
        print("train_clean", best.evaluate(clean_train, verbose=0, return_dict=True))
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Training failed: {error}") from error


if __name__ == "__main__":
    main()
