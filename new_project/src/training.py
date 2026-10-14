"""Shared fit wrapper with each notebook's original history format."""

from pathlib import Path

import pandas as pd

from src.evaluation import plot_learning_curves, save_json


def train_model(model, train_dataset, validation_dataset, checkpoint_path, reports_dir,
                model_name, callbacks, epochs=50):
    checkpoint_path, reports_dir = Path(checkpoint_path), Path(reports_dir)
    if model_name == "simple_cnn":
        history_path = reports_dir / "simple_cnn_6conv_patience10_history.json"
    elif model_name == "efficientnetb2":
        history_path = reports_dir / "history.csv"
    else:
        history_path = reports_dir / f"{model_name}_history.json"
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError("Results already exist. Choose a new output folder to retrain.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    complex_run = model_name not in {"simple_cnn", "efficientnetb2"}
    if complex_run:
        print("Starting fit: the first training batch fills the full shuffle buffer. "
              "Startup may take a minute or longer before batch progress appears.", flush=True)
    result = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs,
        callbacks=callbacks, shuffle=not complex_run, verbose=1,
    )
    history = {key: [float(v) for v in values] for key, values in result.history.items()}
    if model_name == "simple_cnn":
        pd.DataFrame(history).to_json(history_path)
    elif model_name == "efficientnetb2":
        frame = pd.DataFrame(history)
        frame["epoch"] = range(1, len(frame) + 1)
        frame.to_csv(history_path, index=False)
    else:
        save_json(history, history_path)
        plot_learning_curves(history, reports_dir / f"{model_name}_learning_curve.png", model_name)
    return history
