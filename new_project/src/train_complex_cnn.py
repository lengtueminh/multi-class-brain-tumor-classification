"""Train the selected CNN and save the best validation-loss checkpoint."""

from pathlib import Path
from src.evaluation import save_json, plot_learning_curves


def train_model(model, train_dataset, validation_dataset, checkpoint_path, reports_dir,
                model_name, callbacks, epochs=50):
    checkpoint_path, reports_dir = Path(checkpoint_path), Path(reports_dir)
    history_path = reports_dir / f"{model_name}_history.json"
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError("Results already exist. Choose a new output folder to retrain.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result = model.fit(train_dataset, validation_data=validation_dataset,
                       epochs=epochs, callbacks=callbacks, shuffle=False, verbose=2)
    history = {key: [float(v) for v in values] for key, values in result.history.items()}
    save_json(history, history_path)
    plot_learning_curves(history, reports_dir / f"{model_name}_learning_curve.png", model_name)
    return history
