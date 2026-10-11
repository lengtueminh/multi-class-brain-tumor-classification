from types import SimpleNamespace

import pandas as pd
import pytest

from src.training import train_model


@pytest.mark.parametrize("name,filename", [
    ("simple_cnn", "simple_cnn_6conv_patience10_history.json"),
    ("complex_cnn", "complex_cnn_history.json"),
    ("efficientnetb2", "history.csv"),
])
def test_fit_preserves_history_formats_and_prevents_overwrite(tmp_path, name, filename):
    history = {"loss": [1.0], "val_loss": [0.9], "accuracy": [0.5], "val_accuracy": [0.6]}
    calls = []

    class Model:
        def fit(self, dataset, **kwargs):
            calls.append((dataset, kwargs))
            return SimpleNamespace(history=history)

    checkpoint, reports = tmp_path / "models/best.keras", tmp_path / "reports"
    result = train_model(Model(), "train", "val", checkpoint, reports, name, [], epochs=2)
    assert result == history
    assert calls[0][1]["shuffle"] is (name != "complex_cnn")
    assert calls[0][1]["epochs"] == 2
    assert calls[0][1]["verbose"] == 1
    saved = reports / filename
    assert saved.exists()
    if name == "simple_cnn":
        assert saved.read_text() == pd.DataFrame(history).to_json()
    elif name == "efficientnetb2":
        frame = pd.read_csv(saved)
        assert frame["epoch"].tolist() == [1]
        assert frame["val_accuracy"].tolist() == [0.6]
    with pytest.raises(FileExistsError):
        train_model(Model(), "train", "val", checkpoint, reports, name, [], epochs=2)
    assert len(calls) == 1


def test_transfer_scaling_can_reload_safely(tmp_path):
    import numpy as np
    import tensorflow as tf
    from src.models.transfer_learning import to_255

    inputs = tf.keras.Input((3,))
    model = tf.keras.Model(inputs, tf.keras.layers.Lambda(to_255, name="to_255")(inputs))
    checkpoint = tmp_path / "scaling.keras"
    model.save(checkpoint)
    restored = tf.keras.models.load_model(checkpoint)
    sample = np.asarray([[0, 0.5, 1]], dtype=np.float32)
    np.testing.assert_array_equal(restored(sample).numpy(), sample * 255.0)
