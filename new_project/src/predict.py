import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import keras
import numpy as np

from src.preprocessing import CLASS_INDEX_TO_NAME, load_case, prepare_image
from src.models.transfer_learning import to_255  

# python new_project/src/predict.py --model simple_cnn
# python new_project/src/predict.py --model complex_cnn
# python new_project/src/predict.py --model efficientnetb2

MODEL_PATHS = {
    "simple_cnn": ROOT / "models/simple_cnn_6conv_patience10_best.keras",
    "complex_cnn": ROOT / "runs/complex_01/models/complex_cnn_best.keras",
    "efficientnetb2": ROOT / "runs/transfer_01/models/best.keras",
}
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODEL_PATHS, default="simple_cnn")
args = parser.parse_args()
IMAGE_PATH = ROOT / "data/raw/brainTumorDataPublic_1-766/1.mat"

model = keras.models.load_model(MODEL_PATHS[args.model], compile=False)

case = load_case(IMAGE_PATH)
image, _, _ = prepare_image(
    case["image"], case["tumorMask"], image_size=model.input_shape[1]
)
X = np.expand_dims(image, axis=0)

y_pred = model.predict(X, verbose=0)[0]
class_index = int(np.argmax(y_pred))
print("y_pred:", CLASS_INDEX_TO_NAME[class_index])
print("Probability:", float(y_pred[class_index]))
