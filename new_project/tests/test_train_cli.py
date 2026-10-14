import pytest

from src.train import parse_args


@pytest.mark.parametrize("name,backend,device,precision,size,batch,epochs", [
    ("simple_cnn", "tensorflow", "auto", "float32", 224, 32, 50),
    ("complex_cnn", "torch", "cuda", "mixed_float16", 224, 8, 90),
    ("efficientnetb2", "tensorflow", "auto", "float32", 240, 32, 50),
])
def test_model_selection_is_sufficient_and_preserves_defaults(
    name, backend, device, precision, size, batch, epochs,
):
    args = parse_args(["--model", name])
    assert (args.backend, args.device, args.precision) == (backend, device, precision)
    assert (args.image_size, args.batch_size, args.epochs) == (size, batch, epochs)
    assert args.weights == "imagenet"


@pytest.mark.parametrize("option,value", [("--epochs", "0"), ("--batch-size", "-1"), ("--seed", "-1")])
def test_invalid_training_arguments_fail_before_startup(option, value):
    with pytest.raises(SystemExit):
        parse_args(["--model", "simple_cnn", option, value])
