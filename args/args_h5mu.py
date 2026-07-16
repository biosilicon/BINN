import argparse


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and validate NicheTrans with two schema-compliant H5MU files"
    )

    dataset = parser.add_argument_group("H5MU dataset")
    dataset.add_argument("--train-path", required=True, type=str)
    dataset.add_argument("--test-path", required=True, type=str)
    dataset.add_argument("--source-modality", default="rna", type=str)
    dataset.add_argument("--target-modality", default="protein", type=str)
    dataset.add_argument("--n-neighbors", default=12, type=_positive_int)
    dataset.add_argument("--preprocess", default="auto", choices=("auto", "raw"))
    dataset.add_argument("--rna-target-sum", default=1e3, type=_positive_float)
    dataset.add_argument(
        "--task", default="auto", choices=("auto", "regression", "binary")
    )
    dataset.add_argument("-j", "--workers", default=4, type=_non_negative_int)

    model = parser.add_argument_group("Model")
    model.add_argument("--noise-rate", default=0.2, type=_probability)
    model.add_argument("--dropout-rate", default=0.2, type=_probability)
    model.add_argument("--neighbor-mask-probability", default=0.3, type=_probability)
    model.add_argument("--neighbor-keep-probability", default=0.5, type=_probability)

    training = parser.add_argument_group("Training")
    training.add_argument("--max-epoch", default=40, type=_positive_int)
    training.add_argument("--eval-step", default=1, type=_positive_int)
    training.add_argument("--train-batch", default=32, type=_positive_int)
    training.add_argument("--test-batch", default=32, type=_positive_int)
    training.add_argument("--optimizer", default="adam", choices=("adam", "sgd"))
    training.add_argument("--lr", "--learning-rate", default=3e-4, type=_positive_float)
    training.add_argument("--weight-decay", default=5e-4, type=float)
    training.add_argument("--stepsize", default=20, type=_non_negative_int)
    training.add_argument("--gamma", default=0.1, type=_positive_float)

    runtime = parser.add_argument_group("Runtime and output")
    runtime.add_argument("--seed", default=1, type=int)
    runtime.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    runtime.add_argument("--output-dir", default="outputs/h5mu", type=str)
    runtime.add_argument("--run-name", default="nichetrans_h5mu", type=str)
    return parser


def generate_args(argv=None):
    """Parse H5MU training arguments.

    Pass an explicit list from notebooks so Jupyter kernel arguments are not
    interpreted as training arguments.
    """

    args = build_parser().parse_args(argv)
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not args.run_name.strip():
        raise ValueError("run_name must not be empty")
    return args


if __name__ == "__main__":
    print(generate_args())
