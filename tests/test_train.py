import json

from src.train import ModelTrainer, build_argument_parser


def test_train_parser_accepts_custom_arguments():
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--learning-rate",
            "0.01",
            "--architecture",
            "cnn_plus",
            "--device",
            "cpu",
            "--num-workers",
            "2",
            "--seed",
            "123",
            "--weight-decay",
            "0.001",
            "--label-smoothing",
            "0.02",
            "--lr-scheduler-patience",
            "3",
            "--lr-scheduler-factor",
            "0.4",
            "--early-stopping-patience",
            "4",
            "--disable-augmentation",
            "--disable-balanced-sampling",
        ]
    )

    assert args.epochs == 3
    assert args.batch_size == 16
    assert args.learning_rate == 0.01
    assert args.architecture == "cnn_plus"
    assert args.device == "cpu"
    assert args.num_workers == 2
    assert args.seed == 123
    assert args.weight_decay == 0.001
    assert args.label_smoothing == 0.02
    assert args.lr_scheduler_patience == 3
    assert args.lr_scheduler_factor == 0.4
    assert args.early_stopping_patience == 4
    assert args.disable_augmentation is True
    assert args.disable_balanced_sampling is True


def test_write_training_report_saves_json(tmp_path):
    trainer = ModelTrainer(epochs=1, batch_size=2, learning_rate=0.1, architecture="mlp", device="cpu")
    report_path = tmp_path / "report.json"

    trainer._write_training_report(
        report_path=report_path,
        history=[
            {
                "epoch": 1,
                "train_loss": 0.5,
                "train_accuracy": 0.9,
                "val_loss": 0.4,
                "val_accuracy": 0.95,
                "learning_rate": 0.1,
            }
        ],
        best_val_accuracy=0.95,
        best_epoch=1,
        final_val_metrics={
            "loss": 0.4,
            "accuracy": 0.95,
            "confusion_matrix": [[1, 0], [0, 1]],
            "per_class_accuracy": {"0": 1.0},
            "most_confused_pairs": [],
        },
        train_class_distribution={"0": 10},
        val_class_distribution={"0": 2},
        stopped_early=False,
    )

    content = json.loads(report_path.read_text(encoding="utf-8"))

    assert content["best_val_accuracy"] == 0.95
    assert content["best_epoch"] == 1
    assert content["history"][0]["epoch"] == 1
    assert content["history"][0]["learning_rate"] == 0.1
    assert content["final_validation"]["accuracy"] == 0.95
    assert content["train_class_distribution"]["0"] == 10
