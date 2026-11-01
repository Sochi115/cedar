import csv
import json

import h5py
import numpy as np
import torch

from csi_bench.cnn import DeepCnn, load_task_datasets, render_report


def test_loads_csi_bench_hdf5_for_cnn(tmp_path):
    task_dir = tmp_path / "FallDetection"
    metadata_dir = task_dir / "metadata"
    split_dir = task_dir / "splits"
    data_dir = task_dir / "sub_Human"
    metadata_dir.mkdir(parents=True)
    split_dir.mkdir()
    data_dir.mkdir()

    labels = {"fall-1": "Fall", "nonfall-1": "Nonfall", "fall-2": "Fall"}
    rows = []
    for sample_id, label in labels.items():
        filepath = data_dir / f"{sample_id}.h5"
        with h5py.File(filepath, "w") as output:
            output["CSI_amps"] = np.arange(232 * 500, dtype=np.float32).reshape(232, 500, 1)
        rows.append({"id": sample_id, "file_path": f"./sub_Human/{filepath.name}", "label": label})

    with (metadata_dir / "sample_metadata.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("id", "file_path", "label"))
        writer.writeheader()
        writer.writerows(rows)
    (metadata_dir / "label_mapping.json").write_text(
        json.dumps({"label_to_idx": {"Fall": 0, "Nonfall": 1}}), encoding="utf-8"
    )
    for name, sample_id in zip(("train_id", "val_id", "test_id"), labels):
        (split_dir / f"{name}.json").write_text(json.dumps([sample_id]), encoding="utf-8")

    _, class_names, datasets = load_task_datasets(tmp_path, "FallDetection")
    values, label = datasets["train_id"][0]

    assert class_names == ["Fall", "Nonfall"]
    assert values.shape == (1, 500, 232)
    assert torch.isfinite(values).all()
    assert label == 0
    model = DeepCnn(len(class_names))
    assert model(values.unsqueeze(0)).shape == (1, 2)
    assert sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()) >= 17

    report = render_report(
        {
            "task": "FallDetection",
            "model": "csi-bench-cnn",
            "best_epoch": 1,
            "training_history": [{"epoch": 1, "train_loss": 0.8, "val_loss": 0.7}],
            "test_metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "class_names": class_names,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        }
    )
    assert "Training and validation loss" in report
    assert "Confusion matrix" in report
    assert report.count("data:image/png;base64,") == 2
