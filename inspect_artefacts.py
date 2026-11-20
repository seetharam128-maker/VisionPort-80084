
import argparse
import os
from collections import Counter
from pathlib import Path

# 0. OPTIONS AND ROLE TABLE

parser = argparse.ArgumentParser(
    description="Inspect VisionPort model artefacts"
)

parser.add_argument("--weights", default="yolov8n.pt")
parser.add_argument("--onnx", default=None)
parser.add_argument("--export-onnx", action="store_true")

args = parser.parse_args()

ROLE = {
    ".pt": "PyTorch checkpoint",
    ".onnx": "ONNX graph",
    ".engine": "TensorRT engine",
    ".plan": "TensorRT engine (Triton)"
}

# 1. INVENTORY OF ARTEFACTS

print("\n=== 1. ARTEFACT INVENTORY ===")

found = []

for folder, subfolders, files in os.walk("."):
    subfolders[:] = [
        d for d in subfolders
        if d not in (".venv", ".git", "__pycache__", ".idea")
    ]

    for name in files:
        if Path(name).suffix in ROLE:
            found.append(Path(folder) / name)

if not found:
    print("No model artefacts found.")

for path in found:
    print(
        f"{str(path):<35} "
        f"{path.stat().st_size / 1e6:>8.1f} MB "
        f"{ROLE[path.suffix]}"
    )


# 2. PYTORCH CHECKPOINT AND ARCHITECTURE

print("\n=== 2. PYTORCH CHECKPOINT ===")

yolo = None

try:
    import torch
    from ultralytics import YOLO

    yolo = YOLO(args.weights)

    weights_path = Path(args.weights)

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    print("\nTop-level contents:")

    for key, value in checkpoint.items():
        kind = type(value).__name__
        detail = (
            f"{len(value)} entries"
            if isinstance(value, dict)
            else ""
        )
        print(f"{key:<15} {kind:<18} {detail}")

    print(
        "Training state:",
        "Yes" if checkpoint.get("optimizer")
        else "No (inference checkpoint)"
    )

    stored_model = checkpoint.get("ema") or checkpoint.get("model")

    if stored_model is not None:
        print(
            "Stored weights dtype:",
            next(stored_model.parameters()).dtype
        )

    print("\n=== MODEL ARCHITECTURE ===")

    model = yolo.model

    print("Model type:", type(model).__name__)
    print("Total layers:", len(model.model))

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    print("\nLayer details:")

    for i, layer in enumerate(model.model):
        print(f"Layer {i:02d}: {type(layer).__name__}")

except Exception as e:
    print("PyTorch inspection failed:", e)
    print("Install packages using: pip install torch ultralytics")


# 3. ONNX EXPORT AND INSPECTION

print("\n=== 3. ONNX GRAPH ===")

onnx_path = Path(args.onnx) if args.onnx else Path(
    args.weights
).with_suffix(".onnx")

if args.export_onnx:
    try:
        if yolo is None:
            from ultralytics import YOLO
            yolo = YOLO(args.weights)

        print("Exporting model to ONNX...")

        exported = yolo.export(format="onnx")
        onnx_path = Path(exported)

        print("Export completed:", onnx_path)

    except Exception as e:
        print("ONNX export failed:", e)

if onnx_path.exists():

    try:
        import onnx

        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)

        graph = model_onnx.graph

        def describe(value):
            dims = []

            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(dim.dim_value)
                else:
                    dims.append(dim.dim_param or "?")

            return f"{value.name} {dims}"

        weights = {
            w.name for w in graph.initializer
        }

        inputs = [
            describe(v)
            for v in graph.input
            if v.name not in weights
        ]

        outputs = [
            describe(v)
            for v in graph.output
        ]

        operations = Counter(
            node.op_type for node in graph.node
        )

        print("File:", onnx_path)
        print(
            "File size:",
            f"{onnx_path.stat().st_size / 1e6:.2f} MB"
        )
        print("Format check: passed")
        print(
            "Opset version:",
            model_onnx.opset_import[0].version
        )
        print("Inputs:", inputs)
        print("Outputs:", outputs)
        print(
            "Graph:",
            len(graph.node),
            "operations,",
            len(graph.initializer),
            "weight tensors"
        )

        print("Most common operations:")

        for operation, count in operations.most_common(10):
            print(f"{operation} x{count}")

    except Exception as e:
        print("ONNX inspection failed:", e)
        print("Install using: pip install onnx")

else:
    print("No ONNX file yet.")
    print(
        "Create one with: "
        "python inspect_artefacts.py --export-onnx"
    )


# 4. TENSORRT ENGINE

print("\n=== 4. TENSORRT ENGINE ===")

engines = [
    p for p in found
    if p.suffix in (".engine", ".plan")
]

if engines:
    for p in engines:
        print(
            f"{p} "
            f"({p.stat().st_size / 1e6:.1f} MB)"
        )
        print(
            "Only TensorRT on a matching NVIDIA GPU can load it."
        )

else:
    print("No engine found.")
    print(
        "This is expected without a compatible NVIDIA GPU."
    )
    print(
        "An engine is built from ONNX in Session 13."
    )


# 5. COMPARISON TABLE

print("\n=== 5. COMPARISON ===")

rows = [
    (
        "Checkpoint (.pt)",
        "Weights + architecture",
        "PyTorch + Python",
        "Training and export"
    ),
    (
        "ONNX (.onnx)",
        "Graph + weights",
        "ONNX Runtime or TensorRT",
        "Portable model format"
    ),
    (
        "Engine (.engine)",
        "Optimised graph",
        "NVIDIA GPU + TensorRT",
        "Fast model serving"
    )
]

print(
    f"{'Artefact':<22}"
    f"{'Contains':<25}"
    f"{'Needs':<27}"
    f"Used for"
)

for row in rows:
    print(
        f"{row[0]:<22}"
        f"{row[1]:<25}"
        f"{row[2]:<27}"
        f"{row[3]}"
    )

print("\n=== VISIONPORT INSPECTION COMPLETED ===")