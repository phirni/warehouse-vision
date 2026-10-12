"""Export a YOLO .pt model to ONNX. python -m scripts.export_onnx --model yolo11n.pt"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--dynamic", action="store_true", help="dynamic input shapes (can be slower)")
    p.add_argument("--half", action="store_true", help="FP16 (GPU export only)")
    a = p.parse_args()

    path = YOLO(a.model).export(format="onnx", imgsz=a.imgsz, opset=a.opset,
                                dynamic=a.dynamic, half=a.half, simplify=True)
    print(f"exported -> {path}")


if __name__ == "__main__":
    main()
