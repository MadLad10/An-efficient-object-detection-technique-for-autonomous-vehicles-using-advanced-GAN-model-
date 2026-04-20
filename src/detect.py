import argparse
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def run_detection(input_dir, output_dir, model_path='yolov8s.pt', confidence=0.4, device=None):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    images = [f for f in Path(input_dir).glob("*.*") if f.suffix.lower() in VALID_EXTENSIONS]
    print(f"Found {len(images)} images in {input_dir}")

    for image_file in images:
        try:
            results = model.predict(str(image_file), conf=confidence, device=device, verbose=False)
            annotated = results[0].plot()  # returns BGR numpy array
            output_path = Path(output_dir) / image_file.name
            Image.fromarray(annotated[..., ::-1]).save(output_path)  # BGR→RGB
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on a directory of images")
    parser.add_argument("--input", required=True, help="Path to input image directory")
    parser.add_argument("--output", required=True, help="Path to save annotated images")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path or name (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, mps (auto-detected if omitted)")
    args = parser.parse_args()

    run_detection(args.input, args.output, args.model, args.conf, args.device)
