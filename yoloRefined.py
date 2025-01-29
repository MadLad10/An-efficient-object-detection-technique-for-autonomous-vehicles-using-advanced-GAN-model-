import os
import torch
from pathlib import Path
from PIL import Image
import cv2


def run_yolo_on_directory(input_dir, output_dir, model_path='yolov5s', confidence_threshold=0.4):
    """
    Runs YOLO on all images in the input directory and saves annotated results in the output directory.

    Parameters:
        input_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory where results will be saved.
        model_path (str): Path to the YOLO model.
        confidence_threshold (float): Confidence threshold for detections.
    """
    os.makedirs(output_dir, exist_ok=True)

    
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
    model.conf = confidence_threshold  

    
    for image_file in Path(input_dir).glob("*.*"):
        try:
            
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                continue

            print(f"Processing image: {image_file}")

           
            results = model(str(image_file))

          
            output_path = Path(output_dir) / image_file.name

            
            annotated_img = results.render()[0]  
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB) 

            
            Image.fromarray(annotated_img).save(output_path)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    input_directory = "/home/ifaz/Desktop/Study Materials/CSE463/Project/gan/bdd100k/output"  
    output_directory = "/home/ifaz/Desktop/Study Materials/CSE463/Project/YOLO Detection"  

    run_yolo_on_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        model_path='yolov5s',  
        confidence_threshold=0.4  
    )

