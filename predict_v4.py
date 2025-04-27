from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import torch
import torchvision
import argparse
import torchvision.ops as ops
from pathlib import Path
from torchvision.io import decode_image
from torchvision.transforms import v2 as T
from modules.faster_rcnn import StandardFasterRCNN

def load_model(model: torch.nn.Module, target_dir: str, model_name: str, device: torch.device):
        
    """Loads a PyTorch model from a target directory.

    Args:
        model: A target PyTorch model to load.
        target_dir: A directory where the model is located.
        model_name: The name of the model to load. Should include
        ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.

    Returns:
        The loaded PyTorch model.
    """

    # Define the list of valid extensions
    valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

    # Create model save path
    assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
    model_save_path = Path(target_dir) / model_name

    # Load the model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    return model

# Pre-processing transformations
def get_transform():

    """
    Returns a composition of transformations for preprocessing images.
    """

    transforms = []   
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# Function to remove redundant boxes and masks
def prune_predictions(
    pred,
    score_threshold=0.8,
    iou_threshold=0.01,
    best_candidate="area"
    ):

    """
    Filters out redundant predictions.

    Args:
        pred: The raw predictions containing "boxes", "scores", "labels", and "masks".
        score_threshold: The minimum confidence score required to keep a prediction (default: 0.66).
        iou_threshold: The Intersection over Union (IoU) threshold for NMS (default: 0.5).

    Returns:
        A dictionary with filtered and refined predictions:
            "boxes": Tensor of kept bounding boxes.
            "scores": Tensor of kept scores.
            "labels": Tensor of kept labels.
    """
    
    # Filter predictions based on confidence score threshold
    scores = pred["scores"]

    best_idx = scores.argmax()
    high_conf_idx = scores > score_threshold

    # Extract the best bounding box, score, and label
    best_pred = {
        "boxes": pred["boxes"][best_idx].unsqueeze(0).long(), 
        "scores": pred["scores"][best_idx].unsqueeze(0),
        "labels": pred["labels"][best_idx].unsqueeze(0),
    }

    filtered_pred = {
        "boxes":  pred["boxes"][high_conf_idx].long(),
        "scores": pred["scores"][high_conf_idx],
        "labels": pred["labels"][high_conf_idx],
    }

    # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
    if len(filtered_pred["boxes"]) == 0:
        if len(best_pred["boxes"]) > 0:
            return best_pred
        else:
            return filtered_pred 
    
    keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], iou_threshold)

    # Return filtered predictions
    keep_preds = {
        "boxes": filtered_pred["boxes"][keep_idx],
        "scores": filtered_pred["scores"][keep_idx],
        "labels": filtered_pred["labels"][keep_idx],
    }

    # Ensure the best prediction is always included
    best_box = best_pred["boxes"][0]
    if not any(torch.equal(best_box, box) for box in keep_preds["boxes"]):
        keep_preds["boxes"] = torch.cat([keep_preds["boxes"], best_pred["boxes"]])
        keep_preds["scores"] = torch.cat([keep_preds["scores"], best_pred["scores"]])
        keep_preds["labels"] = torch.cat([keep_preds["labels"], best_pred["labels"]])

    # Now we have a set of good candidates. Let's take the best one based on a criterion
    if best_candidate == "score":
        scores = keep_preds['scores']
        idx = scores.argmax()

        # Return only the one with the highest score
        final_pred = {
            "boxes": keep_preds["boxes"][idx].unsqueeze(0),
            "scores": keep_preds["scores"][idx].unsqueeze(0),
            "labels": keep_preds["labels"][idx].unsqueeze(0),
        }
        return final_pred

    # Compute area of each box and get the one with the highest area
    elif best_candidate == "area":
        
        boxes = keep_preds["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = areas.argmax()

        # Return only the largest bounding box
        final_pred = {
            "boxes": keep_preds["boxes"][idx].unsqueeze(0),
            "scores": keep_preds["scores"][idx].unsqueeze(0),
            "labels": keep_preds["labels"][idx].unsqueeze(0),
        }
        return final_pred

    # Return the set
    else:
        return keep_preds 
    
# Function to predict and save images
def predict_and_save(model, image_path, output_path_txt, device):
    """
    Predict bounding boxes using a custom model and save them in YOLO format.
    
    Args:
        model: object detection model.
        image_path: path of the image
        img_width: Width of the image.
        img_height: Height of the image.
        output_path_txt: Path to save the predictions.
    """

    # Load image
    image = decode_image(image_path)

    # Image dimensions
    img_height, img_width = image.shape[1], image.shape[2]

    # Apply image transformation
    transform = get_transform()

    # Make prediction
    model.eval().to(device)
    with torch.no_grad():
        x = transform(image)  # Apply transformations
        x = x[:3, ...].to(device)  # Ensure it's RGB (3 channels)
        pred = model([x])[0]

        # Take the best ones
        if pred["boxes"].nelement() > 0:
            pred = prune_predictions(pred)

    # Save bounding boxes in YOLO format
    with open(output_path_txt, 'w') as f:
        # Unpack values from pred
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        for i in range(len(boxes)):
            cls_id = int(labels[i].item()) - 1 # Always zero
            conf = float(scores[i].item())
            xmin, ymin, xmax, ymax = boxes[i].tolist()
            
            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            f.write(f"{cls_id} {conf} {x_center} {y_center} {width} {height}\n")

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object Detection Kaggle Competition")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()

    # Set working directory
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load test path from YAML
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' not in data or not data['test']:
            print("Add 'test: path/to/test/images' to yolo_params.yaml")
            exit()
        images_dir = Path(data['test'])
    
    # Validate test directory
    if not images_dir.exists():
        print(f"Test directory {images_dir} does not exist")
        exit()
    if not any(images_dir.glob('*')):
        print(f"Test directory {images_dir} is empty")
        exit()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    detect_path = this_dir / "outputs"
    model = StandardFasterRCNN(
        backbone="resnet50_v2",
        num_classes=2,
        device=device
        )
    
    # Load the parameters of the best model
    model = load_model(model, Path(args.model).parent, Path(args.model).name, device)
    
    # Directory with images to generate predictions
    output_dir = this_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create labels subdirectories
    labels_output_dir = output_dir / 'labels'
    
    # images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg','.jpeg']:
            continue
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(model, img_path, output_path_txt, device)

    print(f"Bounding box labels saved in {labels_output_dir}")

if __name__ == '__main__':
    main()