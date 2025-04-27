"""
Contains functionality for creating PyTorch DataLoaders for object dectection.
"""

import os
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# Implement a class that processes the database
class ProcessDatasetCheerios(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading and processing image and mask pairs for object detection.
    """

    def __init__(self, root, image_path, label_path, transforms=None, num_classes=1):
        self.root = root
        self.image_path = image_path
        self.label_path = label_path
        self.transforms = transforms
        self.num_classes = num_classes

        # Load all image and label files, ensuring alignment
        self.imgs = sorted(os.listdir(os.path.join(root, image_path)))
        self.labels = sorted(os.listdir(os.path.join(root, label_path)))

    def yolo_to_xyxy(self, label_path, img_width, img_height):
        bboxes_yolo = []
        bboxes_xyxy = []
        with open(label_path, "r") as file:
            for line in file:
                data = line.strip().split()
                class_id = int(data[0])
                x_centre, y_centre, width, height = map(float, data[1:])

                # Convert YOLO to COCO format
                x_min = (x_centre - width / 2) * img_width
                y_min = (y_centre - height / 2) * img_height
                x_max = (x_centre + width / 2) * img_width
                y_max = (y_centre + height / 2) * img_height
                
                bboxes_yolo.append((class_id, x_centre, y_centre, width, height))
                bboxes_xyxy.append((class_id, x_min, y_min, x_max, y_max))

        return torch.tensor(bboxes_yolo, dtype=torch.float32), torch.tensor(bboxes_xyxy, dtype=torch.float32)

    def __getitem__(self, idx):

        # Load images and labels
        img_path = os.path.join(self.root, self.image_path, self.imgs[idx])
        label_path = os.path.join(self.root, self.label_path, self.labels[idx])

        # Read image
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        # Image dimensions
        img_height, img_width = img.shape[1], img.shape[2]

        # Get bounding boxes
        bboxes_yolo, bboxes_xyxy = self.yolo_to_xyxy(label_path, img_width, img_height)

        # There is only one class
        num_objs = bboxes_xyxy.size(0)

        # Calculate labels
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Ensure bounding boxes exist
        if num_objs > 0:
            bboxes_xyxy = bboxes_xyxy[:, 1:]  # Remove class_id
            xmin, ymin, xmax, ymax = bboxes_xyxy.unbind(dim=1)
            bboxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            bboxes_yolo = bboxes_yolo[:, 1:]
            xc, yc, w, h = bboxes_yolo.unbind(dim=1)
            bboxes_yolo = torch.stack([xc, yc, w, h], dim=1)
        else:
            # Handle empty cases
            bboxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            bboxes_yolo = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        # Create the target
        target = {
            "boxes":      tv_tensors.BoundingBoxes(bboxes_xyxy, format="XYXY", canvas_size=F.get_size(img)),
            "labels":     labels,
        }

        # Preprocess image and target
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
