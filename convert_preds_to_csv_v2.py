from pathlib import Path
import pandas as pd
import csv
import sys

def predictions_to_csv(
    preds_folder: str = "predictions/labels", 
    output_csv: str = "submission.csv", 
    test_images_folder: str = "Synthetic_to_Real_Object_Detection_Full/data/test/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    """
    Convert YOLO prediction files to Kaggle submission CSV format
    with strict validation.
    """
    # Validate inno boxputs
    preds_path = Path(preds_folder)
    if not preds_path.exists():
        print(f"ERROR: Prediction folder '{preds_folder}' does not exist")
        sys.exit(1)

    # Get test image IDs (without extensions)
    test_images_path = Path(test_images_folder)
    if not test_images_path.exists():
        print(f"ERROR: Test images folder '{test_images_folder}' not found")
        sys.exit(1)
        
    test_images = {
        p.stem: True 
        for p in test_images_path.glob("*") 
        if p.suffix.lower() in allowed_extensions
    }
    print(f"Found {len(test_images)} test images")

    # Collect predictions with validation
    predictions = []
    error_count = 0
    
    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        
        # Validate image_id
        if image_id not in test_images:
            print(f"Skipping non-test image prediction: {txt_file.name}")
            continue
            
        with open(txt_file, "r") as f:
            valid_lines = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                    
                parts = line.split()
                # Validate YOLO format: 6 values per line
                if len(parts) != 6:
                    print(f"Invalid prediction in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue
                    
                try:
                    # Validate numerical values
                    [float(x) for x in parts]
                    valid_lines.append(line)
                except ValueError:
                    print(f"Non-numeric values in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue

        pred_str = " ".join(valid_lines) if valid_lines else "no box"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})

    # Create submission dataframe
    submission_df = pd.DataFrame({"image_id": list(test_images.keys())})
    
    if predictions:
        preds_df = pd.DataFrame(predictions)
        final_df = submission_df.merge(preds_df, on="image_id", how="left").fillna("no boxes")
    else:
        final_df = submission_df
        final_df["prediction_string"] = "no boxes"

    # Save with CSV quoting rules
    final_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"\n Success! Submission saved to {output_csv}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Validation errors: {error_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO predictions to Kaggle CSV")
    parser.add_argument("--preds_folder", default="predictions/labels", help="Folder with prediction .txt files")
    parser.add_argument("--output_csv", default="submission.csv", help="Output CSV filename")
    parser.add_argument("--test_images_folder", default="Synthetic_to_Real_Object_Detection_Full/data/test/images", 
                      help="Path to test images directory")
    args = parser.parse_args()

    predictions_to_csv(
        preds_folder=args.preds_folder,
        output_csv=args.output_csv,
        test_images_folder=args.test_images_folder
    )