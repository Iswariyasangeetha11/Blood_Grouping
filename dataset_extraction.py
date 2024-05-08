import pandas as pd
import cv2
import os
import csv
import numpy as np

# Function to calculate color structure features
def calculate_color_structure(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   
    # Calculate mean and standard deviation of each channel
    mean_lab = np.mean(lab_image, axis=(0, 1))
    std_lab = np.std(lab_image, axis=(0, 1))
   
    return mean_lab, std_lab

# Create a folder to store the dataset images
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

# List to store information about each ROI
roi_info = []

# List of image paths
image_paths = ["C:/Users/sedhu/Downloads/20240503122654.jpg",
               "C:/Users/sedhu/Downloads/20240503122653.jpg",
               "C:/Users/sedhu/Downloads/20240503122549 (1).jpg",
               "C:/Users/sedhu/Downloads/20240503122550 (1).jpg"]

# Iterate over image paths
for i, img_path in enumerate(image_paths, start=1):
    # Load image
    img = cv2.imread(img_path)
    rows, cols, _ = img.shape
    print(f"Processing image {i} - Rows: {rows}, Cols: {cols}")

    # Define rectangles for the ROIs
    roi1_rect = [(78, 155), (140, 230)]
    roi2_rect = [(160, 155), (220, 230)]
    roi3_rect = [(260, 155), (320, 230)]  # Adjusted rectangle for roi3

    # Draw rectangles on the image
    cv2.rectangle(img, tuple(roi1_rect[0]), tuple(roi1_rect[1]), (0, 255, 0), 2)
    cv2.rectangle(img, tuple(roi2_rect[0]), tuple(roi2_rect[1]), (0, 255, 0), 2)
    cv2.rectangle(img, tuple(roi3_rect[0]), tuple(roi3_rect[1]), (0, 255, 0), 2)

    # Extract ROIs from the image using rectangles
    roi1 = img[roi1_rect[0][1]:roi1_rect[1][1], roi1_rect[0][0]:roi1_rect[1][0]]
    roi2 = img[roi2_rect[0][1]:roi2_rect[1][1], roi2_rect[0][0]:roi2_rect[1][0]]
    roi3 = img[roi3_rect[0][1]:roi3_rect[1][1], roi3_rect[0][0]:roi3_rect[1][0]]

    # Calculate color structure features for each ROI
    roi1_mean_lab, roi1_std_lab = calculate_color_structure(roi1)
    roi2_mean_lab, roi2_std_lab = calculate_color_structure(roi2)
    roi3_mean_lab, roi3_std_lab = calculate_color_structure(roi3)

    # Ask user to input blood type for each ROI
    blood_type_roi1 = input(f"Enter blood type for ROI 1 in image {i} (0 or 1): ")
    blood_type_roi2 = input(f"Enter blood type for ROI 2 in image {i} (0 or 1): ")
    blood_type_roi3 = input(f"Enter blood type for ROI 3 in image {i} (0 or 1): ")

    # Validate input blood type
    if blood_type_roi1 not in ['0', '1'] or blood_type_roi2 not in ['0', '1'] or blood_type_roi3 not in ['0', '1']:
        print("Invalid input! Blood type must be either 0 or 1.")
        continue

    # Save each ROI as a separate image file in the dataset folder
    cv2.imwrite(os.path.join(dataset_folder, f"image_{i}_roi1.jpg"), roi1)
    cv2.imwrite(os.path.join(dataset_folder, f"image_{i}_roi2.jpg"), roi2)
    cv2.imwrite(os.path.join(dataset_folder, f"image_{i}_roi3.jpg"), roi3)

    # Append information about each ROI to the list
    roi_info.append((f"image_{i}_roi1.jpg", *roi1_mean_lab, *roi1_std_lab, int(blood_type_roi1)))
    roi_info.append((f"image_{i}_roi2.jpg", *roi2_mean_lab, *roi2_std_lab, int(blood_type_roi2)))
    roi_info.append((f"image_{i}_roi3.jpg", *roi3_mean_lab, *roi3_std_lab, int(blood_type_roi3)))

# Write the information to a CSV file with semicolon delimiter
csv_file_path = "H:/Ishwariya/dataset/dataset_info.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Image Name", "Mean L", "Mean A", "Mean B", "Std L", "Std A", "Std B", "Blood Type"])
    writer.writerows(roi_info)

print("CSV file saved to dataset folder.")

# Convert CSV to Excel
excel_file_path = "H:/Ishwariya/dataset/dataset_info.xlsx"
df = pd.read_csv(csv_file_path, delimiter=';')
df.to_excel(excel_file_path, index=False)

print("Excel file saved to dataset folder.")
