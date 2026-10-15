import splitfolders
from pathlib import Path

# TODO: Point this directly to the "Original Images" folder you downloaded
input_folder = "/home/nyx/Desktop/project_a1/dataset/archive/Dataset/BDPapayaLeaf/Original Images/" 
output_folder = "papaya_split"

print("Splitting dataset into train and val...")
# Creates an 80% training and 20% validation split
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.2))

print(f"Done! Dataset ready in: {Path(output_folder).absolute()}")