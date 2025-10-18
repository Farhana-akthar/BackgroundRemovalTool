import cv2
from rembg import remove
from PIL import Image
import os

# Make sure output folder exists
os.makedirs("output_images", exist_ok=True)

# Ask user for input file name
file_name = input("Enter the image filename (inside input_images): ")

input_path = os.path.join("input_images", file_name)

# For OpenCV background removal (just for comparison)
def remove_bg_opencv(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(thresh)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, result)
    print(f"OpenCV: Background removed → {output_path}")

# For AI background removal using rembg
def remove_bg_ai(input_path, output_path):
    with open(input_path, "rb") as i:
        input_image = i.read()
        output_image = remove(input_image)
        with open(output_path, "wb") as o:
            o.write(output_image)
    print(f"AI Model: Background removed → {output_path}")

# Output file paths
output_path_opencv = os.path.join("output_images", "output_opencv.png")
output_path_ai = os.path.join("output_images", "output_ai.png")

# Run both methods
remove_bg_opencv(input_path, output_path_opencv)
remove_bg_ai(input_path, output_path_ai)

print("\n✅ Background removal completed successfully!")


