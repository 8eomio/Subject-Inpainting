import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import argparse
import os 
import torch 

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {box}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")
        

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")
def main():
    warnings.filterwarnings("ignore")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--target_dir", type=str, help = "target directory")
        parser.add_argument("--text_prompt", type=str, help = "subject prompt")
        args = parser.parse_args()
        target_dir = args.target_dir
        text_prompt = args.text_prompt
        image_dir = os.path.join(target_dir, "images")
        bbox_dir = os.path.join(target_dir, "bbox")
        mask_dir = os.path.join(target_dir, "mask")
        images = os.listdir(image_dir)
        if not os.path.exists(bbox_dir):
            os.mkdir(bbox_dir)
            print(f"Directory {bbox_dir} created.")
        else:
            print(f"Directory {bbox_dir} already created.")
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
            print(f"Directory {mask_dir} created.")
        else:
            print(f"Directory {mask_dir} already created.")
        model = LangSAM()
        for idx, image in enumerate(images):
            print(f"Processing {image}")
            if image.startswith('.'):
                continue
            # Suppress warning messages
            warnings.filterwarnings("ignore")
            image_file = os.path.join(image_dir, image)
            image_pil = Image.open(image_file).convert("RGB")
  
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            if len(masks) == 0:
                print(f"No objects of the '{text_prompt}' prompt detected in the image.")
            else:
# Convert masks to numpy arrays
                masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
                
# Save the masks
                for i, mask_np in enumerate(masks_np):
                    mask_path = os.path.join(mask_dir, f"{image[:-4]}.jpg")
                    save_mask(mask_np, mask_path)
            
            for i, box in enumerate(boxes):
                box_name = f"{image[:-4]}.txt"
                file_path = os.path.join(bbox_dir, box_name)
                with open(file_path, "w") as file:
                    file.write(' '.join(map(str, box.tolist())))
                    print(f"bbox file {file_path} created")
            # Print the bounding boxes, phrases, and logits
            #print_bounding_boxes(boxes)
            #print_detected_phrases(phrases)
            #print_logits(logits)
            print(f"finished")
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
