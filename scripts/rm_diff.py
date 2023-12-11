import os

# Set your directories here
dir1 = "./dataset/open-images/bbox/validation"
dir2 = "./dataset/open-images/images/validation"

# Get basenames of all .txt files in dir1 (without the extension)
txt_files = set(os.path.splitext(f)[0] for f in os.listdir(dir1) if f.endswith('.txt'))

# Loop through all .jpg files in dir2
for f in os.listdir(dir2):
    # Skip if not a .jpg file
    if not f.endswith('.jpg'):
        continue
    
    # Extract the basename (name without extension) of the file
    base = os.path.splitext(f)[0]

    # If there's no counterpart in dir1, delete the file
    if base not in txt_files:
        print(f"Deleting {f}...")
        os.remove(os.path.join(dir2, f))
