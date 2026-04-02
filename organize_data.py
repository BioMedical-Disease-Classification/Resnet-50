import os
import zipfile
import shutil
from pathlib import Path

def organize_data():
    """
    Extracts and organizes the Kaggle dataset zip file.
    Separates PNEUMONIA folder into BACTERIAL and VIRAL subfolders based on filename.
    """
    
    # Configuration
    zip_file = "kaggle_dataset.zip"
    extract_dir = "data"
    
    # Check if zip file exists
    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found in the current directory!")
        print("Make sure your zip file is in the folder!")
        return False
    
    print(f"Found {zip_file}. Extracting...")
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted to '{extract_dir}' folder.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False
    
    # Now organize PNEUMONIA folder into BACTERIAL and VIRAL
    base_path = os.path.join(extract_dir, "chest_xray", "chest_xray")
    
    if not os.path.exists(base_path):
        print(f"Warning: Expected data structure not found at {base_path}")
        return False
    
    # Process train, test, and val folders
    for split in ['train', 'test', 'val']:
        pneumonia_path = os.path.join(base_path, split, "PNEUMONIA")
        
        if not os.path.exists(pneumonia_path):
            print(f"Skipping {split} - PNEUMONIA folder not found")
            continue
        
        print(f"\nOrganizing {split} set...")
        
        # Create BACTERIAL and VIRAL subdirectories
        bacterial_path = os.path.join(base_path, split, "BACTERIAL")
        viral_path = os.path.join(base_path, split, "VIRAL")
        
        os.makedirs(bacterial_path, exist_ok=True)
        os.makedirs(viral_path, exist_ok=True)
        
        # Move files based on filename
        for filename in os.listdir(pneumonia_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            
            filepath = os.path.join(pneumonia_path, filename)
            
            if os.path.isfile(filepath):
                if 'bacteria' in filename.lower():
                    dest = os.path.join(bacterial_path, filename)
                    shutil.move(filepath, dest)
                elif 'virus' in filename.lower():
                    dest = os.path.join(viral_path, filename)
                    shutil.move(filepath, dest)
        
        print(f"  ✓ {split} set organized into BACTERIAL and VIRAL")
    
    print("\n" + "="*50)
    print("Data organization complete!")
    print("Your data is organized into:")
    print("  - NORMAL")
    print("  - BACTERIAL") 
    print("  - VIRAL")
    print("="*50)
    return True

if __name__ == "__main__":
    organize_data()
