import zipfile
import os

def unzip_and_rename(zip_file_path, extract_to_dir):
    # Ensure the extraction directory exists
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)

    # Rename extracted files to 'data'
    for filename in os.listdir(extract_to_dir):
        file_path = os.path.join(extract_to_dir, filename)
        new_file_path = os.path.join(extract_to_dir, 'data')
        os.rename(file_path, new_file_path)
        break  # Assuming there's only one file in the zip

if __name__ == "__main__":
    zip_file_path = './file.zip'  # Replace with your zip file path
    extract_to_dir = './data'  # Replace with your extraction directory
    unzip_and_rename(zip_file_path, extract_to_dir)