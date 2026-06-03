import os
import zipfile


def create_replaygain3_zip():
    zip_filename = "replaygain3.zip"
    archive_folder = "replaygain3"

    # Define the files and their targeted names inside the zip archive
    files_to_zip = {
        "__init__.py": "__init__.py",
        "ui_options.py": "ui_options_replaygain3.py",
        "ui_options.ui": "ui_options_replaygain3.ui",
    }

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_file, archive_name in files_to_zip.items():
            if os.path.exists(local_file):
                # Construct the internal path: replaygain3/filename
                internal_path = os.path.join(archive_folder, archive_name)
                zipf.write(local_file, arcname=internal_path)
            else:
                print(f"Warning: {local_file} not found. Skipping.")

    print(f"Successfully created {zip_filename}")


def build():
    create_replaygain3_zip()


if __name__ == "__main__":
    build()
