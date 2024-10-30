import os
import cv2
import json
import ntpath
import zipfile
import logging
import numpy as np
from pathlib import Path
from config import  OUT_DIR_OCR, S3_BUCKET, s3, KEY_PREFIX
from io import BytesIO

BUCKET_NAME = S3_BUCKET
S3_PREFIX = KEY_PREFIX

# Configuration
OUT_DIR = OUT_DIR_OCR
TYPES_TO_EXTRACT = ['measurement', 'parcel', 'year', 'coordinate']

# Define multiple prefixes and postfixes
PREFIX_POSTFIX_PAIRS = [
    ('observations/snapshots/latest/', '.latest.json'),
    ('observations/snapshots/TextBoxReader/', '.TextBoxReader.json'),
]

def upload_json_to_s3(json_data, prefix_name):
    """Upload JSON data to S3 in the specified 'io/Results/' directory."""
    s3_key = f'io/Results/combined_OCR_{prefix_name}.json'
    json_bytes = json.dumps(json_data).encode('utf-8')
    s3.upload_fileobj(BytesIO(json_bytes), BUCKET_NAME, s3_key)
    logging.info(f'Successfully uploaded JSON data to s3://{BUCKET_NAME}/{s3_key}')

def get_rotation_matrix(rotation, img_w, img_h):
    """Compute the rotation matrix for a given angle and image dimensions."""
    cX, cY = img_w // 2, img_h // 2
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), rotation, 1)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    n_width = int((img_h * sin) + (img_w * cos))
    n_height = int((img_h * cos) + (img_w * sin))

    rotation_matrix[0, 2] += (n_width / 2) - cX
    rotation_matrix[1, 2] += (n_height / 2) - cY

    return rotation_matrix


def rotate_point(x, y, rotation_matrix):
    """Rotate a point (x, y) using the given rotation matrix."""
    point = np.array([x, y, 1])
    rotated_point = np.dot(rotation_matrix, point)
    return int(rotated_point[0]), int(rotated_point[1])


def get_attachment_name(name):
    """Extract the attachment name from a given path."""
    return name.split('/')[2]


def read_zip(zip_name, ground_truth):
    """Process a zip file to extract JSON data and images."""
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=zip_name)
        with zipfile.ZipFile(BytesIO(response['Body'].read())) as archive:
            for prefix, postfix in PREFIX_POSTFIX_PAIRS:
                # Ensure there's a dictionary for the current prefix
                if prefix not in ground_truth:
                    ground_truth[prefix] = {}

                sketch_files = [
                    x for x in archive.namelist()
                    if x.startswith(prefix) and x.endswith(postfix)
                ]

                for sketch_file in sketch_files:
                    sketch_name = sketch_file[len(prefix):-len(postfix)]
                    attachment_prefix = 'observations/attachments/'
                    image_files = [
                        x for x in archive.namelist()
                        if x.startswith(attachment_prefix) and sketch_name in x
                    ]

                    # Load attachments as grayscale images
                    attachment_to_image = {
                        get_attachment_name(x): cv2.imdecode(
                            np.frombuffer(archive.read(x), np.uint8), 1
                        )
                        for x in image_files
                    }

                    # Read JSON data
                    json_data = json.loads(archive.read(sketch_file))
                    text = json_data.get('text', {})

                    for text_id, text_box in text.items():
                        text_type = text_box.get('type')
                        if text_type not in TYPES_TO_EXTRACT:
                            continue  # Skip types not in TYPES_TO_EXTRACT

                        value = text_box.get('value', '').strip()
                        if not value:
                            continue  # Skip if value is empty

                        bounding_box = text_box.get('box')
                        attachment = text_box.get('attachment')
                        image = attachment_to_image.get(attachment)

                        # Check if image or bounding box is missing
                        if image is None or bounding_box is None:
                            continue  # Skip if either is missing

                        [[text_x, text_y], [w, h], angle] = bounding_box

                        # Adjust width, height, and angle if necessary
                        if h > w:
                            w, h, angle = h, w, angle + 90

                        # Rotate the point without saving any cropped image
                        matrix = get_rotation_matrix(angle, image.shape[1], image.shape[0])
                        text_x, text_y = rotate_point(text_x, text_y, matrix)

                        # Save data to the ground truth dictionary for the current prefix
                        output_id = f"{sketch_name}_{text_id}_{text_type}"
                        ground_truth[prefix][output_id] = value
    except Exception as e:
        logging.error(f"Error reading zip from S3: {e}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s;%(levelname)s;%(message)s'
    )

    # Initialize ground truth combined dictionary
    ground_truth_combined = {}

    try:
        # List all zip files in the S3 bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
        
        if 'Contents' in response:
            for idx, obj in enumerate(response['Contents'], start=1):
                zip_name = obj['Key']
                logging.info(f'Processing project: {idx}/{len(response["Contents"])}: {zip_name}.')
                
                # Call the function to read the zip file from S3
                read_zip(zip_name, ground_truth_combined)  # Ensure you have a suitable function for this
        else:
            logging.info('No zip files found in the specified S3 bucket and prefix.')

    except Exception as e:
        logging.error(f"Error listing objects in S3: {e}")


    # Ensure the output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save the combined data to JSON files only for the specified prefixes
    for prefix, _ in PREFIX_POSTFIX_PAIRS:
        # Create distinct file names for each prefix
        prefix_name = prefix.strip('/').split('/')[-1]  # Get the last part of the prefix path
        data = ground_truth_combined.get(prefix, {})
        upload_json_to_s3(data, prefix_name)
