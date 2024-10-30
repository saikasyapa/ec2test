import os
import cv2
import json
import zipfile
import logging
import numpy as np
from pathlib import Path
from config import OUT_DIR_TEXT_BOX, OUT_DIR_TEXT_BOX2, S3_BUCKET, KEY_PREFIX, s3, output, KEY_PREFIX1
from io import BytesIO


BUCKET_NAME = S3_BUCKET
S3_PREFIX = KEY_PREFIX

# Configuration Constants
OUT_DIR_1 = OUT_DIR_TEXT_BOX  # Output directory for first prefix/postfix
OUT_DIR_2 = OUT_DIR_TEXT_BOX2  # Output directory for second prefix/postfix


# Color mappings
COLORS = {
    'measurement': [0, 0, 255],    # Red (BGR)
    'red_parcel': [0, 255, 0],     # Green (BGR)
    'black_parcel': [255, 0, 0],   # Blue (BGR)
    'blue_parcel': [0, 255, 255],  # Yellow (BGR)
    'coordinate': [255, 0, 255],   # Magenta (BGR)
    'year': [255, 255, 0],         # Cyan (BGR)
}

def generate_box_mask(box, mask_shape):
    rct = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
    corner_points = cv2.boxPoints(rct).astype(np.int32)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, corner_points, 1)
    return mask

def generate_parcel_mask_from_json(obs, color, image_shape):
    filtered_texts = {
        k: item for k, item in obs['text'].items()
        if item['type'] == 'parcel' and item['color'] == color
    }
    masks = [
        generate_box_mask(filtered_texts[text]['box'], image_shape)
        for text in filtered_texts
    ]
    return masks

def generate_mask_from_json(obs, text_type, image_shape):
    filtered_texts = {
        key: item for key, item in obs['text'].items()
        if item['type'] == text_type
    }
    masks = [
        generate_box_mask(filtered_texts[text]['box'], image_shape)
        for text in filtered_texts
    ]
    return masks

def create_color_mask(categories_to_instances, shape):
    color_mask = np.zeros((*shape, 3), dtype=np.uint8)
    for category, masks in categories_to_instances.items():
        if category in COLORS:
            color = COLORS[category]
            for mask in masks:
                color_mask[mask > 0] = color
    return color_mask

def upload_to_s3(file_path, s3_key):
    """Upload a file to S3."""
    try:
        s3.upload_file(file_path, BUCKET_NAME, s3_key)
        logging.info(f'Successfully uploaded {file_path} to s3://{BUCKET_NAME}/{s3_key}')
    except Exception as e:
        logging.error(f'Error uploading {file_path} to S3: {e}')

def upload_mask_to_s3(mask_image, sketch_name):
    """Upload mask image to S3."""
    s3_key = f'{KEY_PREFIX1}/masks/{sketch_name}_mask.png'  # Adjusted to save in io/Results/
    _, buffer = cv2.imencode('.png', mask_image)
    s3.upload_fileobj(BytesIO(buffer), BUCKET_NAME, s3_key)
    logging.info(f'Successfully uploaded mask to s3://{BUCKET_NAME}/{s3_key}')

def read_zip(zip_name, prefix, postfix):
    logging.info(f'Reading zip file: {zip_name}')
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=zip_name)
        with zipfile.ZipFile(BytesIO(response['Body'].read())) as archive:
            sketch_files = [
                x for x in archive.namelist()
                if x.startswith(prefix) and x.endswith(postfix)
            ]
            logging.info(f'Found {len(sketch_files)} sketch files with prefix "{prefix}" and postfix "{postfix}".')
            for i, sketch_file in enumerate(sketch_files):
                sketch_name = sketch_file[len(prefix):-len(postfix)]
                logging.info(f'Processing sketch: {i + 1}/{len(sketch_files)}: {sketch_name, postfix}.')

                img_pf = f'observations/attachments/front/{sketch_name}'
                img_extensions = ['.JPG', '.png']  # List of possible image extensions
                img_files = []

                # Collect all image files that match the prefix and extensions
                for ext in img_extensions:
                    img_files += [x for x in archive.namelist() if x.startswith(img_pf) and x.endswith(ext)]

                if img_files:
                    try:
                        # Use the first image file found
                        file = archive.read(img_files[0])
                        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

                        # Check if the image was loaded correctly
                        if image is None:
                            raise ValueError(f'Image could not be decoded from file: {img_files[0]}')

                        image_shape = image.shape[:2]

                        # Load JSON data
                        with archive.open(sketch_file, 'r') as fh:
                            json_data = json.loads(fh.read())

                        # Generate masks
                        categories_to_instances = {}
                        for color in ['red', 'blue', 'black']:
                            categories_to_instances[f'{color}_parcel'] = \
                                generate_parcel_mask_from_json(json_data, color, image_shape)

                        categories_to_instances['measurement'] = \
                            generate_mask_from_json(json_data, 'measurement', image_shape)

                        categories_to_instances['coordinate'] = \
                            generate_mask_from_json(json_data, 'coordinate', image_shape)

                        categories_to_instances['year'] = \
                            generate_mask_from_json(json_data, 'year', image_shape)

                        # Create color mask and upload to S3
                        color_mask = create_color_mask(categories_to_instances, image_shape)
                        upload_mask_to_s3(color_mask, sketch_name)  # Call with sketch name

                    except Exception as e:
                        logging.error(f'Error processing image for sketch {sketch_name}: {e}')
                else:
                    logging.warning(f'No image files found for sketch: {sketch_name}')
    except Exception as e:
        logging.error(f'Error reading zip file {zip_name}: {e}')

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    try:
        # List all relevant JSON files in the S3 bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

        if 'Contents' in response:
            projects = response['Contents']
            for j, obj in enumerate(projects):
                zip_name = obj['Key']
                logging.info(f'Processing project {j + 1} / {len(projects)}: {zip_name}.')

                # Define prefixes and postfixes
                prefix_1 = 'observations/snapshots/latest/'
                postfix_1 = '.latest.json'
                prefix_2 = 'observations/snapshots/TextboxDetection/'
                postfix_2 = '.TextboxDetection.json'

                # Read zip for first prefix/postfix
                read_zip(zip_name, prefix_1, postfix_1)
                # Read zip for second prefix/postfix
                read_zip(zip_name, prefix_2, postfix_2)

        else:
            logging.info('No relevant files found in the specified S3 bucket and prefix.')

    except Exception as e:
        logging.error(f"Error listing objects in S3: {e}")

