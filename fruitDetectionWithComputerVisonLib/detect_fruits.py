import json
from pathlib import Path
from typing import Dict
import click
import cv2
from tqdm import tqdm
import numpy as np
from cvlib import detect_common_objects



def detect_fruits(img_path: str) -> Dict[str, int]:

    img_path = cv2.imread(img_path)
    image = cv2.resize(img_path, None, fx=0.2, fy=0.2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    im = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    apples = count_apples(image)
    bbox, label, conf = detect_common_objects(im, confidence=0.2, nms_thresh=0.3, model="yolov3")
    bananas = 0
    oranges = 0
    for i in label:
        if i == "banana":
            bananas += 1

        elif i == "orange":
            oranges += 1


    return {'apple': apples, 'banana': bananas, 'orange': oranges}

def count_apples(image: np.ndarray):

    apples1 = cv2.inRange(image, (0, 109, 55), (8, 255, 157))
    apples2 = cv2.inRange(image, (40, 10, 0), (254, 255, 255))
    apples3 = cv2.inRange(image, (0, 138, 86), (19, 219, 182))
    apples = apples1 | apples2 | apples3

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    apples = cv2.morphologyEx(apples, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    apples = cv2.morphologyEx(apples, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours_apple, _ = cv2.findContours(apples, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    apple_count = 0

    for i in range(len(contours_apple)):

        if calculate_area(contours_apple[i], apples) > 7000:
            apple_count += 1
            cv2.drawContours(image, contours_apple, i, (255, 255, 0), 5)

    return apple_count

def calculate_area(contour, mask):
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], 0, 255, -1)

    return np.count_nonzero(mask[contour_mask == 255])

@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
