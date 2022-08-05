import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np

def calculate_area(contour, mask):
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], 0, 255, -1)

    return np.count_nonzero(mask[contour_mask == 255])

def detect_fruits(img_path: str) -> Dict[str, int]:


    image = cv2.imread(img_path)

    image = cv2.resize(image, None, fx=0.2, fy=0.2)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    bananas = cv2.inRange(image_hsv, (19, 89, 0), (39, 255, 255))
    orange = cv2.inRange(image_hsv, (10, 178, 118), (17, 255, 255))

    apples1 = cv2.inRange(image_hsv, (0, 109, 55), (8, 255, 157))

    apples2 = cv2.inRange(image_hsv, (40, 10, 0), (254, 255, 255))

    apples3 = cv2.inRange(image_hsv, (0, 138, 86), (19, 219, 182))

    apples = apples1 | apples2 | apples3


    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bananas = cv2.morphologyEx(bananas, cv2.MORPH_OPEN, kernel)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, kernel)
    apples = cv2.morphologyEx(apples, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bananas = cv2.morphologyEx(bananas, cv2.MORPH_CLOSE, kernel, iterations=3)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel, iterations=3)
    apples = cv2.morphologyEx(apples, cv2.MORPH_CLOSE, kernel, iterations=3)

    # zliczanie
    contours_banana, _ = cv2.findContours(bananas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_orange, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_apple, _ = cv2.findContours(apples, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    banana_count = 0
    orange_count = 0
    apple_count = 0
    for i in range(len(contours_banana)):

        if calculate_area(contours_banana[i], bananas) > 3000:
            banana_count += 1
            cv2.drawContours(image, contours_banana, i, (255, 0, 0), 5)

    for i in range(len(contours_apple)):

        if calculate_area(contours_apple[i], apples) > 7000:
            apple_count += 1
            cv2.drawContours(image, contours_apple, i, (255, 255, 0), 5)

    for i in range(len(contours_orange)):

        if calculate_area(contours_orange[i], orange) > 7000:
            orange_count += 1
            cv2.drawContours(image, contours_orange, i, (255, 0, 255), 5)

    cv2.waitKey()


    #TODO: Implement detection method.
    


    return {'apple': apple_count, 'banana': banana_count, 'orange': orange_count}


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
