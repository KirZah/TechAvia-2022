from os import getcwd
from pathlib import Path
import json
import pprint
from typing import List
import random

# RELATIVE_DATASET_PATH = r"../dataset"
DATASET_PATH = r"C:\Users\User\Downloads\TechAvia-2022\TechAvia-2022\dataset"
RELATIVE_SAVE_PATH = r"/coco_files"


def get_all_files(directory: Path):
    files = []
    pathlist = Path(directory).glob('*.json')
    for path in pathlist:
         # because dataset_path is object not string
         path_in_str = str(path)
         # print(path_in_str)
         files.append(path_in_str)
    return files


def add_categories(all_data):
    all_data['categories'].append({
        "id": 0,
        "name": "carapina",
        "supercategory": "defect",
    })
    all_data['categories'].append({
        "id": 1,
        "name": "lopatka",
        "supercategory": "object",
    })
    return all_data


def add_image(all_data, id, filename):
    all_data['images'].append({
        "id": id,
        # "license": 1,
        "file_name": filename,
        "height": 3672,
        "width": 5496,
        "date_captured": None,
    })
    return all_data


def add_annotation(all_data, id, image_id, category_id, bbox, segmentation):
    area = bbox[2] * bbox[3]
    all_data['annotations'].append({
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,  # [x, y, w, h],
        "segmentation": segmentation,  # [...]
        "area": area,
        "iscrowd": 0,
    })
    return all_data


def _round_all_points(points: List[List[float]]) -> List[List[int]]:
    points = [
        [round(point[0]), round(point[1])]
        for point in points
    ]
    return points


def _find_bbox(
        points: List[List[int]],
) -> List[int]:
    x_min = float('+inf')
    y_min = float('+inf')
    x_max = float('-inf')
    y_max = float('-inf')
    for x, y in points:
        if x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def make_coco_json(files) -> dict:
    all_data = {}
    all_data['info'] = {
        "year": "2022",
        "version": "1.0",
        "description": "...",
        "contributor": "...",
        "url": "...",
        "date_created": "..."
    }

    all_data['categories'] = []
    all_data['images'] = []
    all_data['annotations'] = []

    all_data = add_categories(all_data)

    image_id = 0
    annotation_id = 0
    for file in files:
        print(f'file #{image_id}')
        with open(file, 'r') as f:
            json_content: dict = json.load(f)
            # print(type(json_content))
            # pprint.pprint(json_content)

            all_data = add_image(
                all_data=all_data,
                id=image_id,
                # filename=file.split('\\')[-1],
                filename=json_content['imagePath'],
            )

            shapes: list = json_content['shapes']
            for shape in shapes:  # dict in list
                if shape['label'] == 'carapina':
                    category_id = 0
                elif shape['label'] == 'lopatka':
                    category_id = 1

                else:
                    raise ValueError(f'Unknown shape {shape = }!!!')

                points: List[List[float]] = shape['points']
                points: List[List[int]] = _round_all_points(points)

                bbox = _find_bbox(points)
                segmentation = points

                all_data = add_annotation(
                    all_data,
                    id=annotation_id,
                    image_id=image_id,
                    category_id=category_id,
                    bbox=bbox,
                    segmentation=segmentation,
                )
                ...
                annotation_id += 1

        ...
        image_id += 1
    return all_data


if __name__ == '__main__':
    print()
    # dataset_path = (Path(getcwd()) / RELATIVE_DATASET_PATH).resolve()
    dataset_path = Path(DATASET_PATH).resolve()
    print(f"{dataset_path = }")
    print("path = {}".format(str(dataset_path).replace('/', '\\')))

    files = get_all_files(dataset_path)
    print(f"{files = }")
    random.shuffle(files)

    TRAIN_PERCENTAGE = 0.8
    train_data_length = round(TRAIN_PERCENTAGE * len(files))
    print(f"{train_data_length = }")

    train_files = []
    test_files = []
    for i, file in enumerate(files):
        if i < train_data_length:
            train_files.append(file)
        else:
            test_files.append(file)


    # for file in files:
    #     if random.random() < 0.8:
    #         train_files.append(file)
    #     else:
    #         test_file

    print(f"train: {len(train_files)} (~{100 * len(train_files) / (len(test_files) + len(train_files))} %)")
    print(f"test: {len(test_files)} (~{100 * len(test_files) / (len(test_files) + len(train_files))} %)")
    # print(f"test: {len(test_files)}")
    same_files = set(train_files) & set(test_files)
    if same_files:
        raise ValueError(f"There are same files in train and test data")
    # input("Press 'Enter' to start making coco files...")

    print("=============================")
    print("=============================")
    print("=============================")
    # pprint.pprint(all_data)
    train_data = make_coco_json(train_files)
    test_data = make_coco_json(test_files)

    save_path = Path(getcwd() + RELATIVE_SAVE_PATH)

    with open(save_path / 'train_data.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(save_path / 'test_data.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)



