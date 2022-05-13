from pathlib import Path
from typing import List
from os import getcwd
from enum import Enum
import json
import pprint
import random

# RELATIVE_DATASET_PATH = r"../dataset"
# RELATIVE_SAVE_PATH = r"/coco_files"
DATASET_PATH = r"C:\Users\User\Downloads\TechAvia-2022\TechAvia-2022\dataset"
SAVE_PATH = r"C:\Users\User\Downloads\TechAvia-2022\TechAvia-2022\data_preprocessing\coco_files"


class SuperCategory:
    DEFECT = 'defect'
    OBJECT = 'object'
    AREA = 'area'


class Category:
    LABELLED = 'labelled'
    CLEAN = 'clean'
    IGNORE = 'ignore'
    BACKGROUND = 'background'
    LOPATKA = 'lopatka'
    SLED_OT_FREZI = 'sled_ot_frezi'
    ZABOINA = 'zaboina'
    CARAPINA = 'carapina'
    RISKA = 'riska'
    NADIR = 'nadir'
    CHERNOTA = 'chernota'


class CategoryId:
    LABELLED = 0
    CLEAN = 1
    IGNORE = 2
    BACKGROUND = 3
    LOPATKA = 4
    SLED_OT_FREZI = 5
    ZABOINA = 6
    CARAPINA = 7
    RISKA = 8
    NADIR = 9
    CHERNOTA = 10





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

    # labelled - area
    all_data['categories'].append({
        "id": CategoryId.LABELLED,
        "name": Category.LABELLED,
        "supercategory": SuperCategory.AREA,
    })

    # clean - area
    all_data['categories'].append({
        "id": CategoryId.CLEAN,
        "name": Category.CLEAN,
        "supercategory": SuperCategory.AREA,
    })

    # ignore - area
    all_data['categories'].append({
        "id": CategoryId.IGNORE,
        "name": Category.IGNORE,
        "supercategory": SuperCategory.AREA,
    })

    # background - area
    all_data['categories'].append({
        "id": CategoryId.BACKGROUND,
        "name": Category.BACKGROUND,
        "supercategory": SuperCategory.AREA,
    })

    # lopatka - object
    all_data['categories'].append({
        "id": CategoryId.LOPATKA,
        "name": Category.LOPATKA,
        "supercategory": SuperCategory.OBJECT,
    })

    # sled_ot_frezi - defect
    all_data['categories'].append({
        "id": CategoryId.SLED_OT_FREZI,
        "name": Category.SLED_OT_FREZI,
        "supercategory": SuperCategory.DEFECT,
    })

    # zaboina - defect
    all_data['categories'].append({
        "id": CategoryId.ZABOINA,
        "name": Category.ZABOINA,
        "supercategory": SuperCategory.DEFECT,
    })

    # carapina - defect
    all_data['categories'].append({
        "id": CategoryId.CARAPINA,
        "name": Category.CARAPINA,
        "supercategory": SuperCategory.DEFECT,
    })

    # riska - defect
    all_data['categories'].append({
        "id": CategoryId.RISKA,
        "name": Category.RISKA,
        "supercategory": SuperCategory.DEFECT,
    })

    # nadir - defect
    all_data['categories'].append({
        "id": CategoryId.NADIR,
        "name": Category.NADIR,
        "supercategory": SuperCategory.DEFECT,
    })

    # chernota - defect
    all_data['categories'].append({
        "id": CategoryId.CHERNOTA,
        "name": Category.CHERNOTA,
        "supercategory": SuperCategory.DEFECT,
    })

    return all_data


def get_category_id(label) -> int:
    if label == 'labelled':
        category_id = CategoryId.LABELLED
    elif label == 'clean':
        category_id = CategoryId.CLEAN
    elif label == 'ignore':
        category_id = CategoryId.IGNORE
    elif label == 'background':
        category_id = CategoryId.BACKGROUND
    elif label == 'lopatka':
        category_id = CategoryId.LOPATKA
    elif label == 'sled_ot_frezi':
        category_id = CategoryId.SLED_OT_FREZI
    elif label == 'zaboina':
        category_id = CategoryId.ZABOINA
    elif label == 'carapina':
        category_id = CategoryId.CARAPINA
    elif label == 'riska':
        category_id = CategoryId.RISKA
    elif label == 'nadir':
        category_id = CategoryId.NADIR
    elif label == 'chernota':
        category_id = CategoryId.CHERNOTA
    else:
        raise ValueError(f'Unknown label: \n\t{label}')
    # category_id = int(category_id)
    return category_id


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


def _find_polygon_bbox(
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


def _find_rectangle_bbox(
        points: List[List[int]],
        image_width: int,
        image_height: int,
) -> List[int]:
    x1, y1 = points[0]
    x2, y2 = points[1]
    # print(f"{points[0] = }")
    # print(f"{points[1] = }")
    if x1 >= x2 and y1 >= y2:
        # print("x1 >= x2 and y1 >= y2")
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    elif x1 < x2 and y1 >= y2:
        # print("x1 < x2 and y1 >= y2")
        y1, y2 = y2, y1
    elif x1 >= x2 and y1 < y2:
        # print("x1 >= x2 and y1 < y2")
        x1, x2 = x2, x1
    else:  # x1 < x2 and y1 < y2:
        # print("x1 < x2 and y1 < y2 (DEFAULT)")
        ...

    bbox = [x1, y1, x2 - x1, y2 - y1]

    if bbox[2] < 0 or bbox[3] < 0:
        print(f"{bbox = }")
        raise ValueError('INCORRECT width or height (cannot be less than zero)')
    # todo correct data into able borders! - DONE!
    elif bbox[0] < 0:
        print(f"{bbox = }")
        bbox[0] = 0
        # raise ValueError('DATA IS INCORRECT x < 0!')
    elif bbox[1] < 0:
        bbox[0] = 0
        print(f"{bbox = }")
        # raise ValueError('DATA IS INCORRECT y < 0!')
    elif bbox[0] + bbox[2] > image_width:
        print(f"{bbox = }")
        raise ValueError('DATA IS INCORRECT x > max!')
    elif bbox[1] + bbox[3] > image_height:
        print(f"{bbox = }")
        bbox[3] = image_height - bbox[1]
        # print(f"{bbox[3] = }")
        # raise ValueError(f'DATA IS INCORRECT! y > max! ({bbox[1] + bbox[3]} > {image_height})')


    #
    return bbox


def _transform_segmentation(points):
    return [[
        num
        for point in points
        for num in point
    ]]


def make_coco_json(files) -> dict:
    all_data = {}
    all_data['info'] = {
        "year": "2022",
        "version": "1.0",
        "description": "...",
        "contributor": "...",
        "url": "...",
        "date_created": "...",
    }

    all_data['categories'] = []
    all_data['images'] = []
    all_data['annotations'] = []

    all_data = add_categories(all_data)

    image_id = 0
    annotation_id = 0
    for file in files[:1]:  # fixme
        print(f'file #{image_id}. "{file}"')
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
                label: str = shape['label']
                print(f"0 {label = }")
                if label.endswith('_questionable'):
                    label = label[:-13]
                    print(f"1 {label = }")
                if label.endswith('_group'):
                    label = label[:-6]
                    print(f"2 {label = }")

                category_id = get_category_id(label)

                points: List[List[float]] = shape['points']
                points: List[List[int]] = _round_all_points(points)

                if shape['shape_type'] == 'polygon':
                    bbox = _find_polygon_bbox(points)
                    segmentation = _transform_segmentation(points)

                elif shape['shape_type'] == 'rectangle':
                    bbox = _find_rectangle_bbox(
                        points=points,
                        image_width=json_content['imageWidth'],
                        image_height=json_content['imageHeight'],
                    )
                    segmentation = None

                else:
                    raise ValueError(f'Unknown shape_type {shape["shape_type"] = }!!!')

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
        ...
        pprint.pprint(all_data)
        exit(1)
    return all_data


if __name__ == '__main__':
    print()
    # dataset_path = (Path(getcwd()) / RELATIVE_DATASET_PATH).resolve()
    dataset_path = Path(DATASET_PATH).resolve()
    print(f"{dataset_path = }")
    print("path = {}".format(str(dataset_path).replace('/', '\\')))

    files = get_all_files(dataset_path)
    # print(f"{files = }")
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

    # save_path = Path(getcwd() + RELATIVE_SAVE_PATH)
    save_path = Path(SAVE_PATH)

    with open(save_path / 'train_data_panoptic.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(save_path / 'test_data_panoptic.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)



