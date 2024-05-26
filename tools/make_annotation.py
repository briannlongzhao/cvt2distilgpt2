from tqdm import tqdm
from pathlib import Path
import json

split = "train"
task = "comparison"
input_path = Path(f"/scratch/xinyangjiang/datasets/MIMIC-CXR/merged_instructions_{split}.json")
output_path = Path(f"datasets/mimic_cxr_chen/merged_annotation.json")

images_path = Path("/scratch/xinyangjiang/datasets/MIMIC-CXR/images_all.json")
images_dict = json.load(open(images_path))
input_file = open(input_path)
input_json = json.load(input_file)["data"]
output_json = {"train": [], "val": [], "test":[]}


count = 0
for report_id, example in tqdm(input_json.items()):
    image_id = example["image_ids"][0]
    data = {
        "id": image_id,
        'study_id': report_id,
        'subject_id': int(images_dict[image_id][11:19]),
        'report': example["answer"],
        'image_path': [images_dict[image_id][6:]],
        "prompt": example["instruction"],
        'split': split
    }
    output_json[split].append(data)

    if count == 10:
        break

if split != "test":
    json_path = Path("/scratch/xinyangjiang/datasets/MIMIC-CXR/instructions_test.json")
    json_file = open(json_path)
    input_json = json.load(json_file)["data"]
    for report_id, example in tqdm(input_json.items()):
        image_id = example["image_ids"][0]
        data = {
            "id": image_id,
            'study_id': int(report_id[1:]),
            'subject_id': int(images_dict[image_id][11:19]),
            'report': example["answer"],
            'image_path': [images_dict[image_id][6:]],
            "prompt": example["instruction"],
            'split': "test"
        }
        output_json["test"].append(data)
        output_json["val"].append(data)

with open(output_path, 'w') as f:
    json.dump(output_json, f)