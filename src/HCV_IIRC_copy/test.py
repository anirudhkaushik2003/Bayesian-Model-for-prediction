import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import json
from PIL import Image

root = "ISIC_2019"
metadata_root = "iirc/metadata"
raw_data_meta_df = pd.read_csv(os.path.join(root, "ISIC_2019_Training_GroundTruth.csv"))

intask_valid_train_ratio = 0.1
posttask_valid_train_ratio= 0.1

isic_data_map = {
    "MEL": "Melanoma",  
    "NV": "Melanocytic_nevus" ,
    "BCC": "Basal_cell_carcinoma",
    "AK": "Actinic_keratosis",
    "BKL": "Benign_keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular_lesion",
    "SCC": "Squamous_cell_carcinoma"
}

labels = list(raw_data_meta_df.columns[1:-1])
class_to_idx = {isic_data_map[label]: idx for idx, label in enumerate(labels)}

X = raw_data_meta_df.iloc[:]['image'] # only image names, not actual images
y = raw_data_meta_df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

raw_data_train = []
for ind  in range(len(X_train)):
    img_name = X_train.iloc[ind]
    labels = y_train.iloc[ind]
    label = labels[labels == 1].index[0]
    if label == "UNK":
        continue
    image = img_name 
    label = class_to_idx[isic_data_map[label]] 
    raw_data_train.append((image, label))


raw_data_test = []
for ind  in range(len(X_test)):
    img_name = X_test.iloc[ind]
    labels = y_test.iloc[ind]
    label = labels[labels == 1].index[0]
    if label == "UNK":
        continue
    image = img_name
    label = class_to_idx[isic_data_map[label]]
    raw_data_test.append((image, label))

train_targets = np.array(list(zip(*raw_data_train))[1])
test_targets = np.array(list(zip(*raw_data_test))[1])

with open(os.path.join(metadata_root, "isic_2019_hierarchy.json"), "r") as f:
    class_hierarchy = json.load(f)

datasets = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}


superclass_to_subclasses = class_hierarchy['super_classes']
subclasses_to_superclass = {subclass: superclass for superclass, subclasses in superclass_to_subclasses.items()
                            for subclass in subclasses}
for subclass in class_hierarchy['other_sub_classes']:
    subclasses_to_superclass[subclass] = None

for subclass in subclasses_to_superclass.keys():
    samples_idx = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}
    superclass = subclasses_to_superclass[subclass]
    subclass_id = class_to_idx[subclass]
    samples_idx["test"] = np.argwhere(test_targets == subclass_id).squeeze(1)
    samples_idx["train"] = np.argwhere(train_targets == subclass_id).squeeze(1)

    original_train_len = len(samples_idx["train"])
    intask_valid_offset = int(intask_valid_train_ratio * original_train_len)
    posttask_valid_offset = int(posttask_valid_train_ratio * original_train_len) + intask_valid_offset
    samples_idx["intask_valid"] = samples_idx["train"][:intask_valid_offset]
    samples_idx["posttask_valid"] = samples_idx["train"][intask_valid_offset:posttask_valid_offset]
    samples_idx["train"] = samples_idx["train"][posttask_valid_offset:]

    assert "test" in datasets.keys()
    for dataset_type in datasets.keys():
        if dataset_type == "test":
            raw_data = raw_data_test
        else:
            raw_data = raw_data_train
        for idx in samples_idx[dataset_type]:
            image = raw_data[idx][0]
            if superclass is None:
                datasets[dataset_type].append((image, (subclass,)))
            else:
                datasets[dataset_type].append((image, (superclass, subclass)))


print()

