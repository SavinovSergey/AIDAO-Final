
import typing as t
import io

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils import (
    get_predictions, 
    create_dataloader,
    PRIVATE_DATA_FOLDER_PATH, 
    PRIVATE_DATA_DESCRIPTION_PATH,
)

BATCH_SIZE = 64
IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = (224, 224)


def pil_open(image_data: bytes) -> Image:
    return Image.open(io.BytesIO(image_data))


def preprocess(image_data: t.Optional[bytes]) -> torch.Tensor:
    return transforms.Compose([
        transforms.Lambda(pil_open),
        transforms.ToTensor(),
        transforms.Resize(RESIZE_SIZE),
        transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_STD),
    ])(image_data)

device = torch.device('cpu')
model = torch.load('baseline_damage.pt', map_location=device)

description = pd.read_csv(PRIVATE_DATA_DESCRIPTION_PATH, index_col='filename').sort_index()
# there is no real target in private data description
dummy_target = {key: 0 for key in description.index}

val_loader = create_dataloader(
    img_dir_path=PRIVATE_DATA_FOLDER_PATH,
    target_map=dummy_target,
    description=description,
    batch_size=BATCH_SIZE,
    preprocessor=preprocess,
    num_load_workers=0,
)

solution = get_predictions(model, device, val_loader)
solution = solution[['pass_id', 'prediction']].groupby('pass_id').max()
solution.to_csv('./predictions.csv')
