import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.RandomRotate90(p=1),
            #A.Rotate(limit=(-10,10), border_mode=0, p=1),
            #A.RandomSizedBBoxSafeCrop(width=224, height=224, p=1),
            A.ShiftScaleRotate(p=0.5, rotate_limit=(0,0), border_mode=0),
            #A.RandomCrop(width=200, height=200, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            #ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            #ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})