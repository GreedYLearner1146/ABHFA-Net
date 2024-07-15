from torchvision import transforms
import torchvision.transforms as T

from torchvision.transforms import (
    Compose,
    RandomApply,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)

###################### Train data augmentation ###########################################

color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
blur = T.GaussianBlur((3, 3), (0.1, 2.0))

data_transform = transforms.Compose(

            [
            transforms.ToTensor(),
            transforms.Resize((128,128)),
            transforms.CenterCrop((128,128)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
            transforms.RandomApply([RandomRotation((270, 270))], p=0.5),
            transforms.RandomApply([RandomRotation((180, 180))], p=0.5),
            transforms.RandomApply([RandomHorizontalFlip(), RandomRotation((90, 90))], p=0.5),
            transforms.RandomApply([RandomHorizontalFlip(), RandomRotation((270, 270))], p=0.5),
            transforms.RandomApply([RandomHorizontalFlip(), RandomRotation((180, 180))], p=0.5),
            T.RandomApply([color_jitter], p=0.5),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ]
)

###################### Test data augmentation ###########################################

data_transform_test = transforms.Compose(

            [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
)
