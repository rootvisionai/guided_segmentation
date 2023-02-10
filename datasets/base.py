import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2, os, torchvision, random

def create_transforms(cfg, eval = True):
    train_transform = A.Compose(
        [
            A.Resize(cfg.input_size, cfg.input_size),
            
            A.ShiftScaleRotate(
                shift_limit=cfg.augmentations.ShiftScaleRotate.shift_limit, 
                scale_limit=cfg.augmentations.ShiftScaleRotate.scale_limit, 
                rotate_limit=cfg.augmentations.ShiftScaleRotate.rotate_limit, 
                p=cfg.augmentations.ShiftScaleRotate.probability
            ),

            A.RGBShift(
                r_shift_limit=cfg.augmentations.RGBShift.r_shift_limit, 
                g_shift_limit=cfg.augmentations.RGBShift.g_shift_limit, 
                b_shift_limit=cfg.augmentations.RGBShift.b_shift_limit, 
                p=cfg.augmentations.RGBShift.probability
            ),

            A.RandomBrightnessContrast(
                brightness_limit=cfg.augmentations.RandomBrightnessContrast.brightness_limit,
                contrast_limit=cfg.augmentations.RandomBrightnessContrast.contrast_limit, 
                p=cfg.augmentations.RandomBrightnessContrast.probability
            ),
            ToTensorV2()
        ]
    )

    if eval:
        val_transform = A.Compose(
            [
                A.Resize(cfg.input_size, cfg.input_size),
                ToTensorV2()
            ]
        )

        return train_transform, val_transform

    else:
        return train_transform

def create_dataloaders(images_directory, masks_directory, transform,
                       batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True):

    set = Set(images_directory, masks_directory, transform=transform)

    data_loader = DataLoader(
        set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return data_loader

class Set(Dataset):
    def __init__(self, images_directory, masks_directory, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.images_filenames = [os.path.join(images_directory, elm) for elm in os.listdir(images_directory)]
        self.masks_filenames  = [os.path.join(masks_directory, elm) for elm in os.listdir(images_directory)]
        self.split_by_class()
        self.to_tensor = torchvision.transforms.ToTensor()

    def split_by_class(self):
        self.class_sets = {}
        for cnt, elm in enumerate(self.images_filenames):
            key = elm.split("-")[-1].split(".")[0]
            if key not in self.class_sets.keys():
                self.class_sets[key] = []
            self.class_sets[key].append([self.images_filenames[cnt], self.masks_filenames[cnt]])

    def __len__(self):
        return len(self.images_filenames)

    # def __getitem__(self, idx):
    #     support_image = cv2.imread(self.images_filenames[idx])
    #     support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
    #
    #     support_mask = cv2.imread(self.masks_filenames[idx], cv2.IMREAD_UNCHANGED)
    #
    #     key = self.images_filenames[idx].split("-")[-1].split(".")[0]
    #     query_path = random.choice(self.class_sets[key])
    #     query_image = cv2.imread(query_path[0])
    #     query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    #
    #     query_mask = cv2.imread(query_path[1], cv2.IMREAD_UNCHANGED)
    #
    #     if self.transform is not None:
    #         t0 = self.transform(image=support_image, mask=support_mask)
    #         support_image = t0["image"]/255
    #         support_mask = t0["mask"]/255
    #         t1 = self.transform(image=query_image, mask=query_mask)
    #         query_image = t1["image"]/255
    #         query_mask = t1["mask"]/255
    #     return support_image, support_mask, query_image, query_mask

    def __getitem__(self, idx):

        # Get Query Images and Masks
        query_image = cv2.imread(self.images_filenames[idx])
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_mask = cv2.imread(self.masks_filenames[idx], cv2.IMREAD_UNCHANGED)

        # Get Support Images and Masks
        key = self.images_filenames[idx].split("-")[-1].split(".")[0]
        support_images = []
        support_masks = []

        support_path = random.choice(self.class_sets[key])
        support_image = cv2.imread(support_path[0])
        support_images.append(cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB))
        support_masks.append(cv2.imread(support_path[1], cv2.IMREAD_UNCHANGED))

        support_path = random.choice(self.class_sets[key])
        support_image = cv2.imread(support_path[0])
        support_images.append(cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB))
        support_masks.append(cv2.imread(support_path[1], cv2.IMREAD_UNCHANGED))

        if self.transform is not None:
            t0 = self.transform(image=support_images[0], mask=support_masks[0])
            support_image_0 = t0["image"]/255
            support_mask_0 = t0["mask"]/255
            t0 = self.transform(image=support_images[0], mask=support_masks[0])
            support_image_1 = t0["image"]/255
            support_mask_1 = t0["mask"]/255
            t1 = self.transform(image=query_image, mask=query_mask)
            query_image = t1["image"]/255
            query_mask = t1["mask"]/255
        return support_image_0, support_mask_0, support_image_1, support_mask_1, query_image, query_mask