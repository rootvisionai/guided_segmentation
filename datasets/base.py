import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2, os, torchvision, random
import tqdm
import numpy as np
from PIL import Image

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

def create_dataloaders(
        dataset_type="pascal",
        images_directory="./",
        masks_directory="./",
        set_type="train",
        transform=None,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
):

    if dataset_type=="simple":
        set = Simple(images_directory, masks_directory, transform=transform)
    elif dataset_type=="pascal":
        set = PascalVoc(
            transform=transform,
            root="./datasets/pascal_voc",
            year="2012",
            image_set=set_type,
            download=False
        )

    data_loader = DataLoader(
        set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return data_loader

class Simple(Dataset):
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

class PascalVoc(torchvision.datasets.VOCSegmentation):
    def __init__(self, transform=None, **kwargs):
        """
        Using Torchvision PASCAL VOC [ https://pytorch.org/vision/0.8/_modules/torchvision/datasets/voc.html#VOCSegmentation ]
        root="./pascal_voc",
        year="2012",
        image_set="train",
        download=True
        """
        super().__init__(**kwargs)
        self.transform = transform
        self.create_class_sets()

    @staticmethod
    def get_labels(mask):
        unqs, cnts = np.unique(np.array(mask, np.uint8), return_counts=True)
        return unqs[1:-1], cnts[1:-1]

    def create_class_sets(self):
        self.class_sets = {}
        pbar = tqdm.tqdm(range(len(self.masks)))
        for i in pbar:
            sample = self.load_images(i)
            unqs, cnts = self.get_labels(sample[1])
            for j, label in enumerate(unqs):
                if not label in self.class_sets:
                    self.class_sets[label] = []
                self.class_sets[label].append([i, self.images[i], self.masks[i], cnts[j]])
            pbar.set_description(f"Creating class sets: [{i}/{len(self.masks)}]")

        for label in self.class_sets:
            self.class_sets[label] = sorted(self.class_sets[label], key= lambda x:x[-1], reverse=True)

    @staticmethod
    def select_label(sample, unqs, label): # if label=None, then get random label & mask of that label
        mask = np.array(sample[1], np.uint8)
        selected_label = random.choice(unqs) if not label else label
        mask[np.where(mask != selected_label)] = 0
        mask[np.where(mask == selected_label)] = 255
        return mask, selected_label

    def load_images(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        return img, target

    def get_single_sample(self, i, label=None):
        if not label:
            sample = self.load_images(i)
        else:
            set_to_select = self.class_sets[label][0:10]
            il = random.choice(set_to_select)[0]
            sample = self.load_images(il)
        if not label:
            labels, _ = self.get_labels(sample[1])
        else:
            labels = None
        mask, label = self.select_label(sample, labels, label)

        image = np.array(sample[0], np.uint8)
        mask = np.array(mask, np.uint8)

        return image, mask, label

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i, label=None):
        try:
            query_image, query_mask, label = self.get_single_sample(i)
            support_image_0, support_mask_0, label = self.get_single_sample(i, label)
            support_image_1, support_mask_1, label = self.get_single_sample(i, label)

            if self.transform is not None:
                t0 = self.transform(image=support_image_0, mask=support_mask_0)
                support_image_0 = t0["image"]/255
                support_mask_0 = t0["mask"]/255
                t0 = self.transform(image=support_image_1, mask=support_mask_1)
                support_image_1 = t0["image"]/255
                support_mask_1 = t0["mask"]/255
                t1 = self.transform(image=query_image, mask=query_mask)
                query_image = t1["image"]/255
                query_mask = t1["mask"]/255
        except:
            support_image_0, support_mask_0, support_image_1, support_mask_1, query_image, query_mask = self.__getitem__(i+1)

        return support_image_0, support_mask_0, support_image_1, support_mask_1, query_image, query_mask

