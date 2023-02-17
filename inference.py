import os
import torch
from PIL import Image
from model import FSS
from inference.utils import json_to_mask, load_config
from torchvision import transforms
import glob, tqdm
import cv2
import numpy as np

class Segment:
    def __init__(self, cfg):

        self.model = FSS(
            input_size=256,
            resnet_arch="resnet50",
            unet_arch="unet3",
            bilinear=False
        )

        self.cfg = cfg
        self.device = cfg.device
        self.model_path = os.path.join("checkpoints", cfg.dataset,
                                       f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')
        self.confidence = cfg.confidence

        if os.path.isfile(os.path.join(os.getcwd(), "checkpoints", cfg.dataset,
                                       f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')):
            checkpoint = torch.load(
                os.path.join(
                    os.getcwd(),
                    "checkpoints",
                    cfg.dataset,
                    f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth'
                )
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            print(f'Model loaded: "{cfg.dataset}/ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')

        self.model.to(self.device)

        self.support_images = []
        for path in cfg.support.image:
            self.support_images.append(Image.open(path))

        self.support_masks = []
        for path in cfg.support.mask:
            if "json" in path:
                path = json_to_mask(json_path=path)
                self.support_masks.append(Image.open(path))
            else:
                self.support_masks.append(Image.open(path))

        self.img_size = cfg.input_size

        self.define_transforms()
        self.prep_support_batch()

    def define_transforms(self):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.ToTensor()
            ]
        )

    def resize(self, img, spatial_size):
        img = img.clone()
        spatial_size = tuple([spatial_size[1], spatial_size[0]])
        img = torch.nn.functional.interpolate(img.unsqueeze(0), spatial_size, mode='bilinear', align_corners=True)
        return img.squeeze(0)

    def prep_support_batch(self):

        support_images = []
        for support_image in self.support_images:
            support_image = self.image_transform(support_image)
            support_images.append(support_image)
        self.support_images = torch.stack(support_images, dim=0).unsqueeze(0)

        support_masks = []
        for support_mask in self.support_masks:
            support_mask = self.mask_transform(support_mask)
            support_masks.append(support_mask)
        self.support_masks = torch.stack(support_masks, dim=0).unsqueeze(0)

    def preprocessing(self, que_img):
        que_img = self.image_transform(que_img).unsqueeze(0)

        batch = {
            'support_imgs': self.support_images.to(self.device) if len(
                self.support_images.to(self.device).shape) == 5 else self.support_images.to(self.device).unsqueeze(1),
            'support_masks': self.support_masks.to(self.device) if len(
                self.support_masks.to(self.device).shape) == 5 else self.support_masks.to(self.device).unsqueeze(1),
            'query_img': que_img.to(self.device)
        }
        return batch

    def predict(self, batch):
        mask_pred = self.model.infer(
            batch['query_img'].to(self.device),
            [elm.to(self.device) for elm in batch["support_imgs"]],
            [elm.to(self.device) for elm in batch["support_masks"]]
        )
        mask_pred = torch.sigmoid(mask_pred[0])

        pred = self.resize(
            mask_pred,
            (np.array(query_image).shape[1],
             np.array(query_image).shape[0])
        )

        pred *= 255
        pred = pred.cpu().detach().numpy().astype(np.uint8)  # 1,N,H,W -> N,H,W

        # post-processing -> erosion to eliminate small dots on mask, dilate to enlarge the mask
        if self.cfg.postprocessing.erosion:
            pred = cv2.erode(pred, np.ones((self.cfg.postprocessing.erosion,
                                            self.cfg.postprocessing.erosion),
                                           np.uint8))
        if self.cfg.postprocessing.dilation:
            pred = cv2.dilate(pred, np.ones((self.cfg.postprocessing.dilation,
                                             self.cfg.postprocessing.dilation),
                                            np.uint8))
        return pred[0]

    def render_mask(self, pred, query_image):
        query_image = np.array(query_image)
        color = np.array([0, 255, 0], dtype='uint8')
        masked_img = np.where(pred[..., None]>self.confidence*255, color, query_image)
        out = cv2.addWeighted(query_image, 0.5, masked_img, 0.5, 0)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    def convert_to_polygon(self, mask):
        coords = np.column_stack(np.where(mask > self.confidence*255))
        return coords

    def convert_to_bbox(self, polygon_points):
        if len(polygon_points) > 0:
            ymin = polygon_points[:, 0].min()
            xmin = polygon_points[:, 1].min()
            ymax = polygon_points[:, 0].max()
            xmax = polygon_points[:, 1].max()
        else:
            ymin = xmin = ymax = xmax = 0
        return {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax
        }

    def render_bbox(self, img, bbox):
        cv2.rectangle(img, (bbox["xmin"], bbox["ymin"]), (bbox["xmax"], bbox["ymax"]), (0, 255, 0), 1)
        return img


if __name__ == '__main__':
    cfg = load_config("./inference/config.yml")

    instance = Segment(cfg=cfg)
    filepaths = glob.glob(os.path.join("query_images", "*.jpeg"))
    filepaths += glob.glob(os.path.join("query_images", "*.jpg"))
    filepaths += glob.glob(os.path.join("query_images", "*.png"))
    pbar = tqdm.tqdm(enumerate(filepaths))
    for i,fp in pbar:
        query_image = Image.open(fp).convert("RGB")
        size = query_image.size
        batch = instance.preprocessing(query_image)
        pred = instance.predict(batch)
        rendered_mask = instance.render_mask(pred, query_image)
        polygon_pts = instance.convert_to_polygon(pred)
        bbox_pts = instance.convert_to_bbox(polygon_pts)
        rendered_bbox = instance.render_bbox(rendered_mask, bbox_pts)

        save_path = fp.replace("query_images", "results")
        out = cv2.imwrite(save_path, rendered_bbox)

        pbar.set_description(f"ITER[{i}] FP[{save_path}]")