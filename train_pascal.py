import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils
from torch import optim
from model import FSS
from model.utils import norm_mask_size

from datasets.base import create_dataloaders, create_transforms
from eval import eval_net
import config

import tqdm
import collections


def main(cfg, net):
    start_epoch = 0

    trns_train, trns_eval = create_transforms(cfg, eval=True)

    train_loader = create_dataloaders(
        dataset_type="pascal",
        images_directory="./",
        masks_directory="./",
        set_type="train",
        transform=trns_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    eval_loader = create_dataloaders(
        dataset_type="pascal",
        images_directory="./",
        masks_directory="./",
        set_type="val",
        transform=trns_eval,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    parameters = [
        {"params": net.resnet.parameters(), "lr": cfg.lr},
        {"params": net.corrfc.parameters(), "lr": cfg.lr},
        {"params": net.fpn.parameters(), "lr": cfg.lr},
        {"params": net.unet.parameters(), "lr": cfg.lr},
    ]

    optimizer = optim.AdamW(parameters)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.01)

    if not os.path.isdir(os.path.join(os.getcwd(), "checkpoints", cfg.dataset)):
        os.makedirs(os.path.join(os.getcwd(), "checkpoints", cfg.dataset))

    if os.path.isfile(os.path.join(os.getcwd(), "checkpoints", cfg.dataset, f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')):
        checkpoint = torch.load(
            os.path.join(
            os.getcwd(), 
            "checkpoints",
            cfg.dataset,
            f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth'
            )
        )
        net.load_state_dict(checkpoint["state_dict"], True)
        print(f'Model loaded: f"{cfg.dataset}/ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')
        start_epoch = checkpoint["last_epoch"]

    net.to(cfg.device)
    for epoch in range(start_epoch, start_epoch+cfg.epochs):

        net.train()
        net.to(cfg.device)
        loss_hist = collections.deque(maxlen=20)
        pbar = tqdm.tqdm(enumerate(train_loader))
        epoch_loss = []
        for i, (sup_images_0, sup_masks_0, sup_images_1, sup_masks_1, query_images, query_masks) in pbar:

            sup_images_0 = sup_images_0.to(cfg.device)
            sup_masks_0 = sup_masks_0.to(cfg.device).unsqueeze(1)

            sup_images_1 = sup_images_1.to(cfg.device)
            sup_masks_1 = sup_masks_1.to(cfg.device).unsqueeze(1)

            query_images = query_images.to(cfg.device)
            query_masks = query_masks.to("cpu").unsqueeze(1)
            query_masks = torch.cat([1-query_masks, query_masks], dim=1)

            for _ in range(1):
                masks_pred = net(query_images, [sup_images_0, sup_images_1], [sup_masks_0, sup_masks_1])
                masks_probs_flat = masks_pred.view(-1)

                query_masks = norm_mask_size(query_masks, target_size=(masks_pred.shape[-2], masks_pred.shape[-1]))
                true_masks_flat = query_masks.view(-1)

                loss = criterion(
                    masks_probs_flat.cpu(),
                    true_masks_flat,
                )

                loss_hist.append(loss.item())
                pbar.set_description(f"EPOCH: {epoch} | ITER:{i} | LOSS: {np.mean(loss_hist)} | LR: {optimizer.param_groups[0]['lr']}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss.append(loss.item())

            if (i)%25==0:
                torchvision.utils.save_image(masks_pred[:, 1].unsqueeze(1), "./keep/pred.png")
                torchvision.utils.save_image(sup_images_0, "./keep/sup_image.png")
                torchvision.utils.save_image(sup_masks_0, "./keep/sup_mask.png")
                torchvision.utils.save_image(query_images, "./keep/query_image.png")
                torchvision.utils.save_image(query_masks[:, 1].unsqueeze(1), "./keep/query_mask.png")
        
        scheduler.step(np.mean(epoch_loss))

        torch.save(
            {
                "state_dict": net.cpu().state_dict(),
                "last_epoch": epoch
            },
            os.path.join("checkpoints", cfg.dataset, f'ckpt_arch[{cfg.unet_arch}]_size[{cfg.input_size}].pth')
        )
        print('Checkpoint {} saved !'.format(epoch))

        eval_net(net.to(cfg.device), eval_loader, cfg.device)


if __name__ == '__main__':
    cfg = config.load("./config/config.json")


    net = FSS(
        input_size=cfg.input_size,
        resnet_arch=cfg.resnet_arch,
        unet_arch=cfg.unet_arch,
        bilinear=False
    )

    main(cfg, net)

    # try:
    #     main(cfg, net)
    #
    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #     print('Saved interrupt')
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)
