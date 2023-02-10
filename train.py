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

import random
import tqdm
import collections



def main(cfg, net):
    start_epoch = 0

    trns_train, trns_eval = create_transforms(cfg, eval=True)

    images_directory = os.path.join("datasets", cfg.dataset, cfg.training_images)
    masks_directory = os.path.join("datasets", cfg.dataset, cfg.training_masks)
    train_loader = create_dataloaders(
        images_directory,
        masks_directory, 
        transform = trns_train,
        batch_size = cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    images_directory = os.path.join("datasets", cfg.dataset, cfg.eval_images)
    masks_directory = os.path.join("datasets", cfg.dataset, cfg.eval_masks)
    eval_loader = create_dataloaders(
        images_directory,
        masks_directory, 
        transform = trns_eval,
        batch_size = 1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    parameters = [
        {"params": net.resnet.parameters(), "lr": cfg.lr/10},
        {"params": net.fpn.parameters(), "lr": cfg.lr},
        {"params": net.unet.parameters(), "lr": cfg.lr},
    ]

    optimizer = optim.AdamW(parameters)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    if not os.path.exists(os.path.join(os.getcwd(), "checkpoints")):
        os.mkdir("checkpoints")

    if not os.path.exists(os.path.join(os.getcwd(), "checkpoints", cfg.dataset)):
        os.mkdir(os.path.join(os.getcwd(), "checkpoints", cfg.dataset))

    if os.path.exists(os.path.join(os.getcwd(), "checkpoints", cfg.dataset, f'ckpt_size[{cfg.input_size}].pth')):
        checkpoint = torch.load(
            os.path.join(
            os.getcwd(), 
            "checkpoints",
            cfg.dataset,
            f'ckpt_size[{cfg.input_size}].pth'
            )
        )
        net.load_state_dict(checkpoint["state_dict"], False)
        print(f'Model loaded: {os.path.join("checkpoints", cfg.dataset, f"ckpt_size[{cfg.input_size}].pth")}')
        start_epoch = checkpoint["last_epoch"]
         
    for epoch in range(start_epoch, start_epoch+cfg.epochs):

        net.train()
        net.to(cfg.device)
        
        loss_hist = collections.deque(maxlen=20)
        pbar = tqdm.tqdm(enumerate(train_loader))
        epoch_loss = []
        for i, (sup_images, sup_masks, query_images, query_masks) in pbar:
            try:
                optimizer.zero_grad()

                sup_images = sup_images.to(cfg.device)
                sup_masks = sup_masks.to(cfg.device).unsqueeze(1)
                query_images = query_images.to(cfg.device)
                query_masks = query_masks.to("cpu").unsqueeze(1)
                
                masks_pred = net(query_images, sup_images, sup_masks)
                masks_probs_flat = masks_pred.view(-1)

                query_masks = norm_mask_size(query_masks, target_size=(masks_pred.shape[-2], masks_pred.shape[-1]))
                true_masks_flat = query_masks.view(-1)

                loss = criterion(masks_probs_flat.cpu(), true_masks_flat)
                loss_hist.append(loss.item())
                pbar.set_description(f"EPOCH: {epoch} | ITER:{i} | LOSS: {np.mean(loss_hist)} | LR: {optimizer.param_groups[0]['lr']}")

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

                if (i)%100==0:
                    torchvision.utils.save_image(masks_pred, "./keep/pred.png")
                    torchvision.utils.save_image(sup_images, "./keep/sup_image.png")
                    torchvision.utils.save_image(sup_masks, "./keep/sup_mask.png")
                    torchvision.utils.save_image(query_images, "./keep/query_image.png")
                    torchvision.utils.save_image(query_masks, "./keep/query_mask.png")
            except Exception as e:
                print(e)
        
        scheduler.step(np.mean(epoch_loss))
        val_dice = eval_net(net, eval_loader, cfg.device)
        print('Validation Dice Coeff: {}'.format(val_dice))

        torch.save(
            {
                "state_dict": net.cpu().state_dict(),
                "last_epoch": epoch
            },
            os.path.join("checkpoints", cfg.dataset, f'ckpt_size[{cfg.input_size}].pth')
        )

        print('Checkpoint {} saved !'.format(epoch))


if __name__ == '__main__':
    cfg = config.load("./config/config.json")


    net = FSS(
        input_size = cfg.input_size,
        resnet_arch = cfg.resnet_arch,
        unet_arch = cfg.unet_arch
    )

    try:
        main(cfg, net)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
