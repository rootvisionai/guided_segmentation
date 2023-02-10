import torch, torchvision
import os
from dice_loss import dice_coeff
import tqdm
from model import FSS

def eval_net(net, eval_loader, device="cuda"):
    """
    For visual/manual evaluation
    predictions are saved to /inference_examples
    """
    if not os.path.isdir("./inference_examples"):
        os.makedirs("./inference_examples")

    net.eval()
    pbar = tqdm.tqdm(enumerate(eval_loader))
    for i, (sup_images, _, sup_masks, _, query_images, query_masks) in pbar:
        with torch.no_grad():

            mask_pred = net.infer(query_images.to(device), [sup_images.to(device)], [sup_masks.to(device)])[:, 0]

            si, sm, qi, qm, p = sup_images.cpu(), sup_masks.cpu(), query_images.cpu(), query_masks.cpu(), mask_pred.cpu()
            qm = torch.stack([qm, qm, qm], dim=1) * 255
            p = torch.stack([p, p, p], dim=1) * 255
            concat_image_to_save = torch.stack([si[0], sm[0],
                                                qi[0], qm[0],
                                                p[0]], dim=0)
            torchvision.utils.save_image(concat_image_to_save, f"./inference_examples/pred{i}.png")

    net.train()

if __name__ == '__main__':
    from datasets.base import create_dataloaders, create_transforms
    import config
    cfg = config.load("./config/config.json")

    images_directory = os.path.join("datasets", cfg.dataset, cfg.eval_images)
    masks_directory = os.path.join("datasets", cfg.dataset, cfg.eval_masks)
    _, trns_eval = create_transforms(cfg, eval=True)
    
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

    net = FSS(
        input_size = cfg.input_size,
        resnet_arch = cfg.resnet_arch,
        unet_arch = cfg.unet_arch
    )

    checkpoint = torch.load(
            os.path.join(
            os.getcwd(), 
            "checkpoints",
            cfg.dataset,
            f'ckpt_size[{cfg.input_size}].pth'
            )
        )
    net.load_state_dict(checkpoint, False)
    print(f'Model loaded: {os.path.join("checkpoints", cfg.dataset, f"ckpt_size[{cfg.input_size}].pth")}')

    net.to(cfg.device)
    val_dice = eval_net(net, eval_loader, cfg.device)