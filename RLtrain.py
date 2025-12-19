import torch
import pywt
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, ConcatDataset
#from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data import SequentialSampler
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from glob import glob
from time import time
import argparse
from pytorch_msssim import  ms_ssim
import os
from RLmodel import Model , MambaConfig
from brisque import BRISQUE
from distillation_model import UMamba, MambaConfig
from colortest import compute_colorfulness
'''训练帮助的函数'''


def reward_function(enhanced_images,gt_images):
    # 使用 NIQE 分数：越低越好（所以取负值作为 reward）
    # enhanced_images shape: (B, 3, H, W)
    rewards = []
    for img1,img2 in zip(enhanced_images,gt_images):
       
        img1 = ((img1 + 1) / 2) * 255
        img2 = ((img2+ 1) / 2) * 255
        img1_1 = img1.unsqueeze(0) 
        img2_1 = img2.unsqueeze(0) 
        
        score1 = ms_ssim(img1_1, img2_1, data_range=255, size_average=False)
        brisque = BRISQUE()
        
        img1 = img1.permute(1, 2, 0).cpu().numpy()
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        
            
        score2 = brisque.score(img2)
        score3 = brisque.score(img1)
        
        score4 = compute_colorfulness(img1)
        score5 = compute_colorfulness(img2)
        
        reward = -3 * score2 - score3 + score1 * 50 + score4 + score5 # 越清晰 reward 越高
        rewards.append(reward)
    return torch.stack(rewards).squeeze()

def requires_grad(model, flag=True):  # 为模型中的所有参数设置 require_grad 标志，需要就可以设置梯度
    for p in model.parameters():
        p.requires_grad = flag



def mean_flat(tensor):  #取所有非批量维度的平均值。
    return tensor.mean(dim=list(range(1, len(tensor.shape)))) #求除了批次维度，从1~len(tensor.shape)的每一个维度上的平均值

def center_crop_arr(pil_image, image_size):  # ADM 的实施中心裁剪。

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def load_checkpoint(checkpoint_path, model, opt):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # 加载模型状态
        model.load_state_dict(checkpoint["model"])
        # 加载优化器状态
        opt.load_state_dict(checkpoint["opt"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return False

'''循环训练'''


def main(args):  # 训练一个新的模型

    assert torch.cuda.is_available(), "至少需要一个GPU"
    print('GPU是否可用：', torch.cuda.is_available())



    # 创造模型
    #assert args.image_size % 8 == 0, "图像大小必须能被 8 整除（对于 VAE 编码器）"

    modelfig = MambaConfig(d_model=1024, n_layers=4)  #1280 1024
    model = UMamba(modelfig)
    #model = Model(modelfig)
    model = model.to('cuda')




    print(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # 设置优化器（使用默认的 Adam betas=（0.9， 0.999） 和 1e-4 的恒定学习率）:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)


    # 设置数据:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])


    data_raw = ImageFolder(args.data_path, transform=transform)  # ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名
    data_gt = ImageFolder(args.data_path_gt, transform=transform)

    val_dataset = ImageFolder(args.data_path_val, transform=transform)
    raw = SequentialSampler(data_raw)
    gt = SequentialSampler(data_gt)
    
    # combined_dataset = ConcatDataset([raw, gt])
    loader_raw = DataLoader(  # 深度学习训练的流程： 1. 创建Dateset 2. Dataset传递给DataLoader 3. DataLoader迭代产生训练数据提供给模型
        # Dataset负责建立索引到样本的映射，DataLoader负责以特定的方式从数据集中迭代的产生 一个个batch的样本集合
        data_raw,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=raw,
        pin_memory=True,
        drop_last=True
    )
    loader_gt = DataLoader(
        data_gt,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=gt,
        pin_memory=True,
        drop_last=True
    )


  
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1),
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    print(f"raw_Data contains {len(data_raw):,} {len(data_gt):,} images ({args.data_path})")

    # 为训练准备模型:

    model.train()

    # 用于监视/日志记录目的的变量:
    train_steps = 0
    log_steps = 0
    running_loss = 0

    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    #writer = SummaryWriter(log_dir='./tf-logs')
    checkpoint_dir = './checkpoint'


    # 获取最新的 checkpoint 文件路径
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoint_files:
            last_checkpoint_path = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_path)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None
    # 加载 checkpoint
    if checkpoint_path:
        success = load_checkpoint(checkpoint_path, model, opt)
    else:
        success = False
    if success:
        print("Mambadiffusion Checkpoint loaded successfully. Continuing training...")
    else:
        print("No checkpoint found. retraining please...")

    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")


        for k, ((x, _), (y, _)) in enumerate(zip(loader_raw, loader_gt)):
            x = x.to('cuda')
            y = y.to('cuda')
            


            out = model(x)
            #policy_loss = mean_flat((y - out) ** 2).mean() + mean_flat(torch.abs(y - out)) .mean()

            # 计算 reward
            with torch.no_grad():
                rewards = reward_function(out.detach(),y.detach())  # 不参与梯度传播
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                rewards = torch.clamp(rewards, -2.0, 2.0)
            

            # 每个样本的损失（B）
            per_sample_l1 = torch.mean(torch.abs(y - out), dim=[1, 2, 3])  # shape: (B,)
            per_sample_l2 = torch.mean((y - out) ** 2, dim=[1, 2, 3])       # shape: (B,)
            per_sample_loss =  per_sample_l1 + per_sample_l2          # shape: (B,)

            # policy loss（weighted by reward）
            mask = rewards > 0  
            policy_loss = torch.mean(per_sample_loss[mask] * rewards[mask])    # shape: scalar
           
            


            opt.zero_grad()
            policy_loss.backward()
            opt.step()

            # 记录损失值:
            running_loss += policy_loss.item()


            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # 明确训练速度:
                torch.cuda.synchronize()  # PyTorch的 torch.cuda.synchronize () 函数来同步GPU上的所有操作
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # 减少所有流程的损失历史:
                avg_loss = torch.tensor(running_loss / log_steps, device='cuda')

                print(f"'epoch': {epoch}  (step={train_steps:07d}) Train Loss: {avg_loss:.4f},Train Steps/Sec:{steps_per_sec:.2f}")
                #writer.add_scalar('avg_loss', avg_loss, train_steps)
                # 重置监控变量:
                running_loss = 0
                log_steps = 0
                start_time = time()
            # 保存checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    "opt": opt.state_dict(),  
                }
                checkpoint_path = f"./checkpoint/model.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        '''验证'''
        if epoch % 20 == 0 and train_steps > 0:
            i=0
            for u, _ in val_loader:
                i = i + 1
                u = u.to('cuda')
                with torch.no_grad():
                     out = model(u)
        
                save_image(out, f"./out/img{i}.png", nrow=4, normalize=True, value_range=(-1, 1))

    model.eval()

    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data/train/raw")
    parser.add_argument("--data-path-gt", type=str, default="./data/train/gt")
    parser.add_argument("--data-path-val", type=str, default="./data/val")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=12)#原本预设的是256
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=125)
    args = parser.parse_args()
    main(args)
