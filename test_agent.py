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

class ImageFolderWithPaths(ImageFolder):
    # 重写 __getitem__，让它返回 (image, label, filepath)
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # 返回文件名（不带路径和扩展名）
        filename = os.path.splitext(os.path.basename(path))[0]
        return image, label, filename



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


def main(args):  

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



    val_dataset = ImageFolderWithPaths(args.data_path_val, transform=transform)
  
   
  
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1),
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    print(f"val_Data contains {len(val_dataset):} ")

    # 准备模型:

    model.eval()


     
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

    
        '''验证'''
       
    i=0
    for u, _, names in val_loader:
          i = i + 1
          u = u.to('cuda')
          with torch.no_grad():
          
               start_time = time()
               out = model(u)
               end_time = time()
          print(f"img{i},  time: {end_time - start_time:.6f} seconds")
          grid_name = "_".join(names)
        
          save_image(out, f"./agentout/SEA/{grid_name}.png", nrow=4, normalize=True, value_range=(-1, 1))

   

    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path-val", type=str, default="./underwaterdata/SEA")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=1)#原本预设的是256
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=125)
    args = parser.parse_args()
    main(args)
