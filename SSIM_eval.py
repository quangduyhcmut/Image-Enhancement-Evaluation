import pytorch_ssim
import tifffile
import matplotlib.pyplot as plt
import torch
from time import time

def SSIM(img1, img2):
    
    return pytorch_ssim.ssim(img1, img2).item()

if __name__ == "__main__":
    img_gt = plt.imread('14_high.png')
    img_in = plt.imread('14_low.png')
    
    img_gt = torch.tensor(img_gt, dtype = torch.float32)
    img_in = torch.tensor(img_in, dtype = torch.float32)
    
    img_gt = torch.unsqueeze(img_gt, dim=0)
    img_in = torch.unsqueeze(img_in, dim=0)

    img_gt = img_gt.permute(0, 3, 1, 2)
    img_in = img_in.permute(0, 3, 1, 2)

    start = time()
    # print(SSIM(img_gt, img_in))
    print(pytorch_ssim.ssim(img_gt, img_in).item())
    print("Time: ", time()-start)
