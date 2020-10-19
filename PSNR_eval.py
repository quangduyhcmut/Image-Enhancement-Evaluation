import numpy as np 
import torch
import tifffile
from time import time 

def PSNR(img1, img2):
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100.0
    # PIXEL_MAX = 255.0
    # PIXEL_MAX = 65536
    PIXEL_MAX = 1.0
    
    return (20 * torch.log(PIXEL_MAX / torch.sqrt(mse))).item()

if __name__ == "__main__":
    img_gt = tifffile.imread(r"00001_00_10s.tiff")/65536.0    #RGB image, ndarray
    img_in = tifffile.imread(r"00076_00_0.1s.tiff")/65536.0

    img_gt = torch.tensor(img_gt, dtype = torch.float32)
    img_in = torch.tensor(img_in, dtype = torch.float32)
    
    img_gt = torch.unsqueeze(img_gt, dim=0)
    img_in = torch.unsqueeze(img_in, dim=0)

    img_gt = img_gt.permute(0, 3, 1, 2)
    img_in = img_in.permute(0, 3, 1, 2)

    start = time()
    print(PSNR(img_gt, img_in))
    print("Time: ", time()-start)
    