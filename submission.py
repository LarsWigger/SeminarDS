import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil
import Datasets
import Utility

checkpoint = "base_no_normalization_50"

ph_ds = Datasets.PhotoDataset()
ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen_ptm = torch.load(os.path.join("checkpoints", checkpoint + ".ckpt"))["gen_ptm"].to(device)

output_folder = os.path.join("submissions", checkpoint)
if os.path.isdir(output_folder):
    #already exists, delete firs
    print(f"Removing old {output_folder}")
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

trans = transforms.ToPILImage()
for i, photo in enumerate(ph_dl):
    with torch.no_grad():
        pred_monet = gen_ptm(photo.to(device)).cpu().detach()
        # pred_monet = Utility.unnorm(pred_monet)
        img = trans(pred_monet[0]).convert("RGB")
        img.save(os.path.join(output_folder, f"{i:04d}.jpg"))
        
# doing this automatically makes no sense, there is little compression and the size is huge. Do in on-demand
# shutil.make_archive(output_folder, "zip", output_folder)
