# src/infer_one.py
import torch, SimpleITK as sitk, numpy as np
from monai.networks.nets import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16,32,64,128,256), strides=(2,2,2,2),
            num_res_units=2).to(device)
model.load_state_dict(torch.load("models/checkpoint_epoch1.pth"))
model.eval()

img = sitk.ReadImage("~/curvas_project/data/processed/<case>/image_processed.nii.gz".replace("~", "/home/minhchau"))
arr = sitk.GetArrayFromImage(img).astype(np.float32)
arr = (arr[np.newaxis, np.newaxis, ...])  # 1,1,z,y,x
import torch
inp = torch.from_numpy(arr).to(device)
with torch.no_grad():
    out = model(inp)
    prob = torch.sigmoid(out).cpu().numpy()[0,0]  # z,y,x

# save probability and binary
prob_img = sitk.GetImageFromArray(prob.astype(np.float32))
prob_img.SetSpacing(img.GetSpacing())
sitk.WriteImage(prob_img, "~/curvas_project/models/<case>_prob.nii.gz".replace("~","/home/minhchau"))

binary = (prob > 0.5).astype(np.uint8)
bin_img = sitk.GetImageFromArray(binary)
bin_img.SetSpacing(img.GetSpacing())
sitk.WriteImage(bin_img, "~/curvas_project/models/<case>_mask.nii.gz".replace("~","/home/minhchau"))
