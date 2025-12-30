import os
import SimpleITK as sitk

preprocessed_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/nnUNet_raw/Dataset001_PDAC/imagesTr"
for case in sorted(os.listdir(preprocessed_dir))[:10]:
    img_path = os.path.join(preprocessed_dir, case, ".nii.gz")
    if os.path.exists(img_path):
        img = sitk.ReadImage(img_path)
        print(f"{case}: {img.GetSize()}")
