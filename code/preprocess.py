import os
from monai.transforms import (
    LoadImaged, Spacingd, CenterSpatialCropd, SpatialPadd, Resized,
    ScaleIntensityRanged, EnsureTyped, EnsureChannelFirstd, Compose
)
from monai.data import Dataset, DataLoader
import SimpleITK as sitk
import torch
import numpy as np

# Thiết lập GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Sử dụng device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Đường dẫn dữ liệu gốc và nơi lưu dữ liệu đã xử lý
data_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/Data"
output_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/Preprocessed"
temp_crop_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/Data_Cropped"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_crop_dir, exist_ok=True)

# Bước 1: Crop ảnh 1024x1024 về 512x512 trước
print("\n📌 Bước 1: Xử lý ảnh 1024x1024...")
for case in sorted(os.listdir(data_dir)):
    case_dir = os.path.join(data_dir, case)
    img_path = os.path.join(case_dir, "image.nii.gz")
    if not os.path.exists(img_path):
        continue
    
    img = sitk.ReadImage(img_path)
    size = img.GetSize()
    
    # Nếu là 1024x1024, crop về 512x512
    if size[0] == 1024 or size[1] == 1024:
        print(f"  Crop {case}: {size} -> (512, 512, {size[2]})")
        img_arr = sitk.GetArrayFromImage(img)  # (z, y, x)
        
        # Crop center 512x512
        z, y, x = img_arr.shape
        start_y = (y - 512) // 2
        start_x = (x - 512) // 2
        img_cropped = img_arr[:, start_y:start_y+512, start_x:start_x+512]
        
        # Lưu lại
        img_new = sitk.GetImageFromArray(img_cropped)
        img_new.SetSpacing(img.GetSpacing())
        img_new.SetOrigin(img.GetOrigin())
        
        case_crop_dir = os.path.join(temp_crop_dir, case)
        os.makedirs(case_crop_dir, exist_ok=True)
        sitk.WriteImage(img_new, os.path.join(case_crop_dir, "image.nii.gz"))
        
        # Copy label
        label_path = os.path.join(case_dir, "annotation_staple.nii.gz")
        if os.path.exists(label_path):
            label = sitk.ReadImage(label_path)
            label_arr = sitk.GetArrayFromImage(label)
            label_cropped = label_arr[:, start_y:start_y+512, start_x:start_x+512]
            label_new = sitk.GetImageFromArray(label_cropped)
            label_new.SetSpacing(label.GetSpacing())
            label_new.SetOrigin(label.GetOrigin())
            sitk.WriteImage(label_new, os.path.join(case_crop_dir, "annotation_staple.nii.gz"))

print("\n📌 Bước 2: Tiền xử lý tất cả ảnh...")

# Tạo danh sách case (ưu tiên dùng ảnh đã crop nếu có)
cases = []
for case in sorted(os.listdir(data_dir)):
    # Kiểm tra xem có ảnh đã crop không
    cropped_case_dir = os.path.join(temp_crop_dir, case)
    if os.path.exists(os.path.join(cropped_case_dir, "image.nii.gz")):
        img_path = os.path.join(cropped_case_dir, "image.nii.gz")
        label_path = os.path.join(cropped_case_dir, "annotation_staple.nii.gz")
    else:
        case_dir = os.path.join(data_dir, case)
        img_path = os.path.join(case_dir, "image.nii.gz")
        label_path = os.path.join(case_dir, "annotation_staple.nii.gz")
    
    if not os.path.exists(label_path):
        print(f"Missing: {case}/annotation_staple.nii.gz")
        continue
    if os.path.exists(img_path) and os.path.exists(label_path):
        cases.append({"image": img_path, "label": label_path, "case_name": case})

# Pipeline tiền xử lý
TARGET_SIZE = (128, 128, 128)  # Z, Y, X

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # Đảm bảo có channel dimension
    # Bước 1: Resize XY từ 512x512 -> 128x128, giữ nguyên Z
    Resized(
        keys=["image", "label"],
        spatial_size=(128, 128, -1),  # (H, W, D) - D=-1 giữ nguyên Z
        mode=("trilinear", "nearest")
    ),
    # Bước 2: Pad Z nếu nhỏ hơn 128
    SpatialPadd(
        keys=["image", "label"],
        spatial_size=(128, 128, 128),  # (H, W, D)
        mode="constant"
    ),
    # Bước 3: Crop Z nếu lớn hơn 128 (crop từ giữa)
    CenterSpatialCropd(
        keys=["image", "label"],
        roi_size=(128, 128, 128)  # (H, W, D)
    ),
    # Normalize intensity
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-50,    # Focus vào soft tissue window
        a_max=200,    # Bao gồm tụy, tumor, mạch máu
        b_min=0.0, 
        b_max=1.0,
        clip=True
    ),
    EnsureTyped(keys=["image", "label"], dtype="float32")
])

# Chạy pipeline cho từng case
dataset = Dataset(data=cases, transform=transforms)
loader = DataLoader(dataset, batch_size=1, num_workers=0)

for batch_data in loader:
    case_name = batch_data["case_name"][0]
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Lưu image (shape: [1, 1, H, W, D] -> [H, W, D])
    img_data = batch_data["image"][0].cpu().numpy()  # [1, H, W, D]
    print(f"Processed: {case_name} -> shape {img_data.shape}")
    img_sitk = sitk.GetImageFromArray(img_data[0])  # Bỏ channel dimension
    sitk.WriteImage(img_sitk, os.path.join(case_output_dir, "image.nii.gz"))
    
    # Lưu label
    label_data = batch_data["label"][0].cpu().numpy()
    label_sitk = sitk.GetImageFromArray(label_data[0])
    sitk.WriteImage(label_sitk, os.path.join(case_output_dir, "annotation_staple.nii.gz"))

print(f"Tiền xử lý hoàn tất. Tất cả ảnh đã được chuẩn hóa về {TARGET_SIZE} (Z, Y, X).")