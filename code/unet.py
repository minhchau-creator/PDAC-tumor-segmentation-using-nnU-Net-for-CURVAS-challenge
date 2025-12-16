# src/train_monai.py
import os, json
import torch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,
    CropForegroundd, RandFlipd, RandRotate90d, ToTensord, Compose
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# Thiết lập GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    # Tối ưu GPU
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    print("⚠️  Đang chạy trên CPU - Training sẽ rất chậm!")
print("=" * 60)

ROOT = os.path.expanduser("/home/minhchau/anaconda3/envs/curvas/curvas/Preprocessed")
with open("split.json") as f:
    split = json.load(f)
train_cases = split["train"]
val_cases = split["val"]

def make_dataset(cases):
    items = []
    for c in cases:
        img = os.path.join(ROOT,c,"image.nii.gz")
        mask = os.path.join(ROOT,c,"annotation_staple.nii.gz")
        if not os.path.exists(mask): continue
        items.append({"image": img, "label": mask})
    return items

train_files = make_dataset(train_cases)
val_files = make_dataset(val_cases)

train_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),  # File lưu là 3D, cần thêm channel
    # Data giờ có shape (1, 128, 128, 128)
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
    ToTensord(keys=["image","label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),  # File lưu là 3D, cần thêm channel
    ToTensord(keys=["image","label"])
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 để tránh lỗi
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16,32,64,128),  # Giảm từ 5 xuống 4 levels
    strides=(2,2,2),  # 3 strides thay vì 4
    num_res_units=2,
).to(device)

loss_fn = DiceLoss(sigmoid=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# train 1 epoch
model.train()
for epoch in range(1):
    print(f"\n📊 Epoch {epoch}")
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        img = batch["image"].to(device)
        label = batch["label"].to(device).float()
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
        if i % 10 == 0:
            print(f"  Iter {i:3d} | Loss: {loss.item():.4f} | GPU Mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        # safety: break quickly to test
        if i >= 20: break
    
    print(f"✅ Epoch {epoch} | Avg Loss: {epoch_loss/(i+1):.4f}")

# quick val (dice)
print("\n📈 Validation...")
model.eval()
dice_metric = DiceMetric(include_background=False, reduction="mean")
with torch.no_grad():
    for j, batch in enumerate(val_loader):
        img = batch["image"].to(device)
        label = batch["label"].to(device).float()
        out = model(img)
        pred = torch.sigmoid(out) > 0.5
        dice_metric(y_pred=pred.float(), y=label)
        if j >= 20: break

print(f"✅ Validation Dice Score: {dice_metric.aggregate().item():.4f}")

# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/checkpoint_epoch1.pth")
print("💾 Model saved to models/checkpoint_epoch1.pth")
