from ultralytics import YOLO
import shutil
import os

# Your paths
loadyolo_path = r'E:\gitrepo\yolo-V8-main\runs\kibo320\krpc_aug_yolov8n_32b_cos_lr_no_rot_with03_mixup_copy_paste\weights\best.pt'
target_path = r'E:\gitrepo\yolo-V8-main\kiborpc\yolo_v8n_400.onnx'

print("=== Different Methods to Export to Specific Path ===")

# ========== METHOD 1: Export then Move (RECOMMENDED) ==========
print("\n🎯 Method 1: Export then move to target location")

# Load your trained model
model = YOLO(loadyolo_path)

# Export to default location first
onnx_path = model.export(
    format='onnx', 
    imgsz=320, 
    opset=15,
    name='yolo_v8n_400'
)

print(f"✅ Model exported to: {onnx_path}")

# Move to your desired location
shutil.move(onnx_path, target_path)
print(f"✅ Model moved to: {target_path}")