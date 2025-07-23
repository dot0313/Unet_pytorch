import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 一次性加载所有数据到内存
# ---------------------------------------------------
def preload_images(file_paths, height, width):
    images = []
    for path in file_paths:
        pixel_values = np.loadtxt(path, delimiter=',', usecols=0, dtype=np.float32)
        images.append(pixel_values.reshape((height, width)))
    return np.stack(images)

def preload_curves(file_paths, num_points, strain_range=(0, 0.35)):
    curves = []
    for path in file_paths:
        data = np.loadtxt(path, dtype=np.float32)
        original_strain = data[:, 0]
        original_stress = data[:, 1]
        target_strain = np.linspace(strain_range[0], strain_range[1], num_points, dtype=np.float32)
        target_stress = np.interp(target_strain, original_strain, original_stress).astype(np.float32)
        curves.append(target_stress)
    return np.stack(curves)

# ---------------------------------------------------
# 自定义数据集（直接从内存读数据）
# ---------------------------------------------------
class MicrostructureDataset(Dataset):
    def __init__(self, images, stresses, stress_mean, stress_std):
        self.images = images
        self.stresses = (stresses - stress_mean) / stress_std
        self.stress_mean = stress_mean
        self.stress_std = stress_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.images[idx]).float().unsqueeze(0)
        stress_tensor = torch.from_numpy(self.stresses[idx]).float()
        return image_tensor, stress_tensor

# ---------------------------------------------------
# UNet编码器和回归器（BN可选）
# ---------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(mid_ch))
            layers.insert(-1, nn.BatchNorm2d(out_ch))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=1, base_ch=16, use_bn=False):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_ch, use_bn=use_bn)
        self.down1 = Down(base_ch, base_ch * 2, use_bn=use_bn)
        self.down2 = Down(base_ch * 2, base_ch * 4, use_bn=use_bn)
        self.down3 = Down(base_ch * 4, base_ch * 8, use_bn=use_bn)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class UNetRegressor(nn.Module):
    def __init__(self, input_shape, output_size, base_ch=16, use_bn=False):
        super().__init__()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape[1:])
            encoder = UNetEncoder(n_channels=input_shape[1], base_ch=base_ch, use_bn=use_bn)
            _, _, _, x4 = encoder(dummy)
            flatten_size = x4.view(1, -1).shape[1]
        self.flatten_size = flatten_size
        self.encoder = UNetEncoder(n_channels=input_shape[1], base_ch=base_ch, use_bn=use_bn)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        _, _, _, x4 = self.encoder(x)
        features = self.regressor(x4)
        return features

# ---------------------------------------------------
# 主流程
# ---------------------------------------------------
if __name__ == "__main__":
    # --- 参数配置 ---
    DATA_DIRS = ["./1", "./2", "./3", "./4"]
    IMAGE_PATTERN = "Case*_Int*_Microstructure.txt"
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    TARGET_CURVE_POINTS = 100
    TARGET_STRAIN_RANGE = (0, 0.35)
    IMAGE_SIZE = (128, 128)

    # --- 1. 查找数据文件 ---
    print("正在查找并配对数据文件...")
    file_pairs = []
    for data_dir in DATA_DIRS:
        image_files_in_dir = glob.glob(os.path.join(data_dir, IMAGE_PATTERN))
        for img_path in image_files_in_dir:
            curve_path = img_path.replace("Microstructure.txt", "Load.txt")
            if os.path.exists(curve_path):
                file_pairs.append((img_path, curve_path))
            else:
                print(f"警告: 找到图片 {img_path} 但未找到曲线文件 {curve_path}")
    if not file_pairs:
        raise FileNotFoundError("未找到数据文件对")
    print(f"总数据对数: {len(file_pairs)}")

    # --- 2. 划分数据集 ---
    train_val_pairs, test_pairs = train_test_split(file_pairs, test_size=TEST_SPLIT, random_state=42)
    relative_val_split = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=relative_val_split, random_state=42)
    print(f"训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}, 测试集: {len(test_pairs)}")

    # --- 3. 数据预加载 ---
    train_img_paths, train_curve_paths = zip(*train_pairs)
    val_img_paths, val_curve_paths = zip(*val_pairs)
    test_img_paths, test_curve_paths = zip(*test_pairs)

    print("正在加载图片和曲线数据到内存...")
    train_images = preload_images(train_img_paths, *IMAGE_SIZE)
    val_images = preload_images(val_img_paths, *IMAGE_SIZE)
    test_images = preload_images(test_img_paths, *IMAGE_SIZE)

    train_stresses = preload_curves(train_curve_paths, TARGET_CURVE_POINTS, strain_range=TARGET_STRAIN_RANGE)
    val_stresses = preload_curves(val_curve_paths, TARGET_CURVE_POINTS, strain_range=TARGET_STRAIN_RANGE)
    test_stresses = preload_curves(test_curve_paths, TARGET_CURVE_POINTS, strain_range=TARGET_STRAIN_RANGE)

    # --- 4. 归一化参数计算 ---
    stress_mean = train_stresses.mean()
    stress_std = train_stresses.std()

    # --- 5. 数据集和Loader ---
    train_dataset = MicrostructureDataset(train_images, train_stresses, stress_mean, stress_std)
    val_dataset = MicrostructureDataset(val_images, val_stresses, stress_mean, stress_std)
    test_dataset = MicrostructureDataset(test_images, test_stresses, stress_mean, stress_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- 6. 模型与训练配置 ---
    output_size = TARGET_CURVE_POINTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRegressor(input_shape=(BATCH_SIZE, 1, *IMAGE_SIZE), output_size=output_size, base_ch=16, use_bn=False).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"模型初始化完成，设备: {device}")

    # --- 7. 训练循环 ---
    train_losses, val_losses, test_losses = [], [], []
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        running_train_loss = 0.0
        for images, stresses in train_loader:
            images, stresses = images.to(device), stresses.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, stresses)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # 验证
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, stresses in val_loader:
                images, stresses = images.to(device), stresses.to(device)
                outputs = model(images)
                loss = criterion(outputs, stresses)
                running_val_loss += loss.item() * images.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        # 测试
        running_test_loss = 0.0
        with torch.no_grad():
            for images, stresses in test_loader:
                images, stresses = images.to(device), stresses.to(device)
                outputs = model(images)
                loss = criterion(outputs, stresses)
                running_test_loss += loss.item() * images.size(0)
        epoch_test_loss = running_test_loss / len(test_dataset)
        test_losses.append(epoch_test_loss)

        # ==== 和test3一样，不带任何单位 ====
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")

    print("训练完成！")

    # --- 8. 损失曲线可视化 ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training, Validation, and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = "./loss_curve.png"
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"损失曲线已保存: {loss_curve_path}")

    # --- 9. 验证集曲线预测效果输出 ---
    print("\n批量输出所有验证样本的预测效果，并保存图片到 ./validation_alldata/ 文件夹...")
    val_output_dir = "./validation_alldata"
    os.makedirs(val_output_dir, exist_ok=True)
    for idx, (img_path, curve_path) in enumerate(val_pairs):
        # 加载原始曲线数据（用于对比真实曲线）
        original_data = np.loadtxt(curve_path, dtype=np.float32)
        original_strain = original_data[:, 0]
        original_stress = original_data[:, 1]
        plot_strain_axis = np.linspace(TARGET_STRAIN_RANGE[0], TARGET_STRAIN_RANGE[1], TARGET_CURVE_POINTS)

        # 输入图片张量
        sample_image = np.loadtxt(img_path, delimiter=',', usecols=0, dtype=np.float32).reshape(*IMAGE_SIZE)
        input_tensor = torch.from_numpy(sample_image).float().unsqueeze(0).unsqueeze(0).to(device)

        # 模型预测（还原应力值）
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).squeeze().cpu().numpy()
        pred_stress = pred_norm * stress_std + stress_mean

        plt.figure(figsize=(12, 7))
        plt.plot(original_strain, original_stress, 'b-', label='Actual Curve (Ground Truth)', linewidth=3, alpha=0.8)
        plt.plot(plot_strain_axis, pred_stress, 'r--', label='Predicted Curve', linewidth=2)
        plt.title(f'Stress-Strain Curve Prediction\nSample: {os.path.basename(img_path)}')
        plt.xlabel('Strain (unitless)')
        plt.ylabel('Stress (MPa)')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(val_output_dir, f"val_curve_{idx + 1:03d}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"已保存验证集图片: {save_path}")

    print("全部验证集图片保存完毕！")

    # --- 10. 测试集曲线预测效果输出 ---
    print("\n批量输出所有测试样本的预测效果，并保存图片到 ./test_alldata/ 文件夹...")
    test_output_dir = "./test_alldata"
    os.makedirs(test_output_dir, exist_ok=True)
    for idx, (img_path, curve_path) in enumerate(test_pairs):
        original_data = np.loadtxt(curve_path, dtype=np.float32)
        original_strain = original_data[:, 0]
        original_stress = original_data[:, 1]
        plot_strain_axis = np.linspace(TARGET_STRAIN_RANGE[0], TARGET_STRAIN_RANGE[1], TARGET_CURVE_POINTS)

        sample_image = np.loadtxt(img_path, delimiter=',', usecols=0, dtype=np.float32).reshape(*IMAGE_SIZE)
        input_tensor = torch.from_numpy(sample_image).float().unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            pred_norm = model(input_tensor).squeeze().cpu().numpy()
        pred_stress = pred_norm * stress_std + stress_mean

        plt.figure(figsize=(12, 7))
        plt.plot(original_strain, original_stress, 'b-', label='Actual Curve (Ground Truth)', linewidth=3, alpha=0.8)
        plt.plot(plot_strain_axis, pred_stress, 'r--', label='Predicted Curve', linewidth=2)
        plt.title(f'Stress-Strain Curve Prediction\nSample: {os.path.basename(img_path)}')
        plt.xlabel('Strain (unitless)')
        plt.ylabel('Stress (MPa)')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(test_output_dir, f"test_curve_{idx + 1:03d}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"已保存测试集图片: {save_path}")

    print("全部测试集图片保存完毕！")

    # --- 11. 参数提醒 ---
    print("""
=============================
1. NUM_EPOCHS: 可根据训练和验证损失曲线进一步调整。
2. BATCH_SIZE: 可根据显存情况调大或调小。
3. LEARNING_RATE: 可微调以加快或稳定收敛。
4. 网络结构: 保持UNet编码器结构不变。
5. Dropout: 若发现过拟合，可适当提高Dropout比例。
6. TARGET_CURVE_POINTS: 如需更高细节，可提升点数。
7. 数据增强: 可以自定义Dataset做图片旋转、翻转等。
8. 曲线插值范围: 请根据实际数据合理设置strain_range。
9. 提前终止: 可加early stopping机制，防止过拟合。
10. 归一化: 本代码采用标准化，评估输出时记得还原。
=============================
""")