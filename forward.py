from src.secatnet import SECatNet
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import random
import pandas as pd
import re
import matplotlib.pyplot as plt

torch.set_printoptions(precision=7, sci_mode=False)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.11932160705327988], std=[0.2542687952518463])
])

parser = argparse.ArgumentParser(description='推理脚本')
parser.add_argument('--model', type=str, help='模型路径')

parser.add_argument('--data_dir', type=str, default=None, help='数据路径')

parser.add_argument('--data_folder', type=str, default=None, help='数据文件夹')
parser.add_argument('--label_dir', type=str, default=None, help='标签路径')
parser.add_argument('--n', type=int, default=3, help='随机选择图片数量')

parser.add_argument('--device', type=str, default=None, help='计算设备')

args = parser.parse_args()


def extract_idx(selected_path):
    filename = os.path.basename(selected_path)
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None


def get_random_image(folder_path, labels, transform=transform, device='cpu'):
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_ext:
                image_files.append(file_path)

    if not image_files:
        raise FileNotFoundError(f"文件夹 {folder_path} 中没有找到合适的图片文件，请检查。")

    selected_path = random.choice(image_files)
    idx = extract_idx(selected_path)
    label = labels[idx]
    image = Image.open(selected_path).convert('L')
    transformed = transform(image).unsqueeze(0).to(device)

    if label == 0:
        label = torch.tensor([[1.0, 0.0]])
    elif label == 1:
        label = torch.tensor([[0.0, 1.0]])
    return transformed, label, selected_path, image


if __name__ == '__main__':
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert args.data_dir or (args.data_folder and args.label_dir), '请指定数据路径或数据文件夹'
    model = SECatNet(num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    model.to(args.device)

    if args.data_dir:
        image = Image.open(args.data_dir).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(image_tensor)
            print(torch.softmax(output, dim=1))
    else:
        labels_df = pd.read_excel(args.label_dir, header=None)
        labels = labels_df.iloc[:, 0].values

        fig, axes = plt.subplots(nrows=args.n, ncols=2, figsize=(12, 4 * args.n))
        if args.n == 1:
            axes = [axes]

        for i in range(args.n):
            image_tensor, label, selected_path, original_image = get_random_image(args.data_folder, labels,
                                                                                  device=args.device)
            with torch.no_grad():
                output = model(image_tensor)
                softmax_output = torch.softmax(output, dim=1).cpu().numpy().flatten()

            ax_image = axes[i, 0] if args.n > 1 else axes[0]
            ax_image.imshow(original_image, cmap='gray')
            ax_image.axis('off')
            ax_image.set_title(f"Image: {selected_path}", fontsize=12, pad=10)

            ax_prob = axes[i, 1] if args.n > 1 else axes[1]
            classes = ['Class 0', 'Class 1']
            colors = ['#1f77b4', '#ff7f0e']
            bars = ax_prob.bar(classes, softmax_output, color=colors, width=0.5)

            for bar, prob in zip(bars, softmax_output):
                height = bar.get_height()
                ax_prob.text(bar.get_x() + bar.get_width() / 2, height,
                             f'{prob:.7f}', ha='center', va='bottom', fontsize=10)

            ax_prob.set_ylim(0, 1.1)
            ax_prob.set_title(f"True Label: {label.cpu().numpy().flatten()}", fontsize=12, pad=10)
            ax_prob.set_ylabel('Probability', fontsize=10)
            ax_prob.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.show()