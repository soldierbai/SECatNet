from src.train import *
from src.dataset import *
from src.secatnet import *
from src.visualization import *
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import os
import argparse

torch.set_printoptions(precision=7, sci_mode=False)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

parser = argparse.ArgumentParser(description='模型评估')

parser.add_argument('--model', type=str, help='模型路径')
parser.add_argument('--data_dir', type=str, help='数据路径')
parser.add_argument('--label_dir', type=str, help='标签路径')

parser.add_argument('--batch_size', type=int, default=64, help='批次大小')

parser.add_argument('--device', type=str, default=None, help='计算设备')

args = parser.parse_args()

if __name__ == '__main__':
    try:
        if args.device:
            assert args.device in ['cpu', 'cuda'], f"--device 必须是 'cpu' 或 'cuda'"

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.device else args.device
        device = args.device

        print("=" * 30 + ' 启动信息 ' + "=" * 30, end='\n\n')
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print()
        print("=" * 70, end="\n\n")

        # print("Using device:", device)

        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.11932160705327988], std=[0.2542687952518463])
        ])


        print("Initiating data reading...")
        dataset = CustomDataset(data_dir=args.data_dir, labels_dir=args.label_dir, transform=transform, prefix='tr')

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SECatNet(num_classes=2)
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        model.eval()
        model.to(args.device)
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate_model(model, dataloader, criterion, device)
        print(f'Loss: {loss:.4f}, Acc: {acc:.2f} %')

    except AssertionError as e:
        print(f"启动参数有误: {e}")
