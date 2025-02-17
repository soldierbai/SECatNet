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

parser = argparse.ArgumentParser(description='训练主入口')

parser.add_argument('--data_dir', type=str, help='数据路径')
parser.add_argument('--labels_dir', type=str, help='标签路径')

parser.add_argument('--num_epochs', type=int, default=100, help='训练轮次')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
parser.add_argument('--weight_decay', type=float, default=0.00005, help='l2正则化')
parser.add_argument('--test_ratio', type=float, default=0.2, help='训练集比例')
parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
parser.add_argument('--random_seed', type=int, default=42, help='随机种子')

parser.add_argument('--device', type=str, default=None, help='计算设备')
parser.add_argument('--output_model_dir', type=str, default=None, help='保存模型路径')
parser.add_argument('--output_result_dir', type=str, default=None, help='保存结果路径')
parser.add_argument('--verbose', action='store_true', help='是否显示详细信息')

args = parser.parse_args()

if __name__ == '__main__':
    try:
        assert args.num_epochs >= 1, "--num_epochs 必须大于等于1"
        assert args.lr > 0, "--lr 必须大于0"
        assert args.batch_size >= 1, "--batch_size 必须大于等于1"
        assert args.weight_decay > 0, "--weight_decay 必须大于0"
        assert 0 < args.test_ratio < 1, "--test_ratio 必须在0～1之间"
        assert 0 < args.val_ratio < 1, "--val_ratio 必须在0～1之间"
        assert args.test_ratio + args.val_ratio < 1, "--test_ratio 和 --val_ratio 之和必须小于1"
        if args.device:
            assert args.device in ['cpu', 'cuda'], f"--device 必须是 'cpu' 或 'cuda'"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.device else args.device

        if args.verbose:
            print("=" * 30 + ' 启动信息 ' + "=" * 30, end='\n\n')
            for arg in vars(args):
                print(f"{arg}: {getattr(args, arg)}")
            print()
            print("=" * 70, end="\n\n")

        print("Using device:", device)

        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.11932160705327988], std=[0.2542687952518463])
        ])

        # 创建数据集实例
        if args.verbose:
            print("Initiating data reading...")
        dataset = CustomDataset(data_dir=args.data_dir, labels_dir=args.labels_dir, transform=transform)

        test_size = int(args.test_ratio * len(dataset))
        train_val_size = len(dataset) - test_size
        train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

        val_size = int(args.val_ratio * train_val_size)
        train_size = train_val_size - val_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SECatNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        if args.verbose:
            print("Start training...")
        for epoch in range(args.num_epochs):
            train_loss, train_acc = train_model(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if args.verbose:
                print(
                    f'Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if args.verbose:
            print("Training done.")
        test_loss, test_acc = evaluate_model(model, test_dataloader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        if args.output_model_dir:
            if not os.path.exists(args.output_model_dir):
                os.makedirs(args.output_model_dir)
            model_path = os.path.join(args.output_model_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"模型参数已保存到 {model_path}")

        visualization(num_epochs=args.num_epochs, train_losses=train_losses, train_accuracies=train_accuracies,
                      val_losses=val_losses, val_accuracies=val_accuracies, save_path=args.output_result_dir)

    except AssertionError as e:
        print(f"启动参数有误: {e}")
