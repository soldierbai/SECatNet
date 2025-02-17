# SECatNet


环境准备：
```bash
pip install -r requirements.txt
```
启动训练：（以 `HC_EMCI` 数据集为例）
```bash
# MacOS
./start_training.sh --data_dir './data/HC_EMCI' --labels_dir './labels/EMCI_HCtrain_list.xlsx' --verbose
# Linux/Windows
./start_training.bat --data_dir './data/HC_EMCI' --labels_dir './labels/EMCI_HCtrain_list.xlsx' --verbose
```
结果演示：
```bash
python forward.py --model './models/HC_EMCI.pth' --data_folder './data/HC_EMCI' --label_path './labels/EMCI_HCtrain_list.xlsx'
```

目录结构：
```
SECatNet
·
├── README.md
├── data  # 数据集
│   ├── AD_EMCI
│   ├── AD_HC
│   └── HC_EMCI
├── labels  # 标签集
│   ├── AD_EMCItrain_list.xlsx
│   ├── AD_HCtrain_list.xlsx
│   └── EMCI_HCtrain_list.xlsx
├── model  # 模型
│   └── HC_EMCI.pth
├── result  # 可视化结果
│   ├── Figure_1.png
│   └── HC_EMCI.png
├── src
│   ├── dataset.py  # 数据集结构
│   ├── secatnet.py  # 模型结构
│   ├── train.py  # 模型训练
│   └── visualization.py  # 可视化
├── forward.py  # 推理脚本
├── main.py  # 训练主入口
├── requirements.txt  # 依赖
├── start_training.bat  # Windows启动脚本
└── start_training.sh  # MacOS启动脚本
```