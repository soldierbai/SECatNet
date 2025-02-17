import matplotlib
matplotlib.use('TkAgg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
from secatnet import SECatNet
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def visualization(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, save_path=None):
    epochs = range(1, num_epochs + 1)

    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到 {save_path}")


# 自定义3D箭头连接器
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


# 3D架构图生成函数
def plot_3d_architecture(model=SECatNet(), input_size=(1, 224, 224)):
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 定义各层三维坐标和尺寸
    layers = [
        {'type': 'Conv2d', 'pos': (0, 0, 0), 'size': (16, 16, 3), 'color': '#FF6B6B'},
        {'type': 'SEBlock', 'pos': (0, 0, 5), 'size': (16, 16, 2), 'color': '#4ECDC4'},
        {'type': 'Conv2d', 'pos': (0, 0, 10), 'size': (32, 32, 3), 'color': '#FF6B6B'},
        {'type': 'Linear', 'pos': (0, 0, 15), 'size': (128, 1, 1), 'color': '#45B7D1'}
    ]

    # 绘制三维模块
    for layer in layers:
        x, y, z = layer['pos']
        dx, dy, dz = layer['size']
        ax.bar3d(x, y, z, dx, dy, dz,
                 color=layer['color'],
                 alpha=0.8,
                 edgecolor='w',
                 linewidth=0.5)

        # 添加文字标注
        ax.text(x + dx / 2, y + dy / 2, z + dz + 0.5,
                f"{layer['type']}\n{layer['size'][0]}→{layer['size'][1]}",
                ha='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))

    # 绘制SE模块内部结构
    se_layer = layers[1]
    ax.plot([se_layer['pos'][0] + 8, se_layer['pos'][0] + 8],
            [se_layer['pos'][1] + 8, se_layer['pos'][1] + 8],
            [se_layer['pos'][2], se_layer['pos'][2] + 5],
            'r--', linewidth=1.5)

    # 添加连接箭头
    for i in range(len(layers) - 1):
        start = [layers[i]['pos'][j] + layers[i]['size'][j] / 2 for j in range(3)]
        end = [layers[i + 1]['pos'][j] + layers[i + 1]['size'][j] / 2 for j in range(3)]
        arrow = Arrow3D([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        mutation_scale=10,
                        lw=1.5,
                        arrowstyle="-|>",
                        color="gray")
        ax.add_artist(arrow)

    # 设置视角和标签
    ax.view_init(elev=25, azim=-45)
    ax.set_xlabel('Channel Dimension', labelpad=15)
    ax.set_ylabel('Spatial Dimension', labelpad=15)
    ax.set_zlabel('Network Depth', labelpad=15)
    ax.set_title('3D Visualization of SECatNet Architecture', pad=20)

    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    model = SECatNet()  # 你的模型类
    plot_3d_architecture(model, input_size=(1, 224, 224))