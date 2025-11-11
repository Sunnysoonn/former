import ssl
import os

# 在导入torchvision之前设置SSL上下文，以处理证书验证问题
try:
    # 创建未验证的SSL上下文（仅用于下载数据集）
    # 注意：这仅在下载数据集时使用，生产环境应使用正确的证书
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # 某些Python版本可能不支持
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNISTNet


def load_mnist_dataset(root='./data', train=True, transform=None, download=True):
    """加载MNIST数据集，处理下载错误"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = datasets.MNIST(
                root=root,
                train=train,
                download=download,
                transform=transform
            )
            return dataset
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print("重试中...")
                # 等待一下再重试
                import time
                time.sleep(2)
            else:
                print(f"\n{'='*60}")
                print("错误: 无法下载MNIST数据集")
                print(f"错误信息: {e}")
                print(f"{'='*60}")
                print("\n解决方案:")
                print("1. 检查网络连接")
                print("2. 如果遇到SSL证书错误，可以尝试:")
                print("   - 更新certifi: pip install --upgrade certifi")
                print("   - 或者在Python中运行以下代码:")
                print("     import ssl")
                print("     ssl._create_default_https_context = ssl._create_unverified_context")
                print("3. 手动下载数据集:")
                print("   - 访问: https://github.com/pytorch/vision/issues/1938")
                print("   - 或使用其他数据源下载MNIST数据集")
                print("   - 将文件放到 ./data/MNIST/raw/ 目录下")
                print("4. 如果数据集已存在，设置 download=False")
                print(f"{'='*60}\n")
                raise RuntimeError("无法下载MNIST数据集，请查看上面的解决方案") from e


def train_model(num_epochs=10, batch_size=64, learning_rate=0.001, 
                embed_dim=128, depth=4, num_heads=4):
    """训练MNIST分类模型（Vision Transformer）"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 检查数据集是否已存在
    data_exists = os.path.exists('./data/MNIST/processed/training.pt') and \
                  os.path.exists('./data/MNIST/processed/test.pt')
    
    if data_exists:
        print("检测到已存在的数据集，跳过下载")
        download_flag = False
    else:
        print("开始下载MNIST数据集...")
        download_flag = True
    
    # 加载训练数据
    try:
        train_dataset = load_mnist_dataset(
            root='./data',
            train=True,
            transform=transform,
            download=download_flag
        )
    except RuntimeError:
        # 如果下载失败，尝试不下载（可能已存在）
        print("尝试加载已存在的数据集...")
        train_dataset = load_mnist_dataset(
            root='./data',
            train=True,
            transform=transform,
            download=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 加载测试数据
    try:
        test_dataset = load_mnist_dataset(
            root='./data',
            train=False,
            transform=transform,
            download=download_flag
        )
    except RuntimeError:
        # 如果下载失败，尝试不下载（可能已存在）
        print("尝试加载已存在的测试集...")
        test_dataset = load_mnist_dataset(
            root='./data',
            train=False,
            transform=transform,
            download=False
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 初始化Vision Transformer模型
    model = MNISTNet(
        num_classes=10,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数数: {total_params:,}')
    print(f'可训练参数数: {trainable_params:,}')
    print('-' * 60)
    
    # 损失函数和优化器（使用较小的学习率，Transformer通常需要warmup）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # 计算训练准确率
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # 测试准确率
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'训练损失: {avg_loss:.4f}, '
              f'训练准确率: {train_accuracy:.2f}%, '
              f'测试准确率: {test_accuracy:.2f}%')
        print('-' * 60)
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('模型已保存到 mnist_model.pth')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    return model


def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print('训练曲线已保存到 training_curves.png')
    plt.close()


if __name__ == '__main__':
    # 训练Vision Transformer模型
    # Transformer参数: embed_dim=128, depth=4, num_heads=4
    model = train_model(
        num_epochs=5, 
        batch_size=64, 
        learning_rate=0.001,
        embed_dim=128,
        depth=4,
        num_heads=4
    )

