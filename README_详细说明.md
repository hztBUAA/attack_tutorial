# 📖 对抗攻击代码库 - 完整说明

## 🎯 这个代码库是做什么的？

这个代码库实现了多种**对抗攻击（Adversarial Attack）**方法，用于：
1. **测试模型鲁棒性**：评估你的模型是否能抵抗对抗攻击
2. **生成对抗样本**：创建看起来正常但能欺骗模型的图像
3. **研究对抗学习**：理解深度学习模型的脆弱性

## 🔍 什么是对抗攻击？

### 简单理解
想象一下：
- 你有一张猫的图片，AI模型正确识别为"猫"
- 我们在这张图片上添加**人眼几乎看不见**的微小变化
- 现在AI模型错误地识别为"狗"（或其他类别）
- 这就是对抗攻击！

### 核心概念

```
原始图像 + 微小扰动 = 对抗样本
    ↓           ↓          ↓
  正常图片   几乎看不见   欺骗AI模型
           的微小变化
```

## 📁 代码结构说明

### 核心文件

1. **`attack.py`** - 攻击基类
   - 所有攻击方法都继承自这个类
   - 定义了统一的接口：`generate()` 方法

2. **攻击方法文件**（每种攻击一个文件）：
   - `fgsm.py` - FGSM（快速梯度符号方法）
   - `pgd.py` - PGD（投影梯度下降）
   - `mifgsm.py` - MI-FGSM（动量迭代FGSM）
   - `tifgsm.py` - TI-FGSM（平移不变FGSM）
   - `mi_smi.py` - MI-SMI（动量空间迭代）
   - 等等...

### 每个攻击类的结构

```python
class 攻击方法(Attack):
    def __init__(self, model, device, IsTargeted, config):
        # 初始化：设置模型、设备、参数等
        
    def _parse_params(self, config):
        # 解析配置参数（epsilon, num_steps等）
        
    def generate(self, xs, ys):
        # 核心方法：生成对抗样本
        # xs: 原始图像
        # ys: 真实标签
        # 返回: 对抗样本
```

## 🚀 如何使用

### 方法1：运行演示脚本（最简单）

```bash
# 运行完整演示
python run_demo.py
```

这会：
- 自动加载模型
- 生成对抗样本
- 显示对比结果
- 保存可视化图像

### 方法2：在代码中使用

```python
# 1. 导入攻击方法
from fgsm import FGSM

# 2. 准备模型和数据
model = your_model  # 你的分类模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = your_images  # 输入图像
labels = your_labels  # 真实标签

# 3. 创建攻击对象
config = {"epsilon": 0.03}  # 扰动大小
attack = FGSM(model=model, device=device, IsTargeted=False, config=config)

# 4. 生成对抗样本
adversarial_images = attack.generate(xs=images, ys=labels)

# 5. 测试效果
original_pred = model(images).argmax(1)
adversarial_pred = model(adversarial_images).argmax(1)
success = (original_pred != adversarial_pred).sum()
print(f"成功攻击了 {success} 个样本")
```

## 📊 攻击方法对比

| 方法 | 特点 | 速度 | 攻击强度 | 适用场景 |
|------|------|------|----------|----------|
| **FGSM** | 单步攻击 | ⚡⚡⚡ 很快 | ⭐⭐ 中等 | 快速测试 |
| **PGD** | 迭代攻击 | ⚡⚡ 中等 | ⭐⭐⭐ 强 | 评估防御 |
| **MI-FGSM** | 带动量 | ⚡⚡ 中等 | ⭐⭐⭐⭐ 很强 | 需要高成功率 |
| **TI-FGSM** | 平移不变 | ⚡⚡ 中等 | ⭐⭐⭐ 强 | 黑盒攻击 |
| **MI-SMI** | 空间动量 | ⚡ 较慢 | ⭐⭐⭐⭐⭐ 最强 | 高迁移性需求 |

## ⚙️ 重要参数说明

### epsilon (ε) - 扰动大小
- **作用**：控制添加到图像上的最大扰动
- **范围**：通常 0.01 - 0.1
- **影响**：
  - 太小 → 攻击可能失败
  - 太大 → 图像变化明显，容易被发现
- **建议**：从 0.03 开始尝试

### num_steps - 迭代次数
- **作用**：迭代攻击的步数
- **范围**：通常 10 - 20
- **影响**：
  - 更多步数 → 更强攻击，但更慢
  - 更少步数 → 更快，但可能较弱

### IsTargeted - 攻击类型
- **False（非目标攻击）**：只要让模型分类错误即可
- **True（目标攻击）**：让模型预测为指定的错误类别

## 🎨 可视化结果

运行演示后，你会看到：

```
┌─────────────┬─────────────┬─────────────┐
│  原始图像   │  对抗样本    │   扰动      │
│             │             │  (放大显示) │
│ 预测: 猫    │ 预测: 狗    │             │
│ 置信度:0.95 │ 置信度:0.87 │             │
│             │ ✓攻击成功   │             │
└─────────────┴─────────────┴─────────────┘
```

## 💡 使用技巧

### 1. 选择合适的攻击方法
- **快速测试**：用 FGSM
- **评估鲁棒性**：用 PGD
- **需要高成功率**：用 MI-FGSM
- **黑盒攻击**：用 TI-FGSM

### 2. 调整参数
```python
# 如果攻击失败，增大epsilon
config = {"epsilon": 0.05}  # 从0.03增加到0.05

# 如果需要更强攻击，增加迭代次数
config = {
    "epsilon": 0.03,
    "num_steps": 20  # 从10增加到20
}
```

### 3. 评估攻击效果
```python
# 计算攻击成功率
success_rate = (original_pred != adversarial_pred).float().mean()
print(f"攻击成功率: {success_rate:.2%}")

# 计算平均扰动大小
avg_perturbation = (adversarial - original).abs().mean()
print(f"平均扰动: {avg_perturbation:.6f}")
```

## 🔧 常见问题

### Q1: 导入错误 `ModuleNotFoundError: No module named 'attack'`
**解决**：使用 `run_demo.py`，它会自动处理导入问题

### Q2: 攻击总是失败
**解决**：
1. 增大 `epsilon` 值
2. 使用更强的攻击方法（如PGD或MI-FGSM）
3. 增加迭代次数

### Q3: 如何攻击自己的模型？
**解决**：
```python
# 加载你的模型
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval().to(device)

# 然后正常使用攻击方法
attack = FGSM(model=model, device=device, IsTargeted=False, config=config)
```

### Q4: 如何加载真实图像？
**解决**：
```python
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("your_image.jpg")
test_image = transform(image).unsqueeze(0)  # 添加batch维度
```

## 📚 学习路径

1. **入门**：运行 `run_demo.py`，看效果
2. **理解**：阅读 `使用指南.md`，了解原理
3. **实践**：修改参数，观察变化
4. **深入**：阅读论文，理解算法细节

## 🎓 相关论文

- **FGSM**: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
- **PGD**: "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., 2018)
- **MI-FGSM**: "Boosting Adversarial Attacks with Momentum" (Dong et al., 2018)
- **TI-FGSM**: "Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks" (Dong et al., 2019)

## 📝 总结

这个代码库提供了：
- ✅ 多种经典对抗攻击方法
- ✅ 统一的接口，易于使用
- ✅ 完整的演示和文档
- ✅ 可视化工具

**现在就开始使用吧！运行 `python run_demo.py` 看看效果！** 🚀

