#!/usr/bin/env python
# coding=UTF-8
"""
简化版对抗攻击演示
使用最简单的FGSM攻击，快速看到效果
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 导入攻击方法（注意：根据实际导入路径调整）
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 尝试不同的导入方式
try:
    # 方式1: 直接导入（如果文件在根目录）
    from fgsm import FGSM
except ImportError:
    try:
        # 方式2: 作为模块导入
        import fgsm
        FGSM = fgsm.FGSM
    except ImportError:
        # 方式3: 动态导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("fgsm", os.path.join(current_dir, "fgsm.py"))
        fgsm_module = importlib.util.module_from_spec(spec)
        # 需要先导入attack模块
        attack_spec = importlib.util.spec_from_file_location("attack", os.path.join(current_dir, "attack.py"))
        attack_module = importlib.util.module_from_spec(attack_spec)
        spec.loader.exec_module(attack_module)
        sys.modules['attack'] = attack_module
        spec.loader.exec_module(fgsm_module)
        FGSM = fgsm_module.FGSM

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def simple_demo():
    """
    简单的对抗攻击演示
    """
    print("="*50)
    print("简单对抗攻击演示 - FGSM")
    print("="*50)
    
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 加载预训练模型
    print("\n正在加载ResNet18模型...")
    model = torchvision.models.resnet18(pretrained=True)
    model.eval().to(device)
    print("模型加载完成！")
    
    # 3. 准备测试图像
    # 使用CIFAR-10或创建一个简单的测试图像
    print("\n正在准备测试图像...")
    
    # 创建一个随机图像（实际使用时可以加载真实图像）
    test_image = torch.rand(1, 3, 224, 224)
    
    # 获取原始预测
    with torch.no_grad():
        outputs = model(test_image.to(device))
        _, predicted = torch.max(outputs, 1)
        original_pred = predicted.item()
        original_conf = torch.softmax(outputs, dim=1)[0, original_pred].item()
    
    print(f"原始图像预测: 类别 {original_pred}, 置信度: {original_conf:.4f}")
    
    # 4. 创建FGSM攻击
    print("\n正在创建FGSM攻击对象...")
    config = {
        "epsilon": 0.05  # 扰动大小，可以调整这个值看效果
    }
    attack = FGSM(
        model=model, 
        device=device, 
        IsTargeted=False,  # 非目标攻击
        config=config
    )
    
    # 5. 生成对抗样本
    print("正在生成对抗样本...")
    labels = torch.tensor([original_pred])  # 使用原始预测作为标签
    adversarial_image = attack.generate(xs=test_image, ys=labels)
    
    # 6. 测试对抗样本
    with torch.no_grad():
        adv_outputs = model(adversarial_image.to(device))
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_pred = adv_predicted.item()
        adv_conf = torch.softmax(adv_outputs, dim=1)[0, adv_pred].item()
    
    print(f"对抗样本预测: 类别 {adv_pred}, 置信度: {adv_conf:.4f}")
    
    # 7. 计算扰动
    perturbation = (adversarial_image - test_image).abs()
    max_pert = perturbation.max().item()
    mean_pert = perturbation.mean().item()
    
    print(f"\n扰动统计:")
    print(f"  最大扰动: {max_pert:.6f}")
    print(f"  平均扰动: {mean_pert:.6f}")
    
    # 8. 判断攻击是否成功
    attack_success = original_pred != adv_pred
    print(f"\n攻击结果: {'✓ 成功！模型被欺骗了' if attack_success else '✗ 失败，需要增加epsilon'}")
    
    # 9. 可视化结果
    print("\n正在生成可视化图像...")
    visualize_comparison(
        test_image, 
        adversarial_image, 
        original_pred, 
        adv_pred,
        original_conf,
        adv_conf,
        max_pert
    )


def visualize_comparison(original, adversarial, orig_pred, adv_pred, orig_conf, adv_conf, max_pert):
    """
    可视化原始图像和对抗样本的对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 转换为numpy格式用于显示
    orig_img = original[0].permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img, 0, 1)
    
    adv_img = adversarial[0].permute(1, 2, 0).cpu().numpy()
    adv_img = np.clip(adv_img, 0, 1)
    
    # 计算扰动（放大显示）
    pert = (adversarial - original)[0].abs()
    pert_img = pert.permute(1, 2, 0).cpu().numpy()
    # 归一化扰动以便可视化
    pert_img = pert_img / pert_img.max() if pert_img.max() > 0 else pert_img
    
    # 绘制原始图像
    axes[0].imshow(orig_img)
    axes[0].set_title(f'原始图像\n预测: 类别 {orig_pred}\n置信度: {orig_conf:.3f}', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 绘制对抗样本
    success_text = "✓ 攻击成功" if orig_pred != adv_pred else "✗ 攻击失败"
    axes[1].imshow(adv_img)
    axes[1].set_title(f'对抗样本 (FGSM)\n预测: 类别 {adv_pred}\n置信度: {adv_conf:.3f}\n{success_text}', 
                     fontsize=12, fontweight='bold', 
                     color='red' if orig_pred != adv_pred else 'black')
    axes[1].axis('off')
    
    # 绘制扰动
    axes[2].imshow(pert_img)
    axes[2].set_title(f'添加的扰动 (放大显示)\n最大扰动: {max_pert:.6f}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('对抗攻击效果对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('simple_attack_demo.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存到: simple_attack_demo.png")
    
    # 显示图像
    plt.show()


if __name__ == "__main__":
    try:
        simple_demo()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n提示: 请确保已安装所需依赖: torch, torchvision, matplotlib, numpy")

