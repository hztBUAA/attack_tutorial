#!/usr/bin/env python
# coding=UTF-8
"""
对抗攻击演示脚本
展示如何使用不同的攻击方法生成对抗样本
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 导入攻击方法
from fgsm import FGSM
from pgd import PGD
from mifgsm import MIFGSM
from tifgsm import TIFGSM
from mi_smi import MISMI

# 设置中文字体（用于显示中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_pretrained_model():
    """
    加载预训练的ResNet模型（使用ImageNet预训练权重）
    """
    print("正在加载预训练模型...")
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    return model


def load_test_image():
    """
    加载测试图像（使用ImageNet验证集或自定义图像）
    """
    # 方法1：使用ImageNet验证集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 这里使用一个示例图像，实际使用时可以加载自己的图像
    # 创建一个随机图像作为示例
    test_image = torch.rand(1, 3, 224, 224)
    
    # 获取预测标签
    model = load_pretrained_model()
    with torch.no_grad():
        outputs = model(test_image)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()
    
    return test_image, label


def get_imagenet_class_name(idx):
    """
    获取ImageNet类别名称（简化版）
    """
    # 这里只返回索引，实际使用时可以加载完整的ImageNet类别列表
    return f"类别 {idx}"


def demo_single_attack(attack_class, attack_name, model, device, images, labels, config):
    """
    演示单个攻击方法
    
    Args:
        attack_class: 攻击类
        attack_name: 攻击方法名称
        model: 目标模型
        device: 设备
        images: 输入图像
        labels: 真实标签
        config: 攻击配置参数
    """
    print(f"\n{'='*50}")
    print(f"正在测试: {attack_name}")
    print(f"{'='*50}")
    
    # 创建攻击对象
    attack = attack_class(model=model, device=device, IsTargeted=False, config=config)
    
    # 获取原始预测
    with torch.no_grad():
        original_outputs = model(images.to(device))
        original_pred = original_outputs.argmax(1).cpu().numpy()
        original_conf = torch.softmax(original_outputs, dim=1).max(1)[0].cpu().numpy()
    
    print(f"原始预测: 类别 {original_pred[0]}, 置信度: {original_conf[0]:.4f}")
    
    # 生成对抗样本
    print("正在生成对抗样本...")
    adversarial_images = attack.generate(xs=images, ys=torch.tensor(labels))
    
    # 获取对抗样本的预测
    with torch.no_grad():
        adv_outputs = model(adversarial_images.to(device))
        adv_pred = adv_outputs.argmax(1).cpu().numpy()
        adv_conf = torch.softmax(adv_outputs, dim=1).max(1)[0].cpu().numpy()
    
    print(f"对抗样本预测: 类别 {adv_pred[0]}, 置信度: {adv_conf[0]:.4f}")
    
    # 计算扰动大小
    perturbation = (adversarial_images - images).abs().max().item()
    print(f"最大扰动: {perturbation:.6f}")
    
    # 判断攻击是否成功
    attack_success = original_pred[0] != adv_pred[0]
    print(f"攻击成功: {'是' if attack_success else '否'}")
    
    return {
        'attack_name': attack_name,
        'original_images': images,
        'adversarial_images': adversarial_images,
        'original_pred': original_pred[0],
        'adversarial_pred': adv_pred[0],
        'original_conf': original_conf[0],
        'adversarial_conf': adv_conf[0],
        'perturbation': perturbation,
        'attack_success': attack_success
    }


def visualize_results(results_list):
    """
    可视化攻击结果
    """
    num_attacks = len(results_list)
    fig, axes = plt.subplots(num_attacks, 3, figsize=(15, 5 * num_attacks))
    
    if num_attacks == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results_list):
        # 原始图像
        orig_img = result['original_images'][0].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        # 对抗样本
        adv_img = result['adversarial_images'][0].permute(1, 2, 0).cpu().numpy()
        adv_img = np.clip(adv_img, 0, 1)
        
        # 扰动（放大显示）
        perturbation = (result['adversarial_images'] - result['original_images'])[0]
        pert_img = perturbation.permute(1, 2, 0).abs().cpu().numpy()
        pert_img = pert_img / pert_img.max() if pert_img.max() > 0 else pert_img
        
        # 绘制
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"原始图像\n预测: {result['original_pred']} ({result['original_conf']:.3f})")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(adv_img)
        success_text = "✓ 攻击成功" if result['attack_success'] else "✗ 攻击失败"
        axes[idx, 1].set_title(f"对抗样本 ({result['attack_name']})\n预测: {result['adversarial_pred']} ({result['adversarial_conf']:.3f})\n{success_text}")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pert_img)
        axes[idx, 2].set_title(f"扰动 (放大50倍)\n最大扰动: {result['perturbation']:.6f}")
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('attack_results.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到: attack_results.png")
    plt.show()


def main():
    """
    主函数：演示多种攻击方法
    """
    print("="*60)
    print("对抗攻击演示程序")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_pretrained_model().to(device)
    
    # 加载测试图像
    print("\n正在加载测试图像...")
    images, true_label = load_test_image()
    print(f"测试图像标签: {true_label}")
    
    # 攻击配置
    base_config = {
        "epsilon": 0.03,  # 扰动大小
    }
    
    pgd_config = {
        "epsilon": 0.03,
        "eps_iter": 0.01,
        "num_steps": 10,
    }
    
    mifgsm_config = {
        "epsilon": 0.03,
        "eps_iter": 0.01,
        "num_steps": 10,
        "decay_factor": 1.0,
    }
    
    tifgsm_config = {
        "epsilon": 0.03,
        "alpha": 0.01,
        "num_steps": 10,
        "decay": 1.0,
        "kernel_name": "gaussian",
    }
    
    misi_config = {
        "epsilon": 0.03,
        "eps_iter": 0.01,
        "num_steps1": 10,
        "num_steps2": 5,
        "decay_factor": 1.0,
    }
    
    # 定义要测试的攻击方法
    attacks_to_test = [
        (FGSM, "FGSM", base_config),
        (PGD, "PGD", pgd_config),
        (MIFGSM, "MI-FGSM", mifgsm_config),
        # (TIFGSM, "TI-FGSM", tifgsm_config),  # 如果接口不同可能需要调整
        # (MISMI, "MI-SMI", misi_config),  # 如果接口不同可能需要调整
    ]
    
    results = []
    
    # 测试每种攻击方法
    for attack_class, attack_name, config in attacks_to_test:
        try:
            result = demo_single_attack(
                attack_class, attack_name, model, device, 
                images, [true_label], config
            )
            results.append(result)
        except Exception as e:
            print(f"攻击 {attack_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 可视化结果
    if results:
        print("\n正在生成可视化结果...")
        visualize_results(results)
        
        # 打印统计信息
        print("\n" + "="*60)
        print("攻击统计")
        print("="*60)
        for result in results:
            print(f"{result['attack_name']:15s} | "
                  f"成功: {'是' if result['attack_success'] else '否':4s} | "
                  f"扰动: {result['perturbation']:.6f} | "
                  f"原始置信度: {result['original_conf']:.3f} | "
                  f"对抗置信度: {result['adversarial_conf']:.3f}")


if __name__ == "__main__":
    main()

