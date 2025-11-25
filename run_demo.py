#!/usr/bin/env python
# coding=UTF-8
"""
可直接运行的对抗攻击演示脚本
自动处理导入问题，展示FGSM攻击效果
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 修复相对导入问题
import importlib.util

# 动态导入attack模块
attack_path = os.path.join(current_dir, "attack.py")
if os.path.exists(attack_path):
    spec = importlib.util.spec_from_file_location("attack", attack_path)
    attack_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attack_module)
    sys.modules['attack'] = attack_module

# 动态导入fgsm模块
fgsm_path = os.path.join(current_dir, "fgsm.py")
if os.path.exists(fgsm_path):
    spec = importlib.util.spec_from_file_location("fgsm", fgsm_path)
    fgsm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fgsm_module)
    FGSM = fgsm_module.FGSM
else:
    print("错误: 找不到 fgsm.py 文件")
    sys.exit(1)

import random
import torch
import torchvision
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ImageNet类别标签
IMAGENET_LABELS = []


def get_label_name(idx):
    """根据索引返回ImageNet类别名称"""
    if 0 <= idx < len(IMAGENET_LABELS):
        return IMAGENET_LABELS[idx]
    return f"类别 {idx}"


def main():
    print("="*60)
    print("对抗攻击演示 - FGSM方法")
    print("="*60)
    
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 2. 加载预训练模型
    print("\n正在加载ResNet18模型...")
    try:
        weights = ResNet18_Weights.DEFAULT
        global IMAGENET_LABELS
        IMAGENET_LABELS = weights.meta.get("categories", [])
        model = torchvision.models.resnet18(weights=weights)
        model.eval().to(device)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("提示: 请确保网络连接正常，或手动下载模型权重")
        return
    
    # 3. 准备测试图像
    print("\n正在准备测试图像...")
    
    # 尝试从images目录加载图片
    images_dir = "images"
    test_image = None
    image_path = None
    
    if os.path.exists(images_dir):
        # 获取images目录中的所有图片文件
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            # 随机选择一张图片
            selected_file = random.choice(image_files)
            image_path = os.path.join(images_dir, selected_file)
            print(f"从 {images_dir}/ 目录随机加载图片: {os.path.basename(image_path)}")
            
            try:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                
                pil_image = Image.open(image_path).convert('RGB')
                test_image = transform(pil_image).unsqueeze(0)  # 添加batch维度
                print(f"✓ 成功加载图片")
            except Exception as e:
                print(f"✗ 加载图片失败: {e}")
                print("  使用随机图片作为替代")
                test_image = None
    
    # 如果没有成功加载，使用随机图片
    if test_image is None:
        print("使用随机生成的测试图片")
        print("提示: 运行 'python download_test_images.py' 下载测试图片")
        test_image = torch.rand(1, 3, 224, 224)
    
    # 获取原始预测
    with torch.no_grad():
        outputs = model(test_image.to(device))
        _, predicted = torch.max(outputs, 1)
        original_pred = predicted.item()
        original_conf = torch.softmax(outputs, dim=1)[0, original_pred].item()
    
    orig_label_name = get_label_name(original_pred)
    print(f"原始图像预测: 类别 {original_pred} ({orig_label_name}), 置信度: {original_conf:.4f}")
    
    # 4. 创建FGSM攻击
    print("\n正在创建FGSM攻击对象...")
    config = {
        "epsilon": 0.05  # 扰动大小
    }
    
    try:
        attack = FGSM(
            model=model, 
            device=device, 
            IsTargeted=False,  # 非目标攻击
            config=config
        )
        print("✓ 攻击对象创建成功")
    except Exception as e:
        print(f"✗ 攻击对象创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 生成对抗样本
    print("\n正在生成对抗样本...")
    try:
        labels = torch.tensor([original_pred])
        adversarial_image = attack.generate(xs=test_image, ys=labels)
        print("✓ 对抗样本生成成功")
    except Exception as e:
        print(f"✗ 对抗样本生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 测试对抗样本
    with torch.no_grad():
        adv_outputs = model(adversarial_image.to(device))
        _, adv_predicted = torch.max(adv_outputs, 1)
        adv_pred = adv_predicted.item()
        adv_conf = torch.softmax(adv_outputs, dim=1)[0, adv_pred].item()
    
    adv_label_name = get_label_name(adv_pred)
    print(f"对抗样本预测: 类别 {adv_pred} ({adv_label_name}), 置信度: {adv_conf:.4f}")
    
    # 7. 计算扰动
    perturbation = (adversarial_image - test_image).abs()
    max_pert = perturbation.max().item()
    mean_pert = perturbation.mean().item()
    
    print(f"\n扰动统计:")
    print(f"  最大扰动: {max_pert:.6f}")
    print(f"  平均扰动: {mean_pert:.6f}")
    
    # 8. 判断攻击是否成功
    attack_success = original_pred != adv_pred
    if attack_success:
        print(f"\n🎉 攻击成功！模型被欺骗了")
        print(f"   原始预测: 类别 {original_pred} ({orig_label_name})")
        print(f"   对抗预测: 类别 {adv_pred} ({adv_label_name})")
    else:
        print(f"\n⚠️  攻击失败，模型仍然正确预测")
        print(f"   提示: 尝试增大 epsilon 值（当前: {config['epsilon']}）")
    
    # 9. 可视化结果
    print("\n正在生成可视化图像...")
    try:
        visualize_results(
            test_image, 
            adversarial_image, 
            original_pred, 
            adv_pred,
            orig_label_name,
            original_conf,
            adv_label_name,
            adv_conf,
            max_pert,
            attack_success
        )
        print("✓ 可视化完成，图像已保存")
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def visualize_results(original, adversarial, orig_pred, adv_pred, orig_label_name, orig_conf, adv_label_name, adv_conf, max_pert, success):
    """可视化攻击结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 转换为numpy格式
    orig_img = original[0].permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img, 0, 1)
    
    adv_img = adversarial[0].permute(1, 2, 0).cpu().numpy()
    adv_img = np.clip(adv_img, 0, 1)
    
    # 计算扰动（放大显示）
    pert = (adversarial - original)[0].abs()
    pert_img = pert.permute(1, 2, 0).cpu().numpy()
    pert_img = pert_img / pert_img.max() if pert_img.max() > 0 else pert_img
    
    # 绘制原始图像
    axes[0].imshow(orig_img)
    label_text = orig_label_name or get_label_name(orig_pred)
    axes[0].set_title(f'原始图像\n预测: {orig_pred} - {label_text}\n置信度: {orig_conf:.3f}', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 绘制对抗样本
    color = 'red' if success else 'black'
    success_text = "✓ 攻击成功" if success else "✗ 攻击失败"
    axes[1].imshow(adv_img)
    adv_label_text = adv_label_name or get_label_name(adv_pred)
    axes[1].set_title(f'对抗样本 (FGSM)\n预测: {adv_pred} - {adv_label_text}\n置信度: {adv_conf:.3f}\n{success_text}', 
                     fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')
    
    # 绘制扰动
    axes[2].imshow(pert_img, cmap='hot')
    axes[2].set_title(f'添加的扰动 (放大显示)\n最大扰动: {max_pert:.6f}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('对抗攻击效果对比 - FGSM', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    output_file = 'attack_demo_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   保存位置: {output_file}")
    
    # 显示图像
    try:
        plt.show()
    except:
        print("   注意: 无法显示图像（可能在没有图形界面的环境中）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n提示: 请确保已安装所需依赖:")
        print("  pip install torch torchvision matplotlib numpy pillow")

