#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CFR2005 项目开发示例

这个脚本演示了如何在您的项目中使用 CFR submodule 进行开发。
"""

import cfr
from cfr import ReconJob
import numpy as np
import sys

def main():
    """主函数 - 演示 CFR 基本功能"""
    
    print("=" * 60)
    print("CFR2005 项目开发示例")
    print("=" * 60)
    
    # 显示 CFR 版本信息
    print(f"CFR 版本: {cfr.__version__}")
    print(f"CFR 安装位置: {cfr.__file__}")
    print()
    
    # 演示基本功能
    print("1. 创建重建作业对象...")
    try:
        job = ReconJob()
        print("✓ ReconJob 创建成功")
    except Exception as e:
        print(f"✗ ReconJob 创建失败: {e}")
        return False
    
    # 演示模块导入
    print("\n2. 测试各个模块导入...")
    modules_to_test = [
        ('cfr.da.enkf', 'EnKF'),
        ('cfr.psm', 'Linear'),
        ('cfr.proxy', 'ProxyDatabase'),
        ('cfr.climate', 'ClimateField'),
        ('cfr.utils', None)  # 工具模块
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name and hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name} 导入成功")
            elif not class_name:
                print(f"✓ {module_name} 模块导入成功")
            else:
                print(f"? {module_name}.{class_name} 类不存在")
        except ImportError as e:
            print(f"✗ {module_name} 导入失败: {e}")
    
    print("\n3. 演示基本数据操作...")
    try:
        # 创建一些示例数据
        time_points = np.arange(1850, 2000)
        temperature_data = 15 + 3 * np.sin(2 * np.pi * time_points / 30) + np.random.normal(0, 0.5, len(time_points))
        
        print(f"✓ 创建了 {len(time_points)} 个时间点的温度数据")
        print(f"  时间范围: {time_points[0]} - {time_points[-1]}")
        print(f"  温度范围: {temperature_data.min():.2f} - {temperature_data.max():.2f}°C")
        
    except Exception as e:
        print(f"✗ 数据操作失败: {e}")
    
    print("\n" + "=" * 60)
    print("开发环境测试完成！")
    print("\n下一步您可以：")
    print("1. 查看 lmr_workflow_example.py 了解完整工作流")
    print("2. 阅读 CFR_LMR_Analysis.md 了解详细分析")
    print("3. 查看 DEVELOPMENT_GUIDE.md 了解开发指南")
    print("4. 开始您自己的气候重建分析！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
