#!/usr/bin/env python3
"""
环境依赖检测脚本
运行此脚本检查当前环境是否满足项目依赖要求
"""

import sys


def check_dependency(package_name, import_name=None, min_version=None):
    """检查单个依赖是否安装及版本"""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        
        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, version, f"需要 >= {min_version}"
        return True, version, "OK"
    except ImportError:
        return False, None, "未安装"


def main():
    print("=" * 60)
    print("MOT20-YOLOv26 项目环境依赖检测")
    print("=" * 60)
    
    # 核心依赖列表
    dependencies = [
        ("torch", "torch", "2.0.0"),
        ("torchvision", "torchvision", "0.15.0"),
        ("numpy", "numpy", "1.21.0"),
        ("opencv-python", "cv2", "4.5.0"),
        ("pillow", "PIL", None),
        ("pyyaml", "yaml", None),
        ("scipy", "scipy", None),
        ("pandas", "pandas", None),
        ("tqdm", "tqdm", None),
        ("matplotlib", "matplotlib", None),
        ("seaborn", "seaborn", None),
        ("scikit-learn", "sklearn", None),
        ("filterpy", "filterpy", None),  # Kalman Filter
        ("lap", "lap", None),  # Linear Assignment Problem
        ("cython", "Cython", None),  # 用于加速
        ("packaging", "packaging", None),  # 版本比较
    ]
    
    # 可选依赖
    optional_deps = [
        ("tensorrt", "tensorrt", None),  # 推理加速
        ("onnx", "onnx", None),
        ("onnxruntime-gpu", "onnxruntime", None),
        ("pycuda", "pycuda", None),
    ]
    
    print("\n【核心依赖】")
    print("-" * 60)
    missing_core = []
    for pkg, imp, min_ver in dependencies:
        installed, version, status = check_dependency(pkg, imp, min_ver)
        symbol = "✓" if installed else "✗"
        ver_str = f"({version})" if version else ""
        print(f"  {symbol} {pkg:20s} {ver_str:15s} - {status}")
        if not installed:
            missing_core.append(pkg)
    
    print("\n【可选依赖】")
    print("-" * 60)
    for pkg, imp, min_ver in optional_deps:
        installed, version, status = check_dependency(pkg, imp, min_ver)
        symbol = "○" if installed else "-"
        ver_str = f"({version})" if version else ""
        print(f"  {symbol} {pkg:20s} {ver_str:15s} - {status}")
    
    print("\n" + "=" * 60)
    print("【检测结果摘要】")
    print("-" * 60)
    
    # CUDA检测
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "N/A"
        
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {'是' if cuda_available else '否'}")
        print(f"  CUDA版本: {cuda_version}")
        print(f"  GPU数量: {device_count}")
        print(f"  GPU型号: {device_name}")
    except Exception as e:
        print(f"  CUDA检测失败: {e}")
    
    print("-" * 60)
    if missing_core:
        print(f"  ✗ 缺少 {len(missing_core)} 个核心依赖: {', '.join(missing_core)}")
        print("\n  请运行以下命令安装缺失依赖:")
        print(f"  pip install {' '.join(missing_core)}")
        return 1
    else:
        print("  ✓ 所有核心依赖已安装")
        return 0


if __name__ == "__main__":
    sys.exit(main())
