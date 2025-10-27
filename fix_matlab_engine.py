#!/usr/bin/env python3
"""
临时脚本：修复MATLAB引擎配置
"""
import os
import sys

# 设置MATLAB路径
MATLAB_ROOT = "/home/aiden/snap/code/app/matlab"
MATLAB_ENGINE_PATH = f"{MATLAB_ROOT}/extern/engines/python/dist"

# 创建_arch.txt文件（如果不存在）
arch_file = f"{MATLAB_ENGINE_PATH}/matlab/engine/_arch.txt"
if not os.path.exists(arch_file):
    try:
        with open(arch_file, 'w') as f:
            f.write("glnxa64\n")
        print(f"已创建 {arch_file}")
    except PermissionError:
        print(f"无法创建 {arch_file}，权限不够")
        # 尝试其他方法
        os.environ['MATLAB_ENGINE_ARCH'] = 'glnxa64'

# 设置环境变量
os.environ['MATLAB_ROOT'] = MATLAB_ROOT
os.environ['LD_LIBRARY_PATH'] = f"{MATLAB_ROOT}/bin/glnxa64:" + os.environ.get('LD_LIBRARY_PATH', '')

# 添加Python路径
if MATLAB_ENGINE_PATH not in sys.path:
    sys.path.insert(0, MATLAB_ENGINE_PATH)

print("MATLAB环境配置完成")
print(f"MATLAB_ROOT: {MATLAB_ROOT}")
print(f"Python路径已添加: {MATLAB_ENGINE_PATH}")

# 测试导入
try:
    import matlab.engine
    print("✓ MATLAB引擎导入成功！")
    
    # 测试启动引擎
    try:
        print("正在启动MATLAB引擎...")
        eng = matlab.engine.start_matlab()
        print("✓ MATLAB引擎启动成功！")
        
        # 简单测试
        result = eng.sqrt(4.0)
        print(f"✓ MATLAB计算测试: sqrt(4) = {result}")
        
        # 关闭引擎
        eng.quit()
        print("✓ MATLAB引擎已关闭")
        
    except Exception as e:
        print(f"✗ MATLAB引擎启动失败: {e}")
        
except ImportError as e:
    print(f"✗ MATLAB引擎导入失败: {e}")

print("\n=== 现在可以测试SPPVT控制器 ===")