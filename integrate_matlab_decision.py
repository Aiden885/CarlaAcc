#!/usr/bin/env python3
"""
集成MATLAB决策模块到项目中
修改acc_updated.py以支持MATLAB和Python决策模块切换
"""

import shutil
import os
from pathlib import Path


def backup_original_file():
    """备份原始文件"""
    original_file = "acc_updated.py"
    backup_file = "acc_updated_backup.py"
    
    if os.path.exists(original_file) and not os.path.exists(backup_file):
        shutil.copy2(original_file, backup_file)
        print(f"✅ 已备份原始文件: {backup_file}")
    elif os.path.exists(backup_file):
        print(f"⚠️ 备份文件已存在: {backup_file}")


def modify_acc_updated():
    """修改acc_updated.py以支持决策模块切换"""
    
    # 读取原始文件
    with open("acc_updated.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 检查是否已经修改过
    if "DecisionModuleFactory" in content:
        print("⚠️ 文件已经被修改过，跳过修改")
        return
    
    print("🔧 修改acc_updated.py以支持MATLAB决策模块...")
    
    # 替换导入部分
    old_import = "from acc_decision import ACCDecisionModule, ACCCommand, ACCState"
    new_import = """# 决策模块导入 - 支持MATLAB和Python版本
from decision_factory import DecisionModuleFactory, UnifiedDecisionModule, DecisionBackend
from acc_decision import ACCCommand, ACCState"""
    
    content = content.replace(old_import, new_import)
    
    # 替换初始化部分
    old_init = "self.acc_decision = ACCDecisionModule(initial_V3_kmh=50.0, initial_G1_m=15.0, initial_time_gap=2.0)"
    new_init = """# === 决策模块初始化 - 支持MATLAB/Python切换 ===
        # 可以通过环境变量DECISION_BACKEND设置后端: python, matlab, auto
        backend_env = os.environ.get('DECISION_BACKEND', 'auto').lower()
        if backend_env == 'python':
            preferred_backend = DecisionBackend.PYTHON
        elif backend_env == 'matlab':
            preferred_backend = DecisionBackend.MATLAB  
        else:
            preferred_backend = DecisionBackend.AUTO
        
        print(f"🧠 初始化决策模块，首选后端: {preferred_backend.value}")
        self.acc_decision = UnifiedDecisionModule(
            preferred_backend=preferred_backend,
            initial_V3_kmh=50.0, 
            initial_G1_m=15.0, 
            initial_time_gap=2.0
        )"""
    
    content = content.replace(old_init, new_init)
    
    # 添加决策后端切换快捷键
    keyboard_section = """                elif event_data == K_p:
                    debug_state = not self.acc_decision.debug
                    self.acc_decision.set_debug(debug_state)
                    print(f"ACC调试模式: {'开启' if debug_state else '关闭'}")"""
    
    new_keyboard_section = """                elif event_data == K_p:
                    debug_state = not self.acc_decision.debug
                    self.acc_decision.set_debug(debug_state)
                    print(f"ACC调试模式: {'开启' if debug_state else '关闭'}")
                
                # === 决策后端切换快捷键 ===
                elif event_data == K_F1:
                    try:
                        self.acc_decision.switch_backend(DecisionBackend.PYTHON)
                        print("🐍 已切换到Python决策模块")
                    except Exception as e:
                        print(f"❌ 切换到Python失败: {e}")
                
                elif event_data == K_F2:
                    try:
                        self.acc_decision.switch_backend(DecisionBackend.MATLAB)
                        print("🚀 已切换到MATLAB决策模块")
                    except Exception as e:
                        print(f"❌ 切换到MATLAB失败: {e}")
                
                elif event_data == K_F3:
                    backend_info = self.acc_decision.get_backend_info()
                    print(f"📋 决策后端信息:")
                    print(f"  当前后端: {backend_info['current_backend']}")
                    print(f"  可用后端: {backend_info['available_backends']}")
                    print(f"  有备用模块: {backend_info['has_fallback']}")"""
    
    content = content.replace(keyboard_section, new_keyboard_section)
    
    # 在文件开头添加os导入
    if "import os" not in content[:500]:  # 检查前500个字符
        content = "import os\n" + content
    
    # 写入修改后的文件
    with open("acc_updated.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ acc_updated.py修改完成")
    print("📋 新增功能:")
    print("  - 支持通过环境变量DECISION_BACKEND设置决策后端")
    print("  - F1键切换到Python决策模块")
    print("  - F2键切换到MATLAB决策模块")
    print("  - F3键显示决策后端信息")


def create_usage_guide():
    """创建使用指南"""
    guide_content = """# MATLAB决策模块集成指南

## 概述
本项目现在支持两种决策实现：
- **Python版本**: 基于 `acc_decision.py` 的纯Python实现
- **MATLAB版本**: 基于 `decision.md` 逻辑的Simulink/Stateflow实现

两个版本都严格遵循 `decision.md` 中定义的4状态7指令逻辑，具有完全相同的接口。

## 快速开始

### 1. 设置环境变量（可选）
```bash
# 强制使用Python版本
export DECISION_BACKEND=python

# 强制使用MATLAB版本（需要MATLAB Engine for Python）
export DECISION_BACKEND=matlab

# 自动选择（默认，优先MATLAB）
export DECISION_BACKEND=auto
```

### 2. 运行主程序
```bash
python acc_updated.py
```

### 3. 运行时切换（键盘快捷键）
- **F1**: 切换到Python决策模块
- **F2**: 切换到MATLAB决策模块  
- **F3**: 显示当前决策后端信息

## MATLAB版本特性

### 文件结构
```
create_decision_md_simulink.m      # 创建Simulink模型
configure_decision_md_stateflow.m  # 配置Stateflow状态机
test_decision_md_simulink.m        # 测试MATLAB实现
matlab_decision_interface.py      # Python-MATLAB接口
decision_factory.py               # 决策模块工厂
```

### MATLAB版本优势
- 🎯 完全基于decision.md的可视化状态机
- 🔧 支持Simulink仿真和代码生成
- 📊 提供状态转移的图形化表示
- 🚀 可能具有更好的实时性能

### Python版本优势
- 🐍 无需MATLAB依赖，部署简单
- 🔧 易于调试和修改
- 📝 代码逻辑清晰可读
- ⚡ 启动快速，内存占用小

## 创建MATLAB模型

### 手动创建
```matlab
% 在MATLAB命令窗口中执行
create_decision_md_simulink();      % 创建基础模型
configure_decision_md_stateflow();  % 配置状态机逻辑
test_decision_md_simulink();        % 运行测试
```

### 自动创建
运行Python程序时，如果选择MATLAB后端但模型不存在，会自动创建。

## 测试验证

### Python测试
```bash
python simple_decision_test.py      # 测试状态转移表
python decision_factory.py          # 工厂模式演示和性能测试
```

### MATLAB测试
```matlab
test_decision_md_simulink()         % Simulink模型测试
```

## 性能比较

运行性能基准测试：
```bash
python decision_factory.py
```

## 故障排除

### MATLAB Engine问题
1. 确保已安装 MATLAB Engine for Python
```bash
cd "matlabroot\extern\engines\python"
python setup.py install
```

2. 检查MATLAB路径设置
3. 确认MATLAB许可证有效

### 模型创建失败
1. 检查MATLAB版本兼容性（推荐R2019b以上）
2. 确认Stateflow工具箱已安装
3. 检查文件权限

### 决策逻辑不一致
1. 运行测试脚本验证逻辑
2. 检查decision.md文档更新
3. 确认两个版本同步

## 开发指南

### 添加新的决策逻辑
1. 更新 `decision.md` 文档
2. 修改 `acc_decision.py` (Python版本)
3. 更新 `configure_decision_md_stateflow.m` (MATLAB版本)
4. 运行测试确保一致性

### 扩展接口
1. 在两个版本中同时添加接口
2. 更新 `matlab_decision_interface.py` 中的转换逻辑
3. 测试接口兼容性

## 注意事项

1. **状态同步**: 运行时切换后端时，状态可能需要重新初始化
2. **性能差异**: MATLAB版本启动较慢，但运行时可能更快
3. **依赖管理**: MATLAB版本需要额外的依赖和许可证
4. **调试**: Python版本更容易调试，MATLAB版本适合可视化分析

## 支持

- 查看日志输出了解当前使用的决策后端
- 使用F3键获取实时后端信息
- 参考测试文件了解用法示例
"""
    
    with open("MATLAB_DECISION_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("✅ 已创建使用指南: MATLAB_DECISION_GUIDE.md")


def main():
    """主集成函数"""
    print("=== MATLAB决策模块集成到项目 ===")
    
    # 检查必要文件
    required_files = [
        "acc_updated.py",
        "acc_decision.py",
        "decision_factory.py",
        "matlab_decision_interface.py",
        "create_decision_md_simulink.m",
        "configure_decision_md_stateflow.m"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return
    
    # 备份原始文件
    backup_original_file()
    
    # 修改acc_updated.py
    modify_acc_updated()
    
    # 创建使用指南
    create_usage_guide()
    
    print("\n🎉 MATLAB决策模块集成完成！")
    print("\n📋 使用方法:")
    print("1. 设置环境变量: export DECISION_BACKEND=matlab|python|auto")
    print("2. 运行程序: python acc_updated.py")
    print("3. 运行时切换: F1(Python) F2(MATLAB) F3(信息)")
    print("\n📖 详细指南: MATLAB_DECISION_GUIDE.md")
    
    # 验证集成
    print("\n🔍 验证集成...")
    try:
        from decision_factory import DecisionModuleFactory, DecisionBackend
        available_backends = DecisionModuleFactory.get_available_backends()
        print(f"✅ 可用决策后端: {[b.value for b in available_backends]}")
        
        if DecisionBackend.MATLAB in available_backends:
            print("✅ MATLAB决策模块可用")
        else:
            print("⚠️ MATLAB决策模块不可用（需要安装MATLAB Engine for Python）")
            
    except Exception as e:
        print(f"❌ 集成验证失败: {e}")


if __name__ == "__main__":
    main()