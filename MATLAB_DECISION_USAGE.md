# MATLAB决策模块使用指南

## 概述

本项目现在支持两种决策实现方式，都严格遵循 `decision.md` 中定义的4状态7指令逻辑：

- **🐍 Python版本**: 基于 `acc_decision.py` 的纯Python实现
- **🧮 MATLAB版本**: 基于Simulink/Stateflow的可视化状态机实现

两个版本具有完全相同的接口，可以在运行时无缝切换。

## 🚀 快速开始

### 1. 设置环境变量（可选）
```bash
# 强制使用Python版本（默认，快速启动）
export DECISION_BACKEND=python

# 强制使用MATLAB版本（需要MATLAB Engine）
export DECISION_BACKEND=matlab

# 自动选择（优先MATLAB，不可用时切换Python）
export DECISION_BACKEND=auto
```

### 2. 运行主程序
```bash
python acc_updated.py
```

程序启动时会显示当前使用的决策后端：
```
🧠 初始化决策模块，首选后端: auto
🐍 使用Python决策模块  # 或 🚀 使用MATLAB决策模块
```

### 3. 运行时控制

#### 决策后端切换
- **F1**: 切换到Python决策模块
- **F2**: 切换到MATLAB决策模块
- **F3**: 显示当前决策后端信息

#### ACC控制（基于decision.md）
- **1**: 激活系统 (I4-THROTTLE指令)
- **2**: 取消系统 (I6-CANCEL指令)  
- **3**: 人工刹车 (I5-BRAKE指令)
- **Q**: 增速 (I1指令)
- **E**: 减速 (I0指令)
- **R**: 增距 (I3指令)
- **T**: 减距 (I2指令)
- **P**: 切换调试模式

## 📁 文件结构

```
CarlaAcc/
├── 🐍 Python决策模块
│   ├── acc_decision.py              # 原始Python决策实现
│   ├── decision_factory.py          # 决策模块工厂
│   ├── matlab_decision_interface.py # Python-MATLAB桥接
│   └── simple_decision_test.py      # 逻辑验证测试
├── 🧮 MATLAB决策模块  
│   ├── create_decision_md_simulink.m      # 创建Simulink模型
│   ├── configure_decision_md_stateflow.m  # 配置Stateflow状态机
│   └── test_decision_md_simulink.m        # MATLAB版本测试
├── 🔧 集成文件
│   ├── acc_updated.py              # 主程序（已修改）
│   └── test_integration_simple.py  # 集成验证测试
└── 📚 文档
    ├── decision.md                 # 决策逻辑规范
    └── MATLAB_DECISION_USAGE.md    # 本文档
```

## 🎯 决策逻辑说明

### 4个核心状态（S0-S3）
| 状态 | 名称 | 说明 |
|------|------|------|
| **S0** | IN_CONTROL | 在控 - ACC系统正在主动控制车辆 |
| **S1** | ADAPTIVE_HISTORY_STANDBY | 适速有史待命 - 有历史参数的待命状态 |
| **S2** | ADAPTIVE_NO_HISTORY_STANDBY | 适速无史待命 - 无历史参数的待命状态 |
| **S3** | LOW_SPEED | 低速 - 车速低于设定阈值 |

### 7种指令（I0-I6）
| 指令 | 名称 | 键盘 | 驾驶意图 | 原则 |
|------|------|------|----------|------|
| **I0** | 降速 | E | ACC降速或当速启控 | 机驾执行 |
| **I1** | 增速 | Q | ACC增速或继承启控 | 机驾执行 |
| **I2** | 降距 | T | ACC降距 | 机驾执行 |
| **I3** | 增距 | R | ACC增距 | 机驾执行 |
| **I4** | 油门 | 1 | 人工加速/系统激活 | 人驾优先 |
| **I5** | 刹车 | 3 | 人工减速 | 人驾优先 |
| **I6** | 取消 | 2 | ACC功能取消 | 无 |

### 8种决策（R1-R8）
| 决策 | 名称 | 目标车速 | 控制状态 | 反馈 |
|------|------|----------|----------|------|
| **R1** | 速度降低 | 速度减 | 控制继续 | 目标车速 |
| **R2** | 速度增加 | 速度增 | 控制继续 | 目标车速 |
| **R3** | 时距降低 | 目标距离减 | 控制继续 | 目标距离 |
| **R4** | 时距增加 | 目标距离增 | 控制继续 | 目标距离 |
| **R5** | 无继控制 | 当前车速 | **进入控制** | 目标车速/距离 |
| **R6** | 继承控制 | 上次目标 | **进入控制** | 目标车速/距离 |
| **R7** | 扭矩仲裁 | 无关 | 控制继续 | 目标车速/距离 |
| **R8** | 系统待命 | 无关 | **进入待命** | ACC待命 |

## 🔧 MATLAB版本特性

### 优势
- **🎯 可视化**: Stateflow图形化状态机，直观展示决策逻辑
- **🚀 性能**: 可能具有更好的实时性能（编译优化）
- **📊 仿真**: 支持Simulink仿真环境和代码生成
- **🔬 分析**: 便于状态转移分析和验证

### 要求
- MATLAB R2019b或更高版本
- Stateflow工具箱
- MATLAB Engine for Python

### 安装MATLAB Engine
```bash
# 在MATLAB安装目录下执行
cd "matlabroot/extern/engines/python"
python setup.py install
```

## 🐍 Python版本特性  

### 优势
- **⚡ 快速**: 无需MATLAB依赖，启动快速
- **🔧 易调试**: 纯Python代码，易于调试和修改
- **📦 轻量**: 内存占用小，部署简单
- **🔄 兼容**: 与现有代码完全兼容

## 💡 使用建议

### 开发阶段
- 使用**Python版本**进行快速开发和调试
- 利用Python的灵活性快速验证逻辑

### 验证阶段  
- 使用**MATLAB版本**进行可视化验证
- 利用Stateflow图形化界面检查状态转移

### 生产部署
- **嵌入式系统**: 优先考虑MATLAB版本（代码生成）
- **服务器部署**: 优先考虑Python版本（轻量化）
- **研发环境**: 使用AUTO模式自动选择

## 🔍 故障排除

### MATLAB相关问题

**问题**: "No module named 'matlab'"
```bash
# 解决: 安装MATLAB Engine for Python
cd "matlabroot/extern/engines/python"  
python setup.py install
```

**问题**: "无法找到MATLAB模型"
- 解决: 程序会自动创建模型，或手动运行 `create_decision_md_simulink.m`

**问题**: "Stateflow编译失败"
- 检查MATLAB版本和Stateflow工具箱
- 确认文件权限和路径正确

### 切换相关问题

**问题**: "切换到MATLAB失败"
- 检查MATLAB Engine是否正常安装
- 查看控制台错误信息
- 尝试先切换到Python版本

**问题**: "状态同步问题"
- 切换后端时状态会重新初始化
- 使用F3查看当前后端状态

## 🧪 测试验证

### 逻辑一致性测试
```bash
python simple_decision_test.py       # Python版本逻辑测试
```

### 集成测试
```bash  
python test_integration_simple.py    # 集成验证测试
```

### MATLAB测试（需要MATLAB环境）
```matlab
test_decision_md_simulink()          % MATLAB版本测试
```

## 📈 性能对比

运行性能基准测试：
```bash
python decision_factory.py
```

典型结果：
- **Python版本**: ~0.1ms per call, 10000+ calls/sec
- **MATLAB版本**: ~1-5ms per call, 200-1000 calls/sec

*注：MATLAB版本启动慢但可能在大规模仿真中表现更好*

## 🔄 开发工作流

### 添加新功能
1. 更新 `decision.md` 规范文档
2. 修改 `acc_decision.py` (Python版本)
3. 更新 `configure_decision_md_stateflow.m` (MATLAB版本)
4. 运行测试确保两版本一致

### 调试技巧
1. 使用**P键**开启调试模式
2. 使用**F3键**查看当前后端信息
3. Python版本：直接打印调试信息
4. MATLAB版本：查看Simulink仿真结果

## 📞 支持信息

- 🐛 问题报告: 查看控制台输出和错误信息
- 📋 状态查看: 使用F3键获取实时信息  
- 🔧 手动测试: 运行相应的测试脚本
- 📚 文档参考: `decision.md` 包含完整的逻辑规范

---

**最后更新**: 2024-08-29  
**版本**: v1.0 - 完整的MATLAB/Python双后端支持