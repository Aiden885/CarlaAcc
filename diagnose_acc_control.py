"""
实时诊断ACC控制问题
检查为什么按E键无法启动控制
"""
import re
import sys

def analyze_log_line(line):
    """分析日志行，提取关键信息"""
    # 提取速度
    speed_match = re.search(r'自车\s+前车\s+.*?(\d+\.?\d*)km/h', line)

    # 提取状态
    state_match = re.search(r'state[=:](\d+)', line)

    # 提取control_enabled
    enabled_match = re.search(r'enabled[=:](\w+)', line)

    # 提取command
    cmd_match = re.search(r'cmd[=:](\d+)', line)

    # 提取Simulink决策输出
    decision_match = re.search(r'🔧 决策UDP: S(\d+), R(\d+), enabled=(\w+)', line)

    # 检测低速转换
    low_speed_match = re.search(r'车速过低.*转入S3低速状态', line)

    # 检测按键
    key_match = re.search(r'⌨️ 已记录键盘指令: ([^(]+)', line)

    return {
        'speed': speed_match.group(1) if speed_match else None,
        'state': state_match.group(1) if state_match else None,
        'enabled': enabled_match.group(1) if enabled_match else None,
        'command': cmd_match.group(1) if cmd_match else None,
        'decision': decision_match.groups() if decision_match else None,
        'low_speed': bool(low_speed_match),
        'key': key_match.group(1).strip() if key_match else None,
        'raw': line.strip()
    }

def main():
    print("="*70)
    print("ACC控制实时诊断工具")
    print("="*70)
    print("\n请执行以下步骤:")
    print("1. 运行 acc_updated.py")
    print("2. 加速到 30 km/h 以上")
    print("3. 按空格键开启ACC系统")
    print("4. 按E键尝试启动控制")
    print("5. 复制最后10-20行日志到这里分析")
    print("\n粘贴日志后按Ctrl+D (Linux/Mac) 或 Ctrl+Z (Windows) 结束输入:")
    print("-"*70)

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    print("\n" + "="*70)
    print("诊断结果")
    print("="*70)

    # 分析每一行
    issues = []
    last_speed = None
    last_state = None
    last_enabled = None
    e_key_pressed = False

    for i, line in enumerate(lines):
        info = analyze_log_line(line)

        # 记录按键
        if info['key'] and 'E' in info['key']:
            e_key_pressed = True
            print(f"\n[帧#{i}] ✅ 检测到E键按下")

        # 检测低速警告
        if info['low_speed']:
            print(f"[帧#{i}] ⚠️  低速警告：被强制转入S3状态")
            issues.append(f"帧#{i}: 速度过低，被强制转S3（可能导致E键无效）")

        # 分析决策输出
        if info['decision']:
            state, decision, enabled = info['decision']
            last_state = int(state)
            last_enabled = enabled

            if e_key_pressed:
                print(f"[帧#{i}] 📊 E键按下后的状态:")
                print(f"    状态: S{state}")

                if state == '0':
                    print(f"    ✅ 进入S0（在控状态）")
                elif state == '2':
                    print(f"    ❌ 仍在S2（待命状态），未进入控制")
                    issues.append(f"帧#{i}: 按E键后仍在S2，控制未启动")
                elif state == '3':
                    print(f"    ❌ 在S3（低速状态），无法启动控制")
                    issues.append(f"帧#{i}: S3低速状态无法通过E键启动")

                print(f"    control_enabled: {enabled}")
                if enabled.lower() in ['true', '1']:
                    print(f"    ✅ 控制已启动")
                else:
                    print(f"    ❌ 控制未启动")
                    issues.append(f"帧#{i}: control_enabled=False")

                e_key_pressed = False  # 重置标志

        # 记录速度
        if info['speed']:
            last_speed = float(info['speed'])

    # 总结
    print("\n" + "="*70)
    print("问题总结")
    print("="*70)

    if not issues:
        print("✅ 未发现明显问题")
        if last_state == 0 and last_enabled in ['True', 'true', '1']:
            print("   控制已成功启动")
    else:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  • {issue}")

        print("\n可能的原因:")
        if any('S3' in issue for issue in issues):
            print("  1. 速度过低（< 20 km/h）导致进入S3低速状态")
            print("     → 解决：加速到30 km/h以上再按E键")
        if any('S2' in issue for issue in issues):
            print("  2. 从S2状态按E键没有转到S0")
            print("     → 可能是Simulink决策表配置问题")
            print("     → 或者command_type传递错误")

    print("\n建议操作:")
    print("  1. 确保速度 > 30 km/h（远高于V_min_kmh=20）")
    print("  2. 按空格开启ACC系统")
    print("  3. 确认状态是S2（不是S3）")
    print("  4. 然后按E键")
    print("  5. 观察状态是否变为S0且enabled=True")

if __name__ == '__main__':
    main()
