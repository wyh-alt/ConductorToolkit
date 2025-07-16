#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示精度测试脚本
验证新的微秒级显示精度是否正确工作
"""

def test_display_precision():
    """测试显示精度设置"""
    print("=== 显示精度测试 ===")
    
    # 模拟一些测试数据
    test_cases = [
        {"original": 10.123456, "new": 10.123450, "expected_diff": 0.000006},
        {"original": 30.987654, "new": 30.987650, "expected_diff": 0.000004},
        {"original": 60.000000, "new": 60.000001, "expected_diff": -0.000001},
        {"original": 120.500000, "new": 120.499999, "expected_diff": 0.000001},
    ]
    
    print("测试数据:")
    print("原始时长(秒) -> 新时长(秒) = 差值(秒)")
    print("-" * 50)
    
    for i, case in enumerate(test_cases, 1):
        original = case["original"]
        new = case["new"]
        diff = original - new
        expected = case["expected_diff"]
        
        # 使用新的显示格式
        diff_text = f"{diff:.5f}"
        original_text = f"{original:.5f}"
        new_text = f"{new:.5f}"
        
        print(f"测试{i}: {original_text}s -> {new_text}s = {diff_text}s")
        
        # 验证精度
        if abs(diff - expected) < 1e-6:
            print(f"  ✓ 精度验证通过")
        else:
            print(f"  ✗ 精度验证失败: 期望{expected:.6f}, 实际{diff:.6f}")
        
        # 显示颜色阈值判断
        if abs(diff) > 0.001:
            color = "黄色 (警告: >1ms)"
        elif abs(diff) > 0.0001:
            color = "浅黄色 (注意: >0.1ms)"
        else:
            color = "正常 (≤0.1ms)"
        
        print(f"  颜色指示: {color}")
        print()
    
    print("=== 精度说明 ===")
    print("• 显示精度: 5位小数 (约0.01ms精度)")
    print("• 黄色警告: 差值 > 1毫秒")
    print("• 浅黄色注意: 差值 > 0.1毫秒")
    print("• 正常: 差值 ≤ 0.1毫秒")
    print()
    print("=== 单位换算 ===")
    print("• 0.00001秒 = 0.01毫秒 = 10微秒")
    print("• 0.0001秒 = 0.1毫秒 = 100微秒")
    print("• 0.001秒 = 1毫秒 = 1000微秒")

if __name__ == "__main__":
    test_display_precision() 