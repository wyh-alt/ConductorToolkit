#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频精度测试脚本
用于验证高精度模式是否能有效减少音频时长误差
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# 添加audioedit模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'audioedit'))

from audio_processor import AudioProcessor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_audio(duration, sample_rate=44100, output_path="test_audio.wav"):
    """创建测试音频文件"""
    import numpy as np
    import soundfile as sf
    
    # 生成测试音频（正弦波）
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    # 生成440Hz的正弦波
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # 确保是立体声
    if len(audio_data.shape) == 1:
        audio_data = np.column_stack((audio_data, audio_data))
    
    # 保存音频文件
    sf.write(output_path, audio_data, sample_rate)
    logger.info(f"创建测试音频: {output_path}, 时长: {duration:.6f}秒")
    return output_path

def test_precision_modes():
    """测试不同精度模式的效果"""
    processor = AudioProcessor()
    
    # 测试参数
    test_durations = [10.0, 30.0, 60.0, 120.0]  # 不同时长
    sample_rates = [44100, 48000]  # 不同采样率
    
    results = []
    
    for duration in test_durations:
        for sample_rate in sample_rates:
            logger.info(f"\n=== 测试时长: {duration}秒, 采样率: {sample_rate}Hz ===")
            
            # 创建测试音频
            test_audio = create_test_audio(duration, sample_rate, f"test_{duration}s_{sample_rate}Hz.wav")
            
            # 测试标准模式
            standard_output = f"standard_{duration}s_{sample_rate}Hz.wav"
            start_time = time.time()
            try:
                operation = processor.trim_audio(test_audio, duration, standard_output, 
                                               sample_rate=sample_rate, high_precision=False)
                standard_time = time.time() - start_time
                
                # 验证时长
                actual_duration = processor.get_audio_duration(standard_output)
                standard_error = abs(actual_duration - duration)
                
                logger.info(f"标准模式: 误差={standard_error:.6f}秒, 耗时={standard_time:.2f}秒")
            except Exception as e:
                logger.error(f"标准模式失败: {e}")
                standard_error = float('inf')
                standard_time = 0
            
            # 测试高精度模式
            high_precision_output = f"high_precision_{duration}s_{sample_rate}Hz.wav"
            start_time = time.time()
            try:
                operation = processor.trim_audio(test_audio, duration, high_precision_output, 
                                               sample_rate=sample_rate, high_precision=True)
                high_precision_time = time.time() - start_time
                
                # 验证时长
                actual_duration = processor.get_audio_duration(high_precision_output)
                high_precision_error = abs(actual_duration - duration)
                
                logger.info(f"高精度模式: 误差={high_precision_error:.6f}秒, 耗时={high_precision_time:.2f}秒")
            except Exception as e:
                logger.error(f"高精度模式失败: {e}")
                high_precision_error = float('inf')
                high_precision_time = 0
            
            # 记录结果
            results.append({
                'duration': duration,
                'sample_rate': sample_rate,
                'standard_error': standard_error,
                'high_precision_error': high_precision_error,
                'standard_time': standard_time,
                'high_precision_time': high_precision_time,
                'improvement': standard_error - high_precision_error if standard_error != float('inf') and high_precision_error != float('inf') else 0
            })
            
            # 清理临时文件
            try:
                os.remove(test_audio)
                if os.path.exists(standard_output):
                    os.remove(standard_output)
                if os.path.exists(high_precision_output):
                    os.remove(high_precision_output)
            except:
                pass
    
    # 输出总结
    print("\n" + "="*80)
    print("精度测试结果总结")
    print("="*80)
    
    total_improvement = 0
    valid_tests = 0
    
    for result in results:
        print(f"时长: {result['duration']:6.1f}秒, 采样率: {result['sample_rate']:5d}Hz")
        print(f"  标准模式误差: {result['standard_error']:10.6f}秒, 耗时: {result['standard_time']:6.2f}秒")
        print(f"  高精度模式误差: {result['high_precision_error']:10.6f}秒, 耗时: {result['high_precision_time']:6.2f}秒")
        
        if result['improvement'] != 0:
            improvement_percent = (result['improvement'] / result['standard_error']) * 100
            print(f"  精度提升: {result['improvement']:10.6f}秒 ({improvement_percent:6.2f}%)")
            total_improvement += result['improvement']
            valid_tests += 1
        print()
    
    if valid_tests > 0:
        avg_improvement = total_improvement / valid_tests
        print(f"平均精度提升: {avg_improvement:.6f}秒")
        print(f"高精度模式平均耗时增加: {sum(r['high_precision_time'] - r['standard_time'] for r in results if r['standard_time'] > 0) / len(results):.2f}秒")
    
    print("="*80)

def test_batch_processing():
    """测试批量处理的累积误差"""
    processor = AudioProcessor()
    
    # 创建多个测试音频
    test_files = []
    base_duration = 10.0
    
    for i in range(5):
        duration = base_duration + i * 0.1  # 每个文件增加0.1秒
        test_file = create_test_audio(duration, 44100, f"batch_test_{i}.wav")
        test_files.append((test_file, duration))
    
    logger.info(f"\n=== 批量处理测试 ===")
    
    # 测试标准模式批量处理
    standard_results = []
    for i, (test_file, original_duration) in enumerate(test_files):
        output_file = f"batch_standard_{i}.wav"
        try:
            operation = processor.trim_audio(test_file, base_duration, output_file, high_precision=False)
            actual_duration = processor.get_audio_duration(output_file)
            error = abs(actual_duration - base_duration)
            standard_results.append(error)
            logger.info(f"标准模式文件{i}: 误差={error:.6f}秒")
        except Exception as e:
            logger.error(f"标准模式文件{i}失败: {e}")
    
    # 测试高精度模式批量处理
    high_precision_results = []
    for i, (test_file, original_duration) in enumerate(test_files):
        output_file = f"batch_high_precision_{i}.wav"
        try:
            operation = processor.trim_audio(test_file, base_duration, output_file, high_precision=True)
            actual_duration = processor.get_audio_duration(output_file)
            error = abs(actual_duration - base_duration)
            high_precision_results.append(error)
            logger.info(f"高精度模式文件{i}: 误差={error:.6f}秒")
        except Exception as e:
            logger.error(f"高精度模式文件{i}失败: {e}")
    
    # 计算累积误差
    if standard_results and high_precision_results:
        standard_accumulated = sum(standard_results)
        high_precision_accumulated = sum(high_precision_results)
        
        print(f"\n批量处理累积误差:")
        print(f"标准模式: {standard_accumulated:.6f}秒")
        print(f"高精度模式: {high_precision_accumulated:.6f}秒")
        print(f"误差减少: {standard_accumulated - high_precision_accumulated:.6f}秒")
    
    # 清理文件
    for test_file, _ in test_files:
        try:
            os.remove(test_file)
        except:
            pass
    
    for i in range(5):
        for prefix in ["batch_standard_", "batch_high_precision_"]:
            try:
                os.remove(f"{prefix}{i}.wav")
            except:
                pass

if __name__ == "__main__":
    print("音频精度测试开始...")
    print("测试将创建临时音频文件并验证不同模式的精度差异")
    print()
    
    try:
        # 测试单个文件的精度
        test_precision_modes()
        
        # 测试批量处理的累积误差
        test_batch_processing()
        
        print("\n测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc() 