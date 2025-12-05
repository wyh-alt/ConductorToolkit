import os
import glob
import librosa
from pydub import AudioSegment
import logging
import re
from datetime import datetime
import numpy as np
import soundfile as sf
import wave
import struct
import math
import concurrent.futures
import time
from functools import lru_cache
from scipy import signal

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.supported_formats = ['.mp3', '.wav', '.flac', '.ogg']
        self.default_sample_rate = 44100
        self.default_channels = 2
        self.default_bit_depth = 16
        self.convert_to_wav = True
        self.max_workers = min(2, os.cpu_count() or 2)  # 限制最大线程数为2
        self.optimize_memory = True  # 确保开启内存优化
        self.batch_size = 5  # 减小批处理大小，避免一次处理过多文件导致内存溢出
    
    def get_audio_files(self, folder_path):
        """获取文件夹中所有支持的音频文件，使用优化的搜索方法"""
        files = []
        logger.info(f"正在搜索文件夹: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"文件夹不存在: {folder_path}")
            return files
            
        # 使用set来存储已找到的文件，避免重复
        found_files = set()
        
        # 一次性遍历目录
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                # 检查文件扩展名（同时处理大小写）
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_formats:
                    full_path = os.path.join(root, filename)
                    found_files.add(full_path)
        
        files = list(found_files)
        logger.info(f"在 {folder_path} 中共找到 {len(files)} 个音频文件")
        if files:
            logger.info(f"示例文件: {files[0]}")
        return files
            
        logger.info(f"在 {folder_path} 中共找到 {len(files)} 个音频文件")
        if len(files) > 0:
            logger.info(f"示例文件: {files[0]}")
        return files
    
    def extract_segment_key(self, filename):
        """从文件名中提取片段模式的匹配关键字
        
        片段模式匹配逻辑：
        - 851793副歌片段01-05.213_vocals_Vocals_No Reverb 与 851793副歌片段01-05.213-干声 匹配
        - 851793自定义片段00-49.081_vocals_Vocals_No Reverb 与 851793自定义片段00-49.081-干声 匹配
        
        通过截取 '-干声' 或 '_vocals_Vocals_No Reverb' 之前的内容作为匹配关键字
        """
        basename = os.path.basename(filename)
        # 去除扩展名
        basename_no_ext = os.path.splitext(basename)[0]
        
        # 定义需要截断的后缀模式
        suffix_patterns = [
            '-干声',
            '_vocals_Vocals_No Reverb',
            '_vocals',
            '-导唱',
            '-原唱',
            '-伴奏',
            '_伴奏',
            '（伴奏）',
            '(伴奏)',
        ]
        
        # 尝试匹配并截断后缀
        key = basename_no_ext
        for suffix in suffix_patterns:
            if suffix in key:
                key = key.split(suffix)[0]
                break
        
        logger.debug(f"片段模式提取: {filename} => {key}")
        return key
    
    def extract_file_id(self, filename):
        """从文件名中提取ID部分（整首模式）
        例如: 12345-原唱.mp3 => 12345
              YPD12345-原唱.wav => YPD12345
              YPD-12345_伴奏.mp3 => YPD12345
              SL_YPD12345（伴奏）.flac => YPD12345
              888446_step3_(No Reverb).wav => 888446
              2022年度YPD12345歌曲合集.wav => YPD12345
        """
        basename = os.path.basename(filename)
        # 去除扩展名，避免误识别扩展名
        basename_no_ext = os.path.splitext(basename)[0]
        
        # 1. 特定公司前缀的ID格式优先处理 (不管位置在哪里)
        for prefix in ["YPD", "SL", "ZX", "KG", "TX"]:
            # 匹配 YPD12345 或 YPD-12345 格式
            match = re.search(f'({prefix}[-_]?\\d+)', basename_no_ext)
            if match:
                raw_id = match.group(1)
                return re.sub(r'[-_]', '', raw_id)
                
        # 2. 如果文件名以数字开头，直接提取起始数字序列
        if re.match(r'^\d{4,}', basename_no_ext):
            match = re.search(r'^(\d+)', basename_no_ext)
            if match:
                return match.group(1)
        
        # 3. 查找文件名中的字母+数字组合
        match = re.search(r'([A-Za-z]{2,}\d{3,})', basename_no_ext)
        if match:
            raw_id = match.group(1)
            return re.sub(r'[-_]', '', raw_id)
            
        # 4. 查找文件名中的年份之外的多位数字
        # 先寻找6位以上的长数字，这些几乎肯定是ID
        match = re.search(r'(\d{6,})', basename_no_ext)
        if match:
            return match.group(1)
            
        # 5. 如果文件名中存在4-5位数字，可能是ID
        match_numbers = re.findall(r'\d{4,5}', basename_no_ext)
        if match_numbers:
            # 过滤可能是年份的数字 (2000-2099)
            non_year_numbers = [num for num in match_numbers if not (2000 <= int(num) <= 2099)]
            if non_year_numbers:
                return non_year_numbers[0]  # 返回第一个非年份的多位数字
            # 如果只有年份，也返回第一个匹配的数字
            return match_numbers[0]
        
        # 6. 如果只有2-3位数字，可能是ID也可能是其他信息
        match = re.search(r'(\d{2,3})', basename_no_ext)
        if match:
            return match.group(1)
            
        # 7. 如果实在找不到有意义的ID，最后尝试单个数字
        match = re.search(r'(\d+)', basename_no_ext)
        if match and len(match.group(1)) > 1:  # 至少两位数字
            return match.group(1)
        
        # 如果没有找到数字ID，则返回文件名（不含扩展名）
        logger.debug(f"未能从 {filename} 中提取到ID，使用基本名称: {basename_no_ext}")
        return basename_no_ext
    
    @lru_cache(maxsize=32)
    def get_audio_duration(self, audio_path):
        """获取音频文件的长度（秒），使用缓存避免重复计算，提高精度"""
        try:
            logger.info(f"正在读取音频文件长度: {audio_path}")
            
            # 针对WAV文件，先尝试使用wave模块
            if audio_path.lower().endswith('.wav'):
                try:
                    with wave.open(audio_path, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        # 使用更高精度的计算，避免浮点误差
                        duration = frames / float(rate)
                        logger.info(f"文件 {os.path.basename(audio_path)} 长度为 {duration:.6f} 秒 (使用wave库)")
                        return duration
                except Exception as wave_error:
                    # 如果wave模块失败（如遇到格式65534），尝试soundfile
                    logger.warning(f"wave库无法处理文件，尝试soundfile: {str(wave_error)}")
                    try:
                        import soundfile as sf
                        info = sf.info(audio_path)
                        duration = info.duration
                        logger.info(f"文件 {os.path.basename(audio_path)} 长度为 {duration:.6f} 秒 (使用soundfile库)")
                        return duration
                    except Exception as sf_error:
                        # 如果soundfile也失败，使用librosa作为最后的后备方案
                        logger.warning(f"soundfile库也无法处理文件，尝试librosa: {str(sf_error)}")
                        duration = librosa.get_duration(path=audio_path)
                        logger.info(f"文件 {os.path.basename(audio_path)} 长度为 {duration:.6f} 秒 (使用librosa库)")
                        return duration
            
            # 对于其他格式，优先使用soundfile
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
                logger.info(f"文件 {os.path.basename(audio_path)} 长度为 {duration:.6f} 秒")
                return duration
            except Exception as sf_error:
                # 如果soundfile失败，使用librosa作为后备方案
                logger.warning(f"soundfile库无法处理文件，尝试librosa: {str(sf_error)}")
                duration = librosa.get_duration(path=audio_path)
                logger.info(f"文件 {os.path.basename(audio_path)} 长度为 {duration:.6f} 秒 (使用librosa库)")
                return duration
        except Exception as e:
            logger.error(f"无法读取音频文件 {audio_path}: {str(e)}")
            raise

    def _calculate_exact_frames(self, duration, sample_rate):
        """精确计算帧数，避免舍入误差"""
        # 使用更高精度的计算，然后进行精确的舍入
        exact_frames = duration * sample_rate
        # 使用四舍五入到最接近的整数，但保持精度
        return int(exact_frames + 0.5) if exact_frames >= 0 else int(exact_frames - 0.5)

    def _verify_output_duration(self, output_path, target_duration, sample_rate, tolerance=1e-6):
        """验证输出文件的时长是否与目标时长一致，容差为微秒级"""
        try:
            actual_duration = self.get_audio_duration(output_path)
            target_frames = self._calculate_exact_frames(target_duration, sample_rate)
            actual_frames = self._calculate_exact_frames(actual_duration, sample_rate)
            
            duration_diff = abs(actual_duration - target_duration)
            frames_diff = abs(actual_frames - target_frames)
            
            logger.info(f"时长验证: 目标={target_duration:.6f}s, 实际={actual_duration:.6f}s, 差值={duration_diff:.6f}s")
            logger.info(f"帧数验证: 目标={target_frames}, 实际={actual_frames}, 差值={frames_diff}")
            
            if duration_diff > tolerance:
                logger.warning(f"输出文件时长与目标时长差异过大: {duration_diff:.6f}s > {tolerance}s")
                return False
            return True
        except Exception as e:
            logger.error(f"验证输出文件时长时出错: {str(e)}")
            return False
    
    def trim_audio(self, audio_path, target_duration, output_path, sample_rate=None, channels=None, bit_depth=None, high_precision=True):
        """以微秒级精度裁剪或延长音频文件至目标长度，确保两个音频的时长完全一致
        
        Args:
            audio_path: 输入音频文件路径
            target_duration: 目标时长（秒）
            output_path: 输出文件路径
            sample_rate: 采样率（Hz）
            channels: 声道数
            bit_depth: 位深
            high_precision: 是否使用高精度模式（减少误差但可能更慢），默认True
        """
        try:
            # 验证输入文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"输入文件不存在: {audio_path}")
                raise FileNotFoundError(f"找不到输入文件: {audio_path}")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                logger.info(f"创建输出目录: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            # 如果需要转换为WAV格式，提前修改输出路径扩展名
            if self.convert_to_wav:
                output_path = os.path.splitext(output_path)[0] + '.wav'
            
            # 应用参数设置（如果没有指定，使用默认值）
            sample_rate = sample_rate or self.default_sample_rate
            channels = channels or self.default_channels
            bit_depth = bit_depth or self.default_bit_depth
            
            logger.info(f"开始精确处理文件: {audio_path} -> {output_path}")
            logger.info(f"目标参数: 采样率={sample_rate}Hz, 声道数={channels}, 位深={bit_depth}bit, 高精度={high_precision}")
            
            # 根据精度要求选择处理方法
            if high_precision:
                operation = self._process_audio_high_precision(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
            else:
                # 修改为更高效的处理方式，避免一次性加载大文件导致内存溢出
                if audio_path.lower().endswith('.wav'):
                    # 对WAV文件使用直接块读取方式，避免加载全部数据到内存
                    operation = self._process_wav_file(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
                else:
                    # 对于其他格式，使用分块加载和处理
                    operation = self._process_audio_file_chunked(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
            
            return operation
        except Exception as e:
            logger.error(f"处理文件 {audio_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _process_wav_file(self, audio_path, target_duration, output_path, sample_rate, channels, bit_depth):
        """直接处理WAV文件，避免将整个文件加载到内存中，使用精确的帧数计算"""
        with wave.open(audio_path, 'rb') as wf:
            # 获取原始WAV参数
            original_channels = wf.getnchannels()
            original_sampwidth = wf.getsampwidth()
            original_framerate = wf.getframerate()
            original_frames = wf.getnframes()
            
            # 使用精确的帧数计算
            target_frames = self._calculate_exact_frames(target_duration, sample_rate)
            
            # 设置重采样标志
            need_resampling = original_framerate != sample_rate
            
            # 如果需要重采样但帧数太多，使用分块处理
            if need_resampling and original_frames > 1000000:
                logger.info(f"大文件需要重采样，使用分块处理")
                return self._process_audio_file_chunked(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
            
            # 确定操作类型
            operation = "unchanged"
            if original_frames > target_frames:
                operation = "trimmed"
                logger.info(f"将对音频进行裁剪: {original_frames} -> {target_frames} 帧")
            elif original_frames < target_frames:
                operation = "extended"
                logger.info(f"将对音频进行延长: {original_frames} -> {target_frames} 帧")
            else:
                logger.info(f"音频长度已经匹配，无需调整: {original_frames} 帧")
            
            # 创建输出WAV文件
            with wave.open(output_path, 'wb') as out_wf:
                out_wf.setnchannels(channels)
                out_wf.setsampwidth(2 if bit_depth == 16 else 3)  # 16位=2字节，24位=3字节
                out_wf.setframerate(sample_rate)
                
                # 如果目标帧数小于原始帧数，裁剪
                if not need_resampling and target_frames <= original_frames:
                    # 直接复制需要的部分，确保精确的帧数
                    frames_data = wf.readframes(target_frames)
                    out_wf.writeframes(frames_data)
                    
                    # 验证输出文件时长
                    if not self._verify_output_duration(output_path, target_duration, sample_rate):
                        logger.warning(f"WAV文件裁剪后时长验证失败，尝试重新处理")
                        # 如果验证失败，使用分块处理重新生成
                        return self._process_audio_file_chunked(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
                else:
                    # 如果需要扩展或重采样，使用分块处理
                    return self._process_audio_file_chunked(audio_path, target_duration, output_path, sample_rate, channels, bit_depth)
            
            return operation
    
    def _process_audio_file_chunked(self, audio_path, target_duration, output_path, sample_rate, channels, bit_depth):
        """使用分块处理方式处理音频文件，避免内存溢出，使用精确的帧数控制"""
        import soundfile as sf
        import numpy as np
        from scipy import signal
        
        try:
            # 使用soundfile获取文件信息，而不加载数据
            # 如果soundfile无法处理（如格式65534），使用librosa获取信息
            try:
                info = sf.info(audio_path)
                original_frames = info.frames
                original_samplerate = info.samplerate
                original_channels = info.channels
            except Exception as sf_error:
                if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                    logger.warning(f"soundfile无法处理文件格式，使用librosa加载: {str(sf_error)}")
                    # 使用librosa加载文件获取信息
                    y, sr = librosa.load(audio_path, sr=None, mono=False)
                    if y.ndim == 1:
                        original_channels = 1
                        original_frames = len(y)
                    else:
                        original_channels = y.shape[0]
                        original_frames = y.shape[1]
                    original_samplerate = sr
                else:
                    raise
            
            # 使用精确的帧数计算
            target_frames = self._calculate_exact_frames(target_duration, sample_rate)
            
            # 确定操作类型
            original_duration = original_frames / original_samplerate
            operation = "unchanged"
            if original_duration > target_duration:
                operation = "trimmed"
                logger.info(f"将对音频进行裁剪: {original_duration:.6f}秒 -> {target_duration:.6f}秒")
            elif original_duration < target_duration:
                operation = "extended"
                logger.info(f"将对音频进行延长: {original_duration:.6f}秒 -> {target_duration:.6f}秒")
            else:
                logger.info(f"音频长度已经匹配，无需调整: {original_duration:.6f}秒")
            
            # 如果原始音频比目标短，需要扩展
            if original_frames / original_samplerate < target_duration:
                # 使用分块处理，而非一次性加载
                # 如果soundfile无法打开，使用librosa
                try:
                    in_file = sf.SoundFile(audio_path)
                    use_soundfile = True
                except Exception as sf_error:
                    if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                        logger.warning(f"soundfile无法打开文件，使用librosa处理: {str(sf_error)}")
                        use_soundfile = False
                        # 使用librosa加载整个文件
                        audio_data, _ = librosa.load(audio_path, sr=original_samplerate, mono=False)
                        if audio_data.ndim == 1:
                            audio_data = audio_data.reshape(1, -1)  # 转换为2D (channels, samples)
                        else:
                            audio_data = audio_data.T  # librosa返回(channels, samples)，需要转置
                    else:
                        raise
                
                if use_soundfile:
                    with in_file:
                        # 创建输出文件
                        output_format = 'WAV'
                        output_subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
                        
                        # 计算需要多少个原始音频帧
                        block_size = 50000  # 减小块大小，降低内存使用
                        
                        # 创建输出文件
                        with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                         format=output_format, subtype=output_subtype) as out_file:
                            
                            # 先复制原始音频
                            while True:
                                # 读取一块数据
                                block_data = in_file.read(block_size, always_2d=True)
                                
                                if len(block_data) == 0:
                                    break
                                    
                                # 重采样（如果需要）- 使用更精确的方法
                                if original_samplerate != sample_rate:
                                    resampled_size = self._calculate_exact_frames(len(block_data) / original_samplerate, sample_rate)
                                    resampled_data = np.zeros((resampled_size, block_data.shape[1]))
                                    for channel in range(block_data.shape[1]):
                                        # 使用更精确的重采样参数
                                        resampled_data[:, channel] = signal.resample(
                                            block_data[:, channel], 
                                            resampled_size,
                                            window='hann'  # 使用汉宁窗减少重采样误差
                                        )
                                    block_data = resampled_data
                                
                                # 通道数转换
                                if block_data.shape[1] != channels:
                                    if channels == 1:
                                        block_data = np.mean(block_data, axis=1, keepdims=True)
                                    else:
                                        if block_data.shape[1] == 1:
                                            block_data = np.column_stack((block_data[:, 0], block_data[:, 0]))
                                            
                                # 写入处理后的数据
                                out_file.write(block_data)
                            
                            # 计算需要添加的静音帧数 - 使用精确计算
                            frames_written = out_file.tell()
                            silent_frames_needed = target_frames - frames_written
                            
                            # 添加静音来扩展
                            if silent_frames_needed > 0:
                                # 分块添加静音，而不是一次性创建大数组
                                silence_block_size = min(50000, silent_frames_needed)
                                remaining_silent_frames = silent_frames_needed
                                
                                while remaining_silent_frames > 0:
                                    current_block = min(silence_block_size, remaining_silent_frames)
                                    silence = np.zeros((current_block, channels))
                                    out_file.write(silence)
                                    remaining_silent_frames -= current_block
                else:
                    # 使用librosa加载的数据进行处理
                    output_format = 'WAV'
                    output_subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
                    
                    # 转换audio_data形状为 (samples, channels)
                    if audio_data.ndim == 2:
                        audio_data = audio_data.T  # 转置为 (samples, channels)
                    
                    # 重采样（如果需要）
                    if original_samplerate != sample_rate:
                        logger.info(f"进行重采样: {original_samplerate}Hz -> {sample_rate}Hz")
                        # 对每个通道进行重采样
                        resampled_data_list = []
                        for ch in range(audio_data.shape[1]):
                            resampled_ch = signal.resample(
                                audio_data[:, ch],
                                int(len(audio_data) * sample_rate / original_samplerate),
                                window='hann'
                            )
                            resampled_data_list.append(resampled_ch)
                        audio_data = np.column_stack(resampled_data_list)
                    
                    # 通道数转换
                    if audio_data.shape[1] != channels:
                        if channels == 1:
                            audio_data = np.mean(audio_data, axis=1, keepdims=True)
                        else:
                            if audio_data.shape[1] == 1:
                                audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0]))
                    
                    # 确保不超过目标帧数，如果短则扩展
                    current_frames = len(audio_data)
                    if current_frames < target_frames:
                        # 添加静音
                        silence_frames = target_frames - current_frames
                        silence = np.zeros((silence_frames, channels))
                        audio_data = np.vstack([audio_data, silence])
                    elif current_frames > target_frames:
                        # 裁剪
                        audio_data = audio_data[:target_frames]
                    
                    # 保存为WAV文件
                    with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                     format=output_format, subtype=output_subtype) as out_file:
                        out_file.write(audio_data)
            
            else:
                # 原始音频较长，需要裁剪 - 使用分块读取，避免内存溢出
                block_size = min(50000, target_frames)  # 减小块大小
                
                # 尝试打开输入文件
                try:
                    in_file = sf.SoundFile(audio_path)
                    use_soundfile_for_trim = True
                except Exception as sf_error:
                    if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                        logger.warning(f"soundfile无法打开文件进行裁剪，使用librosa: {str(sf_error)}")
                        use_soundfile_for_trim = False
                        # 使用librosa加载数据
                        audio_data, _ = librosa.load(audio_path, sr=original_samplerate, mono=False)
                        if audio_data.ndim == 1:
                            audio_data = audio_data.reshape(1, -1)
                        else:
                            audio_data = audio_data.T  # 转为 (samples, channels)
                    else:
                        raise
                
                if use_soundfile_for_trim:
                    with in_file:
                        # 创建新的输出格式
                        output_format = 'WAV'
                        output_subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
                        
                        # 创建输出文件
                        with sf.SoundFile(output_path, 'w', sample_rate, channels, 
                                          format=output_format, subtype=output_subtype) as out_file:
                            
                            # 确定是否需要重采样
                            need_resampling = in_file.samplerate != sample_rate
                            
                            # 计算需要读取的帧数 - 使用精确计算
                            input_frames_to_read = target_frames
                            if need_resampling:
                                input_frames_to_read = self._calculate_exact_frames(target_duration, in_file.samplerate)
                            
                            # 分块处理
                            remaining_frames = input_frames_to_read
                            total_written = 0
                            
                            while remaining_frames > 0 and total_written < target_frames:
                                # 读取块数据，每次读取较小的块
                                current_block_size = min(block_size, remaining_frames)
                                block_data = in_file.read(current_block_size, always_2d=True)
                                
                                # 如果块为空（到达文件末尾），跳出循环
                                if len(block_data) == 0:
                                    break
                                
                                # 重采样（如果需要）- 使用更精确的方法
                                if need_resampling:
                                    # 避免一次性创建过大的数组
                                    resampled_size = self._calculate_exact_frames(len(block_data) / in_file.samplerate, sample_rate)
                                    # 增加错误处理
                                    try:
                                        resampled_data = np.zeros((resampled_size, block_data.shape[1]))
                                        for channel in range(block_data.shape[1]):
                                            # 使用更精确的重采样参数
                                            resampled_data[:, channel] = signal.resample(
                                                block_data[:, channel], 
                                                resampled_size,
                                                window='hann'  # 使用汉宁窗减少重采样误差
                                            )
                                        block_data = resampled_data
                                    except MemoryError:
                                        logger.error("内存不足，无法进行重采样。尝试减小块大小或增加系统内存。")
                                        raise
                                
                                # 通道数转换
                                if block_data.shape[1] != channels:
                                    if channels == 1:
                                        # 转为单声道
                                        block_data = np.mean(block_data, axis=1, keepdims=True)
                                    else:
                                        # 转为双声道（如果原始是单声道）
                                        if block_data.shape[1] == 1:
                                            block_data = np.column_stack((block_data[:, 0], block_data[:, 0]))
                                
                                # 确保不超过目标帧数
                                if total_written + len(block_data) > target_frames:
                                    excess_frames = total_written + len(block_data) - target_frames
                                    block_data = block_data[:-excess_frames]
                                
                                # 写入数据块
                                out_file.write(block_data)
                                total_written += len(block_data)
                                
                                # 更新剩余帧数
                                remaining_frames -= current_block_size
                else:
                    # 使用librosa加载的数据进行裁剪处理
                    output_format = 'WAV'
                    output_subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
                    
                    # 裁剪到目标帧数（原始采样率）
                    input_frames_needed = self._calculate_exact_frames(target_duration, original_samplerate)
                    audio_data = audio_data[:input_frames_needed]  # 裁剪
                    
                    # 重采样（如果需要）
                    if original_samplerate != sample_rate:
                        logger.info(f"进行重采样: {original_samplerate}Hz -> {sample_rate}Hz")
                        resampled_data_list = []
                        for ch in range(audio_data.shape[1]):
                            resampled_ch = signal.resample(
                                audio_data[:, ch],
                                int(len(audio_data) * sample_rate / original_samplerate),
                                window='hann'
                            )
                            resampled_data_list.append(resampled_ch)
                        audio_data = np.column_stack(resampled_data_list)
                    
                    # 确保精确匹配目标帧数
                    if len(audio_data) > target_frames:
                        audio_data = audio_data[:target_frames]
                    
                    # 通道数转换
                    if audio_data.shape[1] != channels:
                        if channels == 1:
                            audio_data = np.mean(audio_data, axis=1, keepdims=True)
                        else:
                            if audio_data.shape[1] == 1:
                                audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0]))
                    
                    # 保存为WAV文件
                    with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                     format=output_format, subtype=output_subtype) as out_file:
                        out_file.write(audio_data)
            
            # 验证输出文件时长
            if not self._verify_output_duration(output_path, target_duration, sample_rate):
                logger.warning(f"分块处理后时长验证失败，可能存在精度误差")
            
            return operation
        
        except MemoryError:
            logger.error("处理音频时内存不足。请尝试关闭其他应用程序，或增加系统虚拟内存。")
            raise
        except Exception as e:
            logger.error(f"处理文件 {audio_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _process_audio_high_precision(self, audio_path, target_duration, output_path, sample_rate, channels, bit_depth):
        """高精度音频处理方法，使用最精确的算法减少误差"""
        import soundfile as sf
        import numpy as np
        from scipy import signal
        
        try:
            logger.info("使用高精度模式处理音频")
            
            # 获取原始音频信息
            # 如果soundfile无法处理（如格式65534），使用librosa获取信息
            try:
                info = sf.info(audio_path)
                original_frames = info.frames
                original_samplerate = info.samplerate
                original_channels = info.channels
            except Exception as sf_error:
                if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                    logger.warning(f"soundfile无法处理文件格式，使用librosa加载: {str(sf_error)}")
                    # 使用librosa加载文件获取信息
                    y, sr = librosa.load(audio_path, sr=None, mono=False)
                    if y.ndim == 1:
                        original_channels = 1
                        original_frames = len(y)
                    else:
                        original_channels = y.shape[0]
                        original_frames = y.shape[1]
                    original_samplerate = sr
                else:
                    raise
            
            # 精确计算目标帧数
            target_frames = self._calculate_exact_frames(target_duration, sample_rate)
            
            # 计算原始时长
            original_duration = original_frames / original_samplerate
            
            logger.info(f"原始时长: {original_duration:.6f}秒, 目标时长: {target_duration:.6f}秒")
            logger.info(f"原始帧数: {original_frames}, 目标帧数: {target_frames}")
            
            # 确定操作类型
            if abs(original_duration - target_duration) < 1e-6:
                operation = "unchanged"
                logger.info("音频长度已经匹配，无需调整")
            elif original_duration > target_duration:
                operation = "trimmed"
                logger.info(f"将对音频进行裁剪: {original_duration:.6f}秒 -> {target_duration:.6f}秒")
            else:
                operation = "extended"
                logger.info(f"将对音频进行延长: {original_duration:.6f}秒 -> {target_duration:.6f}秒")
            
            # 创建输出文件
            output_format = 'WAV'
            output_subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
            
            # 尝试使用soundfile打开文件
            try:
                in_file_test = sf.SoundFile(audio_path)
                use_soundfile_for_hp = True
                in_file_test.close()
            except Exception as sf_error:
                if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                    logger.warning(f"soundfile无法打开文件，使用librosa处理: {str(sf_error)}")
                    use_soundfile_for_hp = False
                else:
                    raise
            
            # 如果不需要重采样且是裁剪操作，使用直接复制
            if use_soundfile_for_hp and original_samplerate == sample_rate and original_channels == channels and operation == "trimmed":
                logger.info("使用直接复制模式进行高精度裁剪")
                try:
                    with sf.SoundFile(audio_path) as in_file:
                        with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                         format=output_format, subtype=output_subtype) as out_file:
                            # 直接读取目标帧数
                            data = in_file.read(target_frames, always_2d=True)
                            out_file.write(data)
                except Exception as sf_error:
                    if "unknown format" in str(sf_error).lower() or "65534" in str(sf_error):
                        logger.warning(f"soundfile无法读取文件，改用librosa: {str(sf_error)}")
                        use_soundfile_for_hp = False
                    else:
                        raise
            
            if not use_soundfile_for_hp or (original_samplerate != sample_rate or original_channels != channels or operation != "trimmed"):
                # 使用高精度重采样和转换
                logger.info("使用高精度重采样模式")
                
                if use_soundfile_for_hp:
                    with sf.SoundFile(audio_path) as in_file:
                        with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                         format=output_format, subtype=output_subtype) as out_file:
                            
                            # 计算需要读取的原始帧数
                            if original_samplerate != sample_rate:
                                input_frames_needed = self._calculate_exact_frames(target_duration, original_samplerate)
                            else:
                                input_frames_needed = target_frames
                            
                            # 读取原始数据
                            data = in_file.read(input_frames_needed, always_2d=True)
                            
                            # 重采样（如果需要）
                            if original_samplerate != sample_rate:
                                logger.info(f"进行高精度重采样: {original_samplerate}Hz -> {sample_rate}Hz")
                                resampled_frames = self._calculate_exact_frames(len(data) / original_samplerate, sample_rate)
                                resampled_data = np.zeros((resampled_frames, data.shape[1]))
                                
                                for channel in range(data.shape[1]):
                                    # 使用最高质量的重采样参数
                                    resampled_data[:, channel] = signal.resample(
                                        data[:, channel], 
                                        resampled_frames,
                                        window='hann',
                                        domain='time'
                                    )
                                data = resampled_data
                            
                            # 通道数转换
                            if data.shape[1] != channels:
                                if channels == 1:
                                    data = np.mean(data, axis=1, keepdims=True)
                                else:
                                    if data.shape[1] == 1:
                                        data = np.column_stack((data[:, 0], data[:, 0]))
                            
                            # 确保帧数精确匹配
                            if len(data) > target_frames:
                                data = data[:target_frames]
                            elif len(data) < target_frames:
                                # 添加静音
                                silence = np.zeros((target_frames - len(data), channels))
                                data = np.vstack([data, silence])
                            
                            # 写入数据
                            out_file.write(data)
                else:
                    # 使用librosa加载和处理
                    logger.info("使用librosa进行高精度处理")
                    # 计算需要读取的原始帧数
                    if original_samplerate != sample_rate:
                        input_frames_needed = self._calculate_exact_frames(target_duration, original_samplerate)
                    else:
                        input_frames_needed = target_frames
                    
                    # 加载音频数据
                    audio_data, _ = librosa.load(audio_path, sr=original_samplerate, mono=False)
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(1, -1)
                    else:
                        audio_data = audio_data.T  # 转为 (samples, channels)
                    
                    # 裁剪或扩展
                    if len(audio_data) > input_frames_needed:
                        audio_data = audio_data[:input_frames_needed]
                    elif len(audio_data) < input_frames_needed and operation == "extended":
                        silence = np.zeros((input_frames_needed - len(audio_data), audio_data.shape[1]))
                        audio_data = np.vstack([audio_data, silence])
                    
                    # 重采样（如果需要）
                    if original_samplerate != sample_rate:
                        logger.info(f"进行高精度重采样: {original_samplerate}Hz -> {sample_rate}Hz")
                        # 使用精确的帧数计算，避免舍入误差
                        duration = len(audio_data) / original_samplerate
                        resampled_frames = self._calculate_exact_frames(duration, sample_rate)
                        resampled_data_list = []
                        for ch in range(audio_data.shape[1]):
                            resampled_ch = signal.resample(
                                audio_data[:, ch],
                                resampled_frames,
                                window='hann',
                                domain='time'
                            )
                            resampled_data_list.append(resampled_ch)
                        audio_data = np.column_stack(resampled_data_list)
                    
                    # 通道数转换
                    if audio_data.shape[1] != channels:
                        if channels == 1:
                            audio_data = np.mean(audio_data, axis=1, keepdims=True)
                        else:
                            if audio_data.shape[1] == 1:
                                audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0]))
                    
                    # 确保帧数精确匹配
                    if len(audio_data) > target_frames:
                        audio_data = audio_data[:target_frames]
                    elif len(audio_data) < target_frames:
                        silence = np.zeros((target_frames - len(audio_data), channels))
                        audio_data = np.vstack([audio_data, silence])
                    
                    # 保存为WAV文件
                    with sf.SoundFile(output_path, 'w', sample_rate, channels,
                                     format=output_format, subtype=output_subtype) as out_file:
                        out_file.write(audio_data)
            
            # 验证输出文件
            if self._verify_output_duration(output_path, target_duration, sample_rate, tolerance=1e-7):
                logger.info("高精度处理完成，时长验证通过")
            else:
                logger.warning("高精度处理后时长验证失败，可能存在微小误差")
            
            return operation
            
        except Exception as e:
            logger.error(f"高精度处理文件 {audio_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def process_folders(self, folder_a, folder_b, output_folder, naming_format, callback=None):
        """处理两个文件夹中的音频文件，使用高效的匹配算法和并行处理"""
        start_time = time.time()
        logger.info(f"开始处理文件夹: A={folder_a}, B={folder_b}, 输出={output_folder}")
        
        # 获取文件夹A和B中的音频文件
        files_a = self.get_audio_files(folder_a)
        files_b = self.get_audio_files(folder_b)
        
        if not files_a:
            logger.warning(f"在文件夹A中没有找到音频文件: {folder_a}")
            return []
            
        if not files_b:
            logger.warning(f"在文件夹B中没有找到音频文件: {folder_b}")
            return []
        
        # 预处理所有文件，创建高效的ID到文件路径映射
        files_a_dict = {}
        files_b_dict = {}
        
        # 预处理文件A (可以并行进行)
        def extract_ids_batch(files, result_dict):
            for file_path in files:
                file_id = self.extract_file_id(file_path)
                if file_id not in result_dict:
                    result_dict[file_id] = file_path
        
        # 使用并行处理提取ID
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 分批处理文件A
            batch_size = min(len(files_a), 100)  # 每批最多100个文件
            batches_a = [files_a[i:i+batch_size] for i in range(0, len(files_a), batch_size)]
            for batch in batches_a:
                executor.submit(extract_ids_batch, batch, files_a_dict)
            
            # 分批处理文件B
            batch_size = min(len(files_b), 100)
            batches_b = [files_b[i:i+batch_size] for i in range(0, len(files_b), batch_size)]
            for batch in batches_b:
                executor.submit(extract_ids_batch, batch, files_b_dict)
        
        # 找出两个文件夹中共有的文件ID
        common_ids = set(files_a_dict.keys()) & set(files_b_dict.keys())
        logger.info(f"文件夹A中有 {len(files_a_dict)} 个唯一文件ID")
        logger.info(f"文件夹B中有 {len(files_b_dict)} 个唯一文件ID")
        logger.info(f"两个文件夹共有 {len(common_ids)} 个相同的文件ID")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"确保输出文件夹存在: {output_folder}")
        
        # 创建处理队列
        process_queue = []
        for file_id in common_ids:
            file_a = files_a_dict[file_id]
            file_b = files_b_dict[file_id]
            
            # 生成输出文件名
            base_name_b = os.path.basename(file_b)
            original_name = os.path.splitext(base_name_b)[0]
            extension = os.path.splitext(file_b)[1][1:]
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            output_filename = naming_format.format(
                original_name=original_name,
                extension=extension,
                file_id=file_id,
                timestamp=timestamp
            )
            output_path = os.path.join(output_folder, output_filename)
            
            # 添加到处理队列
            process_queue.append({
                'file_id': file_id,
                'file_a': file_a,
                'file_b': file_b,
                'output_path': output_path
            })
        
        # 并行处理音频文件
        results = []
        total = len(process_queue)
        processed = 0
        
        # 处理单个文件的函数
        def process_single_file(item):
            file_id = item['file_id']
            file_a = item['file_a']
            file_b = item['file_b']
            output_path = item['output_path']
            base_name_a = os.path.basename(file_a)
            base_name_b = os.path.basename(file_b)
            
            try:
                logger.info(f"处理匹配文件对: {base_name_a} <-> {base_name_b}, ID: {file_id}")
                
                # 获取A文件的长度
                duration_a = self.get_audio_duration(file_a)
                original_duration = self.get_audio_duration(file_b)
                
                # 裁剪B文件
                operation = self.trim_audio(file_b, duration_a, output_path)
                
                return {
                    'file': base_name_a,
                    'matched_with': base_name_b,
                    'original_duration': original_duration,
                    'new_duration': duration_a,
                    'operation': operation,
                    'success': True
                }
            except Exception as e:
                logger.error(f"处理文件 {base_name_a} <-> {base_name_b} 时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    'file': base_name_a,
                    'matched_with': base_name_b,
                    'error': str(e),
                    'success': False
                }
        
        # 批量处理文件，避免内存问题
        for i in range(0, len(process_queue), self.batch_size):
            batch = process_queue[i:i+self.batch_size]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_item = {executor.submit(process_single_file, item): item for item in batch}
                
                # 处理结果
                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    if result['success']:
                        results.append(result)
                    
                    processed += 1
                    if callback:
                        callback(processed / total)
            
            # 如果开启内存优化，在批处理之间清理内存
            if self.optimize_memory and i + self.batch_size < len(process_queue):
                import gc
                gc.collect()
                # 短暂暂停，让系统有时间回收内存
                time.sleep(0.1)
        
        # 记录未匹配的文件
        unmatched_a = set(files_a_dict.keys()) - common_ids
        unmatched_b = set(files_b_dict.keys()) - common_ids
        
        if unmatched_a:
            logger.warning(f"文件夹A中有 {len(unmatched_a)} 个文件在B中没有匹配: {', '.join(list(unmatched_a)[:5])}{'...' if len(unmatched_a) > 5 else ''}")
        
        if unmatched_b:
            logger.warning(f"文件夹B中有 {len(unmatched_b)} 个文件在A中没有匹配: {', '.join(list(unmatched_b)[:5])}{'...' if len(unmatched_b) > 5 else ''}")
        
        # 计算总处理时间
        end_time = time.time()
        total_time = end_time - start_time
        files_per_second = len(results) / total_time if total_time > 0 else 0
        
        logger.info(f"处理完成，共处理了 {len(results)} 个文件对，耗时 {total_time:.2f} 秒")
        logger.info(f"处理速度: {files_per_second:.2f} 文件/秒")
        
        return results
    
    # 批量版本的音频时长获取（并行处理）
    def get_audio_durations_batch(self, audio_files):
        """批量获取多个音频文件的时长，使用并行处理提高效率"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务字典
            future_to_file = {executor.submit(self.get_audio_duration, file): file for file in audio_files}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    duration = future.result()
                    results[file] = duration
                except Exception as e:
                    logger.error(f"获取文件 {file} 长度时出错: {str(e)}")
                    results[file] = None
        
        return results
    
    def test_id_extraction(self, test_filenames):
        """测试ID提取功能，用于验证不同格式的文件名
        
        Args:
            test_filenames: 测试用的文件名列表
            
        Returns:
            包含文件名和提取ID的字典
        """
        results = {}
        for filename in test_filenames:
            file_id = self.extract_file_id(filename)
            results[filename] = file_id
            logger.info(f"测试文件: {filename} => 提取ID: {file_id}")
        return results

# 如果直接运行此文件，执行简单测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建测试实例
    processor = AudioProcessor()
    
    # 测试不同格式的文件名
    test_files = [
        # 基本测试
        "12345-原唱.mp3",
        "12345_伴奏.wav",
        # 字母数字混合ID
        "YPD12345-原唱.wav",
        "YPD12345_伴奏.mp3",
        # 带连接符的混合ID
        "YPD-12345-导唱.flac",
        "YPD_12345-原唱.wav",
        # 带其他前缀的ID
        "SL_YPD12345（伴奏）.flac",
        # ID在文件名中间
        "混合YPD12345混合.mp3",
        "前缀-YPD12345-后缀.mp3",
        # 更复杂情况
        "2022年度YPD12345歌曲合集.wav", 
        "YPD12345&YPD54321合并音轨.mp3",  # 应该提取YPD12345
        # 无法识别的情况
        "无法识别.mp3",
        "backup_20220101.mp3"
    ]
    
    # 执行测试
    results = processor.test_id_extraction(test_files)
    
    # 打印测试结果
    print("\n测试结果:")
    for filename, file_id in results.items():
        print(f"{filename:<35} => {file_id}")