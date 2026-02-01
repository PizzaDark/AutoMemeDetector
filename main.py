import sys
import os
import json
import time
import threading
import queue
import winreg
import numpy as np

# 第三方库
import keyboard
import pygame
import sounddevice as sd
import mss
import pypinyin
import ctypes
import pyaudiowpatch as pyaudio
from vosk import Model, KaldiRecognizer
from rapidocr_onnxruntime import RapidOCR
import cv2

# PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSystemTrayIcon, QMenu, QAction, 
                             QCheckBox, QComboBox, QSlider, QDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QFileDialog, QGroupBox, QStyleFactory, 
                             QMessageBox, QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QPen

# ================= 路径与资源工具 =================
def get_app_dir():
    """获取程序所在的目录 (config.json将生成在这里)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path):
    """获取内部资源路径 (适配PyInstaller打包)"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ================= 配置与常量 =================
APP_TITLE = "自动玩梗器"
AUTHOR_LINK = "https://space.bilibili.com/6297797"
ICON_PATH = get_resource_path("meme_detector.ico")
VOSK_MODEL_PATH = get_resource_path("models/vosk-model-small-cn-0.22")

# 默认规则示例
DEFAULT_RULES = [
    {"keyword": "鸡", "audio": "sounds/你干嘛.mp3"},
    {"keyword": "龙", "audio": "sounds/龙，可是帝王之征啊.mp3"},
    {"keyword": "恭喜", "audio": "sounds/恭喜爹可以称帝了.mp3"},
]

# ================= 工具类：配置管理 =================

class ConfigManager:
    def __init__(self):
        # 创建专用数据文件夹，避免与同目录其他软件冲突
        self.data_dir = os.path.join(get_app_dir(), "AutoMeme_Config")
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except:
                pass
        
        self.config_file = os.path.join(self.data_dir, "config.json")
        self.rules_file = os.path.join(self.data_dir, "rules.json")
        self.settings = self.load_settings()
        self.rules = self.load_rules()

    def load_settings(self):
        default = {
            "volume": 70,
            "input_device": None,
            "output_device": None,  # 扬声器设备
            "hotkey": "alt+m",
            "stop_hotkey": "alt+p", # 停止播放快捷键
            "screen_ocr_enabled": False, # 默认关闭，节省资源
            "microphone_detection_enabled": True, # 麦克风检测，默认开启
            "speaker_detection_enabled": False, # 扬声器检测，默认关闭
            "use_gpu": False,
            "auto_start": False
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return {**default, **json.load(f)}
            except:
                pass
        return default

    def save_settings(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=4)

    def load_rules(self):
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return DEFAULT_RULES

    def save_rules(self, rules):
        self.rules = rules
        with open(self.rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=4)

# ================= 核心线程：检测逻辑 =================

class VoiceWorker(QThread):
    sig_play_audio = pyqtSignal(str)
    sig_log = pyqtSignal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.cm = config_manager
        self.running = False
        self.paused = True
        self.audio_queue = queue.Queue()
        self.vosk_model = None
        self.rec = None

    def init_model(self):
        if not self.vosk_model:
            model_path = VOSK_MODEL_PATH 
            if os.path.exists(model_path):
                try:
                    Model.verbosity = -1
                    self.vosk_model = Model(model_path)
                    self.sig_log.emit("语音模型加载成功")
                except Exception as e:
                    self.sig_log.emit(f"语音模型加载失败: {e}")
            else:
                self.sig_log.emit(f"未找到语音模型文件夹: {VOSK_MODEL_PATH}")

    def run(self):
        self.init_model()
        self.running = True
        input_dev = self.cm.settings.get('input_device')
        
        try:
            with sd.RawInputStream(samplerate=16000, blocksize=2000, device=input_dev, 
                                   dtype='int16', channels=1, callback=self.audio_callback):
                while self.running:
                    # 检查麦克风检测是否开启
                    if self.paused or not self.cm.settings.get('microphone_detection_enabled', True):
                        time.sleep(0.1)
                        with self.audio_queue.mutex:
                            self.audio_queue.queue.clear()
                        continue

                    if self.vosk_model and not self.audio_queue.empty():
                        data = self.audio_queue.get()
                        if self.rec is None:
                            self.rec = KaldiRecognizer(self.vosk_model, 16000)
                        
                        if self.rec.AcceptWaveform(data):
                            res = json.loads(self.rec.Result())
                            text = res.get('text', '').replace(' ', '')
                            if text:
                                self.check_keywords(text)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            self.sig_log.emit(f"语音侦听错误: {e}")
            self.running = False

    def audio_callback(self, indata, frames, time, status):
        if not self.paused and self.cm.settings.get('microphone_detection_enabled', True):
            self.audio_queue.put(bytes(indata))

    def check_keywords(self, text):
        text_pinyin = pypinyin.lazy_pinyin(text)
        
        for rule in self.cm.rules:
            key = rule['keyword']
            key_pinyin = pypinyin.lazy_pinyin(key)
            
            match = False
            if key in text: match = True
            elif self.is_sublist(key_pinyin, text_pinyin): match = True
            
            if match:
                self.sig_log.emit(f"语音检测: {key}")
                self.sig_play_audio.emit(rule['audio'])
                return True
        return False

    def is_sublist(self, sub, full):
        n, m = len(sub), len(full)
        if n > m: return False
        for i in range(m - n + 1):
            if full[i:i+n] == sub: return True
        return False
    
    def stop(self):
        self.running = False
        self.wait(2000)  # 最多等待2秒

class SpeakerWorker(QThread):
    """扬声器音频检测线程 - 使用 PyAudioWPatch 捕获系统音频输出"""
    sig_play_audio = pyqtSignal(str)
    sig_log = pyqtSignal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.cm = config_manager
        self.running = False
        self.paused = True
        self.audio_queue = queue.Queue()
        self.vosk_model = None
        self.rec = None
        self.is_playing = False  # 播放保护标志
        self.last_play_time = 0  # 最后播放时间戳

    def init_model(self):
        if not self.vosk_model:
            model_path = VOSK_MODEL_PATH
            if os.path.exists(model_path):
                try:
                    Model.verbosity = -1
                    self.vosk_model = Model(model_path)
                    self.sig_log.emit("扬声器检测：语音模型加载成功")
                except Exception as e:
                    self.sig_log.emit(f"扬声器检测：语音模型加载失败: {e}")
            else:
                self.sig_log.emit(f"扬声器检测：未找到语音模型文件夹")

    def run(self):
        self.init_model()
        self.running = True
        
        try:
            p = pyaudio.PyAudio()
            
            # 查找 WASAPI loopback 设备（系统音频输出）
            output_dev = self.cm.settings.get('output_device')
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            
            if output_dev is not None:
                # 使用用户指定的设备
                default_speakers = p.get_device_info_by_index(output_dev)
            else:
                # 使用默认输出设备
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            if not default_speakers["isLoopbackDevice"]:
                # 如果默认输出不是 loopback 设备，查找对应的 loopback
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
            
            self.sig_log.emit(f"扬声器检测已启动：{default_speakers['name']}")
            
            # 打开音频流
            stream = p.open(
                format=pyaudio.paInt16,
                channels=default_speakers["maxInputChannels"],
                rate=int(default_speakers["defaultSampleRate"]),
                frames_per_buffer=2000,
                input=True,
                input_device_index=default_speakers["index"],
                stream_callback=self.audio_callback
            )
            
            stream.start_stream()
            
            while self.running:
                if self.paused or not self.cm.settings.get('speaker_detection_enabled', False):
                    time.sleep(0.1)
                    with self.audio_queue.mutex:
                        self.audio_queue.queue.clear()
                    continue

                # 播放保护：如果正在播放或刚播放完不久，跳过检测
                if self.is_playing or (time.time() - self.last_play_time < 2.0):
                    with self.audio_queue.mutex:
                        self.audio_queue.queue.clear()
                    time.sleep(0.1)
                    continue

                if self.vosk_model and not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    if self.rec is None:
                        self.rec = KaldiRecognizer(self.vosk_model, 16000)
                    
                    # 如果采样率不是16000，需要重采样
                    if int(default_speakers["defaultSampleRate"]) != 16000:
                        # 简单的降采样处理
                        audio_np = np.frombuffer(data, dtype=np.int16)
                        # 双声道转单声道
                        if default_speakers["maxInputChannels"] == 2:
                            audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        # 重采样到16000Hz（简单粗暴的跳帧法）
                        ratio = int(default_speakers["defaultSampleRate"]) / 16000
                        indices = (np.arange(0, len(audio_np), ratio)).astype(int)
                        audio_resampled = audio_np[indices]
                        data = audio_resampled.tobytes()
                    else:
                        # 双声道转单声道
                        if default_speakers["maxInputChannels"] == 2:
                            audio_np = np.frombuffer(data, dtype=np.int16)
                            audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
                            data = audio_np.tobytes()
                    
                    if self.rec.AcceptWaveform(data):
                        res = json.loads(self.rec.Result())
                        text = res.get('text', '').replace(' ', '')
                        if text:
                            self.check_keywords(text)
                else:
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            self.sig_log.emit(f"扬声器检测错误: {e}")
            self.running = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.paused and not self.is_playing:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def check_keywords(self, text):
        text_pinyin = pypinyin.lazy_pinyin(text)
        
        for rule in self.cm.rules:
            key = rule['keyword']
            key_pinyin = pypinyin.lazy_pinyin(key)
            
            match = False
            if key in text: match = True
            elif self.is_sublist(key_pinyin, text_pinyin): match = True
            
            if match:
                self.sig_log.emit(f"扬声器识别: {key}")
                self.sig_play_audio.emit(rule['audio'])
                return True
        return False

    def is_sublist(self, sub, full):
        n, m = len(sub), len(full)
        if n > m: return False
        for i in range(m - n + 1):
            if full[i:i+n] == sub: return True
        return False
    
    def set_playing_state(self, is_playing):
        """设置播放状态，用于递归保护"""
        self.is_playing = is_playing
        if not is_playing:
            self.last_play_time = time.time()
    
    def stop(self):
        self.running = False
        self.wait(2000)

class ScreenWorker(QThread):
    sig_play_audio = pyqtSignal(str)
    sig_highlight = pyqtSignal(int, int, int, int) 
    sig_log = pyqtSignal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.cm = config_manager
        self.running = False
        self.paused = True
        self.ocr = None
        self.last_frame_small = None   # 用于存储缩小后的上一帧（做差异对比）
        self.last_ocr_time = 0         # 控制 OCR 触发频率

    def init_model(self):
        # 懒加载：只有用户开启功能时才加载模型
        if self.cm.settings['screen_ocr_enabled'] and not self.ocr:
            try:
                # === 核心优化 1: 限制 AI 线程数 ===
                # 强制单核推理。虽然单次耗时增加 5-10ms，但能避免卡顿前台应用（如游戏）。
                # 必须设置 intra_op_num_threads=1
                use_gpu = self.cm.settings.get('use_gpu', False)
                self.ocr = RapidOCR(det_use_gpu=use_gpu, cls_use_gpu=use_gpu, rec_use_gpu=use_gpu, 
                                    use_angle_cls=False, intra_op_num_threads=1) 
                
                mode_str = "GPU加速" if use_gpu else "CPU节能"
                self.sig_log.emit(f"OCR引擎已就绪 ({mode_str}模式)")
            except Exception as e:
                self.sig_log.emit(f"OCR加载失败: {e}")

    def run(self):
        self.running = True
        # 环境变量双重限制，确保不抢占 CPU
        os.environ["OMP_NUM_THREADS"] = "1"
        
        with mss.mss() as sct:
            while self.running:
                # 暂停状态或功能关闭时，低频空转
                if self.paused or not self.cm.settings['screen_ocr_enabled']:
                    time.sleep(1.0)
                    continue

                if not self.ocr:
                     self.init_model()

                if self.ocr:
                    self.process_screen_full_optimized(sct)
                    # 每次循环后小睡，释放 CPU 时间片
                    time.sleep(0.1) 
                else:
                    time.sleep(1.0)

    def process_screen_full_optimized(self, sct):
        try:
            current_time = time.time()
            # 频率控制：每隔 0.5 秒检测一次（可根据需要调整）
            # 太快了 CPU 还是会高，0.5 - 0.8 秒是比较好的平衡点
            if current_time - self.last_ocr_time < 0.5:
                return

            # 获取主屏幕 (索引 1 通常是主屏，如果是单屏则是 0，这里做个兼容)
            if len(sct.monitors) < 2:
                monitor = sct.monitors[0]
            else:
                monitor = sct.monitors[1]
            
            # 1. 极速截图
            img_bgra = np.array(sct.grab(monitor))
            # 2. 转灰度 (丢弃颜色信息，减少数据量)
            img_gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2GRAY)
            
            # === 核心优化 2: 暴力降采样 ===
            # 将任意分辨率 (2k/4k) 强制缩放到宽度 960 像素
            # 960px 是 OCR 的“甜点分辨率”，速度快且字够大
            h, w = img_gray.shape
            target_w = 960.0
            scale = 1.0
            
            if w > target_w:
                scale = target_w / w
                new_w = int(target_w)
                new_h = int(h * scale)
                # 使用 INTER_AREA 插值，缩放效果最好，字迹最清晰
                img_small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_small = img_gray
                scale = 1.0

            # === 核心优化 3: 视觉差分 (Visual Diff) ===
            # 如果屏幕画面静止不动，绝对不要跑 OCR
            if self.last_frame_small is not None:
                # 尺寸必须一致才能对比
                if self.last_frame_small.shape == img_small.shape:
                    # 计算两帧差异
                    diff = cv2.absdiff(self.last_frame_small, img_small)
                    # 过滤微小噪点
                    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    non_zero = cv2.countNonZero(diff_thresh)
                    
                    # 变化阈值：全屏像素的 0.3% 发生变化才触发
                    # 例如 960x540 的图，大约需要 1500 个像素变化才算变了
                    pixel_total = img_small.shape[0] * img_small.shape[1]
                    if non_zero < (pixel_total * 0.003):
                        return # 画面基本静止，跳过本次 OCR

            # 更新上一帧缓存
            self.last_frame_small = img_small
            
            # === 执行 OCR ===
            self.last_ocr_time = current_time
            # 对缩小后的图进行识别
            result, _ = self.ocr(img_small)
            
            if result:
                for line in result:
                    # RapidOCR 返回: [[[[x1,y1]...]], "text", score]
                    if len(line) >= 2:
                        box = line[0]
                        text = line[1]
                        
                        matched_rule = self.find_keyword_match(text)
                        if matched_rule:
                            try:
                                # === 坐标还原 ===
                                # 我们是在缩小图(img_small)上识别的，坐标是缩小的
                                # 需要除以 scale 还原回 原始屏幕坐标
                                
                                # 获取包围盒的 x, y, w, h
                                xs = [p[0] for p in box]
                                ys = [p[1] for p in box]
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                
                                # 还原计算
                                real_x = monitor['left'] + int(x_min / scale)
                                real_y = monitor['top'] + int(y_min / scale)
                                real_w = int((x_max - x_min) / scale)
                                real_h = int((y_max - y_min) / scale)
                                
                                # 发送信号
                                self.sig_highlight.emit(real_x, real_y, real_w, real_h)
                                self.sig_log.emit(f"屏幕检测: {matched_rule['keyword']}")
                                self.sig_play_audio.emit(matched_rule['audio'])
                                
                                return # 这一帧只要识别到一个关键词就停止，避免重复播放
                            except Exception as e:
                                print(f"坐标换算异常: {e}")

        except Exception as e:
            print(f"屏幕处理循环异常: {e}")

    def find_keyword_match(self, text):
        for rule in self.cm.rules:
            if rule['keyword'] in text:
                return rule
        return None
        
    def stop(self):
        self.running = False
        self.wait(2000)

# ================= 界面组件：透明覆盖层 =================

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 无边框、总在最前、工具窗口、不接受输入
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.box = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.hide_box)

    def show_highlight(self, x, y, w, h):
        # 获取 Qt 的逻辑屏幕几何信息 (主屏幕)
        screen = QApplication.primaryScreen()
        geo = screen.geometry()
        qt_w = geo.width()
        qt_h = geo.height()
        
        # 获取 物理屏幕尺寸 (主屏幕)
        # 使用 ctypes 获取真实物理分辨率，这比推算DPI更可靠
        try:
            user32 = ctypes.windll.user32
            # 这里的 0, 1 分别对应 SM_CXSCREEN, SM_CYSCREEN
            pk_width = user32.GetSystemMetrics(0)
            pk_height = user32.GetSystemMetrics(1)
        except:
            pk_width = qt_w
            pk_height = qt_h
            
        # 计算 物理坐标 -> Qt逻辑坐标 的转换比例
        if pk_width > 0 and pk_height > 0:
            scale_x = qt_w / pk_width
            scale_y = qt_h / pk_height
        else:
            scale_x = 1.0
            scale_y = 1.0
            
        padding = 10
        logical_x = int(x * scale_x) - padding
        logical_y = int(y * scale_y) - padding
        logical_w = int(w * scale_x) + padding * 2
        logical_h = int(h * scale_y) + padding * 2
        
        self.setGeometry(logical_x, logical_y, logical_w, logical_h)
        self.box = (0, 0, self.width(), self.height())
        self.show()
        self.update()
        self.timer.start(800) # 0.8秒后消失，给用户更好的视觉反馈

    def hide_box(self):
        self.hide()
        self.timer.stop()

    def paintEvent(self, event):
        if self.box:
            painter = QPainter(self)
            # 绿色粗框
            pen = QPen(QColor(0, 255, 0), 4)
            painter.setPen(pen)
            # 画空心矩形
            painter.drawRect(2, 2, self.width()-4, self.height()-4)

# ================= 界面组件：规则编辑器 =================

class RuleEditor(QDialog):
    def __init__(self, parent=None, rules=[]):
        super().__init__(parent)
        self.setWindowTitle("编辑规则集")
        self.resize(550, 400) # 窗口调小
        self.rules = rules
        self.layout = QVBoxLayout(self)

        self.table = QTableWidget()
        # 修复崩溃：禁止单击后直接键入，必须双击进入编辑
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["关键词(双击编辑)", "音频文件路径", "选择文件"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 180) # 关键词列变宽
        self.layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加规则")
        self.del_btn = QPushButton("删除选中")
        self.import_btn = QPushButton("导入规则...") # 新增导入按钮
        self.save_btn = QPushButton("保存并生效")
        
        self.add_btn.clicked.connect(self.add_row)
        self.del_btn.clicked.connect(self.del_row)
        self.import_btn.clicked.connect(self.import_rules)
        self.save_btn.clicked.connect(self.save_and_close)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.save_btn)
        self.layout.addLayout(btn_layout)

        self.load_data()

    def load_data(self):
        self.table.setRowCount(0)
        for r in self.rules:
            self.add_row_data(r['keyword'], r['audio'])

    def add_row_data(self, keyword="新关键词", audio_path=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(keyword))
        self.table.setItem(row, 1, QTableWidgetItem(audio_path))
        
        btn = QPushButton("...")
        btn.clicked.connect(lambda _, r=row: self.browse_file(r))
        self.table.setCellWidget(row, 2, btn)

    def add_row(self):
        self.add_row_data()

    def import_rules(self):
        fname, _ = QFileDialog.getOpenFileName(self, '导入规则文件', '', "JSON Files (*.json)")
        if fname:
            # 提示确认覆盖
            reply = QMessageBox.question(self, "确认覆盖", "导入操作将清空并覆盖当前列表中的所有规则，是否继续？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.table.setRowCount(0) # 清空当前列表
                        for item in data:
                            if 'keyword' in item and 'audio' in item:
                                self.add_row_data(item['keyword'], item['audio'])
                        QMessageBox.information(self, "导入成功", f"已导入 {len(data)} 条规则, 请记得点击保存生效。")
                    else:
                        QMessageBox.warning(self, "格式错误", "JSON文件必须包含一个规则列表。")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", str(e))

    def browse_file(self, row):
        # 默认打开目录设为 EXE 所在目录
        # 如果用户是从外部导入的，这里可以记忆上次路径（为了简单，暂定用户常去的目录或者文档）
        start_dir = get_app_dir()
            
        fname, _ = QFileDialog.getOpenFileName(self, '选择音频文件', start_dir, "Audio Files (*.mp3 *.wav *.ogg)")
        if fname:
            # 1. 如果文件在 EXE 同级或子目录下，尝试使用相对路径，保持便携性
            # 2. 如果文件在其他盘符或外部目录，强制使用绝对路径
            try:
                # 检查是否在程序目录下
                app_dir = os.path.abspath(get_app_dir())
                file_abs = os.path.abspath(fname)
                
                if file_abs.startswith(app_dir):
                     rel_path = os.path.relpath(fname, app_dir)
                     self.table.item(row, 1).setText(rel_path)
                else:
                     # 外部文件，使用绝对路径
                     self.table.item(row, 1).setText(fname)
            except:
                self.table.item(row, 1).setText(fname)

    def del_row(self):
        current = self.table.currentRow()
        if current >= 0:
            self.table.removeRow(current)

    def save_and_close(self):
        new_rules = []
        for i in range(self.table.rowCount()):
            key = self.table.item(i, 0).text()
            audio = self.table.item(i, 1).text()
            if key and audio:
                new_rules.append({"keyword": key, "audio": audio})
        self.rules = new_rules
        self.accept()

# ================= 主窗口 =================

class MainWindow(QMainWindow):
    sig_toggle = pyqtSignal()
    sig_stop_audio = pyqtSignal()
    sig_hotkey_recorded = pyqtSignal(str, str) # config_key, new_hotkey_str

    def __init__(self):
        super().__init__()
        # 初始化pygame mixer
        pygame.mixer.init()
        
        self.cm = ConfigManager()
        self.init_ui()
        
        # 覆盖层
        self.overlay = OverlayWindow()

        # 后台线程 - 语音
        self.voice_worker = VoiceWorker(self.cm)
        self.voice_worker.sig_play_audio.connect(self.play_audio)
        self.voice_worker.sig_log.connect(self.update_log)
        self.voice_worker.start()

        # 后台线程 - 扬声器
        self.speaker_worker = SpeakerWorker(self.cm)
        self.speaker_worker.sig_play_audio.connect(self.play_audio)
        self.speaker_worker.sig_log.connect(self.update_log)
        self.speaker_worker.start()

        # 后台线程 - 屏幕
        self.screen_worker = ScreenWorker(self.cm)
        self.screen_worker.sig_play_audio.connect(self.play_audio)
        self.screen_worker.sig_highlight.connect(self.show_highlight)
        self.screen_worker.sig_log.connect(self.update_log)
        self.screen_worker.start()

        # 信号连接
        self.sig_toggle.connect(self.toggle_detection)
        self.sig_stop_audio.connect(self.stop_all_audio)
        self.sig_hotkey_recorded.connect(self.on_hotkey_recorded)

        # 热键
        self.bind_hotkeys()
        # 开机自启状态同步
        self.check_autostart_ui()

    def init_ui(self):
        self.setWindowTitle(APP_TITLE)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        
        self.resize(450, 500)
        # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # 注意：DPI设置必须在QApplication创建前，已移至main入口
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 5)

        # 1. 顶部状态与大按钮
        self.status_label = QLabel("等待启动...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 16px;")
        
        self.toggle_btn = QPushButton("开启检测")
        self.toggle_btn.setMinimumHeight(90)
        self.toggle_btn.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.update_btn_style(False)
        self.toggle_btn.clicked.connect(self.toggle_detection)

        layout.addWidget(self.status_label)
        layout.addWidget(self.toggle_btn)

        # 2. 作者信息
        link = QLabel(f'作者<a href="{AUTHOR_LINK}" style="color: #2196F3; text-decoration: none;">@依然匹萨吧</a>')
        link.setAlignment(Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        link.setFont(QFont("Arial", 10))
        layout.addWidget(link)

        # 3. 设置分组
        group = QGroupBox("设置选项")
        g_layout = QVBoxLayout()

        # 音量
        vol_box = QHBoxLayout()
        vol_box.addWidget(QLabel("播放音量:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(self.cm.settings['volume'])
        self.slider.valueChanged.connect(self.on_volume_change)
        vol_box.addWidget(self.slider)
        g_layout.addLayout(vol_box)

        # 输入设备（麦克风）
        dev_box = QHBoxLayout()
        dev_box.addWidget(QLabel("麦克风:"))
        self.combo_dev = QComboBox()
        self.refresh_devices()
        self.combo_dev.currentIndexChanged.connect(self.on_device_change)
        dev_box.addWidget(self.combo_dev, 1)
        g_layout.addLayout(dev_box)

        # 输出设备（扬声器）
        speaker_box = QHBoxLayout()
        speaker_box.addWidget(QLabel("扬声器:"))
        self.combo_speaker = QComboBox()
        self.refresh_speaker_devices()
        self.combo_speaker.currentIndexChanged.connect(self.on_speaker_device_change)
        speaker_box.addWidget(self.combo_speaker, 1)
        g_layout.addLayout(speaker_box)

        # 开关
        self.chk_mic = QCheckBox("开启麦克风检测")
        self.chk_mic.setToolTip("检测麦克风输入的语音")
        self.chk_mic.setChecked(self.cm.settings.get('microphone_detection_enabled', True))
        self.chk_mic.stateChanged.connect(self.on_microphone_change)
        g_layout.addWidget(self.chk_mic)

        self.chk_speaker = QCheckBox("开启桌面媒体音频(扬声器)检测")
        self.chk_speaker.setToolTip("检测系统音频输出（游戏、视频、音乐等）中的关键词")
        self.chk_speaker.setChecked(self.cm.settings.get('speaker_detection_enabled', False))
        self.chk_speaker.stateChanged.connect(self.on_speaker_change)
        g_layout.addWidget(self.chk_speaker)

        self.chk_ocr = QCheckBox("开启屏幕关键词检测(CPU占用高)")
        self.chk_ocr.setToolTip("使用RapidOCR检测屏幕上的文字，会增加CPU占用")
        self.chk_ocr.setChecked(self.cm.settings['screen_ocr_enabled'])
        self.chk_ocr.stateChanged.connect(self.on_ocr_change)
        g_layout.addWidget(self.chk_ocr)

        self.chk_gpu = QCheckBox("使用 GPU 加速(勾选后需停止检测重新开启)")
        self.chk_gpu.setToolTip("若硬件支持(如NVIDIA显卡)，将大幅降低CPU占用。\n修改后将在下次启动检测或重置模型时生效。")
        self.chk_gpu.setChecked(self.cm.settings.get('use_gpu', False))
        self.chk_gpu.stateChanged.connect(self.on_gpu_change)
        g_layout.addWidget(self.chk_gpu)

        self.chk_auto = QCheckBox("开机自动启动")
        self.chk_auto.stateChanged.connect(self.on_autostart_change)
        g_layout.addWidget(self.chk_auto)

        # 快捷键设置区域
        hk_group = QGroupBox("快捷键设置 (点击按钮修改)")
        hk_layout = QVBoxLayout()
        
        # 1. 检测开关
        hk_layout.addLayout(self.create_hotkey_row("开启/暂停检测:", "hotkey", "alt+m"))
        # 2. 停止音频
        hk_layout.addLayout(self.create_hotkey_row("停止当前播放: ", "stop_hotkey", "alt+p"))
        
        hk_group.setLayout(hk_layout)
        g_layout.addWidget(hk_group)

        # 规则按钮
        btn_layout = QHBoxLayout()
        
        btn_rule = QPushButton("编辑 关键词-音频 规则集")
        btn_rule.setStyleSheet("padding: 8px;")
        btn_rule.clicked.connect(self.open_rules)
        btn_layout.addWidget(btn_rule)
        
        btn_folder = QPushButton("打开配置文件夹")
        btn_folder.setStyleSheet("padding: 8px;")
        btn_folder.setToolTip("打开存放 config.json 和 rules.json 的目录")
        btn_folder.clicked.connect(self.open_rules_folder)
        btn_layout.addWidget(btn_folder)
        
        g_layout.addLayout(btn_layout)

        group.setLayout(g_layout)
        layout.addWidget(group)

        claim = QLabel('本软件免费，禁止商用或贩卖')
        claim.setAlignment(Qt.AlignCenter)
        claim.setFont(QFont("Arial", 8))
        layout.addWidget(claim)

        # 托盘
        self.init_tray()

    def init_tray(self):
        self.tray = QSystemTrayIcon(self)
        if os.path.exists(ICON_PATH):
            self.tray.setIcon(QIcon(ICON_PATH))
        else:
            # 默认图标
            self.tray.setIcon(self.style().standardIcon(QStyleFactory.create("Fusion").SP_ComputerIcon))
        
        menu = QMenu()
        a_show = QAction("显示主界面", self)
        a_show.triggered.connect(self.showNormal)
        a_quit = QAction("退出程序", self)
        a_quit.triggered.connect(self.quit_app)
        
        menu.addAction(a_show)
        menu.addAction(a_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_click)
        self.tray.show()

    def on_tray_click(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.showNormal()

    def closeEvent(self, event):
        if self.tray.isVisible():
            self.hide()
            self.tray.showMessage(APP_TITLE, "程序已最小化到托盘", QSystemTrayIcon.Information, 1500)
            event.ignore()
        else:
            event.accept()

    def quit_app(self):
        self.voice_worker.stop()
        self.speaker_worker.stop()
        self.screen_worker.stop()
        QApplication.quit()

    # ================= 逻辑交互 =================

    def update_btn_style(self, active):
        if active:
            self.toggle_btn.setText("停止检测")
            self.toggle_btn.setStyleSheet("background-color: #f44336; color: white; border-radius: 8px;")
            self.status_label.setText(f"正在监听中... ({self.cm.settings['hotkey']})")
        else:
            self.toggle_btn.setText("开启检测")
            self.toggle_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 8px;")
            self.status_label.setText(f"检测已暂停({self.cm.settings['hotkey']})")

    def toggle_detection(self):
        self.voice_worker.paused = not self.voice_worker.paused
        self.speaker_worker.paused = not self.speaker_worker.paused
        self.screen_worker.paused = not self.screen_worker.paused
        
        # 只要有一个没暂停，就算运行
        is_running = not (self.voice_worker.paused and self.speaker_worker.paused and self.screen_worker.paused)
        
        if is_running:
            # 开启时检查模型
            # 语音模型自动lazy load，屏幕模型手动check
            if self.chk_ocr.isChecked() and not self.screen_worker.ocr:
                 self.status_label.setText("正在加载OCR模型，请稍候...")
                 QApplication.processEvents()
                 # 触发worker内部加载
                 pass 
        self.update_btn_style(is_running)

    def play_audio(self, path):
        # 使用线程播放音频，避免IO阻塞UI
        def _play_thread():
            try:
                # 设置播放标志，防止扬声器检测递归触发
                self.speaker_worker.set_playing_state(True)
                
                # 路径处理逻辑：
                final_path = path
                if not os.path.isabs(path):
                    local_check = os.path.join(get_app_dir(), path)
                    if os.path.exists(local_check):
                        final_path = local_check
                    else:
                        resource_check = get_resource_path(path)
                        if os.path.exists(resource_check):
                            final_path = resource_check
                
                vol = self.cm.settings['volume'] / 100.0
                sound = pygame.mixer.Sound(final_path)
                sound.set_volume(vol)
                channel = sound.play()
                
                # 等待播放完成
                if channel:
                    while channel.get_busy():
                        time.sleep(0.1)
                
                # 播放完成，解除保护
                self.speaker_worker.set_playing_state(False)
            except Exception as e:
                self.speaker_worker.set_playing_state(False)
                pass # print(f"播放失败: {e}")
        
        threading.Thread(target=_play_thread, daemon=True).start()

    def show_highlight(self, x, y, w, h):
        self.overlay.show_highlight(x, y, w, h)

    def update_log(self, text):
        self.statusBar().showMessage(text, 3000)

    # ================= 设置相关 =================

    def refresh_devices(self):
        self.combo_dev.clear()
        try:
            devices = sd.query_devices()
            host_api = sd.default.hostapi
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and dev['hostapi'] == host_api:
                    self.combo_dev.addItem(f"{dev['name']}", idx)
            
            last = self.cm.settings.get('input_device')
            if last is not None:
                idx = self.combo_dev.findData(last)
                if idx >= 0: self.combo_dev.setCurrentIndex(idx)
        except:
            self.combo_dev.addItem("未找到音频设备")

    def on_device_change(self, index):
        val = self.combo_dev.itemData(index)
        if val is not None:
            self.cm.settings['input_device'] = val
            self.cm.save_settings()
            self.update_log("输入设备已变更，请重启检测生效")

    def on_volume_change(self, val):
        self.cm.settings['volume'] = val
        self.cm.save_settings()

    def on_gpu_change(self, state):
        enabled = (state == Qt.Checked)
        self.cm.settings['use_gpu'] = enabled
        self.cm.save_settings()
        # 强制释放OCR对象，以便ScreenWorker在下一帧检测时重建模型
        self.screen_worker.ocr = None
        self.update_log(f"GPU设置已更新: {'开启' if enabled else '关闭'}")

    def on_ocr_change(self, state):
        enabled = (state == Qt.Checked)
        self.cm.settings['screen_ocr_enabled'] = enabled
        self.cm.save_settings()
        # 屏幕识别线程会轮询该配置，无需重启线程

    def on_microphone_change(self, state):
        enabled = (state == Qt.Checked)
        self.cm.settings['microphone_detection_enabled'] = enabled
        self.cm.save_settings()
        if enabled:
            self.update_log("麦克风检测已开启")
        else:
            self.update_log("麦克风检测已关闭")

    def on_speaker_change(self, state):
        enabled = (state == Qt.Checked)
        self.cm.settings['speaker_detection_enabled'] = enabled
        self.cm.save_settings()
        if enabled:
            self.update_log("扬声器检测已开启")
        else:
            self.update_log("扬声器检测已关闭")

    def refresh_speaker_devices(self):
        """刷新扬声器（输出）设备列表"""
        self.combo_speaker.clear()
        try:
            p = pyaudio.PyAudio()
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            
            # 遍历所有 loopback 设备
            for loopback in p.get_loopback_device_info_generator():
                self.combo_speaker.addItem(f"{loopback['name']}", loopback['index'])
            
            p.terminate()
            
            # 恢复上次选择
            last = self.cm.settings.get('output_device')
            if last is not None:
                idx = self.combo_speaker.findData(last)
                if idx >= 0:
                    self.combo_speaker.setCurrentIndex(idx)
                    
            if self.combo_speaker.count() == 0:
                self.combo_speaker.addItem("未找到输出设备")
        except Exception as e:
            self.combo_speaker.addItem(f"设备加载失败: {str(e)}")

    def on_speaker_device_change(self, index):
        """扬声器设备变更回调"""
        val = self.combo_speaker.itemData(index)
        if val is not None:
            self.cm.settings['output_device'] = val
            self.cm.save_settings()
            self.update_log("扬声器设备已变更，请重启检测生效")

    def open_rules(self):
        # 核心修复：打开编辑窗口前卸载全局键盘钩子
        # 避免 keyboard 库钩子与 PyQt5 表格编辑器的输入事件冲突导致卡死崩溃
        try:
            keyboard.unhook_all_hotkeys()
        except:
            pass

        editor = RuleEditor(self, self.cm.rules)
        if editor.exec_() == QDialog.Accepted:
            self.cm.save_rules(editor.rules)
            self.update_log("规则已保存")
        
        # 关闭窗口后恢复热键绑定
        self.bind_hotkeys()

    def open_rules_folder(self):
        """打开规则配置文件所在的文件夹"""
        folder = self.cm.data_dir
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            self.update_log("配置文件夹尚未创建")

    # ================= 热键与自启 =================

    def create_hotkey_row(self, text, key, default):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(text))
        
        btn = QPushButton(self.cm.settings.get(key, default))
        btn.setFixedWidth(120)
        btn.setProperty("config_key", key)  # 标记以便查找
        btn.clicked.connect(lambda: self.start_hotkey_recording(btn, key))
        layout.addWidget(btn)
        
        btn_reset = QPushButton("重置")
        btn_reset.setFixedWidth(50)
        btn_reset.clicked.connect(lambda: self.on_hotkey_recorded(key, default)) # 重置直接调用更新逻辑
        layout.addWidget(btn_reset)
        
        layout.addStretch()
        return layout

    def start_hotkey_recording(self, btn, key):
        btn.setText("按下组合键...")
        btn.setEnabled(False)
        self.update_log("请按下新的快捷键组合...")
        # 启动线程监听按键，避免阻塞UI
        threading.Thread(target=self.wait_for_hotkey, args=(key,), daemon=True).start()

    def wait_for_hotkey(self, key):
        time.sleep(0.3) # 防止误触
        try:
            # 读取下一个热键组合
            hk = keyboard.read_hotkey(suppress=False)
            self.sig_hotkey_recorded.emit(key, hk)
        except Exception as e:
            print(f"Hotkey record error: {e}")
            self.sig_hotkey_recorded.emit(key, "error")

    def on_hotkey_recorded(self, key, new_hotkey):
        if new_hotkey == "error":
            self.update_log("快捷键录制出错")
            # 刷新UI恢复原状
            self.init_ui()
            return

        self.cm.settings[key] = new_hotkey
        self.cm.save_settings()
        self.bind_hotkeys()
        self.update_log(f"快捷键已更新: {new_hotkey}")
        
        for btn in self.findChildren(QPushButton):
            if btn.text() == "按下组合键...":
                 btn.setText(new_hotkey)
                 btn.setEnabled(True)
        self.refresh_hotkey_ui_buttons()

    def refresh_hotkey_ui_buttons(self):
        # 这是一个补丁方法，用于更新那两个按钮的文字
        # 我们需要在 create_hotkey_row 里给按钮设个 objectName
        btns = self.findChildren(QPushButton)
        for btn in btns:
            prop = btn.property("config_key")
            if prop:
                val = self.cm.settings.get(prop)
                btn.setText(str(val))
                btn.setEnabled(True)

    def bind_hotkeys(self):
        try:
            keyboard.unhook_all_hotkeys()
        except Exception as e:
            # 某些情况下 keyboard 会报错，忽略
            pass

        hk_toggle = self.cm.settings.get('hotkey', 'alt+m')
        hk_stop = self.cm.settings.get('stop_hotkey', 'alt+p')

        try:
            keyboard.add_hotkey(hk_toggle, self.sig_toggle.emit)
        except Exception as e:
            self.update_log(f"绑定开关热键失败: {e}")
            
        try:
            keyboard.add_hotkey(hk_stop, self.sig_stop_audio.emit)
        except Exception as e:
            self.update_log(f"绑定停止热键失败: {e}")

    def stop_all_audio(self):
        try:
            pygame.mixer.stop()
            self.update_log("已停止播放")
        except:
            pass

    def on_autostart_change(self, state):
        is_checked = (state == Qt.Checked)
        key_path = r"Software/Microsoft/Windows/CurrentVersion/Run"
        app_path = sys.executable
        
        # 如果是脚本运行，sys.executable 是 python.exe，这里简单判断
        if not getattr(sys, 'frozen', False):
            # 调试模式下不建议写真实注册表
            return 

        try:
            reg = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(reg, key_path, 0, winreg.KEY_ALL_ACCESS)
            if is_checked:
                winreg.SetValueEx(key, APP_TITLE, 0, winreg.REG_SZ, f'"{app_path}"')
            else:
                try:
                    winreg.DeleteValue(key, APP_TITLE)
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
            self.cm.settings['auto_start'] = is_checked
            self.cm.save_settings()
        except Exception as e:
            self.update_log(f"自启设置失败: {e}")

    def check_autostart_ui(self):
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(self.cm.settings['auto_start'])
        self.chk_auto.blockSignals(False)

if __name__ == "__main__":
    # ================= DPI与缩放设置 =================
    # 1. 开启高DPI自适应 (必须在 QApplication 创建前设置)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # 2. 若需手动调整缩放比例 (例如感觉界面太小，想强制放大 1.25 倍)
    # 请取消下一行的注释，并修改数值 (1.0 = 100%, 1.5 = 150%)
    # os.environ["QT_SCALE_FACTOR"] = "0.75"

    # 简单的单实例锁（端口检测法太麻烦，这里仅做简易入口）
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # 确保图标加载，否则使用系统默认
    if not os.path.exists(ICON_PATH):
        pass 

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())