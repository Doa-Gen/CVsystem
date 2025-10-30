"""
摄像头处理模块
"""
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np


class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.paused = False
        self.capture = None
        
    def run(self):
        """运行摄像头捕获"""
        self.capture = cv2.VideoCapture(self.camera_index)
        self.running = True
        
        while self.running:
            if not self.paused:
                ret, frame = self.capture.read()
                if ret:
                    self.frame_ready.emit(frame)
            self.msleep(30)  # 约30fps
        
        if self.capture:
            self.capture.release()
    
    def pause(self):
        """暂停"""
        self.paused = True
    
    def resume(self):
        """继续"""
        self.paused = False
    
    def stop(self):
        """停止"""
        self.running = False
        self.wait()
    
    def get_current_frame(self):
        """获取当前帧"""
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None
