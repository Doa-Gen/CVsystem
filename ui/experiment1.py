"""
实验一面板：基础图像处理
包含：摄像头调用、图像格式转换、图像读写测试、图片融合、图像校正
"""
from typing import Optional
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QScrollArea, QFileDialog,
                             QMessageBox, QLineEdit, QComboBox, QGroupBox,
                             QGridLayout, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QMouseEvent
import cv2
import numpy as np
from .styles import get_style, COLORS
from core.camera import CameraThread
from core.image_processor import ImageProcessor
from utils.helpers import numpy_to_qpixmap, get_display_pixmap, imread_chinese, imwrite_chinese


class Experiment1Panel(QWidget):
    """实验一主面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.original_image = None
        self.processed_image = None
        self.current_task = 0
        
        # 摄像头相关
        self.camera_thread = None
        self.camera_running = False
        
        # 图像读写测试相关
        self.drawing_mode = None
        self.drawing_points = []
        self.shapes_drawn = []
        
        # 图像校正相关
        self.correction_points = [] 
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧控制面板
        self.control_panel = QWidget()
        self.control_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['secondary_bg']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        self.control_panel.setFixedWidth(300)
        
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(12)
        
        # 任务标题
        self.task_label = QLabel('请选择任务')
        self.task_label.setObjectName('title')
        self.task_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {COLORS['text']};
            background: transparent;
            border: none;
        """)
        control_layout.addWidget(self.task_label)
        
        # 控制按钮容器
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setSpacing(8)
        control_layout.addWidget(self.controls_container)
        
        control_layout.addStretch()
        
        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)
        
        # 原始图像
        original_label = QLabel('原始图像')
        original_label.setObjectName('subtitle')
        self.original_display = ImageDisplayLabel()
        self.original_display.setFixedSize(1000, 380)
        self.original_display.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background-color: {COLORS['background']};
        """)
        
        # 处理后图像
        processed_label = QLabel('处理后图像')
        processed_label.setObjectName('subtitle')
        self.processed_display = ImageDisplayLabel()
        self.processed_display.setFixedSize(1000, 380)
        self.processed_display.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background-color: {COLORS['background']};
        """)
        
        # 连接鼠标事件
        self.original_display.mouse_moved.connect(self.on_mouse_moved)
        self.original_display.mouse_clicked.connect(self.on_mouse_clicked)
        
        right_layout.addWidget(original_label)
        right_layout.addWidget(self.original_display)
        right_layout.addWidget(processed_label)
        right_layout.addWidget(self.processed_display)
        
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(right_panel)
    
    def set_task(self, task_index):
        """设置当前任务"""
        self.current_task = task_index
        self.clear_controls()
        
        tasks = {
            1: ('摄像头调用', self.setup_camera_controls),
            2: ('图像格式转换', self.setup_format_controls),
            3: ('图像读写测试', self.setup_readwrite_controls),
            4: ('图片融合', self.setup_blend_controls),
            5: ('颜色阈值抠图', self.setup_color_threshold_controls),
            6: ('图像校正', self.setup_correction_controls),
            7: ('布匹裁剪分割线识别', self.setup_fabric_cut_controls),
        }
        
        if task_index in tasks:
            title, setup_func = tasks[task_index]
            self.task_label.setText(title)
            setup_func()
    
    def clear_controls(self):
        """清空控制面板"""
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
    
    # ==================== 任务1：摄像头调用 ====================
    
    def setup_camera_controls(self):
        """设置摄像头控制"""
        open_btn = QPushButton('打开摄像头')
        open_btn.clicked.connect(self.open_camera)
        
        pause_btn = QPushButton('暂停/继续')
        pause_btn.clicked.connect(self.pause_camera)
        
        capture_btn = QPushButton('获取单帧')
        capture_btn.clicked.connect(self.capture_frame)
        
        close_btn = QPushButton('关闭摄像头')
        close_btn.clicked.connect(self.close_camera)
        
        save_btn = QPushButton('保存当前帧')
        save_btn.clicked.connect(self.save_current_frame)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('摄像头调用'))
        
        self.controls_layout.addWidget(open_btn)
        self.controls_layout.addWidget(pause_btn)
        self.controls_layout.addWidget(capture_btn)
        self.controls_layout.addWidget(close_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def open_camera(self):
        """打开摄像头"""
        if not self.camera_running:
            self.camera_thread = CameraThread(0)
            self.camera_thread.frame_ready.connect(self.update_camera_frame)
            self.camera_thread.start()
            self.camera_running = True
            QMessageBox.information(self, '成功', '摄像头已打开')
    
    def pause_camera(self):
        """暂停/继续摄像头"""
        if self.camera_thread:
            if self.camera_thread.paused:
                self.camera_thread.resume()
            else:
                self.camera_thread.pause()
    
    def capture_frame(self):
        """获取单帧"""
        if self.camera_thread:
            frame = self.camera_thread.get_current_frame()
            if frame is not None:
                self.processed_image = frame.copy()
                self.update_display()
                QMessageBox.information(self, '成功', '已获取单帧图像')
    
    def close_camera(self):
        """关闭摄像头"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            self.camera_running = False
            self.original_display.clear()
            QMessageBox.information(self, '成功', '摄像头已关闭')
    
    def update_camera_frame(self, frame):
        """更新摄像头帧"""
        pixmap = get_display_pixmap(frame, 1000, 380)
        if pixmap:
            self.original_display.setPixmap(pixmap)
    
    def save_current_frame(self):
        """保存当前帧"""
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, '保存图片', '', 
                'Images (*.png *.jpg *.bmp)'
            )
            if file_path:
                if imwrite_chinese(file_path, self.processed_image):
                    QMessageBox.information(self, '成功', f'图片已保存到: {file_path}')
                else:
                    QMessageBox.warning(self, '错误', '图片保存失败')
    
    # ==================== 任务2：图像格式转换 ====================
    
    def setup_format_controls(self):
        """设置格式转换控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        gray_btn = QPushButton('RGB转灰度')
        gray_btn.clicked.connect(self.convert_to_gray)
        
        hsv_btn = QPushButton('RGB转HSV')
        hsv_btn.clicked.connect(self.convert_to_hsv)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('RGB转Gray'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(gray_btn)
        self.controls_layout.addWidget(hsv_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def convert_to_gray(self):
        """RGB转灰度"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        gray = ImageProcessor.rgb_to_gray_manual(self.original_image)
        
        # 显示灰度图
        self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.update_display()
        
        # 在新窗口显示单通道
        self.show_channels_window([gray], ['Gray'])
    
    def convert_to_hsv(self):
        """RGB转HSV"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 处理4通道RGBA图像，转为3通道BGR
        image = self.original_image
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        hsv = ImageProcessor.rgb_to_hsv_manual(image)
        
        # 显示HSV图
        self.processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.update_display()
        
        # 分离并显示通道
        h, s, v = cv2.split(hsv)
        self.show_channels_window([h, s, v], ['H (色调)', 'S (饱和度)', 'V (明度)'])
    
    def show_channels_window(self, channels, titles):
        """显示通道窗口"""
        dialog = ChannelsWindow(channels, titles, self)
        dialog.show_window()
    
    # ==================== 任务3：图像读写测试 ====================
    
    def setup_readwrite_controls(self):
        """设置读写测试控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        # RGB信息显示
        info_group = QGroupBox('像素信息')
        info_layout = QVBoxLayout()
        
        self.pixel_info_label = QLabel('鼠标位置: -\nRGB: -')
        self.pixel_info_label.setStyleSheet('border: none;')
        info_layout.addWidget(self.pixel_info_label)
        
        # RGB修改
        modify_layout = QGridLayout()
        modify_layout.addWidget(QLabel('X:'), 0, 0)
        self.modify_x = QSpinBox()
        self.modify_x.setRange(0, 5000)
        modify_layout.addWidget(self.modify_x, 0, 1)
        
        modify_layout.addWidget(QLabel('Y:'), 1, 0)
        self.modify_y = QSpinBox()
        self.modify_y.setRange(0, 5000)
        modify_layout.addWidget(self.modify_y, 1, 1)
        
        modify_layout.addWidget(QLabel('R:'), 2, 0)
        self.modify_r = QSpinBox()
        self.modify_r.setRange(0, 255)
        modify_layout.addWidget(self.modify_r, 2, 1)
        
        modify_layout.addWidget(QLabel('G:'), 3, 0)
        self.modify_g = QSpinBox()
        self.modify_g.setRange(0, 255)
        modify_layout.addWidget(self.modify_g, 3, 1)
        
        modify_layout.addWidget(QLabel('B:'), 4, 0)
        self.modify_b = QSpinBox()
        self.modify_b.setRange(0, 255)
        modify_layout.addWidget(self.modify_b, 4, 1)
        
        modify_btn = QPushButton('修改像素')
        modify_btn.clicked.connect(self.modify_pixel)
        modify_layout.addWidget(modify_btn, 5, 0, 1, 2)
        
        info_layout.addLayout(modify_layout)
        info_group.setLayout(info_layout)
        
        # 绘制控制
        draw_group = QGroupBox('绘制图形')
        draw_layout = QVBoxLayout()
        
        rect_btn = QPushButton('绘制矩形')
        rect_btn.clicked.connect(lambda: self.set_drawing_mode('rectangle'))
        
        circle_btn = QPushButton('绘制圆形')
        circle_btn.clicked.connect(lambda: self.set_drawing_mode('circle'))
        
        triangle_btn = QPushButton('绘制三角形')
        triangle_btn.clicked.connect(lambda: self.set_drawing_mode('triangle'))
        
        undo_btn = QPushButton('撤回')
        undo_btn.clicked.connect(self.undo_drawing)
        
        draw_layout.addWidget(rect_btn)
        draw_layout.addWidget(circle_btn)
        draw_layout.addWidget(triangle_btn)
        draw_layout.addWidget(undo_btn)
        draw_group.setLayout(draw_layout)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('图像读写'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(info_group)
        self.controls_layout.addWidget(draw_group)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def on_mouse_moved(self, pos, pixel_pos):
        """鼠标移动事件"""
        if hasattr(self, 'pixel_info_label') and self.original_image is not None:
            if 0 <= pixel_pos.x() < self.original_image.shape[1] and 0 <= pixel_pos.y() < self.original_image.shape[0]:
                pixel_value = self.original_image[pixel_pos.y(), pixel_pos.x()]
                # 处理灰度图、彩色图4通道图
                if len(self.original_image.shape) == 2 or isinstance(pixel_value, (int, np.integer)):
                    # 灰度图
                    gray = int(pixel_value)
                    self.pixel_info_label.setText(
                        f'鼠标位置: ({pixel_pos.x()}, {pixel_pos.y()})\n'
                        f'灰度值: {gray}'
                    )
                elif len(pixel_value) == 4:
                    # 4通道RGBA图像
                    b, g, r, a = pixel_value
                    self.pixel_info_label.setText(
                        f'鼠标位置: ({pixel_pos.x()}, {pixel_pos.y()})\n'
                        f'RGBA: ({r}, {g}, {b}, {a})'
                    )
                else:
                    # 3通道BGR图像
                    b, g, r = pixel_value
                    self.pixel_info_label.setText(
                        f'鼠标位置: ({pixel_pos.x()}, {pixel_pos.y()})\n'
                        f'RGB: ({r}, {g}, {b})'
                    )
    
    def on_mouse_clicked(self, pos, pixel_pos):
        """鼠标点击事件"""
        if self.drawing_mode and self.original_image is not None:
            if 0 <= pixel_pos.x() < self.original_image.shape[1] and 0 <= pixel_pos.y() < self.original_image.shape[0]:
                self.drawing_points.append((pixel_pos.x(), pixel_pos.y()))
                self.process_drawing()
        
        # 图像校正点击（4个点）
        if self.current_task == 6 and self.original_image is not None:
            if 0 <= pixel_pos.x() < self.original_image.shape[1] and 0 <= pixel_pos.y() < self.original_image.shape[0]:
                self.correction_points.append((pixel_pos.x(), pixel_pos.y()))
                # 更新点数显示
                if hasattr(self, 'points_status_label'):
                    self.points_status_label.setText(f'已选择: {len(self.correction_points)}/4 个点')
                if len(self.correction_points) == 4:
                    self.apply_correction()
    
    def set_drawing_mode(self, mode):
        """设置绘制模式"""
        self.drawing_mode = mode
        self.drawing_points = []
        QMessageBox.information(self, '提示', f'请在图像上点击以绘制{mode}')
    
    def process_drawing(self):
        """处理绘制"""
        if not self.drawing_mode:
            return
        
        if self.drawing_mode == 'rectangle' and len(self.drawing_points) == 2:
            self.draw_rectangle()
        elif self.drawing_mode == 'circle' and len(self.drawing_points) == 2:
            self.draw_circle()
        elif self.drawing_mode == 'triangle' and len(self.drawing_points) == 3:
            self.draw_triangle()
    
    def draw_rectangle(self):
        """绘制矩形"""
        if self.processed_image is None:
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
            else:
                return
        
        pt1, pt2 = self.drawing_points
        cv2.rectangle(self.processed_image, pt1, pt2, (0, 255, 0), 2)
        self.shapes_drawn.append(('rectangle', self.drawing_points.copy()))
        self.drawing_points = []
        self.drawing_mode = None
        self.update_display()
    
    def draw_circle(self):
        """绘制圆形"""
        if self.processed_image is None:
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
            else:
                return
        
        center, edge = self.drawing_points
        radius = int(np.sqrt((center[0] - edge[0])**2 + (center[1] - edge[1])**2))
        cv2.circle(self.processed_image, center, radius, (0, 255, 0), 2)
        self.shapes_drawn.append(('circle', self.drawing_points.copy()))
        self.drawing_points = []
        self.drawing_mode = None
        self.update_display()
    
    def draw_triangle(self):
        """绘制三角形"""
        if self.processed_image is None:
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
            else:
                return
        
        pts = np.array(self.drawing_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.processed_image, [pts], True, (0, 255, 0), 2)
        self.shapes_drawn.append(('triangle', self.drawing_points.copy()))
        self.drawing_points = []
        self.drawing_mode = None
        self.update_display()
    
    def undo_drawing(self):
        """撤回绘制"""
        if self.shapes_drawn:
            self.shapes_drawn.pop()
            self.redraw_all_shapes()
    
    def redraw_all_shapes(self):
        """重新绘制所有图形"""
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        for shape_type, points in self.shapes_drawn:
            if shape_type == 'rectangle':
                cv2.rectangle(self.processed_image, points[0], points[1], (0, 255, 0), 2)
            elif shape_type == 'circle':
                center, edge = points
                radius = int(np.sqrt((center[0] - edge[0])**2 + (center[1] - edge[1])**2))
                cv2.circle(self.processed_image, center, radius, (0, 255, 0), 2)
            elif shape_type == 'triangle':
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.processed_image, [pts], True, (0, 255, 0), 2)
        self.update_display()
    
    def modify_pixel(self):
        """修改像素值"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        if self.processed_image is None:
            self.processed_image = self.original_image.copy()
        
        x = self.modify_x.value()
        y = self.modify_y.value()
        r = self.modify_r.value()
        g = self.modify_g.value()
        b = self.modify_b.value()
        
        if 0 <= x < self.processed_image.shape[1] and 0 <= y < self.processed_image.shape[0]:
            # 处理4通道图像
            if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 4:
                self.processed_image[y, x] = [b, g, r, 255]  # 保持alpha通道
            else:
                self.processed_image[y, x] = [b, g, r]
            self.update_display()
            QMessageBox.information(self, '成功', f'已修改像素({x}, {y})')
        else:
            QMessageBox.warning(self, '警告', '坐标超出图像范围')
    
    # ==================== 任务4：图片融合 ====================
    
    def setup_blend_controls(self):
        """设置融合控制（只保留透明度融合）"""
        load_btn1 = QPushButton('加载图片1')
        load_btn1.clicked.connect(self.load_image_for_task)
        
        load_btn2 = QPushButton('加载图片2')
        load_btn2.clicked.connect(self.load_second_image)
        
        # 透明度融合
        blend_group = QGroupBox('透明度融合')
        blend_layout = QVBoxLayout()
        
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel('透明度:'))
        self.alpha_input = QLineEdit('0.5')
        alpha_layout.addWidget(self.alpha_input)
        
        blend_btn = QPushButton('确认融合')
        blend_btn.clicked.connect(self.blend_images)
        
        blend_layout.addLayout(alpha_layout)
        blend_layout.addWidget(blend_btn)
        blend_group.setLayout(blend_layout)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('图像融合'))
        
        self.controls_layout.addWidget(load_btn1)
        self.controls_layout.addWidget(load_btn2)
        self.controls_layout.addWidget(blend_group)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def load_second_image(self):
        """加载第二张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '加载图片2', '', 
            'Images (*.png *.jpg *.bmp *.jpeg)'
        )
        if file_path:
            self.second_image = imread_chinese(file_path)
            if self.second_image is None:
                QMessageBox.warning(self, '错误', '无法读取图片，请检查文件路径')
            else:
                QMessageBox.information(self, '成功', '图片2已加载')
    
    def blend_images(self):
        """透明度融合"""
        if self.original_image is None or not hasattr(self, 'second_image') or self.second_image is None:
            QMessageBox.warning(self, '警告', '请先加载两张图片')
            return
        
        try:
            alpha = float(self.alpha_input.text())
            if not 0 <= alpha <= 1:
                raise ValueError
        except:
            QMessageBox.warning(self, '警告', '透明度必须在0-1之间')
            return
        
        self.processed_image = ImageProcessor.blend_images(
            self.original_image, self.second_image, alpha
        )
        self.update_display()
    
    # ==================== 任务5：颜色阈值抠图 ====================
    
    def setup_color_threshold_controls(self):
        """设置颜色阈值抠图控制"""
        load_btn1 = QPushButton('加载前景图片')
        load_btn1.clicked.connect(self.load_image_for_task)
        
        load_btn2 = QPushButton('加载背景图片')
        load_btn2.clicked.connect(self.load_second_image)
        
        # 颜色空间选择
        color_space_layout = QHBoxLayout()
        color_space_layout.addWidget(QLabel('颜色空间:'))
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(['BGR', 'HSV'])
        color_space_layout.addWidget(self.color_space_combo)
        
        # 下限
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(QLabel('下限:'))
        self.lower_1 = QSpinBox()
        self.lower_1.setRange(0, 255)
        self.lower_2 = QSpinBox()
        self.lower_2.setRange(0, 255)
        self.lower_3 = QSpinBox()
        self.lower_3.setRange(0, 255)
        lower_layout.addWidget(self.lower_1)
        lower_layout.addWidget(self.lower_2)
        lower_layout.addWidget(self.lower_3)
        
        # 上限
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(QLabel('上限:'))
        self.upper_1 = QSpinBox()
        self.upper_1.setRange(0, 255)
        self.upper_1.setValue(255)
        self.upper_2 = QSpinBox()
        self.upper_2.setRange(0, 255)
        self.upper_2.setValue(255)
        self.upper_3 = QSpinBox()
        self.upper_3.setRange(0, 255)
        self.upper_3.setValue(255)
        upper_layout.addWidget(self.upper_1)
        upper_layout.addWidget(self.upper_2)
        upper_layout.addWidget(self.upper_3)
        
        threshold_btn = QPushButton('生成掩码')
        threshold_btn.clicked.connect(self.generate_color_mask)
        
        blend_btn = QPushButton('应用抠图融合')
        blend_btn.clicked.connect(self.threshold_blend)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('颜色阈值抠图'))
        
        self.controls_layout.addWidget(load_btn1)
        self.controls_layout.addWidget(load_btn2)
        self.controls_layout.addLayout(color_space_layout)
        self.controls_layout.addLayout(lower_layout)
        self.controls_layout.addLayout(upper_layout)
        self.controls_layout.addWidget(threshold_btn)
        self.controls_layout.addWidget(blend_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def generate_color_mask(self):
        """生成颜色阈值掩码"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载前景图片')
            return
        
        lower = np.array([self.lower_1.value(), self.lower_2.value(), self.lower_3.value()])
        upper = np.array([self.upper_1.value(), self.upper_2.value(), self.upper_3.value()])
        
        color_space = self.color_space_combo.currentText()
        
        # 生成掩码
        mask = ImageProcessor.color_threshold_mask(self.original_image, lower, upper, color_space)
        
        # 显示掩码（转换为3通道显示）
        self.processed_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.update_display()
        
        QMessageBox.information(self, '成功', '掩码已生成，白色区域为选中区域')
    
    def threshold_blend(self):
        """颜色阈值抠图融合"""
        if self.original_image is None or not hasattr(self, 'second_image') or self.second_image is None:
            QMessageBox.warning(self, '警告', '请先加载两张图片')
            return
        
        lower = np.array([self.lower_1.value(), self.lower_2.value(), self.lower_3.value()])
        upper = np.array([self.upper_1.value(), self.upper_2.value(), self.upper_3.value()])
        
        color_space = self.color_space_combo.currentText()
        
        # 生成掩码
        mask = ImageProcessor.color_threshold_mask(self.original_image, lower, upper, color_space)
        
        # 调整第二张图片大小
        img2 = ImageProcessor.resize_and_center(self.second_image, self.original_image.shape[:2])
        
        # 确保所有图像都是3通道
        img1 = self.original_image
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        if len(img2.shape) == 3 and img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        
        # 应用掩码融合
        # 将mask转换为3通道并归一化为float
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        # 将图像转换为float进行计算，避免溢出
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        # 融合计算
        self.processed_image = (img1_float * mask_3channel + img2_float * (1 - mask_3channel)).astype(np.uint8)
        
        self.update_display()
    
    # ==================== 任务5：图像校正 ====================
    
    def setup_correction_controls(self):
        """设置校正控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        info_label = QLabel('请按顺序点击图像上值斜矩形的4个角点：\n1.左上角 2.右上角 3.右下角 4.左下角\n系统将自动进行透视变换校正')
        info_label.setWordWrap(True)
        info_label.setStyleSheet('border: none; padding: 5px;')
                
        # 显示当前选择的点数
        self.points_status_label = QLabel('已选择: 0/4 个点')
        self.points_status_label.setStyleSheet('border: none; color: #0969da; font-weight: bold;')
        
        reset_btn = QPushButton('重置点')
        reset_btn.clicked.connect(self.reset_correction_points)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('图像校正'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(info_label)
        self.controls_layout.addWidget(self.points_status_label)
        self.controls_layout.addWidget(reset_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
        
        self.correction_points = []
    
    def reset_correction_points(self):
        """重置校正点"""
        self.correction_points = []
        if hasattr(self, 'points_status_label'):
            self.points_status_label.setText('已选择: 0/4 个点')
        QMessageBox.information(self, '提示', '已重置，请按顺序选择4个点（左上、右上、右下、左下）')
    
    def apply_correction(self):
        """应用校正"""
        if len(self.correction_points) != 4:
            return
        
        # 四个点定义倾斜矩形：左上、右上、右下、左下
        pts_src = self.correction_points
        
        self.processed_image = ImageProcessor.correct_perspective(self.original_image, pts_src)
        self.update_display()
        
        QMessageBox.information(self, '成功', '图像校正完成')
        self.correction_points = []
        if hasattr(self, 'points_status_label'):
            self.points_status_label.setText('已选择: 0/4 个点')
    
    # ==================== 任务7：布匹裁剪分割线识别 ====================
    
    def setup_fabric_cut_controls(self):
        """设置布匹裁剪分割线识别控制"""
        load_btn = QPushButton('加载布匹图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        info_label = QLabel('系统将基于颜色差异分析，\n自动识别褥皱区域并标记最佳裁剪位置\n（红色线为主分割线，黄色线为辅助参考）')
        info_label.setWordWrap(True)
        info_label.setStyleSheet('border: none; padding: 5px;')
        
        detect_btn = QPushButton('识别分割线')
        detect_btn.clicked.connect(self.detect_fabric_cut_line)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('布匹裁剪分割'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(info_label)
        self.controls_layout.addWidget(detect_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def detect_fabric_cut_line(self):
        """识别布匹裁剪分割线"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载布匹图片')
            return
        
        self.processed_image = ImageProcessor.detect_fabric_cut_line(self.original_image)
        self.update_display()
        
        QMessageBox.information(self, '成功', '分割线识别完成！\n红色线：主分割线（推荐裁剪位置）\n黄色线：辅助参考线（颜色变化点）')
    
    # ==================== 通用方法 ====================
    
    def load_image_for_task(self):
        """为任务加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '加载图片', '', 
            'Images (*.png *.jpg *.bmp *.jpeg)'
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """加载图片"""
        self.original_image = imread_chinese(file_path)
        if self.original_image is None:
            QMessageBox.warning(self, '错误', '无法读取图片，请检查文件路径')
            return
        self.processed_image = None
        self.shapes_drawn = []
        self.update_display()
        
    
    def update_display(self):
        """更新显示"""
        if self.original_image is not None:
            pixmap = get_display_pixmap(self.original_image, 1000, 380)
            if pixmap:
                self.original_display.setPixmap(pixmap)
                self.original_display.original_image = self.original_image
        
        if self.processed_image is not None:
            pixmap = get_display_pixmap(self.processed_image, 1000, 380)
            if pixmap:
                self.processed_display.setPixmap(pixmap)
    
    def save_processed_image(self):
        """保存处理后的图片"""
        if self.processed_image is None:
            QMessageBox.warning(self, '警告', '没有可保存的图片')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, '保存图片', '', 
            'Images (*.png *.jpg *.bmp)'
        )
        if file_path:
            if imwrite_chinese(file_path, self.processed_image):
                QMessageBox.information(self, '成功', f'图片已保存到: {file_path}')
            else:
                QMessageBox.warning(self, '错误', '图片保存失败')
    
    def show_algorithm(self, algorithm_name):
        """显示算法窗口"""
        dialog = AlgorithmWindow(algorithm_name, self)
        dialog.show_window()


class ImageDisplayLabel(QLabel):
    """支持鼠标交互的图像显示标签"""
    from PyQt5.QtCore import pyqtSignal
    
    mouse_moved = pyqtSignal(QPoint, QPoint)  # 显示位置，实际像素位置
    mouse_clicked = pyqtSignal(QPoint, QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.original_image: Optional[np.ndarray] = None
    
    def mouseMoveEvent(self, ev: Optional[QMouseEvent]) -> None:
        """鼠标移动事件"""
        if ev and self.pixmap() and self.original_image is not None:
            pixel_pos = self.map_to_image(ev.pos())
            self.mouse_moved.emit(ev.pos(), pixel_pos)
    
    def mousePressEvent(self, ev: Optional[QMouseEvent]) -> None:
        """鼠标按下事件"""
        if ev and self.pixmap() and self.original_image is not None:
            pixel_pos = self.map_to_image(ev.pos())
            self.mouse_clicked.emit(ev.pos(), pixel_pos)
    
    def map_to_image(self, pos):
        """将显示坐标映射到图像坐标"""
        pixmap = self.pixmap()
        if not pixmap or self.original_image is None:
            return QPoint(0, 0)
        
        # 获取显示的pixmap大小和位置
        label_rect = self.rect()
        
        # 计算pixmap在label中的位置（居中）
        x_offset = (label_rect.width() - pixmap.width()) // 2
        y_offset = (label_rect.height() - pixmap.height()) // 2
        
        # 转换为pixmap坐标
        pixmap_x = pos.x() - x_offset
        pixmap_y = pos.y() - y_offset
        
        # 计算缩放比例
        scale_x = self.original_image.shape[1] / pixmap.width() if pixmap.width() > 0 else 1
        scale_y = self.original_image.shape[0] / pixmap.height() if pixmap.height() > 0 else 1
        
        # 转换为原始图像坐标
        image_x = int(pixmap_x * scale_x)
        image_y = int(pixmap_y * scale_y)
        
        return QPoint(image_x, image_y)


class ChannelsWindow(QWidget):
    """通道显示窗口"""
    
    def __init__(self, channels, titles, parent=None):
        super().__init__(parent)
        self.setWindowTitle('通道显示')
        self.setStyleSheet(get_style())
        self.setWindowFlags(Qt.WindowType.Window)  # 设置为窗口
        
        layout = QHBoxLayout(self)
        
        for channel, title in zip(channels, titles):
            channel_widget = QWidget()
            channel_layout = QVBoxLayout(channel_widget)
            
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setObjectName('subtitle')
            
            image_label = QLabel()
            image_label.setFixedSize(300, 300)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet(f"""
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            """)
            
            # 转换并显示通道
            if len(channel.shape) == 2:
                channel_bgr = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            else:
                channel_bgr = channel
            
            pixmap = get_display_pixmap(channel_bgr, 300, 300)
            if pixmap:
                image_label.setPixmap(pixmap)
            
            channel_layout.addWidget(label)
            channel_layout.addWidget(image_label)
            
            layout.addWidget(channel_widget)
    
    def show_window(self):
        """显示窗口"""
        self.show()


class AlgorithmWindow(QWidget):
    """算法显示窗口"""
    
    def __init__(self, algorithm_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{algorithm_name} - 算法原理')
        self.setStyleSheet(get_style())
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel(f'{algorithm_name}算法原理')
        title_label.setObjectName('subtitle')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 算法内容
        content_text = QTextEdit()
        content_text.setReadOnly(True)
        content_text.setStyleSheet('''
            QTextEdit {
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11pt;
                line-height: 1.6;
                padding: 15px;
            }
        ''')
        content_text.setPlainText(self.get_algorithm_content(algorithm_name))
        
        layout.addWidget(title_label)
        layout.addWidget(content_text)
    
    def get_algorithm_content(self, name):
        """获取算法内容"""
        algorithms = {
            '摄像头调用': '''
【核心技术】
OpenCV VideoCapture + PyQt5 线程

【实现流程】
1. 初始化摄像头
   cap = cv2.VideoCapture(0)  # 0表示默认摄像头

2. 创建独立线程读取帧
   class CameraThread(QThread):
       def run(self):
           while self.running:
               ret, frame = self.cap.read()
               if ret:
                   self.frame_ready.emit(frame)

3. 信号槽机制更新UI
   frame_ready.connect(update_display)

4. 暂停/继续控制
   使用标志位 paused 控制是否发送帧

5. 关闭摄像头
   cap.release()
   thread.quit()

【关键代码】
ret, frame = cap.read()
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emit(frame_rgb)
''',
            'RGB转Gray': '''
【算法原理】
灰度值 = 0.299*R + 0.587*G + 0.114*B
（基于人眼对不同颜色的敏感度）

【实现步骤】
1. 提取RGB三通道
   b, g, r = cv2.split(image)

2. 手动加权计算
   gray = 0.114*b + 0.587*g + 0.299*r

3. 类型转换
   gray = gray.astype(np.uint8)

【核心代码】
def rgb_to_gray_manual(image):
    if len(image.shape) == 2:
        return image
    
    b, g, r = cv2.split(image)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return gray.astype(np.uint8)
''',
            'RGB转HSV': '''
【算法原理】
HSV = (Hue色调, Saturation饱和度, Value明度)

【计算公式】
1. 归一化RGB到[0,1]
   r, g, b = R/255, G/255, B/255

2. 计算最大值、最小值、差值
   cmax = max(r, g, b)
   cmin = min(r, g, b)
   delta = cmax - cmin

3. 计算H（色调）
   if delta == 0:
       h = 0
   elif cmax == r:
       h = 60 * (((g - b) / delta) % 6)
   elif cmax == g:
       h = 60 * (((b - r) / delta) + 2)
   else:
       h = 60 * (((r - g) / delta) + 4)

4. 计算S（饱和度）
   s = 0 if cmax == 0 else delta / cmax

5. 计算V（明度）
   v = cmax

6. 转换到OpenCV范围
   H: [0, 180], S: [0, 255], V: [0, 255]
''',
            '图像读写': '''
【功能说明】
像素读取、修改和图形绘制

【像素读取】
1. 获取像素值
   pixel = image[y, x]  # 注意：y在前，x在后
   
2. BGR到RGB转换
   b, g, r = pixel
   RGB = (r, g, b)

【像素修改】
1. 直接赋值
   image[y, x] = [b, g, r]

2. 处理4通道图像
   image[y, x] = [b, g, r, alpha]

【图形绘制】
1. 绘制矩形
   cv2.rectangle(img, pt1, pt2, color, thickness)

2. 绘制圆形
   cv2.circle(img, center, radius, color, thickness)

3. 绘制多边形
   pts = np.array(points, np.int32)
   cv2.polylines(img, [pts], isClosed, color, thickness)

【核心代码】
# 读取
pixel_value = image[y, x]

# 修改
image[y, x] = [new_b, new_g, new_r]

# 绘制
cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
''',
            '图像融合': '''
【算法原理】
Alpha融合: result = α*img1 + (1-α)*img2

【实现步骤】
1. 尺寸对齐
   将小图像缩放到大图像尺寸
   resize_and_center(img2, img1.shape)

2. 通道数匹配
   如果通道数不同，转换为3通道BGR

3. 类型转换为float32
   img1_f = img1.astype(np.float32)
   img2_f = img2.astype(np.float32)

4. 加权混合
   result = alpha * img1_f + (1 - alpha) * img2_f

5. 转回uint8
   result = result.astype(np.uint8)

【核心代码】
cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
''',
            '颜色阈值抠图': '''
【算法原理】
基于颜色范围创建掩码，实现图像融合

【实现步骤】
1. 颜色空间转换（可选BGR/HSV）
   if color_space == "HSV":
       img_space = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

2. 创建二值掩码
   mask = (img_space >= lower) & (img_space <= upper)
   mask = np.all(mask, axis=2).astype(np.uint8) * 255

3. 掩码融合
   mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
   result = img1 * mask_3ch + img2 * (1 - mask_3ch)

【应用场景】
- 绿幕抠图（HSV空间选择绿色范围）
- 特定颜色区域替换
- 背景替换
''',
            '图像校正': '''
【算法原理】
透视变换 (Perspective Transform)

【数学原理】
通过3×3透视变换矩阵M，将值斜四边形映射为矩形

【实现步骤】
1. 获取四个源点（值斜矩形的四个角）
   pts_src = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
   顺序：左上、右上、右下、左下

2. 计算目标矩形尺寸
   width = max(上边长度, 下边长度)
   height = max(左边长度, 右边长度)

3. 定义目标点（标准矩形）
   pts_dst = [(0,0), (w,0), (w,h), (0,h)]

4. 计算透视变换矩阵
   M = cv2.getPerspectiveTransform(pts_src, pts_dst)

5. 应用变换
   warped = cv2.warpPerspective(img, M, (width, height))

【核心代码】
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
result = cv2.warpPerspective(image, M, (w, h))
''',
            '布匹裁剪分割': '''
【算法原理】
基于颜色差异检测分割线

【实现步骤】
1. 灰度转换
   gray = rgb_to_gray(image)

2. 计算每列的平均灰度值
   column_means = np.mean(gray, axis=0)

3. 高斯平滑降噪
   kernel_size = max(5, width // 100)
   smoothed = GaussianBlur(column_means)

4. 计算梯度（颜色突变）
   gradient = np.abs(np.gradient(smoothed))

5. 自适应阈值
   threshold = mean(gradient) + 1.5 * std(gradient)

6. 找到显著变化点
   changes = gradient > threshold

7. 定位主分割线
   - 单个变化点：直接使用
   - 多个变化点：选择最大间隔的中点

【适用场景】
- 布匹褥皱检测
- 色差区域分割
- 拼接线识别
'''
        }
        
        return algorithms.get(name, '暂无算法说明')
    
    def show_window(self):
        """显示窗口"""
        self.show()
