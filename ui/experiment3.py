"""
实验三面板：高级图像处理
包含：Hough变换检测、傅里叶变换、缺陷检测、划痕检测、PCB检测
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox,
                             QGridLayout, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from .styles import get_style, COLORS
from core.image_processor import ImageProcessor
from utils.helpers import numpy_to_qpixmap, get_display_pixmap, imread_chinese, imwrite_chinese


class ImageDisplayLabel(QLabel):
    """图像显示标签"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)


class Experiment3Panel(QWidget):
    """实验三主面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.original_image = None
        self.processed_image = None
        self.current_task = 0
        
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
            1: ('Hough变换检测', self.setup_hough_controls),
            2: ('傅里叶变换', self.setup_fourier_controls),
            3: ('缺陷检测', self.setup_defect_controls),
            4: ('划痕检测', self.setup_scratch_controls),
            5: ('PCB检测', self.setup_pcb_controls),
        }
        
        if task_index in tasks:
            title, setup_func = tasks[task_index]
            self.task_label.setText(title)
            setup_func()
    
    def clear_controls(self):
        """清空控制面板"""
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
    
    def update_display(self):
        """更新图像显示"""
        if self.original_image is not None:
            size = self.original_display.size()
            pixmap = get_display_pixmap(self.original_image, size.width(), size.height())
            self.original_display.setPixmap(pixmap)
        
        if self.processed_image is not None:
            size = self.processed_display.size()
            pixmap = get_display_pixmap(self.processed_image, size.width(), size.height())
            self.processed_display.setPixmap(pixmap)
    
    def save_processed_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            QMessageBox.warning(self, '警告', '没有处理后的图像可保存')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, '保存图像', '', 'Images (*.png *.jpg *.bmp)'
        )
        
        if file_path:
            success = imwrite_chinese(file_path, self.processed_image)
            if success:
                QMessageBox.information(self, '成功', '图像已保存')
            else:
                QMessageBox.warning(self, '错误', '图像保存失败')
    
    # ==================== 任务1：Hough变换检测 ====================
    
    def setup_hough_controls(self):
        """设置Hough变换控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        line_btn = QPushButton('检测直线')
        line_btn.clicked.connect(self.detect_lines)
        
        circle_btn = QPushButton('检测圆形')
        circle_btn.clicked.connect(self.detect_circles)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        # Canny阈值
        params_layout.addWidget(QLabel('Canny低阈值:'), 0, 0)
        self.canny_low = QSpinBox()
        self.canny_low.setRange(0, 255)
        self.canny_low.setValue(50)
        params_layout.addWidget(self.canny_low, 0, 1)
        
        params_layout.addWidget(QLabel('Canny高阈值:'), 1, 0)
        self.canny_high = QSpinBox()
        self.canny_high.setRange(0, 255)
        self.canny_high.setValue(150)
        params_layout.addWidget(self.canny_high, 1, 1)
        
        # 直线检测参数
        params_layout.addWidget(QLabel('直线阈值:'), 2, 0)
        self.line_threshold = QSpinBox()
        self.line_threshold.setRange(1, 500)
        self.line_threshold.setValue(100)
        params_layout.addWidget(self.line_threshold, 2, 1)
        
        # 圆检测参数
        params_layout.addWidget(QLabel('圆最小半径:'), 3, 0)
        self.circle_min_radius = QSpinBox()
        self.circle_min_radius.setRange(1, 500)
        self.circle_min_radius.setValue(10)
        params_layout.addWidget(self.circle_min_radius, 3, 1)
        
        params_layout.addWidget(QLabel('圆最大半径:'), 4, 0)
        self.circle_max_radius = QSpinBox()
        self.circle_max_radius.setRange(1, 1000)
        self.circle_max_radius.setValue(100)
        params_layout.addWidget(self.circle_max_radius, 4, 1)
        
        params_group.setLayout(params_layout)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(line_btn)
        self.controls_layout.addWidget(circle_btn)
    
    def detect_lines(self):
        """使用Hough变换检测直线"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        result = ImageProcessor.hough_lines(
            self.original_image,
            canny_low=self.canny_low.value(),
            canny_high=self.canny_high.value(),
            threshold=self.line_threshold.value()
        )
        
        self.processed_image = result
        self.update_display()
        QMessageBox.information(self, '完成', '直线检测完成')
    
    def detect_circles(self):
        """使用Hough变换检测圆形"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        result = ImageProcessor.hough_circles(
            self.original_image,
            min_radius=self.circle_min_radius.value(),
            max_radius=self.circle_max_radius.value()
        )
        
        self.processed_image = result
        self.update_display()
        QMessageBox.information(self, '完成', '圆形检测完成')
    
    # ==================== 任务2：傅里叶变换 ====================
    
    def setup_fourier_controls(self):
        """设置傅里叶变换控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        fft_btn = QPushButton('计算傅里叶变换')
        fft_btn.clicked.connect(self.compute_fourier)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(fft_btn)
    
    def compute_fourier(self):
        """计算傅里叶变换并展示幅度谱"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        magnitude_spectrum = ImageProcessor.fourier_transform(self.original_image)
        self.processed_image = magnitude_spectrum
        self.update_display()
        QMessageBox.information(self, '完成', '傅里叶变换计算完成')
    
    # ==================== 任务3：缺陷检测 ====================
    
    def setup_defect_controls(self):
        """设置缺陷检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('检测缺陷')
        detect_btn.clicked.connect(self.detect_defects)
        
        # 结果显示
        self.defect_result_text = QTextEdit()
        self.defect_result_text.setReadOnly(True)
        self.defect_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(detect_btn)
        self.controls_layout.addWidget(QLabel('检测结果:'))
        self.controls_layout.addWidget(self.defect_result_text)
    
    def detect_defects(self):
        """检测圆形中的缺陷和多余小块"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        result, defect_info = ImageProcessor.detect_circle_defects(self.original_image)
        self.processed_image = result
        self.update_display()
        
        # 显示检测结果
        info_text = f"检测到缺陷数量: {defect_info['defect_count']}\n"
        info_text += f"缺陷总面积: {defect_info['total_area']} 像素\n\n"
        info_text += "各缺陷详情:\n"
        for i, area in enumerate(defect_info['areas'], 1):
            info_text += f"缺陷 {i}: {area} 像素\n"
        
        self.defect_result_text.setText(info_text)
        QMessageBox.information(self, '完成', '缺陷检测完成')
    
    # ==================== 任务4：划痕检测 ====================
    
    def setup_scratch_controls(self):
        """设置划痕检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('检测划痕')
        detect_btn.clicked.connect(self.detect_scratches)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('形态学核大小:'), 0, 0)
        self.morph_kernel_size = QSpinBox()
        self.morph_kernel_size.setRange(3, 21)
        self.morph_kernel_size.setValue(5)
        self.morph_kernel_size.setSingleStep(2)
        params_layout.addWidget(self.morph_kernel_size, 0, 1)
        
        params_layout.addWidget(QLabel('阈值:'), 1, 0)
        self.scratch_threshold = QSpinBox()
        self.scratch_threshold.setRange(0, 255)
        self.scratch_threshold.setValue(30)
        params_layout.addWidget(self.scratch_threshold, 1, 1)
        
        params_group.setLayout(params_layout)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(detect_btn)
    
    def detect_scratches(self):
        """检测材料表面划痕"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        result = ImageProcessor.detect_scratches(
            self.original_image,
            kernel_size=self.morph_kernel_size.value(),
            threshold=self.scratch_threshold.value()
        )
        
        self.processed_image = result
        self.update_display()
        QMessageBox.information(self, '完成', '划痕检测完成')
    
    # ==================== 任务5：PCB检测 ====================
    
    def setup_pcb_controls(self):
        """设置PCB检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('检测PCB缺陷')
        detect_btn.clicked.connect(self.detect_pcb_defects)
        
        # 缺陷类型选择
        type_group = QGroupBox('检测类型')
        type_layout = QVBoxLayout()
        
        self.defect_type = QComboBox()
        self.defect_type.addItems(['全部缺陷', '毛刺', '短路', '断路'])
        type_layout.addWidget(self.defect_type)
        
        type_group.setLayout(type_layout)
        
        # 结果显示
        self.pcb_result_text = QTextEdit()
        self.pcb_result_text.setReadOnly(True)
        self.pcb_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(type_group)
        self.controls_layout.addWidget(detect_btn)
        self.controls_layout.addWidget(QLabel('检测结果:'))
        self.controls_layout.addWidget(self.pcb_result_text)
    
    def detect_pcb_defects(self):
        """检测PCB缺陷"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        defect_type = self.defect_type.currentText()
        result, defect_info = ImageProcessor.detect_pcb_defects(
            self.original_image, 
            defect_type=defect_type
        )
        
        self.processed_image = result
        self.update_display()
        
        # 显示检测结果
        info_text = f"检测类型: {defect_type}\n\n"
        for key, value in defect_info.items():
            info_text += f"{key}: {value}\n"
        
        self.pcb_result_text.setText(info_text)
        QMessageBox.information(self, '完成', 'PCB缺陷检测完成')
    
    # ==================== 通用功能 ====================
    
    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', 'Images (*.png *.jpg *.bmp *.jpeg)'
        )
        
        if file_path:
            self.original_image = imread_chinese(file_path)
            if self.original_image is not None:
                self.processed_image = None
                self.update_display()
                QMessageBox.information(self, '成功', '图片加载成功')
            else:
                QMessageBox.warning(self, '错误', '图片加载失败')
