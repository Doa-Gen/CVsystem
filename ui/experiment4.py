"""
实验四面板：深度学习与图像识别
包含：图像分类、目标检测、语义分割、风格迁移、图像生成
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox,
                             QGridLayout, QSpinBox, QTextEdit, QLineEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# 配置matplotlib中文显示
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass

from .styles import get_style, COLORS
from core.image_processor import ImageProcessor
from utils.helpers import numpy_to_qpixmap, get_display_pixmap, imread_chinese, imwrite_chinese
import time


class ImageDisplayLabel(QLabel):
    """图像显示标签"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)


class Experiment4Panel(QWidget):
    """实验四主面板"""
    
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
            1: ('Harris角点检测', self.setup_classification_controls),
            2: ('Shi-Tomasi角点检测', self.setup_detection_controls),
            3: ('SIFT特征点检测', self.setup_segmentation_controls),
            4: ('特征点匹配', self.setup_style_transfer_controls),
            5: ('全景图像拼接', self.setup_image_generation_controls),
            6: ('算法性能对比', self.setup_noise_analysis_controls),
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
    
    # ==================== 任务1：Harris角点检测 ====================
    
    def setup_classification_controls(self):
        """设置Harris角点检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('Harris角点检测')
        detect_btn.clicked.connect(self.harris_corner_detection)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('k值1:'), 0, 0)
        self.k1_value = QLineEdit('0.04')
        params_layout.addWidget(self.k1_value, 0, 1)
        
        params_layout.addWidget(QLabel('k值2:'), 1, 0)
        self.k2_value = QLineEdit('0.06')
        params_layout.addWidget(self.k2_value, 1, 1)
        
        params_group.setLayout(params_layout)
        
        # 结果显示
        self.harris_result_text = QTextEdit()
        self.harris_result_text.setReadOnly(True)
        self.harris_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(detect_btn)
        self.controls_layout.addWidget(QLabel('检测结果:'))
        self.controls_layout.addWidget(self.harris_result_text)
    
    def harris_corner_detection(self):
        """Harris角点检测"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 获取参数
        try:
            k1 = float(self.k1_value.text())
            k2 = float(self.k2_value.text())
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的k值')
            return
        
        # 执行Harris角点检测
        result_img1, corners1 = ImageProcessor.harris_corner_detection(self.original_image, k1)
        result_img2, corners2 = ImageProcessor.harris_corner_detection(self.original_image, k2)
        
        # 显示结果
        self.show_harris_results(self.original_image, result_img1, result_img2, corners1, corners2, k1, k2)
        
        # 更新结果文本
        result_text = f"Harris角点检测结果:\n\n"
        result_text += f"k值 {k1}: 检测到 {len(corners1)} 个角点\n"
        result_text += f"k值 {k2}: 检测到 {len(corners2)} 个角点\n\n"
        result_text += "分析:\n"
        result_text += "k值越大，角点判定越严格，检测到的角点数量越少。"
        
        self.harris_result_text.setText(result_text)
        QMessageBox.information(self, '完成', 'Harris角点检测完成')
    
    def show_harris_results(self, original_img, result_img1, result_img2, corners1, corners2, k1, k2):
        """显示Harris角点检测结果"""
        # 创建matplotlib图形
        fig = Figure(figsize=(12, 4))
        
        # 显示原图
        ax1 = fig.add_subplot(131)
        if len(original_img.shape) == 3:
            # BGR转RGB
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_rgb)
        else:
            ax1.imshow(original_img, cmap='gray')
        ax1.set_title('原图')
        ax1.axis('off')
        
        # 显示k1结果
        ax2 = fig.add_subplot(132)
        if len(result_img1.shape) == 3:
            # BGR转RGB
            result1_rgb = cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB)
            ax2.imshow(result1_rgb)
        else:
            ax2.imshow(result_img1, cmap='gray')
        ax2.set_title(f'k={k1}, 角点数={len(corners1)}')
        ax2.axis('off')
        
        # 显示k2结果
        ax3 = fig.add_subplot(133)
        if len(result_img2.shape) == 3:
            # BGR转RGB
            result2_rgb = cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB)
            ax3.imshow(result2_rgb)
        else:
            ax3.imshow(result_img2, cmap='gray')
        ax3.set_title(f'k={k2}, 角点数={len(corners2)}')
        ax3.axis('off')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('Harris角点检测结果')
        dialog.resize(1200, 400)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
    # ==================== 任务2：Shi-Tomasi角点检测 ====================
    
    def setup_detection_controls(self):
        """设置Shi-Tomasi角点检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('Shi-Tomasi角点检测')
        detect_btn.clicked.connect(self.shi_tomasi_detection)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('最大角点数:'), 0, 0)
        self.max_corners = QSpinBox()
        self.max_corners.setRange(1, 500)
        self.max_corners.setValue(100)
        params_layout.addWidget(self.max_corners, 0, 1)
        
        params_layout.addWidget(QLabel('质量等级:'), 1, 0)
        self.quality_level = QLineEdit('0.01')
        params_layout.addWidget(self.quality_level, 1, 1)
        
        params_layout.addWidget(QLabel('最小距离:'), 2, 0)
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 100)
        self.min_distance.setValue(10)
        params_layout.addWidget(self.min_distance, 2, 1)
        
        params_group.setLayout(params_layout)
        
        # 结果显示
        self.shi_tomasi_result_text = QTextEdit()
        self.shi_tomasi_result_text.setReadOnly(True)
        self.shi_tomasi_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(detect_btn)
        self.controls_layout.addWidget(QLabel('检测结果:'))
        self.controls_layout.addWidget(self.shi_tomasi_result_text)
    
    def shi_tomasi_detection(self):
        """Shi-Tomasi角点检测"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 获取参数
        try:
            max_corners = self.max_corners.value()
            quality_level = float(self.quality_level.text())
            min_distance = self.min_distance.value()
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的参数')
            return
        
        # 执行Shi-Tomasi角点检测
        basic_img, basic_corners = ImageProcessor.shi_tomasi_detection(self.original_image, max_corners, quality_level, min_distance)
        refined_img, refined_corners = ImageProcessor.shi_tomasi_subpix_refinement(self.original_image, basic_corners)
        
        # 计算坐标偏移量
        total_offset = 0.0
        max_offset = 0.0
        if len(basic_corners) > 0 and len(refined_corners) > 0:
            for i in range(min(len(basic_corners), len(refined_corners))):
                bx, by = basic_corners[i]
                rx, ry = refined_corners[i]
                offset = np.sqrt((bx - rx)**2 + (by - ry)**2)
                total_offset += offset
                max_offset = max(max_offset, offset)
            avg_offset = total_offset / len(basic_corners)
        else:
            avg_offset = 0.0
            max_offset = 0.0
        
        # 显示结果
        self.show_shi_tomasi_results(basic_img, refined_img, basic_corners, refined_corners)
        
        # 更新结果文本
        result_text = f"Shi-Tomasi角点检测结果:\n\n"
        result_text += f"基础检测: {len(basic_corners)} 个角点（整数像素精度）\n"
        result_text += f"亚像素优化: {len(refined_corners)} 个角点（亚像素精度）\n\n"
        result_text += f"坐标精度提升:\n"
        result_text += f"平均偏移量: {avg_offset:.4f} 像素\n"
        result_text += f"最大偏移量: {max_offset:.4f} 像素\n\n"
        result_text += "说明:\n"
        result_text += "亚像素优化不改变角点数量，而是将坐标精度\n"
        result_text += "从整数像素级提升到亚像素级（小数精度），\n"
        result_text += "适用于相机标定等高精度任务。"
        
        self.shi_tomasi_result_text.setText(result_text)
        QMessageBox.information(self, '完成', 'Shi-Tomasi角点检测完成')
    
    def show_shi_tomasi_results(self, basic_img, refined_img, basic_corners, refined_corners):
        """显示Shi-Tomasi角点检测结果"""
        # 创建matplotlib图形，增加一个对比图
        fig = Figure(figsize=(15, 5))
        
        # 显示基础检测结果
        ax1 = fig.add_subplot(131)
        if len(basic_img.shape) == 3:
            # BGR转RGB
            basic_rgb = cv2.cvtColor(basic_img, cv2.COLOR_BGR2RGB)
            ax1.imshow(basic_rgb)
        else:
            ax1.imshow(basic_img, cmap='gray')
        ax1.set_title(f'基础检测 ({len(basic_corners)} 个角点)\n整数像素精度')
        ax1.axis('off')
        
        # 显示亚像素优化结果
        ax2 = fig.add_subplot(132)
        if len(refined_img.shape) == 3:
            # BGR转RGB
            refined_rgb = cv2.cvtColor(refined_img, cv2.COLOR_BGR2RGB)
            ax2.imshow(refined_rgb)
        else:
            ax2.imshow(refined_img, cmap='gray')
        ax2.set_title(f'亚像素优化 ({len(refined_corners)} 个角点)\n亚像素精度')
        ax2.axis('off')
        
        # 显示坐标偏移对比图
        ax3 = fig.add_subplot(133)
        # 创建对比图像，同时显示两种角点
        if self.original_image is not None:
            compare_img = self.original_image.copy()
            if len(compare_img.shape) == 2:
                compare_img = cv2.cvtColor(compare_img, cv2.COLOR_GRAY2BGR)
            elif compare_img.shape[2] == 4:
                compare_img = cv2.cvtColor(compare_img, cv2.COLOR_BGRA2BGR)
            
            # 绘制基础检测角点（蓝色方框）和亚像素角点（红色圆点）
            for i in range(min(len(basic_corners), len(refined_corners))):
                bx, by = basic_corners[i]
                rx, ry = refined_corners[i]
                # 基础检测（蓝色方框）
                cv2.rectangle(compare_img, (int(bx)-3, int(by)-3), (int(bx)+3, int(by)+3), (255, 0, 0), 1)
                # 亚像素优化（红色圆点）
                cv2.circle(compare_img, (int(rx), int(ry)), 2, (0, 0, 255), -1)
                # 连线显示偏移
                cv2.line(compare_img, (int(bx), int(by)), (int(rx), int(ry)), (0, 255, 0), 1)
            
            compare_rgb = cv2.cvtColor(compare_img, cv2.COLOR_BGR2RGB)
            ax3.imshow(compare_rgb)
        ax3.set_title('坐标偏移对比\n蓝框=基础 红点=优化 绿线=偏移')
        ax3.axis('off')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('Shi-Tomasi角点检测结果')
        dialog.resize(1500, 500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
    # ==================== 任务3：SIFT特征点检测 ====================
    
    def setup_segmentation_controls(self):
        """设置SIFT特征点检测控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        detect_btn = QPushButton('SIFT特征点检测')
        detect_btn.clicked.connect(self.sift_detection)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QVBoxLayout()
        
        # 尺度变换选项
        scale_group = QGroupBox('尺度变换')
        scale_layout = QVBoxLayout()
        
        self.scale_original = QPushButton('原图')
        self.scale_original.clicked.connect(lambda: self.load_scaled_image(1.0))
        
        self.scale_half = QPushButton('缩小 50%')
        self.scale_half.clicked.connect(lambda: self.load_scaled_image(0.5))
        
        self.scale_double = QPushButton('放大 200%')
        self.scale_double.clicked.connect(lambda: self.load_scaled_image(2.0))
        
        scale_layout.addWidget(self.scale_original)
        scale_layout.addWidget(self.scale_half)
        scale_layout.addWidget(self.scale_double)
        scale_group.setLayout(scale_layout)
        
        params_layout.addWidget(scale_group)
        params_group.setLayout(params_layout)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(detect_btn)
    
    def load_scaled_image(self, scale):
        """加载尺度变换后的图像"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 获取原图尺寸
        height, width = self.original_image.shape[:2]
        
        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 缩放图像
        self.scaled_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # 更新显示
        self.processed_image = self.scaled_image
        self.update_display()
        
        QMessageBox.information(self, '完成', f'图像已缩放至 {new_width}x{new_height}')
    
    def sift_detection(self):
        """SIFT特征点检测"""
        # 确定要处理的图像
        if hasattr(self, 'scaled_image') and self.scaled_image is not None:
            image_to_process = self.scaled_image
        elif self.original_image is not None:
            image_to_process = self.original_image
        else:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 执行SIFT特征点检测
        result_img, keypoints_info = ImageProcessor.sift_detection(image_to_process)
        
        # 显示结果
        self.show_sift_results(result_img, keypoints_info)
        
        QMessageBox.information(self, '完成', 'SIFT特征点检测完成')
    
    def show_sift_results(self, result_img, keypoints_info):
        """显示SIFT特征点检测结果"""
        # 创建matplotlib图形
        fig = Figure(figsize=(8, 6))
        
        # 显示结果图像
        ax = fig.add_subplot(111)
        if len(result_img.shape) == 3:
            # BGR转RGB
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            ax.imshow(result_rgb)
        else:
            ax.imshow(result_img, cmap='gray')
        ax.set_title(f'SIFT特征点检测 ({len(keypoints_info)} 个特征点)')
        ax.axis('off')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('SIFT特征点检测结果')
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
    # ==================== 任务4：特征点匹配 ====================
    
    def setup_style_transfer_controls(self):
        """设置特征点匹配控制"""
        load_btn1 = QPushButton('加载图片1')
        load_btn1.clicked.connect(self.load_image)
        
        load_btn2 = QPushButton('加载图片2')
        load_btn2.clicked.connect(self.load_second_image)
        
        match_btn = QPushButton('特征点匹配')
        match_btn.clicked.connect(self.feature_matching)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('最近邻比率:'), 0, 0)
        self.nn_ratio = QLineEdit('0.8')
        params_layout.addWidget(self.nn_ratio, 0, 1)
        
        params_group.setLayout(params_layout)
        
        # 结果显示
        self.matching_result_text = QTextEdit()
        self.matching_result_text.setReadOnly(True)
        self.matching_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn1)
        self.controls_layout.addWidget(load_btn2)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(match_btn)
        self.controls_layout.addWidget(QLabel('匹配结果:'))
        self.controls_layout.addWidget(self.matching_result_text)
    
    def load_second_image(self):
        """加载第二张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择第二张图片', '', 'Images (*.png *.jpg *.bmp *.jpeg)'
        )
        
        if file_path:
            self.second_image = imread_chinese(file_path)
            if self.second_image is not None:
                QMessageBox.information(self, '成功', '第二张图片加载成功')
            else:
                QMessageBox.warning(self, '错误', '第二张图片加载失败')
    
    def feature_matching(self):
        """特征点匹配"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载第一张图片')
            return
        
        if not hasattr(self, 'second_image') or self.second_image is None:
            QMessageBox.warning(self, '警告', '请先加载第二张图片')
            return
        
        # 获取参数
        try:
            nn_ratio = float(self.nn_ratio.text())
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的最近邻比率')
            return
        
        # 执行特征点匹配
        result_img, match_count = ImageProcessor.feature_matching(self.original_image, self.second_image, nn_ratio)
        
        # 显示结果
        self.show_matching_results(result_img)
        
        # 更新结果文本
        result_text = f"特征点匹配结果:\n\n"
        result_text += f"有效匹配对数量: {match_count}\n\n"
        result_text += "分析:\n"
        result_text += "最近邻比率测试能有效剔除误匹配，提高匹配精度。"
        
        self.matching_result_text.setText(result_text)
        QMessageBox.information(self, '完成', '特征点匹配完成')
    
    def show_matching_results(self, result_img):
        """显示特征点匹配结果"""
        # 创建matplotlib图形
        fig = Figure(figsize=(12, 6))
        
        # 显示结果图像
        ax = fig.add_subplot(111)
        if len(result_img.shape) == 3:
            # BGR转RGB
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            ax.imshow(result_rgb)
        else:
            ax.imshow(result_img, cmap='gray')
        ax.set_title('特征点匹配结果')
        ax.axis('off')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('特征点匹配结果')
        dialog.resize(1200, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
    # ==================== 任务5：全景图像拼接 ====================
    
    def setup_image_generation_controls(self):
        """设置全景图像拼接控制"""
        load_btn1 = QPushButton('加载图片1')
        load_btn1.clicked.connect(self.load_image)
        
        load_btn2 = QPushButton('加载图片2')
        load_btn2.clicked.connect(self.load_second_image)
        
        stitch_btn = QPushButton('全景图像拼接')
        stitch_btn.clicked.connect(self.panoramic_stitching)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('RANSAC阈值:'), 0, 0)
        self.ransac_threshold = QLineEdit('5.0')
        params_layout.addWidget(self.ransac_threshold, 0, 1)
        
        params_layout.addWidget(QLabel('Alpha混合:'), 1, 0)
        self.alpha_blend = QLineEdit('0.5')
        params_layout.addWidget(self.alpha_blend, 1, 1)
        
        params_group.setLayout(params_layout)
        
        # 结果显示
        self.stitching_result_text = QTextEdit()
        self.stitching_result_text.setReadOnly(True)
        self.stitching_result_text.setMaximumHeight(150)
        
        self.controls_layout.addWidget(load_btn1)
        self.controls_layout.addWidget(load_btn2)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(stitch_btn)
        self.controls_layout.addWidget(QLabel('拼接结果:'))
        self.controls_layout.addWidget(self.stitching_result_text)
    
    def panoramic_stitching(self):
        """全景图像拼接"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载第一张图片')
            return
        
        if not hasattr(self, 'second_image') or self.second_image is None:
            QMessageBox.warning(self, '警告', '请先加载第二张图片')
            return
        
        # 获取参数
        try:
            ransac_threshold = float(self.ransac_threshold.text())
            alpha_blend = float(self.alpha_blend.text())
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的参数')
            return
        
        # 执行全景图像拼接
        result_img, stitching_info = ImageProcessor.panoramic_stitching(
            self.original_image, self.second_image, ransac_threshold, alpha_blend)
        
        # 显示结果
        self.show_stitching_results(result_img)
        
        # 更新结果文本
        result_text = f"全景图像拼接结果:\n\n"
        result_text += f"拼接状态: {stitching_info['status']}\n"
        result_text += f"匹配点数: {stitching_info['matches']}\n"
        result_text += f"内点数: {stitching_info['inliers']}\n\n"
        result_text += "分析:\n"
        result_text += "RANSAC算法能有效剔除异常值，提高拼接精度。"
        
        self.stitching_result_text.setText(result_text)
        QMessageBox.information(self, '完成', '全景图像拼接完成')
    
    def show_stitching_results(self, result_img):
        """显示全景图像拼接结果"""
        # 创建matplotlib图形
        fig = Figure(figsize=(10, 6))
        
        # 显示结果图像
        ax = fig.add_subplot(111)
        if len(result_img.shape) == 3:
            # BGR转RGB
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            ax.imshow(result_rgb)
        else:
            ax.imshow(result_img, cmap='gray')
        ax.set_title('全景图像拼接结果')
        ax.axis('off')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('全景图像拼接结果')
        dialog.resize(1000, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
    # ==================== 任务6：算法性能对比 ====================
    
    def setup_noise_analysis_controls(self):
        """设置算法性能对比控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image)
        
        analyze_btn = QPushButton('算法性能对比')
        analyze_btn.clicked.connect(self.algorithm_performance_comparison)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('噪声强度:'), 0, 0)
        self.noise_level = QLineEdit('0.1')
        params_layout.addWidget(self.noise_level, 0, 1)
        
        params_group.setLayout(params_layout)
        
        # 结果显示
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_result_text.setMaximumHeight(200)
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(analyze_btn)
        self.controls_layout.addWidget(QLabel('分析结果:'))
        self.controls_layout.addWidget(self.analysis_result_text)
    
    def algorithm_performance_comparison(self):
        """算法性能对比"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 获取参数
        try:
            noise_level = float(self.noise_level.text())
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的噪声强度')
            return
        
        # 添加噪声
        noisy_image = ImageProcessor.add_gaussian_noise(self.original_image, 0, int(noise_level * 255))
        
        # 执行性能对比
        comparison_results = ImageProcessor.algorithm_performance_comparison(noisy_image)
        
        # 显示结果
        self.show_analysis_results(comparison_results)
        
        # 更新结果文本
        result_text = "算法性能对比结果:\n\n"
        result_text += f"{'算法':<15} {'检测时间(ms)':<15} {'特征点数':<10} {'噪声鲁棒性':<15}\n"
        result_text += "-" * 55 + "\n"
        
        for algo, results in comparison_results.items():
            result_text += f"{algo:<15} {results['time']:<15.2f} {results['points']:<10} {results['robustness']:<15.2f}\n"
        
        result_text += "\n分析:\n"
        result_text += "- Harris: 对噪声敏感，但计算速度快\n"
        result_text += "- Shi-Tomasi: 精度高，适用于相机标定\n"
        result_text += "- SIFT: 尺度不变性好，适用于复杂场景\n"
        
        self.analysis_result_text.setText(result_text)
        QMessageBox.information(self, '完成', '算法性能对比完成')
    
    def show_analysis_results(self, comparison_results):
        """显示算法性能对比结果"""
        # 创建matplotlib图形
        fig = Figure(figsize=(10, 6))
        
        # 提取数据
        algorithms = list(comparison_results.keys())
        detection_times = [comparison_results[algo]['time'] for algo in algorithms]
        point_counts = [comparison_results[algo]['points'] for algo in algorithms]
        
        # 绘制检测时间对比
        ax1 = fig.add_subplot(121)
        bars1 = ax1.bar(algorithms, detection_times, color=['red', 'blue', 'green'])
        ax1.set_title('检测时间对比 (ms)')
        ax1.set_ylabel('时间 (ms)')
        
        # 在柱状图上添加数值标签
        for bar, time in zip(bars1, detection_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{time:.2f}', 
                    ha='center', va='bottom')
        
        # 绘制特征点数对比
        ax2 = fig.add_subplot(122)
        bars2 = ax2.bar(algorithms, point_counts, color=['red', 'blue', 'green'])
        ax2.set_title('特征点数对比')
        ax2.set_ylabel('特征点数')
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars2, point_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}', 
                    ha='center', va='bottom')
        
        # 调整布局
        fig.tight_layout()
        
        # 创建画布并显示
        canvas = FigureCanvasQTAgg(fig)
        
        # 创建新窗口显示结果
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dialog = QDialog(self)
        dialog.setWindowTitle('算法性能对比结果')
        dialog.resize(1000, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.exec_()
    
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