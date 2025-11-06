"""
实验二面板：图像增强
包含：直接灰度变换、图像直方图计算及均衡化、图像中值滤波、低通滤波、目标寻找
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QSpinBox, QGroupBox, QGridLayout, QScrollArea,
                             QListWidget, QListWidgetItem, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from .styles import get_style, COLORS
from core.image_processor import ImageProcessor
from utils.helpers import get_display_pixmap, imread_chinese, imwrite_chinese


class Experiment2Panel(QWidget):
    """实验二主面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.original_image = None
        self.processed_image = None
        self.current_task = 0
        
        # 分段线性变换的点
        self.transform_points = []
        
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
        self.original_display = QLabel()
        self.original_display.setFixedSize(1000, 380)
        self.original_display.setAlignment(Qt.AlignCenter)
        self.original_display.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background-color: {COLORS['background']};
        """)
        
        # 处理后图像
        processed_label = QLabel('处理后图像')
        processed_label.setObjectName('subtitle')
        self.processed_display = QLabel()
        self.processed_display.setFixedSize(1000, 380)
        self.processed_display.setAlignment(Qt.AlignCenter)
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
            1: ('直接灰度变换', self.setup_transform_controls),
            2: ('图像直方图计算及均衡化', self.setup_histogram_controls),
            3: ('图像中值滤波', self.setup_median_controls),
            4: ('低通滤波', self.setup_lowpass_controls),
            5: ('目标寻找', self.setup_find_controls),
        }
        
        if task_index in tasks:
            title, setup_func = tasks[task_index]
            self.task_label.setText(title)
            setup_func()
    
    def clear_controls(self):
        """清空控制面板"""
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    # ==================== 任务1：直接灰度变换 ====================
    
    def setup_transform_controls(self):
        """设置灰度变换控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        # 分段点管理
        points_group = QGroupBox('分段点设置')
        points_layout = QVBoxLayout()
        
        # 点列表
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(150)
        points_layout.addWidget(self.points_list)
        
        # 添加点
        add_layout = QGridLayout()
        add_layout.addWidget(QLabel('X (输入):'), 0, 0)
        self.point_x = QSpinBox()
        self.point_x.setRange(0, 255)
        add_layout.addWidget(self.point_x, 0, 1)
        
        add_layout.addWidget(QLabel('Y (输出):'), 1, 0)
        self.point_y = QSpinBox()
        self.point_y.setRange(0, 255)
        add_layout.addWidget(self.point_y, 1, 1)
        
        add_point_btn = QPushButton('添加点')
        add_point_btn.clicked.connect(self.add_transform_point)
        add_layout.addWidget(add_point_btn, 2, 0, 1, 2)
        
        points_layout.addLayout(add_layout)
        
        # 删除点
        remove_btn = QPushButton('删除选中点')
        remove_btn.clicked.connect(self.remove_transform_point)
        points_layout.addWidget(remove_btn)
        
        # 清空点
        clear_btn = QPushButton('清空所有点')
        clear_btn.clicked.connect(self.clear_transform_points)
        points_layout.addWidget(clear_btn)
        
        points_group.setLayout(points_layout)
        
        # 应用变换
        apply_btn = QPushButton('确认变换')
        apply_btn.clicked.connect(self.apply_transform)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('分段线性变换'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(points_group)
        self.controls_layout.addWidget(apply_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def add_transform_point(self):
        """添加变换点"""
        x = self.point_x.value()
        y = self.point_y.value()
        self.transform_points.append((x, y))
        self.transform_points.sort(key=lambda p: p[0])
        self.update_points_list()
    
    def remove_transform_point(self):
        """删除选中的变换点"""
        current_row = self.points_list.currentRow()
        if 0 <= current_row < len(self.transform_points):
            self.transform_points.pop(current_row)
            self.update_points_list()
    
    def clear_transform_points(self):
        """清空所有变换点"""
        self.transform_points = []
        self.update_points_list()
    
    def update_points_list(self):
        """更新点列表显示"""
        self.points_list.clear()
        for x, y in self.transform_points:
            self.points_list.addItem(f'({x}, {y})')
    
    def apply_transform(self):
        """应用分段线性变换"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        if len(self.transform_points) < 2:
            QMessageBox.warning(self, '警告', '至少需要2个点')
            return
        
        self.processed_image = ImageProcessor.piecewise_linear_transform(
            self.original_image, self.transform_points
        )
        
        # 转换为BGR以便显示
        if len(self.processed_image.shape) == 2:
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        
        self.update_display()
    
    # ==================== 任务2：图像直方图计算及均衡化 ====================
    
    def setup_histogram_controls(self):
        """设置直方图控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        calc_btn = QPushButton('计算直方图')
        calc_btn.clicked.connect(self.calculate_histogram)
        
        equalize_btn = QPushButton('确认均衡化')
        equalize_btn.clicked.connect(self.equalize_histogram)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('直方图均衡化'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(calc_btn)
        self.controls_layout.addWidget(equalize_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def calculate_histogram(self):
        """计算并显示直方图"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 计算直方图
        if len(self.original_image.shape) == 3:
            # 分别计算RGB三个通道
            b_hist = ImageProcessor.calculate_histogram(self.original_image[:, :, 0])
            g_hist = ImageProcessor.calculate_histogram(self.original_image[:, :, 1])
            r_hist = ImageProcessor.calculate_histogram(self.original_image[:, :, 2])
            
            # 显示直方图窗口
            self.show_histogram_window([b_hist, g_hist, r_hist], 
                                      ['Blue', 'Green', 'Red'],
                                      ['b', 'g', 'r'])
        else:
            hist = ImageProcessor.calculate_histogram(self.original_image)
            self.show_histogram_window([hist], ['Gray'], ['gray'])
    
    def equalize_histogram(self):
        """直方图均衡化"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        # 均衡化
        equalized = ImageProcessor.histogram_equalization(self.original_image)
        
        # 转换为BGR以便显示
        if len(equalized.shape) == 2:
            self.processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            self.processed_image = equalized
        
        self.update_display()
        
        # 同时显示均衡化后的直方图
        if len(equalized.shape) == 2:
            hist = ImageProcessor.calculate_histogram(equalized)
            self.show_histogram_window([hist], ['Equalized'], ['gray'])
        else:
            b_hist = ImageProcessor.calculate_histogram(equalized[:, :, 0])
            g_hist = ImageProcessor.calculate_histogram(equalized[:, :, 1])
            r_hist = ImageProcessor.calculate_histogram(equalized[:, :, 2])
            self.show_histogram_window([b_hist, g_hist, r_hist],
                                      ['Blue (Eq)', 'Green (Eq)', 'Red (Eq)'],
                                      ['b', 'g', 'r'])
    
    def show_histogram_window(self, hists, titles, colors):
        """显示直方图窗口"""
        dialog = HistogramWindow(hists, titles, colors, self)
        dialog.show_window()
    
    # ==================== 任务3：图像中值滤波 ====================
    
    def setup_median_controls(self):
        """设置中值滤波控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        # 核大小选择
        kernel_group = QGroupBox('核大小')
        kernel_layout = QVBoxLayout()
        
        kernel_3_btn = QPushButton('3×3')
        kernel_3_btn.clicked.connect(lambda: self.apply_median_filter(3))
        
        kernel_5_btn = QPushButton('5×5')
        kernel_5_btn.clicked.connect(lambda: self.apply_median_filter(5))
        
        kernel_7_btn = QPushButton('7×7')
        kernel_7_btn.clicked.connect(lambda: self.apply_median_filter(7))
        
        kernel_layout.addWidget(kernel_3_btn)
        kernel_layout.addWidget(kernel_5_btn)
        kernel_layout.addWidget(kernel_7_btn)
        kernel_group.setLayout(kernel_layout)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('中值滤波'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(kernel_group)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def apply_median_filter(self, kernel_size):
        """应用中值滤波"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        QMessageBox.information(self, '提示', f'正在应用{kernel_size}×{kernel_size}中值滤波，请稍候...')
        
        self.processed_image = ImageProcessor.median_filter(self.original_image, kernel_size)
        self.update_display()
        
        QMessageBox.information(self, '成功', '中值滤波完成')
    
    # ==================== 任务4：低通滤波 ====================
    
    def setup_lowpass_controls(self):
        """设置低通滤波控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        # 添加噪声
        noise_group = QGroupBox('添加噪声')
        noise_layout = QVBoxLayout()
        
        # 高斯噪声
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(QLabel('高斯噪声σ:'))
        self.gaussian_sigma = QSpinBox()
        self.gaussian_sigma.setRange(1, 100)
        self.gaussian_sigma.setValue(25)
        gaussian_layout.addWidget(self.gaussian_sigma)
        
        gaussian_btn = QPushButton('添加高斯噪声')
        gaussian_btn.clicked.connect(self.add_gaussian_noise)
        
        # 椒盐噪声
        salt_layout = QHBoxLayout()
        salt_layout.addWidget(QLabel('椒盐概率:'))
        self.salt_prob = QSpinBox()
        self.salt_prob.setRange(1, 50)
        self.salt_prob.setValue(5)
        salt_layout.addWidget(self.salt_prob)
        
        salt_btn = QPushButton('添加椒盐噪声')
        salt_btn.clicked.connect(self.add_salt_pepper_noise)
        
        noise_layout.addLayout(gaussian_layout)
        noise_layout.addWidget(gaussian_btn)
        noise_layout.addLayout(salt_layout)
        noise_layout.addWidget(salt_btn)
        noise_group.setLayout(noise_layout)
        
        # 滤波
        filter_group = QGroupBox('滤波处理')
        filter_layout = QVBoxLayout()
        
        median_btn = QPushButton('中值滤波 (5×5)')
        median_btn.clicked.connect(lambda: self.filter_noisy_image(5))
        
        filter_layout.addWidget(median_btn)
        filter_group.setLayout(filter_layout)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('低通滤波'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(noise_group)
        self.controls_layout.addWidget(filter_group)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def add_gaussian_noise(self):
        """添加高斯噪声"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        sigma = self.gaussian_sigma.value()
        self.processed_image = ImageProcessor.add_gaussian_noise(self.original_image, 0, sigma)
        self.update_display()
    
    def add_salt_pepper_noise(self):
        """添加椒盐噪声"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        prob = self.salt_prob.value() / 100.0
        self.processed_image = ImageProcessor.add_salt_pepper_noise(self.original_image, prob)
        self.update_display()
    
    def filter_noisy_image(self, kernel_size):
        """对带噪声图像进行滤波"""
        if self.processed_image is None:
            QMessageBox.warning(self, '警告', '请先添加噪声')
            return
        
        QMessageBox.information(self, '提示', '正在滤波，请稍候...')
        
        filtered = ImageProcessor.median_filter(self.processed_image, kernel_size)
        self.processed_image = filtered
        self.update_display()
        
        QMessageBox.information(self, '成功', '滤波完成')
    
    # ==================== 任务5：目标寻找 ====================
    
    def setup_find_controls(self):
        """设置目标寻找控制"""
        load_btn = QPushButton('加载图片')
        load_btn.clicked.connect(self.load_image_for_task)
        
        # 圆检测参数
        params_group = QGroupBox('检测参数')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('最小半径:'), 0, 0)
        self.min_radius = QSpinBox()
        self.min_radius.setRange(1, 500)
        self.min_radius.setValue(10)
        params_layout.addWidget(self.min_radius, 0, 1)
        
        params_layout.addWidget(QLabel('最大半径:'), 1, 0)
        self.max_radius = QSpinBox()
        self.max_radius.setRange(1, 500)
        self.max_radius.setValue(100)
        params_layout.addWidget(self.max_radius, 1, 1)
        
        params_group.setLayout(params_layout)
        
        find_btn = QPushButton('确认寻找圆形')
        find_btn.clicked.connect(self.find_circles)
        
        save_btn = QPushButton('导出结果')
        save_btn.clicked.connect(self.save_processed_image)
        
        algorithm_btn = QPushButton('📚 查看算法')
        algorithm_btn.setStyleSheet('background-color: #0969da; color: white;')
        algorithm_btn.clicked.connect(lambda: self.show_algorithm('圆形检测'))
        
        self.controls_layout.addWidget(load_btn)
        self.controls_layout.addWidget(params_group)
        self.controls_layout.addWidget(find_btn)
        self.controls_layout.addWidget(save_btn)
        self.controls_layout.addWidget(algorithm_btn)
    
    def find_circles(self):
        """寻找圆形区域"""
        if self.original_image is None:
            QMessageBox.warning(self, '警告', '请先加载图片')
            return
        
        min_r = self.min_radius.value()
        max_r = self.max_radius.value()
        
        circles = ImageProcessor.find_circles(self.original_image, min_r, max_r)
        
        if not circles:
            QMessageBox.information(self, '结果', '未找到圆形区域')
            return
        
        # 在图像上标注圆心和半径
        self.processed_image = self.original_image.copy()
        
        for x, y, r in circles:
            # 红色高亮圆心
            cv2.circle(self.processed_image, (x, y), 5, (0, 0, 255), -1)
            
            # 绘制圆形边缘
            cv2.circle(self.processed_image, (x, y), r, (0, 255, 0), 2)
            
            # 绘制半径线（45度角）
            end_x = int(x + r * np.cos(np.pi / 4))
            end_y = int(y + r * np.sin(np.pi / 4))
            cv2.line(self.processed_image, (x, y), (end_x, end_y), (255, 0, 0), 2)
        
        self.update_display()
        QMessageBox.information(self, '成功', f'找到{len(circles)}个圆形区域')
    
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
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.original_image is not None:
            pixmap = get_display_pixmap(self.original_image, 1000, 380)
            if pixmap:
                self.original_display.setPixmap(pixmap)
        
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


class HistogramWindow(QWidget):
    """直方图显示窗口"""
    
    def __init__(self, hists, titles, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle('直方图')
        self.setStyleSheet(get_style())
        self.resize(1000, 400)
        self.setWindowFlags(Qt.Window)
        
        layout = QHBoxLayout(self)
        
        for hist, title, color in zip(hists, titles, colors):
            # 创建matplotlib画布
            fig = Figure(figsize=(4, 3))
            canvas = FigureCanvasQTAgg(fig)
            ax = fig.add_subplot(111)
            
            # 绘制直方图
            ax.bar(range(256), hist, color=color, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('灰度值')
            ax.set_ylabel('像素数量')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            layout.addWidget(canvas)
    
    def show_window(self):
        """显示窗口"""
        self.show()


class AlgorithmWindow(QWidget):
    """算法显示窗口"""
    
    def __init__(self, algorithm_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{algorithm_name} - 算法原理')
        self.setStyleSheet(get_style())
        self.setWindowFlags(Qt.Window)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel(f'{algorithm_name}算法原理')
        title_label.setObjectName('subtitle')
        title_label.setAlignment(Qt.AlignCenter)
        
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
            '分段线性变换': '''
【算法原理】
分段线性灰度变换：通过多个控制点定义分段线性映射

【数学公式】
对于控制点 (x1,y1) 和 (x2,y2) 之间，输入灰度 x 的输出为：

y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)

【实现步骤】
1. 添加控制点
   用户指定 (x, y) 对，表示输入灰度 x 映射到输出灰度 y

2. 排序控制点
   按 x 值升序排列

3. 建立映射表
   对每个输入灰度值 0-255，找到其所在的分段，计算输出值

4. 应用变换
   遍历图像每个像素，查表映射

【核心代码】
# 构建映射表
for i in range(256):
    for j in range(len(points)-1):
        x1, y1 = points[j]
        x2, y2 = points[j+1]
        if x1 <= i <= x2:
            output = y1 + (i - x1) * (y2 - y1) / (x2 - x1)
            lut[i] = np.clip(output, 0, 255)

# 应用变换
result = lut[image]

【应用场景】
- 对比度增强：S型曲线
- 灰度反转：负片效果
- 亮度调整：提亮/压暗
''',
            '直方图均衡化': '''
【算法原理】
直方图均衡化：使图像直方图分布尽可能均匀

【数学原理】
1. 计算直方图 H(i)
   H(i) = 灰度值 i 的像素数量

2. 计算累积分布函数 CDF(i)
   CDF(i) = ∑(H(j)), j=0 to i

3. 归一化并映射
   output(i) = (CDF(i) - CDF_min) / (total_pixels - CDF_min) * 255

【实现步骤】
1. 计算直方图
   hist = np.zeros(256)
   for pixel in image:
       hist[pixel] += 1

2. 计算CDF
   cdf = np.cumsum(hist)

3. 归一化CDF
   cdf_min = cdf[cdf > 0].min()
   lut = ((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)

4. 应用映射
   result = lut[image]

【效果】
- 增强图像对比度
- 改善图像视觉效果
- 使灰度分布更加均匀

【核心代码】
cdf = np.cumsum(hist)
cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
equalized = cdf_normalized[image]
''',
            '中值滤波': '''
【算法原理】
中值滤波：用邻域像素的中值替换中心像素

【核心思想】
对每个像素，取其邻域窗口内所有像素的中值作为输出

【实现步骤】
1. 选择窗口大小（如 3×3、5×5、7×7）

2. 对每个像素：
   a. 提取邻域窗口
   b. 对窗口内像素排序
   c. 取中间值

3. 边界处理：填充或镜像

【核心代码】
def median_filter(image, kernel_size):
    pad = kernel_size // 2
    result = np.zeros_like(image)
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            window = image[i-pad:i+pad+1, j-pad:j+pad+1]
            result[i, j] = np.median(window)
    
    return result

【优点】
- 去除椒盐噪声效果好
- 保持边缘清晰
- 非线性滤波，不模糊

【缺点】
- 计算量大，速度慢
- 大核在大图像上耗时较长
''',
            '低通滤波': '''
【算法原理】
低通滤波：去除高频噪声，保留低频信息

【噪声类型】

1. 高斯噪声
   n(x,y) ~ N(μ, σ²)
   noisy = image + gaussian_noise

2. 椒盐噪声
   随机将像素设为 0（椒）或 255（盐）
   if rand() < prob:
       pixel = 0 or 255

【滤波方法】

1. 中值滤波（非线性）
   - 适合椒盐噪声
   - 保持边缘

2. 高斯滤波（线性）
   - 适合高斯噪声
   - 会模糊边缘

3. 均值滤波
   output = mean(window)

【实现步骤】
1. 添加噪声（模拟）
2. 选择适合的滤波器
3. 应用滤波
4. 对比原图、噪声图、滤波后图像

【核心代码】
# 高斯噪声
noise = np.random.normal(mean, sigma, image.shape)
noisy = np.clip(image + noise, 0, 255).astype(np.uint8)

# 椒盐噪声
mask = np.random.rand(*image.shape) < prob
noisy[mask] = np.random.choice([0, 255])
''',
            '圆形检测': '''
【算法原理】
Hough圆变换：检测图像中的圆形区域

【数学原理】
圆的方程：(x - a)² + (y - b)² = r²
其中 (a,b) 是圆心，r 是半径

【实现步骤】
1. 灰度化与预处理
   gray = rgb_to_gray(image)
   blurred = GaussianBlur(gray)

2. 边缘检测
   edges = Canny(blurred, threshold1, threshold2)

3. Hough圆变换
   circles = cv2.HoughCircles(
       gray, cv2.HOUGH_GRADIENT,
       dp=1, minDist=50,
       param1=100, param2=30,
       minRadius=min_r, maxRadius=max_r
   )

4. 结果标注
   - 红色圆点：圆心
   - 绿色圆圈：圆形边缘
   - 蓝色直线：半径指示

【参数调节】
- minRadius: 最小半径，过滤小圆
- maxRadius: 最大半径，过滤大圆
- minDist: 圆心最小距离，避免重复检测
- param1: Canny高阈值
- param2: 累加器阈值，越小检测越多

【应用场景】
- 硬币检测计数
- 圆形物体识别
- 工业检测
'''
        }
        
        return algorithms.get(name, '暂无算法说明')
    
    def show_window(self):
        """显示窗口"""
        self.show()
