"""
主窗口 - 包含实验和任务的二级菜单
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QPushButton, QLabel, QStackedWidget)
from PyQt5.QtCore import Qt
from .styles import get_style, COLORS
from .experiment1 import Experiment1Panel
from .experiment2 import Experiment2Panel
from .experiment3 import Experiment3Panel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_task_panel = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('机器视觉图像处理实验平台')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(get_style())
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 内容区域 - 使用堆叠窗口
        self.content_stack = QStackedWidget()
        
        # 欢迎页面
        self.welcome_page = self.create_welcome_page()
        self.content_stack.addWidget(self.welcome_page)
        
        # 实验一面板
        self.exp1_panel = Experiment1Panel(self)
        self.content_stack.addWidget(self.exp1_panel)
        
        # 实验二面板
        self.exp2_panel = Experiment2Panel(self)
        self.content_stack.addWidget(self.exp2_panel)
        
        # 实验三面板
        self.exp3_panel = Experiment3Panel(self)
        self.content_stack.addWidget(self.exp3_panel)
        
        main_layout.addWidget(self.content_stack)
        
        # 默认显示欢迎页面
        self.content_stack.setCurrentWidget(self.welcome_page)
    
    def create_welcome_page(self):
        """创建欢迎页面"""
        content = QWidget()
        content.setStyleSheet(f"background-color: {COLORS['secondary_bg']};")
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 欢迎信息
        welcome_label = QLabel('欢迎使用机器视觉图像处理实验平台')
        welcome_label.setObjectName('title')
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet(f"""
            font-size: 32px;
            font-weight: 600;
            color: {COLORS['text']};
            margin: 40px;
        """)
        
        subtitle_label = QLabel('请从顶部菜单选择实验和任务开始')
        subtitle_label.setObjectName('subtitle')
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(f"""
            font-size: 16px;
            color: {COLORS['text_secondary']};
        """)
        
        content_layout.addWidget(welcome_label)
        content_layout.addWidget(subtitle_label)
        
        return content
        
    def create_toolbar(self):
        """创建顶部工具栏"""
        toolbar = QWidget()
        toolbar.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
                border-bottom: 1px solid {COLORS['border']};
                padding: 8px;
            }}
        """)
        toolbar.setFixedHeight(60)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        
        # 实验选择
        exp_label = QLabel('实验:')
        exp_label.setStyleSheet("border: none; padding: 0;")
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(['请选择实验', '实验一：基础图像处理', '实验二：图像增强', '实验三：高级图像处理'])
        self.experiment_combo.setFixedWidth(200)
        self.experiment_combo.currentIndexChanged.connect(self.on_experiment_changed)
        
        # 任务选择
        task_label = QLabel('任务:')
        task_label.setStyleSheet("border: none; padding: 0;")
        self.task_combo = QComboBox()
        self.task_combo.addItem('请先选择实验')
        self.task_combo.setFixedWidth(200)
        self.task_combo.setEnabled(False)
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        
        # 导出图片按钮
        self.export_btn = QPushButton('导出处理后图片')
        self.export_btn.clicked.connect(self.export_image)
        
        layout.addWidget(exp_label)
        layout.addWidget(self.experiment_combo)
        layout.addWidget(task_label)
        layout.addWidget(self.task_combo)
        layout.addStretch()
        layout.addWidget(self.export_btn)
        
        return toolbar
    
    def on_experiment_changed(self, index):
        """实验选择改变"""
        self.task_combo.clear()
        self.task_combo.setEnabled(False)
        
        # 返回欢迎页面
        self.content_stack.setCurrentWidget(self.welcome_page)
        
        if index == 1:  # 实验一
            self.task_combo.setEnabled(True)
            self.task_combo.addItems([
                '请选择任务',
                '摄像头调用',
                '图像格式转换',
                '图像读写测试',
                '图片融合',
                '颜色阈值抠图',
                '图像校正',
                '布匹裁剪分割线识别'
            ])
        elif index == 2:  # 实验二
            self.task_combo.setEnabled(True)
            self.task_combo.addItems([
                '请选择任务',
                '直接灰度变换',
                '图像直方图计算及均衡化',
                '图像中值滤波',
                '低通滤波',
                '目标寻找'
            ])
        elif index == 3:  # 实验三
            self.task_combo.setEnabled(True)
            self.task_combo.addItems([
                '请选择任务',
                'Hough变换检测',
                '傅里叶变换',
                '缺陷检测',
                '划痕检测',
                'PCB检测'
            ])
    
    def on_task_changed(self, index):
        """任务选择改变"""
        if index == 0:
            self.content_stack.setCurrentWidget(self.welcome_page)
            return
            
        exp_index = self.experiment_combo.currentIndex()
        
        if exp_index == 1:  # 实验一
            self.content_stack.setCurrentWidget(self.exp1_panel)
            self.exp1_panel.set_task(index)
            
        elif exp_index == 2:  # 实验二
            self.content_stack.setCurrentWidget(self.exp2_panel)
            self.exp2_panel.set_task(index)
        
        elif exp_index == 3:  # 实验三
            self.content_stack.setCurrentWidget(self.exp3_panel)
            self.exp3_panel.set_task(index)
    
    def export_image(self):
        """导出图片"""
        # 获取当前激活的面板
        current_widget = self.content_stack.currentWidget()
        if current_widget is not None and hasattr(current_widget, 'save_processed_image'):
            current_widget.save_processed_image()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, '提示', '请先选择任务并处理图片')
