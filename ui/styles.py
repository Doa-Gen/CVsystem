"""
统一样式配置 - 浅色GitHub风格主题
"""

# 颜色配置
COLORS = {
    'background': '#ffffff',
    'secondary_bg': '#f6f8fa',
    'border': '#d0d7de',
    'text': '#24292f',
    'text_secondary': '#57606a',
    'primary': '#0969da',
    'primary_hover': '#0550ae',
    'button_bg': '#f6f8fa',
    'button_hover': '#f3f4f6',
    'success': '#1a7f37',
    'danger': '#cf222e',
}

# 主样式表
MAIN_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    font-size: 14px;
}}

QPushButton {{
    background-color: {COLORS['button_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 5px 16px;
    color: {COLORS['text']};
    font-weight: 500;
    min-height: 28px;
}}

QPushButton:hover {{
    background-color: {COLORS['button_hover']};
    border-color: {COLORS['text_secondary']};
}}

QPushButton:pressed {{
    background-color: {COLORS['secondary_bg']};
}}

QPushButton:disabled {{
    background-color: {COLORS['secondary_bg']};
    color: {COLORS['text_secondary']};
    border-color: {COLORS['border']};
}}

QPushButton#primary {{
    background-color: {COLORS['primary']};
    color: white;
    border-color: {COLORS['primary']};
}}

QPushButton#primary:hover {{
    background-color: {COLORS['primary_hover']};
    border-color: {COLORS['primary_hover']};
}}

QComboBox {{
    background-color: {COLORS['button_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 28px;
}}

QComboBox:hover {{
    border-color: {COLORS['text_secondary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {COLORS['text']};
    margin-right: 5px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['secondary_bg']};
    selection-color: {COLORS['text']};
    outline: none;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 28px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['primary']};
    outline: none;
}}

QLabel {{
    background-color: transparent;
    color: {COLORS['text']};
}}

QLabel#title {{
    font-size: 16px;
    font-weight: 600;
    color: {COLORS['text']};
}}

QLabel#subtitle {{
    font-size: 14px;
    color: {COLORS['text_secondary']};
}}

QScrollArea {{
    border: none;
    background-color: {COLORS['background']};
}}

QScrollBar:vertical {{
    border: none;
    background: {COLORS['secondary_bg']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 5px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    border: none;
    background: {COLORS['secondary_bg']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['border']};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_secondary']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QMenuBar {{
    background-color: {COLORS['background']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 6px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['secondary_bg']};
}}

QMenu {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS['secondary_bg']};
}}

QGroupBox {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {COLORS['text']};
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLORS['border']};
    height: 4px;
    background: {COLORS['secondary_bg']};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS['primary_hover']};
}}

QTextEdit {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px;
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    background-color: {COLORS['background']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['text_secondary']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['primary']};
    border-color: {COLORS['primary']};
    image: none;
}}

QRadioButton {{
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    background-color: {COLORS['background']};
}}

QRadioButton::indicator:hover {{
    border-color: {COLORS['text_secondary']};
}}

QRadioButton::indicator:checked {{
    background-color: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}
"""


def get_style():
    """获取主样式表"""
    return MAIN_STYLE
