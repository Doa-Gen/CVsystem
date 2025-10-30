"""
辅助函数
"""
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def numpy_to_qimage(image):
    """将numpy数组转换为QImage"""
    if image is None:
        return None
    
    q_image = None
    if len(image.shape) == 2:  # 灰度图
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(image.shape) == 3:  # 彩色图
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        if channel == 3:  # BGR to RGB
            image = image[:, :, ::-1].copy()
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channel == 4:  # BGRA to RGBA
            image = image[:, :, [2, 1, 0, 3]].copy()
            q_image = QImage(image.data, width, height, 4 * width, QImage.Format_RGBA8888)
    
    return q_image


def numpy_to_qpixmap(image):
    """将numpy数组转换为QPixmap"""
    q_image = numpy_to_qimage(image)
    if q_image is None:
        return None
    return QPixmap.fromImage(q_image)


def resize_image_for_display(image, max_width=800, max_height=600):
    """调整图像大小以适应显示区域"""
    if image is None:
        return None
        
    height, width = image.shape[:2]
    
    # 计算缩放比例
    scale_w = max_width / width if width > max_width else 1
    scale_h = max_height / height if height > max_height else 1
    scale = min(scale_w, scale_h)
    
    if scale < 1:
        import cv2
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def get_display_pixmap(image, max_width=800, max_height=600):
    """获取用于显示的QPixmap"""
    if image is None:
        return None
    
    display_image = resize_image_for_display(image, max_width, max_height)
    return numpy_to_qpixmap(display_image)


def imread_chinese(file_path):
    """读取包含中文路径的图像文件"""
    try:
        # 使用numpy读取文件，支持中文路径
        stream = open(file_path, "rb")
        bytes_array = bytearray(stream.read())
        numpy_array = np.asarray(bytes_array, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
        stream.close()
        return image
    except Exception as e:
        print(f"读取图像失败: {e}")
        return None


def imwrite_chinese(file_path, image):
    """保存图像到包含中文路径的文件"""
    try:
        # 使用imencode编码后写入，支持中文路径
        ext = file_path[file_path.rfind('.'):]
        success, encoded_image = cv2.imencode(ext, image)
        if success:
            encoded_image.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"保存图像失败: {e}")
        return False
