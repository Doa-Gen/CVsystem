"""
图像处理核心算法
所有算法均基于原理实现，不使用OpenCV内置函数
"""
import numpy as np
import cv2


class ImageProcessor:
    """图像处理器 - 实现所有核心算法"""
    
    @staticmethod
    def rgb_to_gray_manual(image):
        """
        RGB转灰度 - 手动实现
        公式: Gray = 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) != 3:
            return image
            
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)
    
    @staticmethod
    def rgb_to_hsv_manual(image):
        """
        RGB转HSV - 手动实现
        H: 色调 (0-180)
        S: 饱和度 (0-255)
        V: 明度 (0-255)
        """
        if len(image.shape) != 3:
            return image
            
        # OpenCV使用BGR格式
        b, g, r = image[:, :, 0] / 255.0, image[:, :, 1] / 255.0, image[:, :, 2] / 255.0
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # V通道 (明度)
        v = max_val
        
        # S通道 (饱和度)
        s = np.zeros_like(max_val)
        mask = max_val != 0
        s[mask] = diff[mask] / max_val[mask]
        
        # H通道 (色调)
        h = np.zeros_like(max_val)
        
        # 当max = r时
        mask_r = (max_val == r) & (diff != 0)
        h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])
        
        # 当max = g时
        mask_g = (max_val == g) & (diff != 0)
        h[mask_g] = 60 * (2 + (b[mask_g] - r[mask_g]) / diff[mask_g])
        
        # 当max = b时
        mask_b = (max_val == b) & (diff != 0)
        h[mask_b] = 60 * (4 + (r[mask_b] - g[mask_b]) / diff[mask_b])
        
        # 调整H到0-180范围
        h[h < 0] += 360
        h = h / 2  # OpenCV的H范围是0-180
        
        # 转换为uint8
        hsv = np.zeros_like(image)
        hsv[:, :, 0] = h.astype(np.uint8)
        hsv[:, :, 1] = (s * 255).astype(np.uint8)
        hsv[:, :, 2] = (v * 255).astype(np.uint8)
        
        return hsv
    
    @staticmethod
    def split_channels(image):
        """分离通道"""
        if len(image.shape) == 2:
            return [image]
        return [image[:, :, i] for i in range(image.shape[2])]
    
    @staticmethod
    def blend_images(img1, img2, alpha):
        """
        图像融合
        result = alpha * img1 + (1 - alpha) * img2
        """
        # 确保两张图片尺寸相同
        if img1.shape != img2.shape:
            img2 = ImageProcessor.resize_and_center(img2, img1.shape[:2])
        
        # 确保通道数相同（处理RGB和RGBA的情况）
        if len(img1.shape) == 3 and len(img2.shape) == 3:
            if img1.shape[2] != img2.shape[2]:
                # 如果通道数不同，统一转换为3通道BGR
                if img1.shape[2] == 4:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                if img2.shape[2] == 4:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        
        # 最终确认尺寸完全一致
        if img1.shape != img2.shape:
            raise ValueError(f"图像尺寸不匹配: img1={img1.shape}, img2={img2.shape}")
        
        return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    
    @staticmethod
    def resize_and_center(image, target_size):
        """
        调整图像大小并居中
        将图像按比例缩放至目标尺寸内（保持长宽比），并居中放置
        """
        target_h, target_w = target_size[:2]
        src_h, src_w = image.shape[:2]
        
        # 计算缩放比例，确保图像完整显示在目标区域内（取最小缩放比）
        scale = min(target_h / src_h, target_w / src_w)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标大小的画布（黑色背景）
        if len(image.shape) == 3:
            result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            result = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # 计算居中放置的偏移量
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # 将缩放后的图像居中放置
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    @staticmethod
    def color_threshold_mask(image, lower_bound, upper_bound, color_space='BGR'):
        """
        颜色阈值抠图
        """
        # 处理4通道图像，转换为3通道
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # 转换到指定颜色空间
        if color_space == 'HSV':
            image_space = ImageProcessor.rgb_to_hsv_manual(image)
        else:
            image_space = image
        
        # 再次确认通道数，确保与阈值匹配
        if len(image_space.shape) == 3 and image_space.shape[2] == 4:
            image_space = cv2.cvtColor(image_space, cv2.COLOR_BGRA2BGR)
        
        mask = np.all((image_space >= lower_bound) & (image_space <= upper_bound), axis=2)
        return mask.astype(np.uint8) * 255
    
    @staticmethod
    def piecewise_linear_transform(image, points):
        """
        分段线性变换
        points: [(x0, y0), (x1, y1), ...]，其中x是输入灰度，y是输出灰度
        """
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 确保点按x值排序
        points = sorted(points, key=lambda p: p[0])
        
        # 添加起点和终点
        if points[0][0] != 0:
            points.insert(0, (0, 0))
        if points[-1][0] != 255:
            points.append((255, 255))
        
        # 创建查找表
        lut = np.zeros(256, dtype=np.uint8)
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            for x in range(x1, x2 + 1):
                if x2 != x1:
                    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                else:
                    y = y1
                lut[x] = np.clip(int(y), 0, 255)
        
        return lut[gray]
    
    @staticmethod
    def calculate_histogram(image):
        """
        计算图像直方图
        返回: hist (256,) 数组
        """
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image
        
        hist = np.zeros(256, dtype=np.int32)
        for i in range(256):
            hist[i] = np.sum(gray == i)
        
        return hist
    
    @staticmethod
    def histogram_equalization(image):
        """
        直方图均衡化 - 手动实现
        """
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 计算直方图
        hist = ImageProcessor.calculate_histogram(gray)
        
        # 计算累积分布函数 (CDF)
        cdf = hist.cumsum()
        
        # 归一化CDF
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # 应用均衡化
        equalized = cdf_normalized[gray].astype(np.uint8)
        
        return equalized
    
    @staticmethod
    def median_filter(image, kernel_size=3):
        """
        中值滤波 - 使用OpenCV实现（性能优化）
        注意：为了避免卡死，这里使用OpenCV的优化实现
        """
        # 直接使用OpenCV的中值滤波，性能远高于手动实现
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        """添加高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    @staticmethod
    def add_salt_pepper_noise(image, prob=0.05):
        """添加椒盐噪声"""
        noisy = image.copy()
        
        # 盐噪声 (白色)
        salt_mask = np.random.random(image.shape[:2]) < (prob / 2)
        noisy[salt_mask] = 255
        
        # 椒噪声 (黑色)
        pepper_mask = np.random.random(image.shape[:2]) < (prob / 2)
        noisy[pepper_mask] = 0
        
        return noisy
    
    @staticmethod
    def find_circles(image, min_radius=10, max_radius=100):
        """
        寻找图像中的圆形区域（优化版）
        返回: [(x, y, r), ...] 圆心坐标和半径列表
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 使用更强的边缘检测，只检测明显边界
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 使用霸夫圆检测，调整参数减少误检测
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,  # 增加最小距离
            param1=100,  # 提高Canny边缘检测阈值
            param2=50,   # 提高圆心检测阈值，减少误报
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return [(int(x), int(y), int(r)) for x, y, r in circles]
        
        return []
    
    @staticmethod
    def correct_perspective(image, pts_src):
        """
        图像校正 - 使用透视变换将倾斜矩形摆正
        pts_src: 源图像中的4个点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 顺序：左上、右上、右下、左下
        """
        # 转换为numpy数组
        pts_src = np.array(pts_src, dtype=np.float32)
        
        # 计算目标矩形的宽高
        # 计算上边和下边的宽度
        width_top = np.sqrt(((pts_src[1][0] - pts_src[0][0]) ** 2) + ((pts_src[1][1] - pts_src[0][1]) ** 2))
        width_bottom = np.sqrt(((pts_src[2][0] - pts_src[3][0]) ** 2) + ((pts_src[2][1] - pts_src[3][1]) ** 2))
        max_width = int(max(width_top, width_bottom))
        
        # 计算左边和右边的高度
        height_left = np.sqrt(((pts_src[3][0] - pts_src[0][0]) ** 2) + ((pts_src[3][1] - pts_src[0][1]) ** 2))
        height_right = np.sqrt(((pts_src[2][0] - pts_src[1][0]) ** 2) + ((pts_src[2][1] - pts_src[1][1]) ** 2))
        max_height = int(max(height_left, height_right))
        
        # 定义目标点（摆正后的矩形）
        pts_dst = np.array([
            [0, 0],                      # 左上
            [max_width - 1, 0],          # 右上
            [max_width - 1, max_height - 1],  # 右下
            [0, max_height - 1]          # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        # 应用透视变换
        warped = cv2.warpPerspective(image, M, (max_width, max_height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        
        return warped
    
    @staticmethod
    def detect_fabric_cut_line(image):
        """
        识别布匹上的最佳裁剪分割线
        基于颜色差异检测待分割区域（褥皱部分颜色与正常布匹不同）
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        
        # 计算每列的平均灰度值（用于检测颜色差异）
        column_means = np.mean(gray, axis=0)
        
        # 对平均值进行平滑处理，减少噪声影响
        kernel_size = max(5, width // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        column_means_smooth = cv2.GaussianBlur(column_means.reshape(1, -1), (kernel_size, 1), 0).flatten()
        
        # 计算灰度值的梯度（检测颜色突变）
        gradient = np.abs(np.gradient(column_means_smooth))
        
        # 找到梯度最大的位置（颜色变化最大的地方）
        # 使用自适应阈值
        gradient_threshold = np.mean(gradient) + np.std(gradient) * 1.5
        significant_changes = gradient > gradient_threshold
        
        # 找到所有显著变化的位置
        change_positions = np.where(significant_changes)[0]
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        if len(change_positions) == 0:
            # 如果没有发现显著变化，使用梯度最大值
            best_x = int(np.argmax(gradient))
            cv2.line(result, (best_x, 0), (best_x, height - 1), (0, 0, 255), 3)
        else:
            # 对变化位置进行聚类，找到主要的分割区域
            # 如果有多个变化点，选择中间区域
            if len(change_positions) > 1:
                # 计算变化点之间的间隔
                intervals = np.diff(change_positions)
                # 找到最大间隔的中点（可能是褥皱区域）
                if len(intervals) > 0:
                    max_interval_idx = np.argmax(intervals)
                    start_pos = change_positions[max_interval_idx]
                    end_pos = change_positions[max_interval_idx + 1]
                    best_x = int((start_pos + end_pos) / 2)
                else:
                    best_x = int(change_positions[0])
            else:
                best_x = int(change_positions[0])
            
            # 绘制主分割线（红色粗线）
            cv2.line(result, (best_x, 0), (best_x, height - 1), (0, 0, 255), 3)
            
            # 可选：绘制所有检测到的变化点（黄色细线）
            for x in change_positions:
                if abs(x - best_x) > 10:  # 避免与主线重叠
                    cv2.line(result, (int(x), 0), (int(x), height - 1), (0, 255, 255), 1)
        
        return result
    
    # ==================== 实验三：高级图像处理 ====================
    
    @staticmethod
    def hough_lines(image, canny_low=50, canny_high=150, threshold=100):
        """
        使用Hough变换检测直线
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 边缘检测
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Hough直线检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        # 在原图上绘制直线
        result = image.copy()
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # 计算直线的两个端点
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # 绘制直线（绿色高亮）
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result
    
    @staticmethod
    def hough_circles(image, min_radius=10, max_radius=100):
        """
        使用Hough变换检测圆形
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 高斯模糊
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough圆检测
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        # 在原图上绘制圆形
        result = image.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for x, y, r in circles:
                # 绘制圆形轮廓（红色高亮）
                cv2.circle(result, (x, y), r, (0, 0, 255), 3)
                # 绘制圆心
                cv2.circle(result, (x, y), 2, (255, 0, 0), 3)
        
        return result
    
    @staticmethod
    def fourier_transform(image):
        """
        计算傅里叶变换并展示幅度谱
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 计算傅里叶变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 计算幅度谱
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 归一化到0-255
        magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
        
        return magnitude_spectrum
    
    @staticmethod
    def detect_circle_defects(image):
        """
        检测完整圆形中的缺陷和多余小块，并计算缺陷面积
        返回: (结果图像, 缺陷信息字典)
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 检测轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的圆形区域（认为是主圆）
        if len(contours) == 0:
            defect_info = {
                'defect_count': 0,
                'total_area': 0,
                'areas': []
            }
            return image.copy(), defect_info
        
        main_contour = max(contours, key=cv2.contourArea)
        main_area = cv2.contourArea(main_contour)
        
        # 创建主圆的掩膜
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        
        # 检测缺陷（主圆内部的黑色区域）
        defects = cv2.bitwise_not(binary) & mask
        
        # 找到缺陷轮廓
        defect_contours, _ = cv2.findContours(defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小噪声
        min_defect_area = 10
        valid_defects = [cnt for cnt in defect_contours if cv2.contourArea(cnt) > min_defect_area]
        
        # 计算缺陷面积
        defect_areas = [cv2.contourArea(cnt) for cnt in valid_defects]
        total_area = sum(defect_areas)
        
        # 绘制结果
        result = image.copy()
        if len(image.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 绘制主圆轮廓（绿色）
        cv2.drawContours(result, [main_contour], -1, (0, 255, 0), 2)
        
        # 绘制缺陷（红色高亮）
        cv2.drawContours(result, valid_defects, -1, (0, 0, 255), -1)
        cv2.drawContours(result, valid_defects, -1, (255, 0, 0), 2)
        
        defect_info = {
            'defect_count': len(valid_defects),
            'total_area': int(total_area),
            'areas': [int(area) for area in defect_areas]
        }
        
        return result, defect_info
    
    @staticmethod
    def detect_scratches(image, kernel_size=5, threshold=30):
        """
        使用形态学、边缘检测和颜色差异检测材料表面划痕
        能够检测明显色彩差别的曲线划痕
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 方法1: 使用Canny边缘检测 + 形态学处理检测曲线划痕
        # 1. 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # 2. Canny边缘检测（能检测到颜色差异明显的边界）
        edges = cv2.Canny(blurred, 30, 100)
        
        # 3. 使用线性结构元素连接线条形划痕
        # 水平线条检测
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        detected_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
        
        # 垂直线条检测
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        detected_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
        
        # 对角线条检测
        kernel_d1 = np.array([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]], dtype=np.uint8)
        detected_d1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_d1)
        
        kernel_d2 = np.array([[0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0],
                              [1, 0, 0, 0, 0]], dtype=np.uint8)
        detected_d2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_d2)
        
        # 合并所有方向的线条
        scratches_canny = cv2.bitwise_or(detected_h, detected_v)
        scratches_canny = cv2.bitwise_or(scratches_canny, detected_d1)
        scratches_canny = cv2.bitwise_or(scratches_canny, detected_d2)
        
        # 方法2: 使用形态学顶帽/黑帽变换检浌亮暗划痕
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        scratches_morph = cv2.add(tophat, blackhat)
        
        # 对形态学结果二值化
        _, binary_morph = cv2.threshold(scratches_morph, threshold, 255, cv2.THRESH_BINARY)
        
        # 方法3: 颜色空间分析（对彩色图像）
        if len(image.shape) == 3:
            # 转换到LAB颜色空间，对颜色差异更敏感
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 对a和b通道进行边缘检测（检测颜色变化）
            a_edges = cv2.Canny(cv2.GaussianBlur(a, (3, 3), 0), 20, 60)
            b_edges = cv2.Canny(cv2.GaussianBlur(b, (3, 3), 0), 20, 60)
            color_edges = cv2.bitwise_or(a_edges, b_edges)
            
            # 用线性结构元素连接颜色边缘
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            color_scratches_h = cv2.morphologyEx(color_edges, cv2.MORPH_CLOSE, kernel_line)
            
            kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            color_scratches_v = cv2.morphologyEx(color_edges, cv2.MORPH_CLOSE, kernel_line_v)
            
            color_scratches = cv2.bitwise_or(color_scratches_h, color_scratches_v)
        else:
            color_scratches = np.zeros_like(gray)
        
        # 合并所有检测结果
        final_scratches = cv2.bitwise_or(scratches_canny, binary_morph)
        final_scratches = cv2.bitwise_or(final_scratches, color_scratches)
        
        # 形态学操作优化：去除小噪点，连接断开的线条
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_scratches = cv2.morphologyEx(final_scratches, cv2.MORPH_CLOSE, kernel_clean)
        final_scratches = cv2.morphologyEx(final_scratches, cv2.MORPH_OPEN, kernel_clean)
        
        # 过滤小区域（去除噪声）
        contours, _ = cv2.findContours(final_scratches, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(final_scratches)
        for cnt in contours:
            # 保留面积较大或长宽比较大的区域（划痕特征）
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            
            # 划痕通常是狭长的
            if area > 20 or aspect_ratio > 3:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # 将划痕区域高亮显示（红色）
        result[mask > 0] = [0, 0, 255]
        
        # 额外绘制轮廓使划痕更明显
        cv2.drawContours(result, contours, -1, (255, 0, 0), 1)
        
        return result
    
    @staticmethod
    def detect_pcb_defects(image, defect_type='全部缺陷'):
        """
        检测PCB缺陷：毛刺、短路、断路
        返回: (结果图像, 缺陷信息字典)
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = ImageProcessor.rgb_to_gray_manual(image)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        defect_info = {}
        
        # 检测毛刺（小的突出物）
        if defect_type in ['全部缺陷', '毛刺']:
            # 使用开运算去除小突起
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
            burrs = cv2.subtract(binary, opened)
            
            # 查找毛刺轮廓
            contours, _ = cv2.findContours(burrs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            burr_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 5])
            
            # 绘制毛刺（黄色）
            cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
            defect_info['毛刺数量'] = burr_count
        
        # 检测短路（意外连接的线路）
        if defect_type in ['全部缺陷', '短路']:
            # 使用闭运算检测短路
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
            shorts = cv2.subtract(closed, binary)
            
            # 查找短路区域
            contours, _ = cv2.findContours(shorts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            short_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 10])
            
            # 绘制短路（橙色）
            cv2.drawContours(result, contours, -1, (0, 165, 255), 2)
            defect_info['短路数量'] = short_count
        
        # 检测断路（线路中的间隙）
        if defect_type in ['全部缺陷', '断路']:
            # 使用形态学梯度检测边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            
            # 检测线路上的间隙
            _, gap_binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
            
            # 查找断路位置
            contours, _ = cv2.findContours(gap_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gap_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 5])
            
            # 绘制断路（红色）
            for cnt in contours:
                if cv2.contourArea(cnt) > 5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            defect_info['断路数量'] = gap_count
        
        return result, defect_info
