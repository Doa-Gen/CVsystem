@echo off
chcp 65001 >nul
echo ====================================
echo 图像处理学习平台 - 安装依赖
echo ====================================
echo.

echo 正在安装Python依赖包...
pip install -r requirements.txt

echo.
echo ====================================
echo 安装完成！
echo ====================================
echo.
pause
