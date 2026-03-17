# page4.py
import streamlit as st
import sys
import os

# 关键：禁用 OpenCV 的系统依赖检测（核心修复）
os.environ["OPENCV_IO_ENABLE_JASPER"] = "false"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "false"
os.environ["OPENCV_DNN_DISABLE_OPENCL"] = "true"

# 页面配置
st.set_page_config(page_title="无packages.txt测试", page_icon="📸")
st.title("OpenCV 导入测试（无系统依赖）")

# 分步导入，捕获详细错误
try:
    # 1. 先导入基础库
    import numpy as np
    from PIL import Image
    st.success("✅ 基础库导入成功")

    # 2. 导入 OpenCV（核心步骤）
    import cv2
    st.success(f"✅ OpenCV 导入成功！版本：{cv2.__version__}")

    # 3. 测试 OpenCV 基础功能（验证可用）
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img = cv2.putText(test_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    st.image(test_img, caption="OpenCV 绘图测试", width=200)

    # 4. 可选：导入 Ultralytics YOLO
    try:
        from ultralytics import YOLO
        st.success("✅ YOLO 导入成功")
        model = YOLO('yolov8n.pt')
        st.success("✅ YOLOv8n 模型加载完成")
    except Exception as e:
        st.warning(f"⚠️ YOLO 导入/加载警告：{str(e)}")

except ImportError as e:
    st.error(f"❌ 导入错误：{str(e)}")
    st.error(f"📝 Python 路径：{sys.path}")
except Exception as e:
    st.error(f"❌ 运行错误：{str(e)}")
    st.error(f"📝 错误类型：{type(e).__name__}")
