import streamlit as st
import os
from PIL import Image

# 配置页面
st.set_page_config(
    page_title="基于深度学习的口罩佩戴系统设计与实现",
    layout="wide"
)

# 使用 markdown 插入全局的自定义 CSS 样式
st.markdown("""
    <style>
        /* 全局调整 Streamlit 页面顶部的间距 */
        .main .block-container {
            padding-top: 10px;  /* 去除整个页面顶部的默认间距 */
        }

        /* 自定义图片标题的字体样式 */
        .image-caption {
            font-size: 20px;  /* 设置字号 */
            color: #FF6347;  /* 设置字体颜色 */
            font-weight: bold;  /* 设置字体加粗 */
            text-align: center;  /* 设置文字居中 */
        }

        /* 为第四张图片的标题设置左对齐 */
        .left-align {
            font-size: 20px;  /* 设置字号 */
            color: #FF6347;  /* 设置字体颜色 */
            font-weight: bold;  /* 设置字体加粗 */
            margin-left: 300px;  /* 将标题向左移动 */
        }
    </style>
""", unsafe_allow_html=True)

# 标题
st.title("👩‍⚕ 基于深度学习的口罩佩戴系统设计与实现")

# 创建四列布局，调整间距
col1, col2, col3, col4 = st.columns([4, 3, 3, 3])  # 第一列窄，后三列宽

# 第一列：展示系统介绍
with col1:
    # 创建一个容器，内嵌样式
    st.markdown("""
    <div style='
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
    '>
    <h4 style='color: #2c3e50; margin-top: 0;'>系统简介</h4>
    <p style='font-size: 16px; line-height: 1.6; color: #34495e;'>
    本系统基于深度学习和计算机视觉技术，实现对口罩佩戴情况的自动检测与分类。
    </p>

    <ul style='font-size: 15px; line-height: 1.8; color: #2c3e50;'>
        <li><b>基于YOLOv8目标检测算法</b>，实现实时人脸识别与定位</li>
        <li><b>支持多平台部署</b>，包括Web端、移动端和嵌入式设备</li>
    </ul>

    <p style='font-size: 16px; line-height: 1.6; color: #34495e;'>
    <b>系统目标是帮助用户快速判断个人或他人是否佩戴口罩，提升公共安全和健康管理效率。</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

# 读取 photo 文件夹中的所有图片
image_folder = "webpage/photo"  # 图片文件夹路径
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 自定义图片标题（可以手动设置）
image_titles = ["规范佩戴", "未佩戴", "不规范佩戴", "人群场景"]

# 第二列到第四列：显示前三张图片
columns = [col2, col3, col4]  # 将后三列存入一个列表

# 前三张图片按列显示
for i in range(3):
    image_path = os.path.join(image_folder, image_files[i])
    img = Image.open(image_path)  # 打开图片

    # 显示图片
    columns[i].image(img, use_column_width=False, width=300)

    # 显示图片标题，并应用自定义 CSS 样式
    columns[i].markdown(f'<p class="image-caption">{image_titles[i]}</p>', unsafe_allow_html=True)

# 将第四张图片放置在第二列和第三列之间
col_middle = st.columns([2, 3])  # 通过调整比例可以优化间距

with col_middle[0]:
    # 创建一个容器，内嵌样式
    st.markdown("""
    <div style='
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
    '>
    <h4 style='color: #2c3e50; margin-top: 0;'>应用场景</h4>
    <p style='font-size: 16px; line-height: 1.6; color: #34495e;'>
    系统可广泛应用于公共场所、企事业单位、学校等场景，助力疫情防控常态化管理。
    </p>

    <ul style='font-size: 15px; line-height: 1.8; color: #2c3e50;'>
        <li><b>图片上传预测</b>：可提供图片批量进行识别</li>
        <li><b>实时预警机制</b>：自动识别未佩戴口罩人员</li>
        <li><b>隐私保护设计</b>：本地化处理，不存储个人生物信息</li>
    </ul>

    <p style='font-size: 16px; line-height: 1.6; color: #34495e;'>
    <b>可根据不同行业需求调整检测参数和功能模块。</b>
    </p>
    <div style='
        background-color: #e8f4fc;
        padding: 10px;
        border-left: 4px solid #3498db;
        margin-top: 15px;
        font-size: 15px;
    '>
    👈 <b>请从左侧选择功能模块，开始使用系统</b>
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_middle[1]:
    image_path = os.path.join(image_folder, image_files[3])
    img = Image.open(image_path)  # 打开图片
    st.image(img, use_column_width=False, width=700)  # 使其适应宽度

    # 显示第四张图片的标题，并使用左对齐的 CSS 类
    st.markdown(f'<p class="left-align">{image_titles[3]}</p>', unsafe_allow_html=True)