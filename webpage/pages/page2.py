import streamlit as st
import os
from PIL import Image
import pandas as pd

# 配置页面
st.set_page_config(layout="wide")

# 使用 markdown 插入全局的自定义 CSS 样式
st.markdown("""
    <style>
        /* 全局调整 Streamlit 页面顶部的间距 */
        .main .block-container {
            padding-top: 10px;  /* 去除整个页面顶部的默认间距 */
        }
    </style>
""", unsafe_allow_html=True)

# 标题
st.title("🧠 训练结果")

# 使用标签（tabs）
tab1, tab2, tab3, tab4 = st.tabs(["1.混淆矩阵图", "2.标签图", "3.性能评估", "4.结果图"])

# 在 tab1 中显示内容
with tab1:
    # 创建两列布局来显示图片
    col1, col2 = st.columns(2)

    # 图片路径
    image_folder = "runs/detect/train"  # 图片所在文件夹路径
    image_files = ["confusion_matrix.png", "confusion_matrix_normalized.png"]  # 图片文件名

    # 检查图片是否存在
    try:
        # 在第一列显示 confusion_matrix.png
        with col1:
            image_path1 = os.path.join(image_folder, image_files[0])  # 获取第一张图片路径
            if os.path.exists(image_path1):
                img1 = Image.open(image_path1)  # 打开图片
                st.image(img1, caption="混淆矩阵", use_column_width=True)  # 显示图片
            else:
                st.warning(f"图片未找到: {image_files[0]}")

        # 在第二列显示 confusion_matrix_normalized.png
        with col2:
            image_path2 = os.path.join(image_folder, image_files[1])  # 获取第二张图片路径
            if os.path.exists(image_path2):
                img2 = Image.open(image_path2)  # 打开图片
                st.image(img2, caption="归一化混淆矩阵", use_column_width=True)  # 显示图片
            else:
                st.warning(f"图片未找到: {image_files[1]}")
    except Exception as e:
        st.error(f"加载图片时出错: {e}")

# 在 tab2 中显示内容
with tab2:
    # 创建两列布局来显示图片
    col1, col2 = st.columns(2)

    # 图片路径
    image_folder = "runs/detect/train"  # 图片所在文件夹路径
    image_files = ["labels.jpg", "labels_correlogram.jpg"]  # 图片文件名

    # 检查图片是否存在
    try:
        # 在第一列显示 labels.jpg
        with col1:
            image_path1 = os.path.join(image_folder, image_files[0])  # 获取第一张图片路径
            if os.path.exists(image_path1):
                img1 = Image.open(image_path1)  # 打开图片
                st.image(img1, caption="标签分布图", use_column_width=True)  # 显示图片
            else:
                st.warning(f"图片未找到: {image_files[0]}")

        # 在第二列显示 labels_correlogram.jpg
        with col2:
            image_path2 = os.path.join(image_folder, image_files[1])  # 获取第二张图片路径
            if os.path.exists(image_path2):
                img2 = Image.open(image_path2)  # 打开图片
                st.image(img2, caption="标签相关性热图", use_column_width=True)  # 显示图片
            else:
                st.warning(f"图片未找到: {image_files[1]}")
    except Exception as e:
        st.error(f"加载图片时出错: {e}")

# 在 tab3 中显示四个评估图
with tab3:
    # 创建两列布局来显示图片
    col1, col2 = st.columns(2)

    # 图片路径
    image_folder = "runs/detect/train"  # 图片所在文件夹路径
    image_files = [
        "F1_curve.png",  # F1 曲线
        "P_curve.png",  # 精确度曲线
        "R_curve.png",  # 召回率曲线
        "PR_curve.png"  # 精确度-召回率曲线
    ]  # 评估相关图片文件名

    # 检查图片是否存在
    try:
        # 在第一列显示 F1 曲线和精确度曲线
        with col1:
            # 显示 F1 曲线
            image_path1 = os.path.join(image_folder, image_files[0])
            if os.path.exists(image_path1):
                img1 = Image.open(image_path1)
                st.image(img1, caption="F1 分数变化曲线", use_column_width=True)
                st.caption("F1分数是精确度和召回率的调和平均数")
            else:
                st.warning(f"图片未找到: {image_files[0]}")

            # 显示精确度曲线
            image_path2 = os.path.join(image_folder, image_files[1])
            if os.path.exists(image_path2):
                img2 = Image.open(image_path2)
                st.image(img2, caption="精确度变化曲线", use_column_width=True)
                st.caption("精确度表示检测正确的比例")
            else:
                st.warning(f"图片未找到: {image_files[1]}")

        # 在第二列显示召回率曲线和精确度-召回率曲线
        with col2:
            # 显示召回率曲线
            image_path3 = os.path.join(image_folder, image_files[2])
            if os.path.exists(image_path3):
                img3 = Image.open(image_path3)
                st.image(img3, caption="召回率变化曲线", use_column_width=True)
                st.caption("召回率表示检测到所有正样本的比例")
            else:
                st.warning(f"图片未找到: {image_files[2]}")

            # 显示精确度-召回率曲线
            image_path4 = os.path.join(image_folder, image_files[3])
            if os.path.exists(image_path4):
                img4 = Image.open(image_path4)
                st.image(img4, caption="精确度-召回率关系图", use_column_width=True)
                st.caption("PR曲线显示了精确度和召回率的权衡关系")
            else:
                st.warning(f"图片未找到: {image_files[3]}")
    except Exception as e:
        st.error(f"加载图片时出错: {e}")

# 在 tab4 中显示一张图片
with tab4:
    # 图片路径
    image_folder = "runs/detect/train"  # 图片所在文件夹路径
    image_file = "results.png"  # 单个图片文件名
    csv_file = "results.csv"  # CSV 文件名

    # 检查并显示图片
    try:
        image_path = os.path.join(image_folder, image_file)  # 获取图片路径
        if os.path.exists(image_path):
            img = Image.open(image_path)  # 打开图片
            st.image(img, caption="训练结果图", use_column_width=True)  # 显示图片
            st.caption("显示训练过程中的各种指标变化趋势")
        else:
            st.warning(f"结果图片未找到: {image_file}")
    except Exception as e:
        st.error(f"加载结果图片时出错: {e}")

    # 显示 result.csv 表格
    csv_path = os.path.join(image_folder, csv_file)  # 获取 CSV 文件路径

    # 检查文件是否存在
    if os.path.exists(csv_path):
        try:
            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(csv_path)

            st.subheader("📊 训练数据详情")

            # 显示表格
            st.dataframe(df)  # 显示 CSV 表格

            # 添加一些简单的统计信息
            st.subheader("📈 关键指标统计")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'metrics/precision(B)' in df.columns:
                    latest_precision = df['metrics/precision(B)'].iloc[-1]
                    st.metric("最新精确度", f"{latest_precision:.3f}")

            with col2:
                if 'metrics/recall(B)' in df.columns:
                    latest_recall = df['metrics/recall(B)'].iloc[-1]
                    st.metric("最新召回率", f"{latest_recall:.3f}")

            with col3:
                if 'metrics/mAP50(B)' in df.columns:
                    latest_map50 = df['metrics/mAP50(B)'].iloc[-1]
                    st.metric("最新mAP50", f"{latest_map50:.3f}")

            with col4:
                if 'metrics/mAP50-95(B)' in df.columns:
                    latest_map95 = df['metrics/mAP50-95(B)'].iloc[-1]
                    st.metric("最新mAP50-95", f"{latest_map95:.3f}")

        except Exception as e:
            st.error(f"读取CSV文件时出错: {e}")
    else:
        st.info("📝 CSV 文件未找到，训练数据详情将在这里显示")

# 侧边栏添加额外信息
with st.sidebar:
    st.header("ℹ️ 关于训练结果")
    st.write("""
    **训练结果分析工具**

    展示YOLO模型训练过程中的各种评估指标和可视化结果。

    ### 标签页说明:

    **1. 混淆矩阵图**
    - 显示模型在各类别上的分类性能
    - 对角线值越高表示分类越准确
    - 归一化版本更易比较不同类别

    **2. 标签图**
    - 标签分布：显示训练数据中各类别的分布情况
    - 标签相关性：显示不同类别之间的相关性

    **3. 性能评估曲线**
    - F1曲线：精确度和召回率的平衡指标
    - 精确度曲线：随置信度阈值变化的精确度
    - 召回率曲线：随置信度阈值变化的召回率
    - PR曲线：精确度-召回率权衡曲线

    **4. 结果图**
    - 训练过程指标变化趋势
    - 详细的训练数据表格
    - 关键性能指标统计

    ### 指标解释:
    - **精确度 (Precision)**: 正确检测的比例
    - **召回率 (Recall)**: 检测到所有目标的比例
    - **F1分数**: 精确度和召回率的调和平均数
    - **mAP**: 平均精确度，综合评估指标

    ### 使用方法:
    1. 在各标签页查看不同的可视化结果
    2. 分析模型的性能表现
    3. 查看训练数据详情
    4. 根据结果调整训练策略
    """)

    # 添加文件路径信息
    st.markdown("---")
    st.subheader("📁 文件路径信息")

    # 检查主要文件是否存在
    files_to_check = {
        "混淆矩阵": "runs/detect/train/confusion_matrix.png",
        "标签图": "runs/detect/train/labels.jpg",
        "结果图": "runs/detect/train/results.png",
        "CSV数据": "runs/detect/train/results.csv"
    }

    for file_name, file_path in files_to_check.items():
        if os.path.exists(file_path):
            st.success(f"✅ {file_name}: 已找到")
        else:
            st.warning(f"⚠️ {file_name}: 未找到")

    # 添加刷新按钮
    st.markdown("---")
    if st.button("🔄 刷新页面"):
        st.markdown("""
        <script>
        window.location.reload();
        </script>
        """, unsafe_allow_html=True)

    # 添加版本信息
    st.caption("训练结果分析 v1.0")