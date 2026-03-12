import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
from PIL import Image
import base64
from io import BytesIO

# 配置页面
st.set_page_config(layout="wide")

# 使用 markdown 插入全局的自定义 CSS 样式
st.markdown("""
    <style>
        /* 全局调整 Streamlit 页面顶部的间距 */
        .main .block-container {
            padding-top: 10px;  /* 去除整个页面顶部的默认间距 */
        }

        /* 自定义按钮样式 */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            width: 100%;
        }

        .stButton > button:hover {
            background-color: #45a049;
        }

        .primary-button {
            background-color: #FF4B4B !important;
        }

        .primary-button:hover {
            background-color: #FF3333 !important;
        }

        .secondary-button {
            background-color: #008CBA !important;
        }

        .secondary-button:hover {
            background-color: #007B9A !important;
        }

        .folder-selection {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #4CAF50;
        }

        .path-display {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-family: monospace;
            margin: 10px 0;
            word-break: break-all;
        }
    </style>
""", unsafe_allow_html=True)

# 标题
st.title("🤖 图片识别")


# 加载训练好的 YOLO 模型
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


model = load_model()

# 在 session_state 中初始化变量
if 'input_folder' not in st.session_state:
    st.session_state.input_folder = "111/111_input"
if 'output_folder' not in st.session_state:
    st.session_state.output_folder = "111/111_outputs"
if 'show_browse_input' not in st.session_state:
    st.session_state.show_browse_input = False
if 'show_browse_output' not in st.session_state:
    st.session_state.show_browse_output = False

# 创建两个主要区域
col1, col2 = st.columns([1, 2])

# 左侧列：上传和设置
with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "选择图片文件",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支持格式: JPG, JPEG, PNG, BMP"
    )

    st.subheader("⚙️ 设置")
    process_option = st.radio(
        "处理方式",
        ["单张图片处理", "批量处理文件夹"],
        help="选择单张图片处理或批量处理整个文件夹"
    )

# 根据选择的处理方式显示不同内容
if process_option == "单张图片处理":
    # 单张图片处理逻辑
    with col1:
        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_column_width=True)

            # 处理按钮
            if st.button("🚀 开始识别"):
                st.markdown(
                    """
                    <script>
                    var buttons = document.querySelectorAll('.stButton > button');
                    if (buttons.length > 0) {
                        buttons[buttons.length - 1].classList.add('primary-button');
                    }
                    </script>
                    """,
                    unsafe_allow_html=True
                )

                if model is not None:
                    with st.spinner("正在识别中..."):
                        try:
                            # 保存上传的文件到临时文件
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            # 读取图像
                            image_cv = cv2.imread(tmp_path)

                            # 使用 YOLO 模型对图像进行推理
                            results = model(image_cv)

                            # 确保结果是一个 Results 对象
                            if isinstance(results, list):
                                results = results[0]  # 如果返回的是列表，取第一个元素

                            # 使用 plot() 渲染预测框到图像
                            rendered_image = results.plot()  # 获取渲染后的图像

                            # 转换颜色空间从BGR到RGB
                            rendered_image_rgb = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

                            # 在右侧显示结果
                            with col2:
                                st.subheader("🔍 识别结果")

                                # 显示处理后的图片
                                st.image(rendered_image_rgb, caption="识别结果", use_column_width=True)

                                # 显示检测统计信息
                                if hasattr(results, 'boxes') and results.boxes is not None:
                                    num_detections = len(results.boxes)
                                    st.success(f"✅ 检测到 {num_detections} 个目标")

                                    # 显示检测详情
                                    if num_detections > 0:
                                        st.subheader("📋 检测详情")
                                        for i, box in enumerate(results.boxes):
                                            class_id = int(box.cls[0])
                                            confidence = float(box.conf[0])
                                            class_name = model.names[class_id] if hasattr(model,
                                                                                          'names') else f"Class {class_id}"
                                            st.write(f"{i + 1}. {class_name}: 置信度 {confidence:.2%}")

                                # 提供下载链接
                                output_image = Image.fromarray(rendered_image_rgb)
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as output_tmp:
                                    output_image.save(output_tmp.name, 'JPEG')
                                    with open(output_tmp.name, 'rb') as file:
                                        if st.button("📥 下载结果图片"):
                                            st.markdown(
                                                """
                                                <script>
                                                var buttons = document.querySelectorAll('.stButton > button');
                                                if (buttons.length > 0) {
                                                    buttons[buttons.length - 1].classList.add('secondary-button');
                                                }
                                                </script>
                                                """,
                                                unsafe_allow_html=True
                                            )

                                            buffered = BytesIO()
                                            output_image.save(buffered, format="JPEG")
                                            img_str = base64.b64encode(buffered.getvalue()).decode()

                                            href = f'<a href="data:image/jpeg;base64,{img_str}" download="result_{uploaded_file.name}" style="background-color:#4CAF50;color:white;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;border-radius:4px;margin-top:10px;">📥 点击下载结果图片</a>'
                                            st.markdown(href, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"处理失败: {e}")
                else:
                    st.error("模型未加载成功，无法进行识别")

    # 如果没有上传文件，在右侧显示说明
    if uploaded_file is None:
        with col2:
            st.info("👈 请在左侧上传图片并点击'开始识别'按钮")
            st.image("https://images.unsplash.com/photo-1579546929662-711aa81148cf?w=800&auto=format&fit=crop",
                     caption="示例图片", use_column_width=True)

else:  # 批量处理文件夹
    # 批量处理文件夹逻辑 - 独立于列结构
    st.markdown("---")

    # 创建文件夹选择区域
    folder_container = st.container()

    with folder_container:
        st.markdown('<div class="folder-selection">', unsafe_allow_html=True)
        st.info("📁 批量处理功能")

        # 输入文件夹设置
        st.write("### 输入文件夹设置")

        # 使用expander来组织内容，避免嵌套columns
        with st.expander("📂 输入文件夹配置", expanded=True):
            # 手动输入路径
            input_folder = st.text_input(
                "输入文件夹路径",
                value=st.session_state.input_folder,
                key="input_folder_input",
                help="请输入包含图片的文件夹完整路径"
            )

            if input_folder != st.session_state.input_folder:
                st.session_state.input_folder = input_folder

            # 显示当前路径
            st.markdown(f'<div class="path-display">📁 当前输入路径: {st.session_state.input_folder}</div>',
                        unsafe_allow_html=True)

            # 浏览当前目录内容
            if st.button("📋 查看输入目录内容", key="browse_input"):
                st.session_state.show_browse_input = not st.session_state.show_browse_input

            if st.session_state.show_browse_input:
                if os.path.exists(st.session_state.input_folder):
                    try:
                        items = os.listdir(st.session_state.input_folder)
                        folders = [item for item in items if
                                   os.path.isdir(os.path.join(st.session_state.input_folder, item))]
                        files = [item for item in items if
                                 os.path.isfile(os.path.join(st.session_state.input_folder, item))]

                        col_folders, col_files = st.columns(2)

                        with col_folders:
                            st.write("**子文件夹:**")
                            if folders:
                                for folder in folders[:10]:  # 只显示前10个
                                    if st.button(f"📁 {folder}", key=f"input_folder_{folder}"):
                                        new_path = os.path.join(st.session_state.input_folder, folder)
                                        st.session_state.input_folder = new_path
                                        st.rerun()
                            else:
                                st.write("无子文件夹")

                        with col_files:
                            st.write("**文件:**")
                            if files:
                                image_files = [f for f in files if
                                               f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                                if image_files:
                                    st.write(f"找到 {len(image_files)} 张图片:")
                                    for file in image_files[:5]:  # 只显示前5个
                                        st.code(file)
                                    if len(image_files) > 5:
                                        st.write(f"... 还有 {len(image_files) - 5} 个图片文件")
                            else:
                                st.write("无文件")
                    except Exception as e:
                        st.error(f"无法读取目录: {e}")
                else:
                    st.warning("文件夹路径不存在")

        # 输出文件夹设置
        st.write("### 输出文件夹设置")

        with st.expander("📂 输出文件夹配置", expanded=True):
            # 手动输入路径
            output_folder = st.text_input(
                "输出文件夹路径",
                value=st.session_state.output_folder,
                key="output_folder_input",
                help="请输入保存结果的文件夹完整路径"
            )

            if output_folder != st.session_state.output_folder:
                st.session_state.output_folder = output_folder

            # 显示当前路径
            st.markdown(f'<div class="path-display">📁 当前输出路径: {st.session_state.output_folder}</div>',
                        unsafe_allow_html=True)

            # 浏览当前目录内容
            if st.button("📋 查看输出目录内容", key="browse_output"):
                st.session_state.show_browse_output = not st.session_state.show_browse_output

            if st.session_state.show_browse_output:
                if os.path.exists(st.session_state.output_folder):
                    try:
                        items = os.listdir(st.session_state.output_folder)
                        folders = [item for item in items if
                                   os.path.isdir(os.path.join(st.session_state.output_folder, item))]

                        if folders:
                            st.write("**子文件夹:**")
                            for folder in folders[:10]:  # 只显示前10个
                                if st.button(f"📁 {folder}", key=f"output_folder_{folder}"):
                                    new_path = os.path.join(st.session_state.output_folder, folder)
                                    st.session_state.output_folder = new_path
                                    st.rerun()
                        else:
                            st.write("无子文件夹")
                    except Exception as e:
                        st.error(f"无法读取目录: {e}")
                else:
                    st.info("输出文件夹不存在，将在处理时创建")

        st.markdown('</div>', unsafe_allow_html=True)

        # 显示统计信息
        if os.path.exists(st.session_state.input_folder):
            try:
                image_files = [f for f in os.listdir(st.session_state.input_folder)
                               if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
                if image_files:
                    st.success(f"✅ 输入文件夹中包含 {len(image_files)} 张图片")
                else:
                    st.warning("⚠️ 输入文件夹中没有找到图片文件")
            except:
                pass

        # 批量处理按钮
        if st.button("🔄 开始批量处理", key="batch_process"):
            if model is not None:
                # 创建处理结果区域
                result_container = st.container()

                with result_container:
                    with st.spinner("正在批量处理中..."):
                        try:
                            input_folder = st.session_state.input_folder
                            output_folder = st.session_state.output_folder

                            # 检查输入文件夹是否存在
                            if not os.path.exists(input_folder):
                                st.error(f"❌ 输入文件夹不存在: {input_folder}")
                            else:
                                # 确保输出文件夹存在
                                os.makedirs(output_folder, exist_ok=True)

                                # 获取文件夹中所有图像文件
                                image_files = [f for f in os.listdir(input_folder)
                                               if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]

                                if not image_files:
                                    st.warning("指定文件夹中没有找到图片文件")
                                else:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    processed_count = 0
                                    total_count = len(image_files)

                                    st.info(f"📊 开始处理 {total_count} 张图片...")

                                    # 创建两列显示处理进度和预览
                                    progress_col, preview_col = st.columns([2, 1])

                                    example_image_path = None

                                    for idx, image_file in enumerate(image_files):
                                        # 更新进度
                                        progress = (idx + 1) / total_count
                                        progress_bar.progress(progress)
                                        status_text.text(f"正在处理: {image_file} ({idx + 1}/{total_count})")

                                        # 读取图像
                                        image_path = os.path.join(input_folder, image_file)
                                        image = cv2.imread(image_path)

                                        if image is not None:
                                            # 使用 YOLO 模型对图像进行推理
                                            results = model(image)

                                            # 确保结果是一个 Results 对象
                                            if isinstance(results, list):
                                                results = results[0]  # 如果返回的是列表，取第一个元素

                                            # 使用 plot() 渲染预测框到图像
                                            rendered_image = results.plot()  # 获取渲染后的图像

                                            # 保存带有预测框的图像
                                            output_image_path = os.path.join(output_folder, image_file)
                                            cv2.imwrite(output_image_path, rendered_image)

                                            processed_count += 1

                                            # 保存第一张处理后的图片作为示例
                                            if idx == 0:
                                                example_image_path = output_image_path

                                    progress_bar.empty()
                                    status_text.empty()

                                    # 显示处理结果
                                    st.success(f"✅ 批量处理完成！共处理 {processed_count}/{total_count} 张图片")
                                    st.info(f"💾 结果已保存到: {os.path.abspath(output_folder)}")

                                    # 显示处理后的示例图片
                                    if processed_count > 0 and example_image_path and os.path.exists(
                                            example_image_path):
                                        st.subheader("🎯 处理结果示例")

                                        example_col1, example_col2 = st.columns([2, 1])

                                        with example_col1:
                                            example_image = Image.open(example_image_path)
                                            st.image(example_image,
                                                     caption=f"示例: {os.path.basename(example_image_path)}",
                                                     use_column_width=True)

                                        with example_col2:
                                            # 提供下载示例结果的链接
                                            buffered = BytesIO()
                                            example_image.save(buffered, format="JPEG")
                                            img_str = base64.b64encode(buffered.getvalue()).decode()

                                            st.markdown("### 📥 下载示例")
                                            href = f'<a href="data:image/jpeg;base64,{img_str}" download="sample_result.jpg" style="background-color:#4CAF50;color:white;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;border-radius:4px;margin-top:10px;">下载示例结果</a>'
                                            st.markdown(href, unsafe_allow_html=True)

                                            # 显示处理统计
                                            st.metric("处理图片数", processed_count)
                                            st.metric("成功率", f"{(processed_count / total_count) * 100:.1f}%")

                        except Exception as e:
                            st.error(f"❌ 批量处理失败: {str(e)}")
            else:
                st.error("❌ 模型未加载成功，无法进行批量处理")

# 侧边栏添加额外信息
with st.sidebar:
    st.header("ℹ️ 关于")
    st.write("""
    **图片识别工具**

    使用 YOLO 模型进行目标检测。

    ### 功能特点:
    - 支持单张图片上传和识别
    - 支持批量处理文件夹中的图片
    - 显示检测结果和置信度
    - 可下载处理后的图片

    ### 使用说明:
    1. 在左侧选择处理方式
    2. 上传图片或指定文件夹路径
    3. 点击识别按钮开始处理
    4. 查看结果并下载
    """)

    if model is not None:
        st.success("✅ 模型加载成功")
    else:
        st.error("❌ 模型加载失败")

# 添加自定义JavaScript来增强按钮样式
st.markdown("""
<script>
// 页面加载完成后为所有按钮添加样式
document.addEventListener('DOMContentLoaded', function() {
    // 定期检查并应用样式（因为Streamlit会动态更新DOM）
    setInterval(function() {
        var buttons = document.querySelectorAll('.stButton > button');
        buttons.forEach(function(button, index) {
            // 根据按钮文本内容添加不同的类
            var buttonText = button.textContent;
            if (buttonText.includes('开始识别') || buttonText.includes('开始批量处理')) {
                button.classList.add('primary-button');
            } else if (buttonText.includes('下载')) {
                button.classList.add('secondary-button');
            }
        });
    }, 500);
});
</script>
""", unsafe_allow_html=True)