import streamlit as st
import cv2
import time
from ultralytics import YOLO

# ======================
# 页面配置
# ======================
st.set_page_config(
    page_title="YOLO 实时视频识别",
    layout="wide"
)

st.title("🎥 YOLO 实时摄像头目标检测")

# ======================
# 模型加载（1.12.0 兼容）
# ======================
@st.cache(allow_output_mutation=True)
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

# ======================
# Session State
# ======================
if "run" not in st.session_state:
    st.session_state.run = False

# ======================
# 控制面板
# ======================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📷 控制面板")

    camera_id = st.selectbox("选择摄像头", [0, 1], index=0)

    if not st.session_state.run:
        if st.button("▶ 开始识别"):
            st.session_state.run = True
    else:
        if st.button("⏹ 停止识别"):
            st.session_state.run = False

    st.markdown("---")
    st.write("**运行状态：**", "🟢 运行中" if st.session_state.run else "🔴 已停止")

# ======================
# 视频显示区
# ======================
with col2:
    st.subheader("🎬 实时画面")
    frame_area = st.empty()

# ======================
# 主循环（关键）
# ======================
if st.session_state.run:
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        st.error("❌ 无法打开摄像头")
        st.stop()

    prev_time = time.time()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ 摄像头读取失败")
            break

        # YOLO 推理
        results = model(frame, verbose=False)[0]
        frame = results.plot()

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        frame_area.image(frame, use_column_width=False, width=768)

        time.sleep(0.01)

    cap.release()