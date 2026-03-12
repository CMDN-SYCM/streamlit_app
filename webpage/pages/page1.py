import streamlit as st
import subprocess
import os
import sys
from streamlit_autorefresh import st_autorefresh

# ======================
# 页面配置
# ======================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.main .block-container {
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("💻 模型训练")

tab1, tab2 = st.tabs(["模型训练", "训练日志"])

# ======================
# 路径配置
# ======================
SCRIPT_PATH = os.path.join("execute", "3.model.py")
LOG_PATH = os.path.join("train.log")

# ======================
# Tab 1：启动训练
# ======================
with tab1:
    st.subheader("📄 训练脚本")
    st.code(SCRIPT_PATH)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ 开始训练模型"):
            if os.path.exists(LOG_PATH):
                os.remove(LOG_PATH)

            process = subprocess.Popen(
                [sys.executable, SCRIPT_PATH],
                stdout=open(LOG_PATH, "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )

            st.session_state["train_process"] = process
            st.success("✅ 训练进程已启动（后台运行）")

    with col2:
        if st.button("🛑 停止训练"):
            if "train_process" in st.session_state:
                p = st.session_state["train_process"]
                if p.poll() is None:
                    p.terminate()  # 🔥 关键
                    st.warning("⛔ 训练已手动停止")
                else:
                    st.info("训练进程已结束")
                del st.session_state["train_process"]
            else:
                st.info("当前没有训练进程")

# ======================
# Tab 2：训练日志（自动刷新）
# ======================
with tab2:
    st.subheader("📜 训练日志")

    # 🔥 每 2 秒自动刷新一次页面
    st_autorefresh(interval=5000, key="log_refresh")

    if "train_process" not in st.session_state:
        st.info("尚未启动训练")
    else:
        log_area = st.empty()

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # 只显示最后 8000 字符，防止页面卡死
                log_area.text(content[-8000:])
        else:
            st.info("⏳ 训练进行中（日志尚未生成）")

        process = st.session_state["train_process"]
        if process.poll() is not None:
            st.success("🎉 训练已完成")
        else:
            st.info("🚀 训练进行中（日志每 2 秒自动刷新）")