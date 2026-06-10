import streamlit as st
import os

st.set_page_config(
    page_title="Ý TƯỞNG CỦA BÉ",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #FFF9F0 0%, #FFF0FB 50%, #F0F6FF 100%);
    min-height: 100vh;
}

/* Header */
.hero-title {
    font-family: 'Baloo 2', cursive;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B9D, #FF9A3C, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0;
}
.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
}

/* Cards */
.step-card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 2px solid #F5E6FF;
    margin-bottom: 1rem;
}
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #A855F7, #EC4899);
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    text-align: center;
    line-height: 32px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #A855F7, #EC4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Baloo 2', cursive !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(168,85,247,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(168,85,247,0.4) !important;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 16px;
    border: 2px dashed #D8B4FE !important;
    padding: 1rem;
}

/* Progress / spinner */
.stSpinner > div {
    border-top-color: #A855F7 !important;
}

/* Settings sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #A855F7 !important;
}
.stSelectbox [data-baseweb="select"] {
    border-radius: 12px !important;
}

/* Result section */
.result-box {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 30px rgba(168,85,247,0.12);
    border: 2px solid #E9D5FF;
    text-align: center;
}

/* Status messages */
.status-msg {
    background: #F5F0FF;
    border-left: 4px solid #A855F7;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    color: #6B21A8;
    font-size: 0.95rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">✨ Tranh Động Của Bé ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Biến tranh vẽ tay của bé thành video hoạt hình có lời kể chuyện</p>', unsafe_allow_html=True)

# ── Check API key ─────────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    with st.expander("⚙️ Cài đặt API Key", expanded=True):
        api_key = st.text_input(
            type="password",
            placeholder="sk-...",
            help="Key chỉ dùng trong phiên này, không được lưu lại"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ Đã lưu API key cho phiên này!")

st.divider()

# ── Layout: 2 cột ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### 🖼️ Bước 1 — Tải ảnh tranh vẽ của bé')

    uploaded = st.file_uploader(
        "Chụp ảnh hoặc chọn file ảnh",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        st.image(uploaded, caption="Tranh vẽ của bé", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### ⚙️ Bước 2 — Tuỳ chỉnh video')

    animation_style = st.selectbox(
        "Kiểu chuyển động",
        ["🌈 Nhẹ nhàng, vui tươi", "🌊 Mượt mà, thư giãn", "⚡ Năng động, sôi nổi"],
        help="Ảnh hưởng đến tốc độ và cách vật thể di chuyển"
    )

    narrator_voice = st.selectbox(
        "Giọng kể chuyện",
        ["alloy — Trung tính, dễ nghe", "nova — Nữ, ấm áp", "echo — Nam, trầm ấm", "shimmer — Nữ, nhẹ nhàng"],
        help="Giọng AI đọc lời kể câu chuyện trong tranh"
    )

    language = st.selectbox(
        "Ngôn ngữ lời kể",
        ["Tiếng Việt", "Tiếng Anh"],
    )

    duration = st.slider(
        "Độ dài video (giây)",
        min_value=10, max_value=30, value=15, step=5,
        help="Video càng dài càng tốn thời gian xử lý"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### 🎬 Bước 3 — Tạo video')

    # Nút tạo video
    can_run = bool(uploaded and api_key)
    if not uploaded:
        st.info("👈 Tải ảnh tranh vẽ trước để tiếp tục")
    elif not api_key:
        st.warning("🔑 Cần nhập API key OpenAI trước")
    else:
        if st.button("✨ Tạo Video Hoạt Hình!", use_container_width=True):
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from utils.pipeline import run_pipeline

            # Lưu ảnh tạm
            img_bytes = uploaded.read()
            tmp_path = f"/tmp/drawing_{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)

            voice_id = narrator_voice.split(" — ")[0].strip()
            style_map = {
                "🌈 Nhẹ nhàng, vui tươi": "gentle",
                "🌊 Mượt mà, thư giãn":   "calm",
                "⚡ Năng động, sôi nổi":  "energetic"
            }
            style_key = style_map[animation_style]

            status_box = st.empty()
            progress_bar = st.progress(0)

            def update_status(msg, pct):
                status_box.markdown(f'<div class="status-msg">⏳ {msg}</div>', unsafe_allow_html=True)
                progress_bar.progress(pct)

            result = run_pipeline(
                image_path=tmp_path,
                voice=voice_id,
                language=language,
                style=style_key,
                duration=duration,
                status_cb=update_status,
                api_key=api_key
            )

            progress_bar.progress(100)

            if result["ok"]:
                status_box.markdown('<div class="status-msg">✅ Xong rồi!</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("#### 🎉 Video của bé đã sẵn sàng!")
                st.video(result["video_path"])

                with open(result["video_path"], "rb") as vf:
                    st.download_button(
                        label="⬇️ Tải video về máy",
                        data=vf,
                        file_name=f"tranh_dong_{uploaded.name.split('.')[0]}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

                if result.get("story"):
                    with st.expander("📖 Câu chuyện AI tạo ra"):
                        st.write(result["story"])
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                status_box.error(f"❌ Lỗi: {result['error']}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hướng dẫn ──────────────────────────────────────────────────────────────
    with st.expander("💡 Mẹo để có video đẹp nhất"):
        st.markdown("""
        - **Chụp ảnh thẳng góc**, tránh nghiêng hoặc bóng đổ
        - **Nền trắng** giúp AI nhận diện tranh tốt hơn
        - Tranh có **nhân vật rõ ràng** (người, con vật) sẽ có animation đẹp hơn
        - **Ánh sáng đủ** — tránh chụp thiếu sáng
        - Tranh vẽ bằng **bút màu/sáp màu** cho kết quả sinh động nhất
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#bbb;font-size:0.85rem;">🎨 Tranh Động Của Bé · Giữ nguyên nét vẽ, thổi hồn chuyển động</p>',
    unsafe_allow_html=True
)
