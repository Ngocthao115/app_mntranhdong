import streamlit as st
import os
import sys
import base64
import json
import time
import math
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Tranh Động Của Bé",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: linear-gradient(135deg, #FFF9F0 0%, #FFF0FB 50%, #F0F6FF 100%); min-height: 100vh; }
.hero-title {
    font-family: 'Baloo 2', cursive; font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B9D, #FF9A3C, #A855F7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center; line-height: 1.2; margin-bottom: 0;
}
.hero-sub { text-align: center; color: #888; font-size: 1.1rem; margin-top: 0.3rem; margin-bottom: 2rem; }
.step-card {
    background: white; border-radius: 20px; padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 2px solid #F5E6FF; margin-bottom: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #A855F7, #EC4899) !important;
    color: white !important; border: none !important; border-radius: 50px !important;
    padding: 0.7rem 2rem !important; font-family: 'Baloo 2', cursive !important;
    font-weight: 600 !important; font-size: 1.05rem !important;
    box-shadow: 0 4px 15px rgba(168,85,247,0.3) !important; width: 100% !important;
}
.result-box {
    background: white; border-radius: 20px; padding: 2rem;
    box-shadow: 0 4px 30px rgba(168,85,247,0.12); border: 2px solid #E9D5FF; text-align: center;
}
.status-msg {
    background: #F5F0FF; border-left: 4px solid #A855F7;
    border-radius: 0 12px 12px 0; padding: 0.8rem 1rem; color: #6B21A8; font-size: 0.95rem; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PIPELINE (gộp trực tiếp vào app.py)
# ══════════════════════════════════════════════════════

def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _img_to_base64_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
    return f"data:{mime};base64,{_encode_image(path)}"

def analyze_drawing(client, image_path: str, language: str) -> dict:
    lang_instruction = "Viết hoàn toàn bằng tiếng Việt." if language == "Tiếng Việt" else "Write everything in English."
    prompt = f"""Bạn là chuyên gia giáo dục mầm non và AI sáng tạo.
Hãy phân tích bức tranh vẽ tay của trẻ và trả về JSON với cấu trúc sau (không có markdown, chỉ JSON thuần):
{{
  "objects": [{{"name": "tên vật thể", "type": "character|animal|plant|sky|ground|other", "motion": "walk|fly|bounce|sway|shine|flow|spin|float", "description": "mô tả ngắn"}}],
  "scene": "mô tả tổng quan cảnh trong tranh (1-2 câu)",
  "story": "câu chuyện ngắn 3-4 câu cho trẻ mầm non, dựa trên tranh, kết thúc vui vẻ",
  "mood": "happy|calm|adventurous|peaceful",
  "colors": ["màu chủ đạo 1", "màu chủ đạo 2"]
}}
{lang_instruction}
Câu chuyện đơn giản, tích cực, phù hợp trẻ 3-6 tuổi."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":_img_to_base64_url(image_path),"detail":"high"}},
            {"type":"text","text":prompt}
        ]}],
        max_tokens=800, temperature=0.7
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n",1)[1].rsplit("```",1)[0]
    return json.loads(raw)

def generate_audio(client, story: str, voice: str, output_path: str) -> str:
    resp = client.audio.speech.create(model="tts-1", voice=voice, input=story, speed=0.9)
    resp.stream_to_file(output_path)
    return output_path

def _apply_gentle_effect(img, t):
    from PIL import Image
    w, h = img.size
    bob = int(math.sin(t * 2 * math.pi) * 3)
    scale = 1.0 + 0.012 * math.sin(t * 2 * math.pi * 0.7)
    nw, nh = int(w*scale), int(h*scale)
    frame = img.resize((nw, nh), Image.LANCZOS)
    left = (nw-w)//2; top = max(0, min((nh-h)//2 - bob, nh-h))
    return frame.crop((left, top, left+w, top+h))

def _apply_calm_effect(img, t):
    from PIL import Image, ImageEnhance
    w, h = img.size
    scale = 1.0 + 0.008 * math.sin(t * 2 * math.pi * 0.5)
    nw, nh = int(w*scale), int(h*scale)
    frame = img.resize((nw, nh), Image.LANCZOS)
    left = (nw-w)//2; top = (nh-h)//2
    frame = frame.crop((left, top, left+w, top+h))
    brightness = 1.0 + 0.04 * math.sin(t * 2 * math.pi * 0.3)
    return ImageEnhance.Brightness(frame).enhance(brightness)

def _apply_energetic_effect(img, t):
    from PIL import Image
    w, h = img.size
    bob = int(math.sin(t * 2 * math.pi * 1.5) * 6)
    sway = int(math.sin(t * 2 * math.pi * 0.8) * 4)
    scale = 1.0 + 0.018 * abs(math.sin(t * 2 * math.pi))
    nw, nh = int(w*scale), int(h*scale)
    frame = img.resize((nw, nh), Image.LANCZOS)
    left = max(0, min((nw-w)//2 - sway, nw-w))
    top  = max(0, min((nh-h)//2 - bob,  nh-h))
    return frame.crop((left, top, left+w, top+h))

EFFECTS = {"gentle":_apply_gentle_effect, "calm":_apply_calm_effect, "energetic":_apply_energetic_effect}

def generate_frames(image_path: str, style: str, duration: int, fps: int = 24):
    import numpy as np
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((720, 720), Image.LANCZOS)
    w, h = img.size
    w = w if w%2==0 else w-1
    h = h if h%2==0 else h-1
    img = img.crop((0,0,w,h))
    effect_fn = EFFECTS.get(style, _apply_gentle_effect)
    total = duration * fps
    frames = [np.array(effect_fn(img.copy(), i/total)) for i in range(total)]
    return frames, (w, h), fps

def assemble_video(frames, audio_path: str, output_path: str, fps: int) -> str:
    import moviepy.editor as mpy
    video_clip = mpy.ImageSequenceClip(frames, fps=fps)
    audio_clip = mpy.AudioFileClip(audio_path)
    if audio_clip.duration < video_clip.duration:
        audio_clip = audio_clip.audio_loop(duration=video_clip.duration)
    else:
        audio_clip = audio_clip.subclip(0, video_clip.duration)
    final = video_clip.set_audio(audio_clip)
    final.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac",
                          logger=None, temp_audiofile="/tmp/temp_audio.m4a", remove_temp=True)
    return output_path

def run_pipeline(image_path, voice, language, style, duration, status_cb, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        out_dir = Path("/tmp/drawing_animator_output")
        out_dir.mkdir(exist_ok=True)
        ts = int(time.time())

        status_cb("Đang phân tích tranh vẽ của bé...", 10)
        analysis = analyze_drawing(client, image_path, language)
        story = analysis.get("story", "Bé vẽ một bức tranh thật đẹp!")

        status_cb("Đang tạo giọng kể chuyện...", 35)
        audio_path = str(out_dir / f"audio_{ts}.mp3")
        generate_audio(client, story, voice, audio_path)

        status_cb("Đang tạo hiệu ứng chuyển động cho tranh...", 60)
        frames, size, fps = generate_frames(image_path, style, duration)

        status_cb("Đang ghép video hoàn chỉnh...", 85)
        video_path = str(out_dir / f"video_{ts}.mp4")
        assemble_video(frames, audio_path, video_path, fps)

        return {"ok": True, "video_path": video_path, "story": story, "analysis": analysis}
    except Exception as e:
        import traceback
        return {"ok": False, "error": f"{str(e)}\n{traceback.format_exc()}"}

# ══════════════════════════════════════════════════════
#  GIAO DIỆN
# ══════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title">✨ Tranh Động Của Bé ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Biến tranh vẽ tay của bé thành video hoạt hình có lời kể chuyện</p>', unsafe_allow_html=True)

# API Key
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if not api_key:
    with st.expander("⚙️ Cài đặt API Key", expanded=True):
        api_key = st.text_input("Nhập OpenAI API Key của bạn", type="password", placeholder="sk-...",
                                help="Key chỉ dùng trong phiên này, không được lưu lại")
        if api_key:
            st.success("✅ Đã lưu API key cho phiên này!")

st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### 🖼️ Bước 1 — Tải ảnh tranh vẽ của bé')
    uploaded = st.file_uploader("Chụp ảnh hoặc chọn file ảnh", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, caption="Tranh vẽ của bé", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### ⚙️ Bước 2 — Tuỳ chỉnh video')
    animation_style = st.selectbox("Kiểu chuyển động",
        ["🌈 Nhẹ nhàng, vui tươi", "🌊 Mượt mà, thư giãn", "⚡ Năng động, sôi nổi"])
    narrator_voice = st.selectbox("Giọng kể chuyện",
        ["alloy — Trung tính, dễ nghe", "nova — Nữ, ấm áp", "echo — Nam, trầm ấm", "shimmer — Nữ, nhẹ nhàng"])
    language = st.selectbox("Ngôn ngữ lời kể", ["Tiếng Việt", "Tiếng Anh"])
    duration = st.slider("Độ dài video (giây)", min_value=10, max_value=30, value=15, step=5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('### 🎬 Bước 3 — Tạo video')

    if not uploaded:
        st.info("👈 Tải ảnh tranh vẽ trước để tiếp tục")
    elif not api_key:
        st.warning("🔑 Cần nhập API key OpenAI trước")
    else:
        if st.button("✨ Tạo Video Hoạt Hình!", use_container_width=True):
            img_bytes = uploaded.read()
            tmp_path = f"/tmp/drawing_{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)

            voice_id = narrator_voice.split(" — ")[0].strip()
            style_map = {"🌈 Nhẹ nhàng, vui tươi":"gentle","🌊 Mượt mà, thư giãn":"calm","⚡ Năng động, sôi nổi":"energetic"}
            style_key = style_map[animation_style]

            status_box = st.empty()
            progress_bar = st.progress(0)

            def update_status(msg, pct):
                status_box.markdown(f'<div class="status-msg">⏳ {msg}</div>', unsafe_allow_html=True)
                progress_bar.progress(pct)

            result = run_pipeline(
                image_path=tmp_path, voice=voice_id, language=language,
                style=style_key, duration=duration, status_cb=update_status, api_key=api_key
            )
            progress_bar.progress(100)

            if result["ok"]:
                status_box.markdown('<div class="status-msg">✅ Xong rồi!</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("#### 🎉 Video của bé đã sẵn sàng!")
                st.video(result["video_path"])
                with open(result["video_path"], "rb") as vf:
                    st.download_button(
                        label="⬇️ Tải video về máy", data=vf,
                        file_name=f"tranh_dong_{uploaded.name.split('.')[0]}.mp4",
                        mime="video/mp4", use_container_width=True
                    )
                if result.get("story"):
                    with st.expander("📖 Câu chuyện AI tạo ra"):
                        st.write(result["story"])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                status_box.error(f"❌ Lỗi: {result['error']}")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 Mẹo để có video đẹp nhất"):
        st.markdown("""
        - **Chụp ảnh thẳng góc**, tránh nghiêng hoặc bóng đổ
        - **Nền trắng** giúp AI nhận diện tranh tốt hơn
        - Tranh có **nhân vật rõ ràng** (người, con vật) sẽ có animation đẹp hơn
        - **Ánh sáng đủ** — tránh chụp thiếu sáng
        """)

st.divider()
st.markdown('<p style="text-align:center;color:#bbb;font-size:0.85rem;">🎨 Tranh Động Của Bé · Giữ nguyên nét vẽ, thổi hồn chuyển động</p>', unsafe_allow_html=True)
