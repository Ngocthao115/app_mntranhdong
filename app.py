import streamlit as st
import os, sys, base64, json, time, math, io, re
from pathlib import Path

st.set_page_config(page_title="Tranh Động Của Bé", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: linear-gradient(135deg, #FFF9F0 0%, #FFF0FB 50%, #F0F6FF 100%); min-height: 100vh; }
.hero-title {
    font-family: 'Baloo 2', cursive; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #FF6B9D, #FF9A3C, #A855F7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center; line-height: 1.2; margin-bottom: 0;
}
.hero-sub { text-align: center; color: #888; font-size: 1.05rem; margin-top: 0.3rem; margin-bottom: 1.5rem; }
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
    border-radius: 0 12px 12px 0; padding: 0.8rem 1rem;
    color: #6B21A8; font-size: 0.95rem; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img_to_data_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
    return f"data:{mime};base64,{encode_image_b64(path)}"

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1: GPT-4o phân tích tranh → danh sách vật thể + bounding box
# ══════════════════════════════════════════════════════════════════

def analyze_drawing(client, image_path: str, language: str) -> dict:
    lang = "Viết hoàn toàn bằng tiếng Việt." if language == "Tiếng Việt" else "Write in English."
    prompt = f"""Bạn là chuyên gia phân tích tranh vẽ trẻ em.
Phân tích bức tranh và trả về JSON thuần (không markdown):
{{
  "objects": [
    {{
      "id": "obj_1",
      "name": "tên vật thể (ví dụ: con cá, thuyền, mặt biển)",
      "name_en": "english name",
      "motion_type": "swim|sail|wave|fly|walk|float|sway|spin|bounce|glow",
      "motion_desc": "mô tả chuyển động cụ thể (ví dụ: bơi từ trái sang phải)",
      "layer": "foreground|midground|background",
      "bbox_percent": {{"x": 10, "y": 20, "w": 30, "h": 25}},
      "point_xy_percent": {{"x": 25, "y": 32}}
    }}
  ],
  "scene_type": "ocean|sky|forest|city|home|farm|other",
  "story": "câu chuyện 3-4 câu cho trẻ mầm non, vui vẻ",
  "background_motion": "wave|wind|rain|shine|none"
}}
bbox_percent và point_xy_percent là % so với chiều rộng/cao ảnh (0-100).
Nhận diện TẤT CẢ vật thể có thể chuyển động trong tranh.
{lang}"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":img_to_data_url(image_path),"detail":"high"}},
            {"type":"text","text":prompt}
        ]}],
        max_tokens=1500, temperature=0.5
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?","", raw); raw = re.sub(r"\n?```$","", raw)
    return json.loads(raw)

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 2: Replicate SAM2 → mask từng vật thể
# ══════════════════════════════════════════════════════════════════

def segment_objects(replicate_client, image_path: str, objects: list, img_w: int, img_h: int) -> dict:
    """
    Dùng SAM2 (Segment Anything Model 2) trên Replicate
    để tạo mask cho từng vật thể dựa trên điểm click (point prompt).
    Trả về dict: {obj_id: PIL.Image mask (RGBA)}
    """
    import replicate
    from PIL import Image
    import requests, numpy as np

    masks = {}

    # Upload ảnh lên Replicate
    with open(image_path, "rb") as f:
        image_data = f.read()

    for obj in objects:
        try:
            px = obj["point_xy_percent"]["x"] / 100.0
            py = obj["point_xy_percent"]["y"] / 100.0
            point_x = int(px * img_w)
            point_y = int(py * img_h)

            output = replicate.run(
                "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
                input={
                    "image": io.BytesIO(image_data),
                    "points": f"[[{point_x},{point_y}]]",
                    "point_labels": "[1]",
                    "multimask_output": False
                }
            )

            # output là URL của mask PNG
            if output and len(output) > 0:
                mask_url = output[0] if isinstance(output, list) else output
                r = requests.get(str(mask_url), timeout=30)
                mask_img = Image.open(io.BytesIO(r.content)).convert("L")
                mask_arr = np.array(mask_img)

                # Cắt vùng vật thể từ ảnh gốc với mask
                orig = Image.open(image_path).convert("RGBA")
                orig_arr = np.array(orig)

                # Áp dụng mask: vùng ngoài mask thành trong suốt
                result_arr = orig_arr.copy()
                result_arr[:,:,3] = mask_arr
                masks[obj["id"]] = Image.fromarray(result_arr, "RGBA")

        except Exception as e:
            # Nếu SAM lỗi với 1 vật thể → dùng bbox thay thế
            try:
                orig = Image.open(image_path).convert("RGBA")
                bx = int(obj["bbox_percent"]["x"]/100*img_w)
                by = int(obj["bbox_percent"]["y"]/100*img_h)
                bw = int(obj["bbox_percent"]["w"]/100*img_w)
                bh = int(obj["bbox_percent"]["h"]/100*img_h)
                bx = max(0,bx); by = max(0,by)
                bw = min(bw, img_w-bx); bh = min(bh, img_h-by)
                if bw > 5 and bh > 5:
                    crop = orig.crop((bx, by, bx+bw, by+bh))
                    masks[obj["id"]] = crop
            except:
                pass

    return masks


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 3: Tính toán vị trí từng frame cho từng vật thể
# ══════════════════════════════════════════════════════════════════

def compute_transform(obj: dict, t: float, img_w: int, img_h: int, mask_size: tuple) -> tuple:
    """
    Tính (dx, dy, angle, scale) cho vật thể tại thời điểm t (0→1).
    Trả về offset (dx, dy) tính từ vị trí gốc, góc xoay, và tỉ lệ.
    """
    motion = obj.get("motion_type", "float")
    mw, mh = mask_size

    # Vị trí gốc từ bbox
    base_x = int(obj["bbox_percent"]["x"]/100*img_w)
    base_y = int(obj["bbox_percent"]["y"]/100*img_h)

    dx, dy, angle, scale = 0, 0, 0.0, 1.0

    if motion == "swim":
        # Bơi: dao động ngang + lên xuống nhẹ
        amplitude = min(img_w * 0.08, 40)
        dx = int(math.sin(t * 2 * math.pi) * amplitude)
        dy = int(math.sin(t * 4 * math.pi) * 6)
        angle = math.sin(t * 2 * math.pi) * 8  # nghiêng nhẹ

    elif motion == "sail":
        # Thuyền: lắc lư + nhô nhẹ
        dx = int(math.sin(t * 2 * math.pi * 0.5) * 15)
        dy = int(math.sin(t * 2 * math.pi) * 5)
        angle = math.sin(t * 2 * math.pi * 0.5) * 5

    elif motion == "wave":
        # Sóng: gợn lên xuống theo phase
        dy = int(math.sin(t * 2 * math.pi * 1.5) * 8)
        scale = 1.0 + 0.03 * math.sin(t * 2 * math.pi)

    elif motion == "fly":
        # Bay: hình sin lên xuống + lắc nhẹ
        amplitude = min(img_h * 0.06, 25)
        dy = int(math.sin(t * 2 * math.pi * 1.2) * amplitude)
        dx = int(math.sin(t * 2 * math.pi * 0.4) * 20)
        angle = math.sin(t * 2 * math.pi) * 10

    elif motion == "walk":
        dx = int(math.sin(t * 2 * math.pi * 0.7) * 12)
        dy = int(abs(math.sin(t * 4 * math.pi)) * 4)  # nhún nhảy

    elif motion == "float":
        dy = int(math.sin(t * 2 * math.pi * 0.8) * 10)
        angle = math.sin(t * 2 * math.pi * 0.5) * 5

    elif motion == "sway":
        angle = math.sin(t * 2 * math.pi * 0.6) * 15
        dy = int(math.sin(t * 2 * math.pi * 0.3) * 5)

    elif motion == "spin":
        angle = t * 360 * 0.3  # xoay chậm

    elif motion == "bounce":
        dy = -int(abs(math.sin(t * 2 * math.pi * 1.5)) * 20)
        scale = 1.0 + 0.05 * abs(math.sin(t * 2 * math.pi * 1.5))

    elif motion == "glow":
        scale = 1.0 + 0.08 * math.sin(t * 2 * math.pi)

    return base_x + dx, base_y + dy, angle, scale


def apply_background_motion(bg: "Image.Image", t: float, bg_motion: str) -> "Image.Image":
    """Áp dụng chuyển động nền (sóng, gió...)"""
    from PIL import Image
    import numpy as np

    if bg_motion == "wave":
        arr = np.array(bg.convert("RGBA"), dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for y in range(h):
            offset = int(math.sin((y / h * 4 * math.pi) + t * 2 * math.pi) * 3)
            src_x = np.clip(np.arange(w) - offset, 0, w-1).astype(int)
            result[y] = arr[y, src_x]
        return Image.fromarray(result.astype(np.uint8), "RGBA")

    elif bg_motion == "wind":
        arr = np.array(bg.convert("RGBA"), dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for x in range(w):
            offset = int(math.sin((x / w * 3 * math.pi) + t * 2 * math.pi) * 2)
            src_y = np.clip(np.arange(h) - offset, 0, h-1).astype(int)
            result[:, x] = arr[src_y, x]
        return Image.fromarray(result.astype(np.uint8), "RGBA")

    elif bg_motion == "shine":
        from PIL import ImageEnhance
        brightness = 1.0 + 0.08 * math.sin(t * 2 * math.pi * 0.5)
        return ImageEnhance.Brightness(bg).enhance(brightness)

    return bg


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 4: Render từng frame
# ══════════════════════════════════════════════════════════════════

def render_frame(
    base_img: "Image.Image",
    masks: dict,
    objects: list,
    t: float,
    bg_motion: str,
    img_w: int,
    img_h: int
) -> "Image.Image":
    from PIL import Image

    # Nền với motion
    canvas = apply_background_motion(base_img.convert("RGBA"), t, bg_motion)

    # Sắp xếp objects theo layer: background → midground → foreground
    layer_order = {"background": 0, "midground": 1, "foreground": 2}
    sorted_objs = sorted(objects, key=lambda o: layer_order.get(o.get("layer","midground"), 1))

    for obj in sorted_objs:
        obj_id = obj["id"]
        if obj_id not in masks:
            continue

        mask_img = masks[obj_id].convert("RGBA")
        mw, mh = mask_img.size
        if mw < 3 or mh < 3:
            continue

        new_x, new_y, angle, scale = compute_transform(obj, t, img_w, img_h, (mw, mh))

        # Scale
        if scale != 1.0:
            new_size = (max(1, int(mw*scale)), max(1, int(mh*scale)))
            layer = mask_img.resize(new_size, Image.LANCZOS)
        else:
            layer = mask_img

        # Rotate (xoay quanh tâm)
        if abs(angle) > 0.1:
            layer = layer.rotate(-angle, expand=True, resample=Image.BICUBIC)

        # Paste lên canvas
        lw, lh = layer.size
        paste_x = new_x - lw//2 + mw//2
        paste_y = new_y - lh//2 + mh//2

        # Giới hạn trong canvas
        paste_x = max(-lw, min(paste_x, img_w))
        paste_y = max(-lh, min(paste_y, img_h))

        canvas.paste(layer, (paste_x, paste_y), layer)

    return canvas.convert("RGB")


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 5: Tạo audio
# ══════════════════════════════════════════════════════════════════

def generate_audio(client, story: str, voice: str, output_path: str):
    resp = client.audio.speech.create(model="tts-1-hd", voice=voice, input=story, speed=0.88)
    resp.stream_to_file(output_path)

# ══════════════════════════════════════════════════════════════════
#  PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════

def run_pipeline(image_path, voice, language, duration, status_cb, openai_key, replicate_key):
    try:
        import numpy as np
        from PIL import Image
        import moviepy.editor as mpy
        from openai import OpenAI
        import replicate as replicate_module
        import os as _os

        _os.environ["REPLICATE_API_TOKEN"] = replicate_key
        client = OpenAI(api_key=openai_key)
        out_dir = Path("/tmp/drawing_anim_v2"); out_dir.mkdir(exist_ok=True)
        ts = int(time.time())

        # 1. Load ảnh
        orig = Image.open(image_path).convert("RGBA")
        orig.thumbnail((800, 800), Image.LANCZOS)
        w, h = orig.size
        w = w if w%2==0 else w-1; h = h if h%2==0 else h-1
        orig = orig.crop((0,0,w,h))

        # 2. Phân tích
        status_cb("🔍 AI đang đọc và hiểu bức tranh của bé...", 8)
        analysis = analyze_drawing(client, image_path, language)
        objects = analysis.get("objects", [])
        bg_motion = analysis.get("background_motion", "none")
        story = analysis.get("story", "Bé vẽ một bức tranh thật đẹp!")

        if not objects:
            return {"ok": False, "error": "Không nhận diện được vật thể trong tranh. Thử ảnh có nét vẽ rõ hơn nhé!"}

        status_cb(f"✅ Nhận diện được {len(objects)} vật thể: {', '.join([o['name'] for o in objects])}", 20)

        # 3. Segment từng vật thể
        status_cb("✂️ Đang tách từng vật thể ra để làm animation riêng...", 30)
        masks = segment_objects(None, image_path, objects, w, h)
        status_cb(f"✅ Đã tách được {len(masks)}/{len(objects)} vật thể thành công", 50)

        # 4. Tạo audio
        status_cb("🎙️ Đang tạo giọng kể chuyện...", 55)
        audio_path = str(out_dir / f"audio_{ts}.mp3")
        generate_audio(client, story, voice, audio_path)

        # 5. Render frames
        status_cb("🎨 Đang vẽ từng khung hình animation...", 62)
        fps = 24
        total_frames = duration * fps
        frames = []

        for i in range(total_frames):
            t = i / total_frames
            frame = render_frame(orig, masks, objects, t, bg_motion, w, h)
            frames.append(np.array(frame))

            if i % (fps*2) == 0:
                pct = 62 + int((i/total_frames) * 28)
                status_cb(f"🎨 Đang render frame {i+1}/{total_frames}...", pct)

        # 6. Ghép video
        status_cb("🎬 Đang ghép video hoàn chỉnh...", 92)
        video_path = str(out_dir / f"video_{ts}.mp4")
        video_clip = mpy.ImageSequenceClip(frames, fps=fps)
        audio_clip = mpy.AudioFileClip(audio_path)
        if audio_clip.duration < video_clip.duration:
            audio_clip = audio_clip.audio_loop(duration=video_clip.duration)
        else:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        final = video_clip.set_audio(audio_clip)
        final.write_videofile(video_path, fps=fps, codec="libx264", audio_codec="aac",
                              logger=None, temp_audiofile="/tmp/tmp_audio.m4a", remove_temp=True)

        return {"ok": True, "video_path": video_path, "story": story,
                "objects": [o["name"] for o in objects], "masks_count": len(masks)}

    except Exception as e:
        import traceback
        return {"ok": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════════
#  GIAO DIỆN
# ══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title">✨ Tranh Động Của Bé ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Mỗi nét vẽ của bé đều sống động — từng con cá bơi, từng con thuyền lắc lư, từng ngọn sóng gợn lên</p>', unsafe_allow_html=True)

# API Keys
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_key = os.environ.get("OPENAI_API_KEY","")
try:
    replicate_key = st.secrets["REPLICATE_API_TOKEN"]
except:
    replicate_key = os.environ.get("REPLICATE_API_TOKEN","")

if not openai_key or not replicate_key:
    with st.expander("⚙️ Cài đặt API Keys", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", value=openai_key)
        with c2:
            replicate_key = st.text_input("Replicate API Token", type="password",
                                          placeholder="r8_...",
                                          help="Đăng ký miễn phí tại replicate.com — nhận $5 credit", value=replicate_key)
        if openai_key and replicate_key:
            st.success("✅ Đã có đủ API keys!")

st.divider()

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Bước 1 — Tải ảnh tranh vẽ của bé")
    uploaded = st.file_uploader("Chụp ảnh hoặc chọn file", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, caption="Tranh vẽ của bé", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Bước 2 — Tuỳ chỉnh")
    narrator_voice = st.selectbox("Giọng kể chuyện",
        ["nova — Nữ, ấm áp", "alloy — Trung tính", "echo — Nam, trầm", "shimmer — Nữ, nhẹ nhàng"])
    language = st.selectbox("Ngôn ngữ lời kể", ["Tiếng Việt", "Tiếng Anh"])
    duration = st.slider("Độ dài video (giây)", 10, 30, 20, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🎬 Bước 3 — Tạo video")

    if not uploaded:
        st.info("👈 Tải ảnh tranh vẽ trước")
    elif not openai_key or not replicate_key:
        st.warning("🔑 Cần nhập đủ 2 API keys")
    else:
        if st.button("✨ Tạo Video Hoạt Hình!", use_container_width=True):
            img_bytes = uploaded.read()
            tmp_path = f"/tmp/drawing_{uploaded.name}"
            with open(tmp_path,"wb") as f: f.write(img_bytes)

            voice_id = narrator_voice.split(" — ")[0].strip()
            status_box = st.empty()
            progress_bar = st.progress(0)

            def update_status(msg, pct):
                status_box.markdown(f'<div class="status-msg">{msg}</div>', unsafe_allow_html=True)
                progress_bar.progress(min(pct,99))

            result = run_pipeline(
                image_path=tmp_path, voice=voice_id, language=language,
                duration=duration, status_cb=update_status,
                openai_key=openai_key, replicate_key=replicate_key
            )
            progress_bar.progress(100)

            if result["ok"]:
                status_box.success("✅ Xong! Tranh của bé đã sống động rồi!")
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("#### 🎉 Video của bé đã sẵn sàng!")

                if result.get("objects"):
                    st.caption(f"🎭 Các nhân vật chuyển động: {' · '.join(result['objects'])}")

                st.video(result["video_path"])
                with open(result["video_path"],"rb") as vf:
                    st.download_button("⬇️ Tải video về máy", data=vf,
                        file_name=f"tranh_dong_{uploaded.name.split('.')[0]}.mp4",
                        mime="video/mp4", use_container_width=True)
                if result.get("story"):
                    with st.expander("📖 Câu chuyện AI tạo ra"):
                        st.write(result["story"])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                status_box.error(f"❌ Lỗi: {result['error']}")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 Mẹo để animation đẹp nhất"):
        st.markdown("""
        - **Nét vẽ rõ ràng**, tách biệt giữa các vật thể
        - **Nền trắng** hoặc màu đơn giản giúp AI tách layer tốt hơn
        - Mỗi vật thể nên **không chồng chéo** nhau quá nhiều
        - Ảnh **chụp thẳng góc**, đủ sáng
        - Tranh có **3-6 vật thể** cho kết quả animation đẹp nhất
        """)

st.divider()
st.markdown('<p style="text-align:center;color:#bbb;font-size:0.85rem;">🎨 Tranh Động Của Bé · Từng nét vẽ đều có hồn</p>', unsafe_allow_html=True)
