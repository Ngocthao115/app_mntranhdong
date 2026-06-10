import streamlit as st
import os, base64, json, time, math, io, re
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

def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img_to_data_url(path):
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
    return f"data:{mime};base64,{encode_image_b64(path)}"

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1: GPT-4o phân tích tranh
# ══════════════════════════════════════════════════════════════════

def analyze_drawing(client, image_path, language):
    lang = "Viết hoàn toàn bằng tiếng Việt." if language == "Tiếng Việt" else "Write in English."
    prompt = f"""Bạn là chuyên gia phân tích tranh vẽ trẻ em.
Phân tích bức tranh và trả về JSON thuần (không markdown, không backtick):
{{
  "objects": [
    {{
      "id": "obj_1",
      "name": "tên vật thể",
      "motion_type": "swim|sail|wave|fly|walk|float|sway|bounce|glow",
      "layer": "foreground|midground|background",
      "bbox_percent": {{"x": 10, "y": 20, "w": 30, "h": 25}},
      "center_percent": {{"x": 25, "y": 32}}
    }}
  ],
  "background_motion": "wave|wind|shine|none",
  "story": "câu chuyện 3-4 câu cho trẻ mầm non, vui vẻ, kết thúc tích cực"
}}
Quy tắc bbox_percent: x,y là góc trên trái (% so với width/height ảnh), w,h là kích thước (%).
Nhận diện TẤT CẢ vật thể rõ ràng trong tranh. Tối đa 8 vật thể.
{lang}"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":img_to_data_url(image_path),"detail":"high"}},
            {"type":"text","text":prompt}
        ]}],
        max_tokens=1500, temperature=0.3
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?","",raw); raw = re.sub(r"\n?```$","",raw)
    return json.loads(raw)

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 2: Tách vật thể bằng OpenCV GrabCut
#  Giữ NGUYÊN nét vẽ — chỉ tách vùng để di chuyển
# ══════════════════════════════════════════════════════════════════

def segment_with_grabcut(image_path, objects, img_w, img_h):
    """
    Dùng GrabCut để tách từng vật thể theo bbox GPT-4o cung cấp.
    Kết quả: dict {obj_id: PIL.Image RGBA với nền trong suốt}
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Đọc ảnh gốc bằng OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {}
    img_bgr = cv2.resize(img_bgr, (img_w, img_h))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masks = {}

    for obj in objects:
        try:
            bp = obj["bbox_percent"]
            x  = max(0, int(bp["x"]/100 * img_w))
            y  = max(0, int(bp["y"]/100 * img_h))
            w  = max(10, int(bp["w"]/100 * img_w))
            h  = max(10, int(bp["h"]/100 * img_h))

            # Clamp
            x2 = min(x+w, img_w); y2 = min(y+h, img_h)
            w  = x2-x; h = y2-y
            if w < 8 or h < 8:
                continue

            rect = (x, y, w, h)

            # GrabCut — tách vật thể khỏi nền
            gc_mask  = np.zeros(img_rgb.shape[:2], np.uint8)
            bgd_model = np.zeros((1,65), np.float64)
            fgd_model = np.zeros((1,65), np.float64)

            cv2.grabCut(img_rgb, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Vùng foreground (chắc chắn + có thể)
            fg_mask = np.where((gc_mask==2)|(gc_mask==0), 0, 255).astype(np.uint8)

            # Làm mịn mask — giữ nguyên nét vẽ
            kernel = np.ones((3,3), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg_mask = cv2.GaussianBlur(fg_mask, (3,3), 0)

            # Kiểm tra mask có nội dung không
            if fg_mask.sum() < 500:
                # Fallback: dùng bbox trực tiếp nếu GrabCut thất bại
                fg_mask = np.zeros(img_rgb.shape[:2], np.uint8)
                fg_mask[y:y2, x:x2] = 255

            # Tạo ảnh RGBA: vật thể giữ nguyên màu, nền trong suốt
            rgba = np.dstack([img_rgb, fg_mask])
            obj_img = Image.fromarray(rgba, "RGBA")

            # Crop vừa vặn với vùng bbox (có padding nhỏ)
            pad = 5
            crop_x1 = max(0, x-pad); crop_y1 = max(0, y-pad)
            crop_x2 = min(img_w, x2+pad); crop_y2 = min(img_h, y2+pad)
            cropped = obj_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            masks[obj["id"]] = {
                "image": cropped,
                "origin_x": crop_x1,
                "origin_y": crop_y1,
            }

        except Exception as e:
            # Fallback tuyệt đối: crop bbox thô
            try:
                from PIL import Image as PILImage
                pil = PILImage.open(image_path).convert("RGBA").resize((img_w, img_h))
                bp = obj["bbox_percent"]
                fx = max(0,int(bp["x"]/100*img_w)); fy = max(0,int(bp["y"]/100*img_h))
                fw = max(10,int(bp["w"]/100*img_w)); fh = max(10,int(bp["h"]/100*img_h))
                crop = pil.crop((fx,fy,min(fx+fw,img_w),min(fy+fh,img_h)))
                masks[obj["id"]] = {"image": crop, "origin_x": fx, "origin_y": fy}
            except:
                pass

    return masks

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 3: Tính chuyển động cho từng vật thể
# ══════════════════════════════════════════════════════════════════

def get_offset(obj, t, img_w, img_h):
    """Tính (dx, dy, angle_deg, scale) tại thời điểm t (0→1)"""
    motion = obj.get("motion_type","float")
    dx, dy, angle, scale = 0, 0, 0.0, 1.0

    if motion == "swim":
        dx    = int(math.sin(t * 2*math.pi) * min(img_w*0.07, 35))
        dy    = int(math.sin(t * 4*math.pi) * 5)
        angle = math.sin(t * 2*math.pi) * 10

    elif motion == "sail":
        dx    = int(math.sin(t * math.pi) * 12)
        dy    = int(math.sin(t * 2*math.pi) * 5)
        angle = math.sin(t * math.pi) * 6

    elif motion == "wave":
        dy    = int(math.sin(t * 2*math.pi * 1.5) * 9)
        scale = 1.0 + 0.02*math.sin(t * 2*math.pi)

    elif motion == "fly":
        dy    = int(math.sin(t * 2*math.pi * 1.2) * min(img_h*0.05,22))
        dx    = int(math.sin(t * 2*math.pi * 0.4) * 18)
        angle = math.sin(t * 2*math.pi) * 12

    elif motion == "walk":
        dx    = int(math.sin(t * 2*math.pi * 0.6) * 10)
        dy    = int(abs(math.sin(t * 4*math.pi)) * 4)

    elif motion == "float":
        dy    = int(math.sin(t * 2*math.pi * 0.7) * 12)
        angle = math.sin(t * 2*math.pi * 0.4) * 6

    elif motion == "sway":
        angle = math.sin(t * 2*math.pi * 0.5) * 18
        dy    = int(math.sin(t * 2*math.pi * 0.3) * 4)

    elif motion == "bounce":
        dy    = -int(abs(math.sin(t * 2*math.pi * 1.5)) * 18)
        scale = 1.0 + 0.04*abs(math.sin(t * 2*math.pi * 1.5))

    elif motion == "glow":
        scale = 1.0 + 0.09*math.sin(t * 2*math.pi)

    return dx, dy, angle, scale


def apply_bg_motion(bg_rgba, t, bg_motion):
    """Chuyển động nền: sóng biển, gió, ánh sáng"""
    import numpy as np
    from PIL import Image, ImageEnhance

    if bg_motion == "wave":
        arr = np.array(bg_rgba, dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for row in range(h):
            offset = int(math.sin((row/h*5*math.pi) + t*2*math.pi) * 4)
            cols = np.clip(np.arange(w) - offset, 0, w-1).astype(int)
            result[row] = arr[row, cols]
        return Image.fromarray(result.astype(np.uint8), "RGBA")

    elif bg_motion == "wind":
        arr = np.array(bg_rgba, dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for col in range(w):
            offset = int(math.sin((col/w*4*math.pi) + t*2*math.pi) * 3)
            rows = np.clip(np.arange(h) - offset, 0, h-1).astype(int)
            result[:, col] = arr[rows, col]
        return Image.fromarray(result.astype(np.uint8), "RGBA")

    elif bg_motion == "shine":
        b = 1.0 + 0.07*math.sin(t * 2*math.pi * 0.4)
        return ImageEnhance.Brightness(bg_rgba).enhance(b)

    return bg_rgba

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 4: Render từng frame
# ══════════════════════════════════════════════════════════════════

def render_frame(base_img_rgba, masks, objects, t, bg_motion, img_w, img_h):
    from PIL import Image
    import numpy as np

    # Nền với hiệu ứng chuyển động
    canvas = apply_bg_motion(base_img_rgba.copy(), t, bg_motion)

    # Thứ tự vẽ: background → midground → foreground
    layer_order = {"background":0, "midground":1, "foreground":2}
    sorted_objs = sorted(objects, key=lambda o: layer_order.get(o.get("layer","midground"),1))

    for obj in sorted_objs:
        oid = obj["id"]
        if oid not in masks:
            continue

        info   = masks[oid]
        layer  = info["image"].convert("RGBA")
        orig_x = info["origin_x"]
        orig_y = info["origin_y"]
        lw, lh = layer.size

        if lw < 4 or lh < 4:
            continue

        # Tính offset chuyển động
        dx, dy, angle, scale = get_offset(obj, t, img_w, img_h)

        # Scale
        if abs(scale-1.0) > 0.001:
            new_lw = max(2, int(lw*scale))
            new_lh = max(2, int(lh*scale))
            layer = layer.resize((new_lw, new_lh), Image.LANCZOS)
            lw, lh = new_lw, new_lh

        # Rotate quanh tâm vật thể
        if abs(angle) > 0.2:
            layer = layer.rotate(-angle, expand=True, resample=Image.BICUBIC)
            lw, lh = layer.size

        # Vị trí paste
        px = orig_x + dx
        py = orig_y + dy

        # Giới hạn trong canvas
        px = max(-lw//2, min(px, img_w))
        py = max(-lh//2, min(py, img_h))

        canvas.paste(layer, (px, py), layer)

    return canvas.convert("RGB")

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 5: Audio TTS
# ══════════════════════════════════════════════════════════════════

def generate_audio(client, story, voice, output_path):
    resp = client.audio.speech.create(model="tts-1-hd", voice=voice, input=story, speed=0.88)
    resp.stream_to_file(output_path)

# ══════════════════════════════════════════════════════════════════
#  PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════

def run_pipeline(image_path, voice, language, duration, status_cb, openai_key):
    try:
        import numpy as np
        from PIL import Image
        import moviepy.editor as mpy
        from openai import OpenAI

        client  = OpenAI(api_key=openai_key)
        out_dir = Path("/tmp/drawing_anim_v3"); out_dir.mkdir(exist_ok=True)
        ts = int(time.time())

        # Load & chuẩn hoá ảnh
        orig = Image.open(image_path).convert("RGBA")
        orig.thumbnail((800,800), Image.LANCZOS)
        w, h = orig.size
        w = w if w%2==0 else w-1
        h = h if h%2==0 else h-1
        orig = orig.crop((0,0,w,h))

        # Lưu bản resize để OpenCV dùng
        resized_path = str(out_dir / f"resized_{ts}.png")
        orig.convert("RGB").save(resized_path)

        # 1. Phân tích tranh
        status_cb("🔍 GPT-4o đang đọc và hiểu bức tranh...", 8)
        analysis   = analyze_drawing(client, image_path, language)
        objects    = analysis.get("objects", [])
        bg_motion  = analysis.get("background_motion", "none")
        story      = analysis.get("story", "Bé vẽ một bức tranh thật đẹp!")

        if not objects:
            return {"ok":False, "error":"Không nhận diện được vật thể. Hãy thử ảnh có nét vẽ rõ hơn!"}

        names = [o["name"] for o in objects]
        status_cb(f"✅ Nhận diện: {' · '.join(names)}", 18)

        # 2. Segment bằng GrabCut
        status_cb("✂️ Đang tách từng vật thể bằng GrabCut (giữ nguyên nét vẽ)...", 28)
        masks = segment_with_grabcut(resized_path, objects, w, h)
        status_cb(f"✅ Tách được {len(masks)}/{len(objects)} vật thể", 42)

        # 3. Audio
        status_cb("🎙️ Đang tạo giọng kể chuyện...", 48)
        audio_path = str(out_dir / f"audio_{ts}.mp3")
        generate_audio(client, story, voice, audio_path)

        # 4. Render frames
        fps          = 24
        total_frames = duration * fps
        frames       = []

        status_cb("🎨 Đang render animation từng vật thể...", 55)

        for i in range(total_frames):
            t     = i / total_frames
            frame = render_frame(orig.copy(), masks, objects, t, bg_motion, w, h)
            frames.append(np.array(frame))
            if i % (fps*3) == 0:
                pct = 55 + int((i/total_frames)*33)
                status_cb(f"🎨 Đang vẽ frame {i+1}/{total_frames}...", pct)

        # 5. Ghép video
        status_cb("🎬 Đang ghép video hoàn chỉnh...", 90)
        video_path = str(out_dir / f"video_{ts}.mp4")
        vc = mpy.ImageSequenceClip(frames, fps=fps)
        ac = mpy.AudioFileClip(audio_path)
        ac = ac.audio_loop(duration=vc.duration) if ac.duration < vc.duration else ac.subclip(0, vc.duration)
        vc.set_audio(ac).write_videofile(
            video_path, fps=fps, codec="libx264", audio_codec="aac",
            logger=None, temp_audiofile="/tmp/tmp_audio_v3.m4a", remove_temp=True
        )

        return {"ok":True, "video_path":video_path, "story":story, "objects":names}

    except Exception as e:
        import traceback
        return {"ok":False, "error":f"{str(e)}\n{traceback.format_exc()}"}

# ══════════════════════════════════════════════════════════════════
#  GIAO DIỆN
# ══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title">✨ Tranh Động Của Bé ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Mỗi nét vẽ của bé đều sống động — từng con cá bơi, từng con thuyền lắc lư, từng ngọn sóng gợn lên</p>', unsafe_allow_html=True)

# API Key
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_key = os.environ.get("OPENAI_API_KEY","")

if not openai_key:
    with st.expander("⚙️ Cài đặt API Key", expanded=True):
        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if openai_key:
            st.success("✅ Đã lưu!")

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
        ["nova — Nữ, ấm áp","alloy — Trung tính","echo — Nam, trầm","shimmer — Nữ, nhẹ nhàng"])
    language = st.selectbox("Ngôn ngữ lời kể", ["Tiếng Việt","Tiếng Anh"])
    duration = st.slider("Độ dài video (giây)", 10, 30, 20, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🎬 Bước 3 — Tạo video")

    if not uploaded:
        st.info("👈 Tải ảnh tranh vẽ trước")
    elif not openai_key:
        st.warning("🔑 Cần nhập API key OpenAI")
    else:
        if st.button("✨ Tạo Video Hoạt Hình!", use_container_width=True):
            img_bytes = uploaded.read()
            tmp_path  = f"/tmp/drawing_{uploaded.name}"
            with open(tmp_path,"wb") as f: f.write(img_bytes)

            voice_id     = narrator_voice.split(" — ")[0].strip()
            status_box   = st.empty()
            progress_bar = st.progress(0)

            def update_status(msg, pct):
                status_box.markdown(f'<div class="status-msg">{msg}</div>', unsafe_allow_html=True)
                progress_bar.progress(min(pct,99))

            result = run_pipeline(
                image_path=tmp_path, voice=voice_id, language=language,
                duration=duration, status_cb=update_status, openai_key=openai_key
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
        - **Nét vẽ rõ ràng**, màu đậm tách biệt với nền
        - **Nền trắng hoặc màu nhạt** giúp tách vật thể tốt hơn
        - Mỗi vật thể nên **không chồng chéo** nhau quá nhiều
        - Ảnh **chụp thẳng góc**, đủ sáng, tránh bóng đổ
        - Tranh có **3-6 vật thể** cho animation đẹp nhất
        """)

st.divider()
st.markdown('<p style="text-align:center;color:#bbb;font-size:0.85rem;">🎨 Tranh Động Của Bé · Từng nét vẽ đều có hồn</p>', unsafe_allow_html=True)
