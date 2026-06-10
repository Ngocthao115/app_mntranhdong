import streamlit as st
import os, base64, json, time, math, re
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
.step-card { background: white; border-radius: 20px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 2px solid #F5E6FF; margin-bottom: 1rem; }
.stButton > button {
    background: linear-gradient(135deg, #A855F7, #EC4899) !important;
    color: white !important; border: none !important; border-radius: 50px !important;
    padding: 0.7rem 2rem !important; font-family: 'Baloo 2', cursive !important;
    font-weight: 600 !important; font-size: 1.05rem !important;
    box-shadow: 0 4px 15px rgba(168,85,247,0.3) !important; width: 100% !important;
}
.result-box { background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 4px 30px rgba(168,85,247,0.12); border: 2px solid #E9D5FF; text-align: center; }
.status-msg { background: #F5F0FF; border-left: 4px solid #A855F7; border-radius: 0 12px 12px 0; padding: 0.8rem 1rem; color: #6B21A8; font-size: 0.95rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════

def img_to_data_url(path):
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
    with open(path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1: GPT-4o phân tích tranh
# ══════════════════════════════════════════════════════════════════

def analyze_drawing(client, image_path, language):
    lang = "Viết hoàn toàn bằng tiếng Việt." if language == "Tiếng Việt" else "Write in English."
    prompt = f"""Phân tích tranh vẽ tay của trẻ. Trả về JSON thuần (không markdown):
{{
  "objects": [
    {{
      "id": "obj_1",
      "name": "tên vật thể",
      "motion_type": "swim|sail|wave|fly|walk|float|sway|bounce|glow",
      "layer": "foreground|midground|background",
      "bbox_percent": {{"x": 10, "y": 20, "w": 30, "h": 25}},
      "dominant_color_hint": "blue|red|yellow|green|orange|purple|brown|white|black|mixed"
    }}
  ],
  "background_motion": "wave|wind|shine|none",
  "story": "câu chuyện 3-4 câu vui cho trẻ mầm non"
}}
bbox_percent: x,y góc trên trái (% của width/height), w,h kích thước (%).
Nhận diện TẤT CẢ vật thể rõ ràng. Tối đa 8 vật thể.
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
#  BƯỚC 2: TÁCH VẬT THỂ — CẢI TIẾN VỚI 3 KỸ THUẬT
# ══════════════════════════════════════════════════════════════════

def preprocess_for_grabcut(img_bgr):
    """
    KỸ THUẬT 1: Tiền xử lý ảnh
    - Tăng contrast nét vẽ bằng CLAHE
    - Làm mịn nhiễu nền giấy bằng bilateral filter (GIỮ NGUYÊN nét vẽ)
    - Chuyển sang LAB color space (tách màu tốt hơn BGR)
    """
    import cv2
    import numpy as np

    # Bilateral filter: làm mịn nền giấy nhưng KHÔNG làm mờ nét vẽ
    smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE tăng contrast trên kênh L của LAB
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # Trả về cả BGR gốc (để GrabCut) và LAB (để phân tích màu)
    return smooth, lab_enhanced


def build_grabcut_mask(img_bgr, bbox, img_w, img_h, color_hint="mixed"):
    """
    KỸ THUẬT 2: GrabCut nâng cao
    - Khởi tạo với GC_INIT_WITH_MASK (chính xác hơn RECT)
    - Đánh dấu probable foreground dựa trên màu sắc
    - Tăng iterCount lên 10
    - Multi-pass: chạy lại 1 lần nữa với mask đã được tinh chỉnh
    """
    import cv2
    import numpy as np

    x  = max(0, int(bbox["x"]/100 * img_w))
    y  = max(0, int(bbox["y"]/100 * img_h))
    w  = max(20, int(bbox["w"]/100 * img_w))
    h  = max(20, int(bbox["h"]/100 * img_h))
    x2 = min(x+w, img_w); y2 = min(y+h, img_h)
    w  = x2-x; h = y2-y

    if w < 10 or h < 10:
        return None

    # Khởi tạo mask với 3 vùng rõ ràng:
    # GC_BGD (0) = chắc chắn nền (ngoài bbox + padding)
    # GC_PR_BGD (2) = có thể nền (vùng rìa bbox)
    # GC_PR_FGD (3) = có thể foreground (vùng giữa bbox)
    # GC_FGD (1) = chắc chắn foreground (trung tâm bbox)
    gc_mask = np.zeros(img_bgr.shape[:2], np.uint8)  # toàn bộ = GC_BGD

    # Vùng bbox ngoài = có thể nền
    pad = max(5, min(15, w//8, h//8))
    gc_mask[y:y2, x:x2] = cv2.GC_PR_BGD  # vùng bbox = có thể nền

    # Vùng trong bbox (co vào pad) = có thể foreground
    inner_x1 = min(x+pad, x2-1); inner_y1 = min(y+pad, y2-1)
    inner_x2 = max(x2-pad, x+1); inner_y2 = max(y2-pad, y+1)
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        gc_mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_PR_FGD

    # Tâm bbox = chắc chắn foreground
    cx1 = x + w//3; cy1 = y + h//3
    cx2 = x + 2*w//3; cy2 = y + 2*h//3
    if cx2 > cx1 and cy2 > cy1:
        gc_mask[cy1:cy2, cx1:cx2] = cv2.GC_FGD

    # Hint màu sắc: vùng trong bbox có màu gần với color_hint → probable FGD
    color_map = {
        "blue":   ([90,50,20],  [130,255,255]),   # HSV
        "red":    ([0,100,100], [10,255,255]),
        "yellow": ([20,100,100],[35,255,255]),
        "green":  ([40,50,20],  [80,255,255]),
        "orange": ([10,100,100],[20,255,255]),
        "purple": ([130,50,20], [160,255,255]),
        "brown":  ([5,50,20],   [15,200,200]),
    }
    if color_hint in color_map:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lo, hi = color_map[color_hint]
        color_mask = cv2.inRange(hsv, tuple(lo), tuple(hi))
        # Chỉ áp dụng trong vùng bbox
        roi_color = color_mask[y:y2, x:x2]
        roi_gc    = gc_mask[y:y2, x:x2]
        roi_gc[roi_color > 0] = cv2.GC_PR_FGD
        gc_mask[y:y2, x:x2] = roi_gc

    # GrabCut lần 1 — 10 lần lặp
    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)
    except Exception:
        # Fallback về RECT mode nếu MASK lỗi
        rect = (x, y, w, h)
        gc_mask2 = np.zeros(img_bgr.shape[:2], np.uint8)
        bgd_model2 = np.zeros((1,65), np.float64)
        fgd_model2 = np.zeros((1,65), np.float64)
        cv2.grabCut(img_bgr, gc_mask2, rect, bgd_model2, fgd_model2, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((gc_mask2==2)|(gc_mask2==0), 0, 255).astype(np.uint8)

    # Multi-pass: chạy lại 5 lần nữa để tinh chỉnh
    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_EVAL)

    fg_mask = np.where((gc_mask==2)|(gc_mask==0), 0, 255).astype(np.uint8)
    return fg_mask


def postprocess_mask(fg_mask, img_bgr, bbox, img_w, img_h):
    """
    KỸ THUẬT 3: Hậu xử lý mask
    - Morphological close để lấp lỗ hổng bên trong vật thể
    - Connected components để loại vùng nhỏ nhiễu
    - Feathering viền để mask mềm, tự nhiên
    """
    import cv2
    import numpy as np

    if fg_mask is None or fg_mask.sum() < 200:
        return None

    # 3a. Morphological close: lấp lỗ nhỏ bên trong
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # 3b. Dilate nhẹ để bao phủ nét vẽ ở rìa
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)

    # 3c. Connected components: giữ lại vùng lớn nhất (vật thể chính)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if n_labels > 2:  # có hơn 1 vùng (label 0 = nền)
        # Tìm vùng lớn nhất (không tính nền)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        # Giữ vùng lớn nhất + các vùng gần bbox center > 10% max_area
        max_area = areas[np.argmax(areas)]
        clean_mask = np.zeros_like(fg_mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] > max_area * 0.1:
                clean_mask[labels == lbl] = 255
        fg_mask = clean_mask

    # 3d. Feathering: làm mềm viền để tránh viền cứng như hình chữ nhật
    fg_float = fg_mask.astype(np.float32) / 255.0
    blurred  = cv2.GaussianBlur(fg_float, (11,11), 0)
    # Kết hợp: vùng lõi giữ nguyên, viền được làm mềm
    alpha = np.where(fg_mask > 200, fg_float, blurred)
    alpha = np.clip(alpha, 0, 1)
    fg_soft = (alpha * 255).astype(np.uint8)

    return fg_soft


def segment_objects_v4(image_path, objects, img_w, img_h):
    """
    Pipeline tách vật thể đầy đủ với 3 kỹ thuật cải thiện.
    Trả về dict {obj_id: {"image": PIL.RGBA, "origin_x": int, "origin_y": int}}
    """
    import cv2
    import numpy as np
    from PIL import Image

    img_bgr_orig = cv2.imread(image_path)
    if img_bgr_orig is None:
        return {}
    img_bgr_orig = cv2.resize(img_bgr_orig, (img_w, img_h))

    # KỸ THUẬT 1: Tiền xử lý
    img_smooth, img_lab = preprocess_for_grabcut(img_bgr_orig)
    img_rgb = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

    masks = {}
    for obj in objects:
        try:
            bbox  = obj["bbox_percent"]
            color = obj.get("dominant_color_hint", "mixed")

            # KỸ THUẬT 2: GrabCut nâng cao (dùng ảnh đã smooth)
            fg_mask = build_grabcut_mask(img_smooth, bbox, img_w, img_h, color)

            # KỸ THUẬT 3: Hậu xử lý
            fg_soft = postprocess_mask(fg_mask, img_smooth, bbox, img_w, img_h)

            if fg_soft is None or fg_soft.sum() < 100:
                raise ValueError("Mask rỗng")

            # Tạo ảnh RGBA: giữ NGUYÊN màu sắc nét vẽ gốc, chỉ thêm alpha
            rgba = np.dstack([img_rgb, fg_soft])
            obj_img_full = Image.fromarray(rgba, "RGBA")

            # Crop về vùng bbox có padding
            x  = max(0, int(bbox["x"]/100*img_w) - 8)
            y  = max(0, int(bbox["y"]/100*img_h) - 8)
            x2 = min(img_w, int((bbox["x"]+bbox["w"])/100*img_w) + 8)
            y2 = min(img_h, int((bbox["y"]+bbox["h"])/100*img_h) + 8)
            cropped = obj_img_full.crop((x, y, x2, y2))

            masks[obj["id"]] = {"image": cropped, "origin_x": x, "origin_y": y}

        except Exception:
            # Fallback: crop bbox trực tiếp (không tách nền)
            try:
                pil = Image.fromarray(img_rgb).convert("RGBA")
                bx = max(0, int(obj["bbox_percent"]["x"]/100*img_w))
                by = max(0, int(obj["bbox_percent"]["y"]/100*img_h))
                bw = max(10, int(obj["bbox_percent"]["w"]/100*img_w))
                bh = max(10, int(obj["bbox_percent"]["h"]/100*img_h))
                crop = pil.crop((bx, by, min(bx+bw,img_w), min(by+bh,img_h)))
                masks[obj["id"]] = {"image": crop, "origin_x": bx, "origin_y": by}
            except:
                pass

    return masks


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 3: Chuyển động từng vật thể
# ══════════════════════════════════════════════════════════════════

def get_offset(obj, t, img_w, img_h):
    motion = obj.get("motion_type","float")
    dx, dy, angle, scale = 0, 0, 0.0, 1.0
    if motion == "swim":
        dx    = int(math.sin(t*2*math.pi) * min(img_w*0.07,35))
        dy    = int(math.sin(t*4*math.pi) * 5)
        angle = math.sin(t*2*math.pi) * 10
    elif motion == "sail":
        dx    = int(math.sin(t*math.pi) * 12)
        dy    = int(math.sin(t*2*math.pi) * 5)
        angle = math.sin(t*math.pi) * 6
    elif motion == "wave":
        dy    = int(math.sin(t*2*math.pi*1.5) * 9)
        scale = 1.0 + 0.02*math.sin(t*2*math.pi)
    elif motion == "fly":
        dy    = int(math.sin(t*2*math.pi*1.2) * min(img_h*0.05,22))
        dx    = int(math.sin(t*2*math.pi*0.4) * 18)
        angle = math.sin(t*2*math.pi) * 12
    elif motion == "walk":
        dx    = int(math.sin(t*2*math.pi*0.6) * 10)
        dy    = int(abs(math.sin(t*4*math.pi)) * 4)
    elif motion == "float":
        dy    = int(math.sin(t*2*math.pi*0.7) * 12)
        angle = math.sin(t*2*math.pi*0.4) * 6
    elif motion == "sway":
        angle = math.sin(t*2*math.pi*0.5) * 18
        dy    = int(math.sin(t*2*math.pi*0.3) * 4)
    elif motion == "bounce":
        dy    = -int(abs(math.sin(t*2*math.pi*1.5)) * 18)
        scale = 1.0 + 0.04*abs(math.sin(t*2*math.pi*1.5))
    elif motion == "glow":
        scale = 1.0 + 0.09*math.sin(t*2*math.pi)
    return dx, dy, angle, scale


def apply_bg_motion(bg_rgba, t, bg_motion):
    import numpy as np
    from PIL import Image, ImageEnhance
    if bg_motion == "wave":
        arr = np.array(bg_rgba, dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for row in range(h):
            offset = int(math.sin((row/h*5*math.pi) + t*2*math.pi) * 4)
            cols = np.clip(np.arange(w)-offset, 0, w-1).astype(int)
            result[row] = arr[row, cols]
        return Image.fromarray(result.astype(np.uint8), "RGBA")
    elif bg_motion == "wind":
        arr = np.array(bg_rgba, dtype=np.float32)
        h, w = arr.shape[:2]
        result = np.zeros_like(arr)
        for col in range(w):
            offset = int(math.sin((col/w*4*math.pi) + t*2*math.pi) * 3)
            rows = np.clip(np.arange(h)-offset, 0, h-1).astype(int)
            result[:, col] = arr[rows, col]
        return Image.fromarray(result.astype(np.uint8), "RGBA")
    elif bg_motion == "shine":
        b = 1.0 + 0.07*math.sin(t*2*math.pi*0.4)
        return ImageEnhance.Brightness(bg_rgba).enhance(b)
    return bg_rgba


def render_frame(base_rgba, masks, objects, t, bg_motion, img_w, img_h):
    from PIL import Image
    canvas = apply_bg_motion(base_rgba.copy(), t, bg_motion)
    layer_order = {"background":0,"midground":1,"foreground":2}
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
        dx, dy, angle, scale = get_offset(obj, t, img_w, img_h)
        if abs(scale-1.0) > 0.001:
            layer = layer.resize((max(2,int(lw*scale)), max(2,int(lh*scale))), Image.LANCZOS)
            lw, lh = layer.size
        if abs(angle) > 0.2:
            layer = layer.rotate(-angle, expand=True, resample=Image.BICUBIC)
            lw, lh = layer.size
        px = max(-lw//2, min(orig_x+dx, img_w))
        py = max(-lh//2, min(orig_y+dy, img_h))
        canvas.paste(layer, (px, py), layer)
    return canvas.convert("RGB")


# ══════════════════════════════════════════════════════════════════
#  AUDIO
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
        out_dir = Path("/tmp/drawing_anim_v4"); out_dir.mkdir(exist_ok=True)
        ts = int(time.time())

        orig = Image.open(image_path).convert("RGBA")
        orig.thumbnail((800,800), Image.LANCZOS)
        w, h = orig.size
        w = w if w%2==0 else w-1; h = h if h%2==0 else h-1
        orig = orig.crop((0,0,w,h))
        resized_path = str(out_dir/f"resized_{ts}.png")
        orig.convert("RGB").save(resized_path)

        status_cb("🔍 GPT-4o đang phân tích từng vật thể trong tranh...", 8)
        analysis  = analyze_drawing(client, image_path, language)
        objects   = analysis.get("objects", [])
        bg_motion = analysis.get("background_motion","none")
        story     = analysis.get("story","Bé vẽ một bức tranh thật đẹp!")

        if not objects:
            return {"ok":False,"error":"Không nhận diện được vật thể. Thử ảnh có nét vẽ rõ hơn!"}

        names = [o["name"] for o in objects]
        status_cb(f"✅ Nhận diện: {' · '.join(names)}", 18)

        status_cb("✂️ Đang tách từng vật thể (GrabCut nâng cao + feathering)...", 28)
        masks = segment_objects_v4(resized_path, objects, w, h)
        good  = len([m for m in masks.values() if m["image"].size[0] > 5])
        status_cb(f"✅ Tách được {good}/{len(objects)} vật thể — giữ nguyên nét vẽ gốc", 44)

        status_cb("🎙️ Đang tạo giọng kể chuyện...", 50)
        audio_path = str(out_dir/f"audio_{ts}.mp3")
        generate_audio(client, story, voice, audio_path)

        fps          = 24
        total_frames = duration * fps
        frames       = []
        status_cb("🎨 Đang render animation từng vật thể chuyển động...", 57)
        for i in range(total_frames):
            t     = i / total_frames
            frame = render_frame(orig.copy(), masks, objects, t, bg_motion, w, h)
            frames.append(np.array(frame))
            if i % (fps*3) == 0:
                status_cb(f"🎨 Frame {i+1}/{total_frames}...", 57+int((i/total_frames)*30))

        status_cb("🎬 Đang ghép video MP4...", 90)
        video_path = str(out_dir/f"video_{ts}.mp4")
        vc = mpy.ImageSequenceClip(frames, fps=fps)
        ac = mpy.AudioFileClip(audio_path)
        ac = ac.audio_loop(duration=vc.duration) if ac.duration < vc.duration else ac.subclip(0, vc.duration)
        vc.set_audio(ac).write_videofile(
            video_path, fps=fps, codec="libx264", audio_codec="aac",
            logger=None, temp_audiofile="/tmp/tmp_audio_v4.m4a", remove_temp=True
        )
        return {"ok":True,"video_path":video_path,"story":story,"objects":names}

    except Exception as e:
        import traceback
        return {"ok":False,"error":f"{str(e)}\n{traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════════
#  GIAO DIỆN
# ══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title">✨ Tranh Động Của Bé ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Mỗi nét vẽ của bé đều sống động — từng con cá bơi, từng con thuyền lắc lư, từng ngọn sóng gợn lên</p>', unsafe_allow_html=True)

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
    uploaded = st.file_uploader("Chụp ảnh hoặc chọn file", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
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
                    st.caption(f"🎭 Các nhân vật: {' · '.join(result['objects'])}")
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
        - **Nét vẽ đậm, rõ ràng** — GrabCut nhận diện tốt hơn
        - **Nền trắng** hoặc màu nhạt giữa các vật thể
        - Mỗi vật thể **không chồng chéo** nhau quá nhiều
        - Ảnh **chụp thẳng góc**, đủ sáng, tránh bóng
        - Tốt nhất: **3-6 vật thể** với màu sắc khác nhau
        """)

st.divider()
st.markdown('<p style="text-align:center;color:#bbb;font-size:0.85rem;">🎨 Tranh Động Của Bé · Từng nét vẽ đều có hồn</p>', unsafe_allow_html=True)
