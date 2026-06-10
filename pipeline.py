"""
pipeline.py — Luồng xử lý chính
  1. Phân tích tranh bằng GPT-4o Vision
  2. Tạo lời kể chuyện
  3. Tạo audio bằng TTS
  4. Tạo các frame animation từ ảnh gốc
  5. Ghép thành video MP4
"""

import os
import base64
import json
import time
import math
import tempfile
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import moviepy.editor as mpy


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    """Encode ảnh sang base64 để gửi lên API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _img_to_base64_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return f"data:{mime};base64,{_encode_image(path)}"


# ─── Bước 1: Phân tích tranh ──────────────────────────────────────────────────

def analyze_drawing(client: OpenAI, image_path: str, language: str) -> dict:
    """
    Dùng GPT-4o Vision để:
    - Nhận diện nhân vật, cảnh vật, màu sắc trong tranh
    - Tạo câu chuyện ngắn phù hợp với trẻ mầm non
    - Gợi ý loại chuyển động cho từng đối tượng
    """
    lang_instruction = "Viết hoàn toàn bằng tiếng Việt." if language == "Tiếng Việt" else "Write everything in English."

    prompt = f"""Bạn là chuyên gia giáo dục mầm non và AI sáng tạo.
Hãy phân tích bức tranh vẽ tay của trẻ và trả về JSON với cấu trúc sau (không có markdown, chỉ JSON thuần):

{{
  "objects": [
    {{"name": "tên vật thể", "type": "character|animal|plant|sky|ground|other", "motion": "walk|fly|bounce|sway|shine|flow|spin|float", "description": "mô tả ngắn"}}
  ],
  "scene": "mô tả tổng quan cảnh trong tranh (1-2 câu)",
  "story": "câu chuyện ngắn 3-4 câu cho trẻ mầm non, dựa trên tranh, kết thúc vui vẻ",
  "mood": "happy|calm|adventurous|peaceful",
  "colors": ["màu chủ đạo 1", "màu chủ đạo 2", "màu chủ đạo 3"]
}}

{lang_instruction}
Lưu ý: Giữ câu chuyện đơn giản, tích cực, phù hợp trẻ 3-6 tuổi."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _img_to_base64_url(image_path), "detail": "high"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=800,
        temperature=0.7
    )

    raw = resp.choices[0].message.content.strip()
    # Loại bỏ markdown code block nếu có
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


# ─── Bước 2: Tạo audio ────────────────────────────────────────────────────────

def generate_audio(client: OpenAI, story: str, voice: str, output_path: str) -> str:
    """Tạo file MP3 từ câu chuyện bằng OpenAI TTS."""
    resp = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=story,
        speed=0.9  # Hơi chậm hơn — phù hợp trẻ em
    )
    resp.stream_to_file(output_path)
    return output_path


# ─── Bước 3: Tạo animation frames ─────────────────────────────────────────────

def _apply_gentle_effect(img: Image.Image, t: float) -> Image.Image:
    """
    Tạo chuyển động nhẹ nhàng trực tiếp trên ảnh gốc,
    KHÔNG thay đổi màu sắc hay nét vẽ — chỉ thêm chuyển động tinh tế.
    """
    w, h = img.size

    # 1. Nhấp nhô nhẹ (bob) — toàn bộ ảnh lên xuống nhẹ
    bob_offset = int(math.sin(t * 2 * math.pi) * 3)

    # 2. Thở (breathe) — phóng to thu nhỏ rất nhẹ
    scale = 1.0 + 0.012 * math.sin(t * 2 * math.pi * 0.7)
    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = img.resize((new_w, new_h), Image.LANCZOS)

    # Crop về kích thước gốc, căn giữa + bob_offset
    left = (new_w - w) // 2
    top = (new_h - h) // 2 - bob_offset
    top = max(0, min(top, new_h - h))
    frame = frame.crop((left, top, left + w, top + h))

    return frame


def _apply_calm_effect(img: Image.Image, t: float) -> Image.Image:
    """Mượt mà, thư giãn — dao động rất nhỏ, thêm shimmer sáng nhẹ."""
    w, h = img.size
    scale = 1.0 + 0.008 * math.sin(t * 2 * math.pi * 0.5)
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    frame = frame.crop((left, top, left + w, top + h))

    # Shimmer — sáng hơn nhẹ theo chu kỳ
    brightness = 1.0 + 0.04 * math.sin(t * 2 * math.pi * 0.3)
    enhancer = ImageEnhance.Brightness(frame)
    frame = enhancer.enhance(brightness)

    return frame


def _apply_energetic_effect(img: Image.Image, t: float) -> Image.Image:
    """Năng động — nảy mạnh hơn, lắc nhẹ."""
    w, h = img.size

    # Nảy
    bob = int(math.sin(t * 2 * math.pi * 1.5) * 6)

    # Lắc ngang
    sway = int(math.sin(t * 2 * math.pi * 0.8) * 4)

    scale = 1.0 + 0.018 * abs(math.sin(t * 2 * math.pi))
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - w) // 2 - sway
    top = (new_h - h) // 2 - bob
    left = max(0, min(left, new_w - w))
    top = max(0, min(top, new_h - h))
    frame = frame.crop((left, top, left + w, top + h))

    return frame


EFFECTS = {
    "gentle":   _apply_gentle_effect,
    "calm":     _apply_calm_effect,
    "energetic": _apply_energetic_effect,
}


def generate_frames(image_path: str, style: str, duration: int, fps: int = 24) -> list:
    """
    Tạo danh sách frames numpy từ ảnh gốc.
    Ảnh gốc KHÔNG bị thay đổi — chỉ thêm chuyển động tinh tế.
    """
    img = Image.open(image_path).convert("RGB")

    # Chuẩn hoá kích thước — giữ tỉ lệ, padding trắng nếu cần
    max_size = 720
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Đảm bảo kích thước là bội số 2 (yêu cầu của H.264)
    w, h = img.size
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    img = img.crop((0, 0, w, h))

    effect_fn = EFFECTS.get(style, _apply_gentle_effect)
    total_frames = duration * fps
    frames = []

    for i in range(total_frames):
        t = i / total_frames  # 0.0 → 1.0
        frame = effect_fn(img.copy(), t)
        frames.append(np.array(frame))

    return frames, (w, h), fps


# ─── Bước 4: Ghép video ───────────────────────────────────────────────────────

def assemble_video(frames: list, audio_path: str, output_path: str, fps: int, size: tuple) -> str:
    """Ghép frames + audio thành file MP4 bằng MoviePy."""

    # Tạo video clip từ frames
    video_clip = mpy.ImageSequenceClip(frames, fps=fps)

    # Thêm audio
    audio_clip = mpy.AudioFileClip(audio_path)

    # Nếu audio ngắn hơn video → loop; nếu dài hơn → cắt
    if audio_clip.duration < video_clip.duration:
        audio_clip = audio_clip.audio_loop(duration=video_clip.duration)
    else:
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    final = video_clip.set_audio(audio_clip)
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
        temp_audiofile="/tmp/temp_audio.m4a",
        remove_temp=True
    )

    return output_path


# ─── Hàm chính ────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    voice: str,
    language: str,
    style: str,
    duration: int,
    status_cb,
    api_key: str
) -> dict:
    """
    Chạy toàn bộ pipeline và trả về:
      {"ok": True, "video_path": "...", "story": "..."}
      {"ok": False, "error": "..."}
    """
    try:
        client = OpenAI(api_key=api_key)
        out_dir = Path("/tmp/drawing_animator_output")
        out_dir.mkdir(exist_ok=True)
        ts = int(time.time())

        # Bước 1
        status_cb("Đang phân tích tranh vẽ của bé...", 10)
        analysis = analyze_drawing(client, image_path, language)
        story = analysis.get("story", "Bé vẽ một bức tranh thật đẹp!")

        # Bước 2
        status_cb("Đang tạo giọng kể chuyện...", 35)
        audio_path = str(out_dir / f"audio_{ts}.mp3")
        generate_audio(client, story, voice, audio_path)

        # Bước 3
        status_cb("Đang tạo hiệu ứng chuyển động cho tranh...", 60)
        frames, size, fps = generate_frames(image_path, style, duration)

        # Bước 4
        status_cb("Đang ghép video hoàn chỉnh...", 85)
        video_path = str(out_dir / f"video_{ts}.mp4")
        assemble_video(frames, audio_path, video_path, fps, size)

        return {"ok": True, "video_path": video_path, "story": story, "analysis": analysis}

    except Exception as e:
        import traceback
        return {"ok": False, "error": f"{str(e)}\n{traceback.format_exc()}"}
