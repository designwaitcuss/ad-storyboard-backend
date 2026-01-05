import base64
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import whisper

app = FastAPI()

JOBS: Dict[str, dict] = {}
WORKDIR = "/tmp/jobs"
os.makedirs(WORKDIR, exist_ok=True)

WHISPER_MODEL = None

def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore"))

def probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return 0.0
    try:
        return float(p.stdout.decode().strip())
    except:
        return 0.0

def extract_scene_frames(video_path: str, out_dir: str, max_frames: int = 20) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)

    # Scene detection. Adjust threshold if you get too many or too few frames.
    # Higher threshold -> fewer frames.
    threshold = "0.35"

    # Create images with timestamp in filename using pts.
    # We first extract a bunch, then cap to max_frames evenly.
    tmp_pattern = os.path.join(out_dir, "frame_%06d.jpg")
    vf = f"select='gt(scene,{threshold})',scale=640:-1"

    run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", "3",
        tmp_pattern
    ])

    frames = sorted([f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    if not frames:
        # Fallback, extract start frame
        run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "scale=640:-1",
            "-frames:v", "1",
            os.path.join(out_dir, "frame_000001.jpg")
        ])
        frames = ["frame_000001.jpg"]

    # Cap frames by sampling evenly if too many
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int).tolist()
        frames = [frames[i] for i in idxs]

    # Get timestamps for each selected frame via ffprobe frame metadata is annoying.
    # Simple MVP approximation:
    # Map frames evenly across duration.
    duration = probe_duration(video_path)
    n = len(frames)
    results = []
    for i, fname in enumerate(frames):
        t = 0.0 if n == 1 else (duration * i / (n - 1))
        results.append({"path": os.path.join(out_dir, fname), "timestamp": float(t)})
    return results

def ocr_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img)
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text.strip()

def load_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = whisper.load_model("tiny")
    return WHISPER_MODEL

def transcribe(video_path: str) -> List[dict]:
    model = load_whisper()
    result = model.transcribe(video_path, fp16=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": str(s["text"]).strip()
        })
    return segments

def vo_snippet_at(t: float, segments: List[dict], window: float = 2.0) -> str:
    start = t - window / 2
    end = t + window / 2
    parts = []
    for s in segments:
        if s["end"] >= start and s["start"] <= end:
            if s["text"]:
                parts.append(s["text"])
    text = " ".join(parts).strip()
    return text

def wav_rms_loudness(video_path: str) -> str:
    wav_path = video_path + ".wav"
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    import wave
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    os.remove(wav_path)

    if audio.size == 0:
        return "low"

    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Simple thresholds for MVP
    if rms < 0.03:
        return "low"
    if rms < 0.08:
        return "medium"
    return "high"

def estimate_pace(duration: float, frame_count: int) -> dict:
    if duration <= 0 or frame_count <= 0:
        return {"avg_shot_length": 0.0, "total_cuts": 0, "cuts_per_10s": 0.0, "label": "unknown"}

    total_cuts = max(0, frame_count - 1)
    avg_shot = duration / max(1, frame_count)
    cuts_per_10s = total_cuts / (duration / 10.0) if duration > 0 else 0.0

    if avg_shot < 1.2:
        label = "fast"
    elif avg_shot <= 2.5:
        label = "medium"
    else:
        label = "slow"

    return {
        "avg_shot_length": round(avg_shot, 2),
        "total_cuts": total_cuts,
        "cuts_per_10s": round(cuts_per_10s, 2),
        "label": label
    }

def img_to_data_url_jpg(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()

def make_storyboard_png(frame_paths: List[str]) -> bytes:
    thumbs = [Image.open(p).convert("RGB") for p in frame_paths]
    thumb_w = 360
    resized = []
    for im in thumbs:
        w, h = im.size
        new_h = int(h * (thumb_w / w))
        resized.append(im.resize((thumb_w, new_h)))

    padding = 16
    text_h = 0
    total_w = sum(im.size[0] for im in resized) + padding * (len(resized) + 1)
    max_h = max(im.size[1] for im in resized) + padding * 2 + text_h

    canvas = Image.new("RGB", (total_w, max_h), (16, 16, 16))
    x = padding
    for im in resized:
        canvas.paste(im, (x, padding))
        x += im.size[0] + padding

    import io
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()

def background_process(job_id: str, video_path: str):
    try:
        job_dir = os.path.join(WORKDIR, job_id)
        frames_dir = os.path.join(job_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        duration = probe_duration(video_path)
        extracted = extract_scene_frames(video_path, frames_dir, max_frames=20)

        # OCR
        frames_out = []
        for i, fr in enumerate(extracted, start=1):
            ocr = ocr_image(fr["path"])
            frames_out.append({
                "index": i,
                "timestamp": fr["timestamp"],
                "image_url": img_to_data_url_jpg(fr["path"]),
                "ocr_text": ocr
            })

        # Transcript
        segments = transcribe(video_path)
        for fr in frames_out:
            fr["vo_snippet"] = vo_snippet_at(fr["timestamp"], segments)

        pace = estimate_pace(duration, len(frames_out))

        loudness = wav_rms_loudness(video_path)
        speech_detected = True if segments else False

        sound = {
            "speech_detected": speech_detected,
            "loudness": loudness,
            "music_likely": (not speech_detected and loudness in ["medium", "high"])
        }

        storyboard_png = make_storyboard_png([f["path"] for f in extracted])
        storyboard_png_url = png_bytes_to_data_url(storyboard_png)

        JOBS[job_id] = {
            "status": "done",
            "duration_seconds": round(duration, 2),
            "pace": pace,
            "sound": sound,
            "transcript": segments,
            "frames": frames_out,
            "storyboard_png_url": storyboard_png_url
        }

    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}

    finally:
        try:
            os.remove(video_path)
        except:
            pass

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "processing"}

    # Run processing in a separate process using subprocess via python -c is overkill.
    # Simple FastAPI background task alternative:
    from threading import Thread
    Thread(target=background_process, args=(job_id, video_path), daemon=True).start()

    return {"job_id": job_id}

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job

@app.get("/health")
async def health():
    return {"ok": True}
