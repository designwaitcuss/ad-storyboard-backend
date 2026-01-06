import base64
import os
import shutil
import subprocess
import uuid
import re
from threading import Thread
from typing import Dict, List
from fastapi import Request
from starlette.datastructures import UploadFile as StarletteUploadFile

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract


app = FastAPI()

# CORS for Lovable and any browser client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, dict] = {}

WORKDIR = "/tmp/jobs"
os.makedirs(WORKDIR, exist_ok=True)


def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        out = p.stdout.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err}\n{out}")


def probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return 0.0
    try:
        return float(p.stdout.decode().strip())
    except Exception:
        return 0.0


def extract_scene_frames(video_path: str, out_dir: str, max_frames: int = 20) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)

    def extract_at_time(t: float, out_path: str) -> None:
        run([
            "ffmpeg", "-y",
            "-ss", f"{t}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "3",
            "-vf", "scale=640:-1",
            out_path
        ])

    def extract_scene_frames_with_timestamps(threshold: float) -> List[dict]:
        pattern = os.path.join(out_dir, "scene_%06d.jpg")

        vf = f"select='gt(scene,{threshold})',showinfo,scale=640:-1"

        p = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-vsync", "vfr", "-q:v", "3", pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", errors="ignore")
            out = p.stdout.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Scene extraction failed\n{err}\n{out}")

        # Parse pts_time from showinfo for each output frame
        stderr = p.stderr.decode("utf-8", errors="ignore")
        pts_times = [float(x) for x in re.findall(r"pts_time:([0-9]+\.[0-9]+)", stderr)]

        files = sorted([f for f in os.listdir(out_dir) if f.startswith("scene_") and f.endswith(".jpg")])
        results = []
        for i, fname in enumerate(files):
            ts = pts_times[i] if i < len(pts_times) else 0.0
            results.append({"path": os.path.join(out_dir, fname), "timestamp": float(ts)})
        return results

    # 1) Always include early frames (fixes your issue)
    early = []
    first_path = os.path.join(out_dir, "early_000000.jpg")
    second_path = os.path.join(out_dir, "early_000001.jpg")

    extract_at_time(0.0, first_path)
    early.append({"path": first_path, "timestamp": 0.0})

    # 1.0s catches most talking head intros without being too redundant
    extract_at_time(1.0, second_path)
    early.append({"path": second_path, "timestamp": 1.0})

    # 2) Scene detection frames with real timestamps
    scene = extract_scene_frames_with_timestamps(threshold=0.35)

    # 3) Merge and de-duplicate by timestamp proximity
    merged = early + scene
    merged.sort(key=lambda x: x["timestamp"])

    deduped = []
    min_gap = 0.6  # seconds. prevents near-identical frames
    for item in merged:
        if not deduped:
            deduped.append(item)
            continue
        if abs(item["timestamp"] - deduped[-1]["timestamp"]) >= min_gap:
            deduped.append(item)

    # 4) Cap frames by sampling evenly, while keeping the first frame
    if len(deduped) > max_frames:
        keep_first = deduped[0]
        rest = deduped[1:]
        target_rest = max_frames - 1
        idxs = np.linspace(0, len(rest) - 1, target_rest).astype(int).tolist()
        deduped = [keep_first] + [rest[i] for i in idxs]

    return deduped


def ocr_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def img_to_data_url_jpg(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")


def wav_rms_loudness(video_path: str) -> str:
    wav_path = video_path + ".wav"
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    import wave
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        os.remove(wav_path)
    except Exception:
        pass

    if audio.size == 0:
        return "low"

    rms = float(np.sqrt(np.mean(audio ** 2)))

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
        "label": label,
    }


def background_process(job_id: str, video_path: str) -> None:
    try:
        job_dir = os.path.join(WORKDIR, job_id)
        frames_dir = os.path.join(job_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        duration = probe_duration(video_path)
        extracted = extract_scene_frames(video_path, frames_dir, max_frames=20)

        frames_out = []
        for i, fr in enumerate(extracted, start=1):
            frames_out.append({
                "index": i,
                "timestamp": fr["timestamp"],
                "image_url": img_to_data_url_jpg(fr["path"]),
                "ocr_text": ocr_image(fr["path"]),
                "vo_snippet": "",
            })

        pace = estimate_pace(duration, len(frames_out))

        loudness = wav_rms_loudness(video_path)
        speech_detected = False

        sound = {
            "speech_detected": speech_detected,
            "loudness": loudness,
            "music_likely": (not speech_detected and loudness in ["medium", "high"]),
        }

        JOBS[job_id] = {
            "status": "done",
            "duration_seconds": round(duration, 2),
            "pace": pace,
            "sound": sound,
            "transcript": [],
            "frames": frames_out,
            "storyboard_png_url": "",
        }

    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}

    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


@app.get("/")
async def root():
    return {"ok": True, "message": "Ad Storyboard Backend running. Use /health, /process, /job/{id}."}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/process")
async def process_video(request: Request):
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    form = await request.form()

    upload = None
    for key in ["file", "video", "media", "upload"]:
        if key in form:
            upload = form[key]
            break

    # If the client used a random field name, try first file-like entry
    if upload is None:
        for v in form.values():
            upload = v
            break

    if upload is None or not isinstance(upload, StarletteUploadFile):
        return JSONResponse(
            {"error": "No uploaded file found. Expected multipart form with a file field."},
            status_code=422,
        )

    filename = upload.filename or "upload.mp4"
    video_path = os.path.join(job_dir, filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    JOBS[job_id] = {"status": "processing"}
    Thread(target=background_process, args=(job_id, video_path), daemon=True).start()

    return {"job_id": job_id}


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job
