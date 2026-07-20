import streamlit as st
import tempfile
import subprocess
import math
import json
import os
from datetime import datetime, timezone
from groq import Groq

st.set_page_config(page_title="Chuyển giọng nói thành văn bản", page_icon="🎙️")

# Groq giới hạn 25MB/request, nên cắt audio thành từng đoạn 20 phút trước khi gửi.
CHUNK_SECONDS = 20 * 60

# Giới hạn thật của Groq free tier (không phải "phút/tháng" - đây là 2 giới hạn riêng):
HOURLY_AUDIO_SECONDS_LIMIT = 7200   # ~120 phút audio / giờ (rolling, không phải cố định 0h-1h)
DAILY_REQUEST_LIMIT = 2000          # số request / ngày

USAGE_FILE = "/tmp/groq_usage.json"


def load_usage():
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"hour_key": None, "audio_seconds_this_hour": 0, "date_key": None, "requests_today": 0}


def save_usage(usage):
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)


def current_hour_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")


def current_date_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_reset_usage():
    """Đọc usage hiện tại, tự reset nếu đã sang giờ mới / ngày mới."""
    usage = load_usage()
    if usage.get("hour_key") != current_hour_key():
        usage["hour_key"] = current_hour_key()
        usage["audio_seconds_this_hour"] = 0
    if usage.get("date_key") != current_date_key():
        usage["date_key"] = current_date_key()
        usage["requests_today"] = 0
    return usage


def record_usage(audio_seconds, n_requests):
    usage = get_reset_usage()
    usage["audio_seconds_this_hour"] += audio_seconds
    usage["requests_today"] += n_requests
    save_usage(usage)
    return usage


def get_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "Chưa cấu hình GROQ_API_KEY. Vào Settings > Secrets trên Streamlit Cloud "
            "để thêm API key lấy từ console.groq.com."
        )
        st.stop()
    return Groq(api_key=api_key)


def format_ts(seconds):
    total_seconds = int(seconds)
    m, s = divmod(total_seconds, 60)
    return f"{m:02d}:{s:02d}"


def seg_field(seg, key):
    return seg[key] if isinstance(seg, dict) else getattr(seg, key)


def get_duration_seconds(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def cut_chunk(input_path, start_sec, duration_sec, output_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", input_path,
            "-acodec", "libmp3lame",
            "-b:a", "128k",
            output_path,
        ],
        capture_output=True, check=True,
    )


def transcribe_chunk(client, path):
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(path, f.read()),
            model="whisper-large-v3",
            language="vi",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return result


st.title("🎙️ Chuyển ghi âm thành văn bản (tiếng Việt)")
st.write("Tải lên file audio (mp3, wav, m4a...), chờ xử lý, nhận lại văn bản kèm timestamp.")
st.caption("Dùng Groq API (whisper-large-v3) — chất lượng cao, xử lý trên server Groq, không phụ thuộc RAM của app này.")

# --- Hiển thị usage hiện tại ---
usage = get_reset_usage()
audio_minutes_used = usage["audio_seconds_this_hour"] / 60
audio_minutes_limit = HOURLY_AUDIO_SECONDS_LIMIT / 60

col1, col2 = st.columns(2)
with col1:
    st.caption(f"Phút audio đã dùng (giờ này): {audio_minutes_used:.1f} / {audio_minutes_limit:.0f} phút")
    st.progress(min(1.0, audio_minutes_used / audio_minutes_limit))
with col2:
    st.caption(f"Request đã dùng (hôm nay): {usage['requests_today']} / {DAILY_REQUEST_LIMIT}")
    st.progress(min(1.0, usage["requests_today"] / DAILY_REQUEST_LIMIT))

st.divider()

uploaded_file = st.file_uploader(
    "Chọn file audio",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
)

if uploaded_file is not None:
    if st.button("Chuyển văn bản", type="primary"):
        client = get_client()

        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            input_path = tmp.name

        with st.spinner("Đang xử lý audio, tùy độ dài file có thể mất vài phút..."):
            try:
                total_seconds = get_duration_seconds(input_path)
            except subprocess.CalledProcessError as e:
                st.error(f"Không đọc được file audio: {e.stderr}")
                st.stop()

            n_chunks = max(1, math.ceil(total_seconds / CHUNK_SECONDS))

            full_text = ""
            progress = st.progress(0, text="Đang gửi từng đoạn tới Groq...")

            for i in range(n_chunks):
                start_sec = i * CHUNK_SECONDS
                chunk_len = min(CHUNK_SECONDS, total_seconds - start_sec)

                chunk_path = f"/tmp/chunk_{i}.mp3"
                try:
                    cut_chunk(input_path, start_sec, chunk_len, chunk_path)
                except subprocess.CalledProcessError as e:
                    st.error(f"Lỗi khi cắt đoạn {i + 1}: {e.stderr}")
                    st.stop()

                result = transcribe_chunk(client, chunk_path)
                record_usage(audio_seconds=chunk_len, n_requests=1)

                segments = getattr(result, "segments", None) or []
                if segments:
                    for seg in segments:
                        seg_start = format_ts(start_sec + seg_field(seg, "start"))
                        seg_end = format_ts(start_sec + seg_field(seg, "end"))
                        text = seg_field(seg, "text").strip()
                        full_text += f"[{seg_start} - {seg_end}] {text}\n"
                else:
                    # Groq đôi khi không trả về segments — fallback đọc text tổng.
                    text = (getattr(result, "text", "") or "").strip()
                    if text:
                        full_text += text + "\n"

                progress.progress((i + 1) / n_chunks, text=f"Đã xử lý đoạn {i + 1}/{n_chunks}")

        st.success("Xong!")
        st.text_area("Kết quả", value=full_text.strip(), height=400)
        st.download_button(
            "Tải văn bản (.txt)",
            data=full_text.strip(),
            file_name="transcript.txt",
        )
