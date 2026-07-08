import streamlit as st
import tempfile
import subprocess
import math
from groq import Groq

st.set_page_config(page_title="Chuyển giọng nói thành văn bản", page_icon="🎙️")

CHUNK_SECONDS = 20 * 60


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
    return getattr(result, "segments", []) or []


st.title("🎙️ Chuyển ghi âm thành văn bản (tiếng Việt)")
st.write("Tải lên file audio (mp3, wav, m4a...), chờ xử lý, nhận lại văn bản kèm timestamp.")
st.caption("Dùng Groq API (whisper-large-v3) — chất lượng cao, xử lý trên server Groq, không phụ thuộc RAM của app này.")

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

                segments = transcribe_chunk(client, chunk_path)
                for seg in segments:
                    seg_start = format_ts(start_sec + seg_field(seg, "start"))
                    seg_end = format_ts(start_sec + seg_field(seg, "end"))
                    text = seg_field(seg, "text").strip()
                    full_text += f"[{seg_start} - {seg_end}] {text}\n"

                progress.progress((i + 1) / n_chunks, text=f"Đã xử lý đoạn {i + 1}/{n_chunks}")

        st.success("Xong!")
        st.text_area("Kết quả", value=full_text.strip(), height=400)
        st.download_button(
            "Tải văn bản (.txt)",
            data=full_text.strip(),
            file_name="transcript.txt",
        )
