import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Chuyển giọng nói thành văn bản", page_icon="🎙️")

MODEL_SIZE = "large-v3"


@st.cache_resource(show_spinner="Đang tải model (chỉ lần đầu)...")
def load_model():
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


st.title("🎙️ Chuyển ghi âm thành văn bản (tiếng Việt)")
st.write("Tải lên file audio (mp3, wav, m4a...), chờ xử lý, nhận lại văn bản kèm timestamp.")

uploaded_file = st.file_uploader(
    "Chọn file audio",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
)

if uploaded_file is not None:
    if st.button("Chuyển văn bản", type="primary"):
        model = load_model()

        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Đang xử lý audio, tùy độ dài file có thể mất vài phút..."):
            segments, info = model.transcribe(
                temp_path,
                language="vi",
                beam_size=5,
                vad_filter=True,
            )

            full_text = ""
            for segment in segments:
                start = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
                end = f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d}"
                full_text += f"[{start} - {end}] {segment.text.strip()}\n"

        st.success("Xong!")
        st.text_area("Kết quả", value=full_text.strip(), height=400)
        st.download_button(
            "Tải văn bản (.txt)",
            data=full_text.strip(),
            file_name="transcript.txt",
        )
