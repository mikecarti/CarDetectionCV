import os
from pathlib import Path
import streamlit as st
from count_direction_tracker import process_video
from ultralytics import YOLO
import torch
import ffmpeg

# CONSTANTS
VIDEO_DIR = "videos"  # Папка с видео
MODEL_PATH = "last.pt"
OUTPUT_DIR = "results"  # Папка для результатов
SHOW_VIDEO = False  # Показывать ли видео в процессе обработки
LIMIT_SIZE_MB = 10000


# fix torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("Choose a Video File")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    print(f"Received file, named {uploaded_file.name}")
    input_video_path = Path(os.path.join(VIDEO_DIR, uploaded_file.name))
    with open(input_video_path, "wb") as f:
        f.write(bytes_data)

    output_video_path = input_video_path.with_stem(input_video_path.stem + "_264")

    print(f"Storing to path: {output_video_path}")
    st.write("Converting codex to 264...")

    ffmpeg.input(input_video_path).output(
        str(output_video_path),
        vcodec="libx264",
        preset="ultrafast",  # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        crf=28,  # Constant Rate Factor (lower value = better quality, higher value = faster conversion)
        # loglevel="quiet",
    ).run(overwrite_output=True)


video_files = [
    f for f in os.listdir(VIDEO_DIR) if f.endswith(("_264.mp4", "_264.avi", "_264.mov"))
]
# print(os.listdir(VIDEO_DIR), video_files)

if len(video_files) != 0:
    assert len(video_files) == 1, len(video_files)
    video_file = video_files[0]
    video_path = os.path.join(VIDEO_DIR, video_file)

    st.write("Inferen...")
    results = process_video(
        video_path, model, output_dir=OUTPUT_DIR, show_video=SHOW_VIDEO, save_video=True
    )

    video_data = video_file.split(".")
    processed_video_file_name = video_data[0] + "_processed" + "." + video_data[1]
    processed_video_path = Path(os.path.join(OUTPUT_DIR, processed_video_file_name))
    print(f"Path to processed video: {processed_video_path}")

    with open(processed_video_path, "rb") as v:
        st.video(v)
