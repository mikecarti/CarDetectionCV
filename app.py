import os
from pathlib import Path
import streamlit as st
from count_direction_tracker import process_video
from ultralytics import YOLO
import torch
import ffmpeg
import polars as pl

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

placeholder = st.empty()

if uploaded_file is not None:
    # Save file
    bytes_data = uploaded_file.getvalue()
    print(f"Received file, named {uploaded_file.name}")
    client_video_path = Path(os.path.join(VIDEO_DIR, uploaded_file.name))
    with open(client_video_path, "wb") as f:
        f.write(bytes_data)

    video_data = client_video_path.name.split(".")
    processed_video_file_name = video_data[0] + "_processed" + "." + video_data[1]
    processed_video_path = Path(os.path.join(OUTPUT_DIR, processed_video_file_name))

    # Inference
    with st.spinner(show_time=False):
        placeholder.write(f"Inference for video {client_video_path.name} ...")
        if not processed_video_path.exists():
            directions = process_video(
                client_video_path,
                model,
                output_dir=OUTPUT_DIR,
                show_video=SHOW_VIDEO,
                save_video=True,
            )
        directions_df = pl.read_csv(
            Path(OUTPUT_DIR) / (video_data[0] + "_counts" + ".csv"), separator=";"
        )
    st.write(directions_df)

    # Convert to 264
    with st.spinner(show_time=False):
        print(f"Path to processed video: {processed_video_path}")

        placeholder.write("Converting codec to 264...")
        processed_video_264_path = processed_video_path.with_stem(
            processed_video_path.stem + "_264"
        )
        if not processed_video_264_path.exists():
            ffmpeg.input(str(processed_video_path)).output(
                str(processed_video_264_path),
                vcodec="libx264",
                preset="ultrafast",  # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
                crf=28,  # Constant Rate Factor (lower value = better quality, higher value = faster conversion)
                # loglevel="quiet",
            ).run(overwrite_output=True)

    placeholder.write("Processing Done!")

    # Video Playback
    play_button, download_button = st.columns(2)

    if play_button.button("Play", use_container_width=True):
        with open(processed_video_264_path, "rb") as v:
            st.video(v)

    video_file = open(processed_video_264_path, "rb")
    # Download the result
    if download_button.download_button("Download", video_file):
        placeholder.write("Video is downloaded!")
