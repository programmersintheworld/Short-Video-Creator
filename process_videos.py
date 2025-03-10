import os
import subprocess
import logging
from moviepy.editor import VideoFileClip, concatenate_videoclips
from config import (
    INPUT_VIDEOS_DIR, OUTPUT_VIDEOS_DIR, PROCESS_INPUT_VIDEOS_DIR, PROCESS_OUTPUT_VIDEOS_DIR, SEGMENT_DURATION
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_video(input_path):
    """Divide un video en segmentos más pequeños usando FFmpeg y los guarda en input_videos uno por uno."""
    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    
    duration_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path]
    duration = float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip())
    
    for start_time in range(0, int(duration), SEGMENT_DURATION):
        segment_path = os.path.join(INPUT_VIDEOS_DIR, f"{os.path.basename(input_path)}_part_{start_time}.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(SEGMENT_DURATION), "-c", "copy", segment_path, "-y"
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        yield segment_path  # Devuelve cada segmento a medida que se crea
        os.remove(segment_path)  # Elimina el segmento después de procesarlo

def merge_videos(output_video_name):
    """Une los segmentos procesados en output_videos en un solo archivo final usando moviepy."""
    os.makedirs(PROCESS_OUTPUT_VIDEOS_DIR, exist_ok=True)
    processed_segments = sorted([os.path.join(OUTPUT_VIDEOS_DIR, f) for f in os.listdir(OUTPUT_VIDEOS_DIR) if f.endswith(".mp4")])
    
    if not processed_segments:
        logging.error("No hay archivos válidos para unir.")
        return
    
    clips = [VideoFileClip(segment) for segment in processed_segments]
    final_video = concatenate_videoclips(clips, method="compose")
    final_output_path = os.path.join(PROCESS_OUTPUT_VIDEOS_DIR, output_video_name)
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    logging.info(f"Video final guardado en {final_output_path}")

def process_video(input_video):
    """Procesa cada segmento individualmente antes de generar el siguiente."""
    for segment in split_video(input_video):
        logging.info(f"Procesando segmento: {segment}")
        subprocess.run(["python", "main.py", segment])  # Procesa cada segmento antes de generar el siguiente
    
    final_output_name = f"final_{os.path.basename(input_video)}"
    merge_videos(final_output_name)
    logging.info(f"Video final guardado en {PROCESS_OUTPUT_VIDEOS_DIR}/{final_output_name}")

if __name__ == "__main__":
    video_files = [f for f in os.listdir(PROCESS_INPUT_VIDEOS_DIR) if f.endswith(".mp4")]
    
    for video in video_files:
        process_video(os.path.join(PROCESS_INPUT_VIDEOS_DIR, video))
