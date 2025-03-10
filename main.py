import math
import multiprocessing
import os
import random
import shutil
import time
import logging
import subprocess
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (VideoFileClip, clips_array, concatenate_videoclips,
                           ImageClip, CompositeVideoClip, VideoClip)
from moviepy.video.fx.all import crop as moviepy_crop
import whisper_timestamped as whisper

from config import (
    BACKGROUND_VIDEOS_DIR,
    FONT_BORDER_WEIGHT,
    FONTS_DIR,
    FONT_NAME,
    FONT_SIZE,
    FULL_RESOLUTION,
    INPUT_VIDEOS_DIR,
    MAX_NUMBER_OF_PROCESSES,
    OUTPUT_VIDEOS_DIR,
    PERCENT_MAIN_CLIP,
    TEXT_POSITION_PERCENT,
    MODEL_NAME,
    LANGUAGE,
    NUM_THREADS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Check for GPU availability
def has_gpu():
    try:
        # Check if CUDA is available (for NVIDIA GPUs)
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

USE_GPU = has_gpu()
if USE_GPU:
    logging.info("GPU detected and will be used for processing")
else:
    logging.info("No GPU detected, using CPU for processing")

class VideoTools:
    # Initialize the VideoTools class with a VideoFileClip
    clip: VideoFileClip = None

    def __init__(self, clip: VideoFileClip) -> None:
        """Constructor to initialize the VideoFileClip."""
        self.clip = clip

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        if self.clip:
            self.clip.close()  # Close the clip to free resources
            self.clip = None  # Set clip to None to avoid dangling reference

    def crop(self, width: int, height: int) -> VideoFileClip:
        """Crop the video clip to the specified width and height.

        Args:
            width (int): The desired width of the cropped video.
            height (int): The desired height of the cropped video.

        Returns:
            VideoFileClip: The cropped video clip.
        """
        # Get the original dimensions of the video clip
        original_width, original_height = self.clip.size

        # Calculate the change ratios for width and height
        width_change_ratio = width / original_width
        height_change_ratio = height / original_height

        # Determine the maximum ratio to maintain aspect ratio
        max_ratio = max(width_change_ratio, height_change_ratio)

        # Resize the clip based on the maximum ratio
        self.clip = self.clip.resize((
            original_width * max_ratio,
            original_height * max_ratio,
        ))

        # Get the new dimensions after resizing
        new_width, new_height = self.clip.size

        # Crop the video based on the aspect ratio
        if width_change_ratio > height_change_ratio:
            # Calculate the vertical crop
            height_change = new_height - height
            new_y1 = round(height_change / 2)  # Calculate the starting y-coordinate
            new_y2 = min(new_y1 + height, new_height)  # Calculate the ending y-coordinate
            self.clip = moviepy_crop(self.clip, y1=new_y1, y2=new_y2)  # Crop the video
        elif height_change_ratio > width_change_ratio:
            # Calculate the horizontal crop
            width_change = new_width - width
            new_x1 = round(width_change / 2)  # Calculate the starting x-coordinate
            new_x2 = min(new_x1 + width, new_width)  # Calculate the ending x-coordinate
            self.clip = moviepy_crop(self.clip, x1=new_x1, x2=new_x2)  # Crop the video
            self.clip = self.clip.resize((width, height))  # Resize to the final dimensions

        return self.clip  # Return the cropped video clip


class Tools:
    @staticmethod
    def round_down(num: float, decimals: int = 0) -> float:
        """
        Rounds down a number to a specified number of decimal places.

        :param num: The number to round down.
        :param decimals: The number of decimal places to round to (default is 0).
        :return: The rounded down number.
        """
        return math.floor(num * 10 ** decimals) / 10 ** decimals

    @staticmethod
    def get_file_hash(filepath):
        """
        Generate a simple hash for a file based on its path, size, and modification time.
        This is used for caching purposes.
        """
        try:
            stats = os.stat(filepath)
            return f"{filepath}_{stats.st_size}_{stats.st_mtime}"
        except:
            return filepath


class BackgroudVideo:
    @staticmethod
    def get_clip(duration: float) -> VideoFileClip:
        """
        Retrieves a random background video clip, trims it to the specified duration,
        and crops it to the target resolution.

        :param duration: The desired duration of the video clip.
        :return: A cropped and trimmed VideoFileClip object.
        """
        # Select a random clip from the background videos directory
        full_clip = VideoFileClip(BackgroudVideo.select_clip())
        
        # Trim the selected clip to the specified duration
        trimmed_clip = BackgroudVideo.trim_clip(full_clip, duration)

        # Crop the trimmed clip to 90% of its width
        width, height = trimmed_clip.size
        trimmed_clip = VideoTools(trimmed_clip).crop(round(width * 0.9), height)

        # Get the target resolution for the final clip
        target_resolution = BackgroudVideo.get_target_resolution()
        
        # Crop the trimmed clip to the target resolution
        cropped_clip = VideoTools(trimmed_clip).crop(target_resolution[0], target_resolution[1])

        # Return the cropped clip without audio
        return cropped_clip.set_audio(None)
    
    @staticmethod
    def select_clip() -> str:
        """
        Selects a random video clip from the background videos directory.

        :return: The file path of the selected video clip.
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        clip = random.choice(clips)
        return os.path.join(BACKGROUND_VIDEOS_DIR, clip)
    
    @staticmethod
    def trim_clip(clip: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Trims a video clip to a specified duration.

        :param clip: The VideoFileClip to trim.
        :param duration: The desired duration of the trimmed clip.
        :return: A trimmed VideoFileClip object.
        :raises ValueError: If the clip's duration is less than the specified duration.
        """
        if clip.duration < duration:
            raise ValueError(f"Clip duration {clip.duration} is less than duration {duration}")
        
        # Randomly select a start time for the subclip
        clip_start_time = Tools.round_down(random.uniform(0, clip.duration - duration))
        return clip.subclip(clip_start_time, clip_start_time + duration)

    @staticmethod
    def get_target_resolution():
        """
        Calculates the target resolution for the video clip based on the full resolution
        and the percentage reduction for the main clip.

        :return: A tuple containing the target width and height.
        """
        return (
            FULL_RESOLUTION[0], 
            round(FULL_RESOLUTION[1] * (1 - (PERCENT_MAIN_CLIP / 100)))
        )
    
    @staticmethod
    def format_all_background_clips():
        """
        Formats all background video clips in the specified directory by cropping them
        to the full resolution and saving them back to the directory.

        :return: None
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        for clip_name in clips:
            # Load each clip and crop it to the full resolution
            clip = VideoFileClip(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name))
            clip = VideoTools(clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])

            # Save the formatted clip back to the directory
            clip.write_videofile(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name), codec="libx264", audio_codec="aac")
            

class VideoCreation:
    # Class attributes for video and audio clips
    clip = None
    audio = None
    background_clip = None
    font_cache = {}  # Cache for loaded fonts

    def __init__(self, clip: VideoFileClip) -> None:
        # Initialize the VideoCreation object with a video clip
        self.clip = clip
        self.audio = clip.audio  # Extract audio from the video clip

    def __del__(self) -> None:
        # Clean up resources by closing video and background clips
        if self.clip:
            self.clip.close()
            self.clip = None
        if self.background_clip:
            self.background_clip.close()
            self.background_clip = None

    def process(self) -> VideoClip:
        # Main processing function to create the final video
        self.clip = self.create_final_clip()  # Create the final video clip
        transcription = self.create_transcription(self.audio)  # Generate transcription from audio
        self.clip = self.add_captions_to_video(self.clip, transcription)  # Add captions to the video

        return self.clip  # Return the processed video clip
    
    def split_video(self, segment_duration=30):
        # Split the video into segments of the specified duration
        segments = []
        for i in range(0, math.ceil(self.clip.duration / segment_duration)):
            end_time = min(start_time + segment_duration, self.clip.duration)
            segment = self.clip.subclip(start_time, end_time)
            segments.append(segment)
        return segments

    def create_final_clip(self):
        # Just return the original clip without background
        return self.clip
    
    @staticmethod
    def transcribe_audio(audio, file_dir):
        loaded_audio = whisper.load_audio(audio)
        model = whisper.load_model(MODEL_NAME, device="cuda" if USE_GPU else "cpu")

        logging.info("Transcribing audio...")
        
        # Dividir audios largos en segmentos para procesamiento
        chunk_size = 10 * 60  # 10 minutos en segundos
        audio_duration = len(loaded_audio) / 16000  # Whisper usa 16kHz
        
        if audio_duration > 15 * 60:  # Si dura más de 15 minutos
            logging.info(f"Audio largo detectado ({audio_duration:.2f}s), procesando en segmentos")
            
            all_timestamps = []
            for i in range(0, int(audio_duration), chunk_size):
                end_time = min(i + chunk_size, audio_duration)
                logging.info(f"Procesando segmento {i/60:.1f}m - {end_time/60:.1f}m")
                
                # Extraer el segmento de audio
                segment_samples = loaded_audio[int(i * 16000):int(end_time * 16000)]
                
                # Transcribir el segmento
                segment_result = whisper.transcribe(
                    model,
                    segment_samples,
                    language=LANGUAGE,
                    verbose=None,
                    beam_size=3,
                    temperature=0
                )
                
                # Ajustar timestamps para este segmento
                for segment in segment_result['segments']:
                    for word in segment['words']:
                        word['start'] += i  # Ajustar el tiempo de inicio
                        word['end'] += i    # Ajustar el tiempo de fin
                        all_timestamps.append({
                            'timestamp': (word['start'], word['end']),
                            'text': word['text']
                        })
            
            return all_timestamps
        
        # Para audios cortos, usa el método original
        result = whisper.transcribe(
            model,
            loaded_audio,
            language=LANGUAGE,
            verbose=None,
            beam_size=3,
            temperature=0
        )

        timestamps = []
        for segment in result['segments']:
            for word in segment['words']:
                timestamps.append({
                    'timestamp': (word['start'], word['end']),
                    'text': word['text']
                })
                
        return timestamps
    
    def preprocess_audio(self, input_audio_path, output_audio_path):
        logging.info(f"Preprocessing audio: {input_audio_path}")
        subprocess.run([
            "ffmpeg", "-i", input_audio_path,
            "-ac", "1",  # Convierte a mono
            "-ar", "16000",  # Reduce la frecuencia a 16 kHz
            "-b:a", "32k",  # Reduce tasa de bits para mejorar velocidad
            "-y", output_audio_path  # Sobrescribe si ya existe
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.remove(input_audio_path)  # Remove the original audio file
        logging.info(f"Preprocessed audio saved to {output_audio_path}")
        
    def create_transcription(self, audio):
        # Generate transcription from the audio
        os.makedirs("temp", exist_ok=True)  # Create a temporary directory for audio files

        # Create a unique file name for the audio file
        file_dir = f"temp/{time.time() * 10**20:.0f}.mp3"
        preprocessed_audio_path = f"temp/preprocessed_{time.time() * 10**20:.0f}.mp3"

        audio.write_audiofile(file_dir, codec="mp3", verbose=False, logger=None)  # Save original audio

        # Reduce audio quality for faster processing
        self.preprocess_audio(file_dir, preprocessed_audio_path)

        # Check if we have a cached transcription
        cache_key = Tools.get_file_hash(preprocessed_audio_path)
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}_transcription.pkl")

        if os.path.exists(cache_file):
            logging.info(f"Loading transcription from cache")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load transcription from cache: {e}")

        with Pool(processes=4) as pool:
            transcription_result = pool.apply(VideoCreation.transcribe_audio, args=(preprocessed_audio_path, file_dir))
            
        timestamps = transcription_result  # Get the timestamps from the transcription
        
        # Cache the transcription for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(timestamps, f)
        except Exception as e:
            logging.warning(f"Failed to cache transcription: {e}")
            
        # Clean up temporary audio files
        try:
            os.remove(file_dir)
            os.remove(audio)
        except FileNotFoundError:
            pass

        return timestamps  # Return the list of timestamps and words

    def add_captions_to_video(self, clip, timestamps):
        """
        Add subtitles to the video by overlaying text at specific times
        without modifying the structure of the original video.
        
        Args:
            clip: The original video clip
            timestamps: List of dictionaries with 'timestamp' (start, end) and 'text'
        
        Returns:
            A video clip with overlaid subtitles
        """
        if not timestamps:
            return clip  # Return original clip if no timestamps
        
        # List to store all text clips (subtitles)
        text_clips = []
        
        # Process timestamps to combine nearby words
        processed_timestamps = []
        queued_texts = []
        full_start = None
        
        # Optimize by pre-processing all timestamps in one pass
        for pos, timestamp in enumerate(timestamps):
            start, end = timestamp["timestamp"]
            text = timestamp["text"]
            
            # Adjust end time if there is a next timestamp
            if pos + 1 < len(timestamps):
                next_timestamp_start = timestamps[pos + 1]['timestamp'][0]
                if next_timestamp_start > end:
                    if next_timestamp_start - end > 0.5:
                        end += 0.5
                    else:
                        end = next_timestamp_start
            
            # If the difference is small, accumulate the text
            if pos > 0 and end - timestamps[pos-1]['timestamp'][1] < 0.3:
                if full_start is None:
                    full_start = start
                queued_texts.append(text)
                continue
            
            # Add accumulated text
            if queued_texts:
                text = " ".join(queued_texts) + " " + text
                queued_texts = []
                if full_start is not None:
                    start = full_start
                    full_start = None
            
            # Make sure times are within video limits
            start = max(0, start)
            end = min(clip.duration, end)
            
            if start >= end or start >= clip.duration:
                continue  # Skip invalid time ranges
            
            processed_timestamps.append({
                'timestamp': (start, end),
                'text': text
            })
        
        # Add any remaining text in the queue
        if queued_texts and full_start is not None:
            text = " ".join(queued_texts)
            start = full_start
            processed_timestamps.append({
                'timestamp': (start, min(clip.duration, end)),
                'text': text
            })
        
        # Use batch processing for text image creation
        text_images = {}
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            # Start all text image creation tasks
            future_to_text = {
                executor.submit(
                    self.create_text_image, 
                    timestamp['text'],
                    os.path.join(FONTS_DIR, FONT_NAME),
                    FONT_SIZE,
                    clip.size[0]
                ): timestamp['text'] for timestamp in processed_timestamps
            }
            
            # Collect results as they complete
            for future in future_to_text:
                text = future_to_text[future]
                try:
                    text_images[text] = future.result()
                except Exception as e:
                    logging.error(f"Error creating text image: {e}")
        
        # Create text clips for each processed timestamp
        for timestamp in processed_timestamps:
            start, end = timestamp['timestamp']
            text = timestamp['text']
            
            # Get the pre-created text image
            text_image = text_images.get(text)
            if text_image is None:
                continue
            
            # Convert image to clip and set its properties
            txt_clip = (ImageClip(np.array(text_image))
                    .set_start(start)
                    .set_duration(end - start))
            
            # Calculate Y position
            y_offset = round(FULL_RESOLUTION[1] * 0.85)
            txt_clip = txt_clip.set_position(("center", y_offset))
            
            text_clips.append(txt_clip)
        
        # Overlay all text clips on the original video
        if text_clips:
            # Use optimized method to create composite clip
            return CompositeVideoClip([clip] + text_clips)
        else:
            return clip

    def add_text_to_video(self, clip, text):
        # Add text overlay to the video clip
        text_image = self.create_text_image(
            text,
            os.path.join(FONTS_DIR, FONT_NAME),
            FONT_SIZE,
            clip.size[0]
        )

        image_clip = ImageClip(np.array(text_image), duration=clip.duration)  # Create an image clip for the text

        y_offset = round(FULL_RESOLUTION[1] * 0.85)  # Calculate vertical position for text
        clip = CompositeVideoClip([clip, image_clip.set_position(("center", y_offset,))])  # Overlay text on the video

        return clip  # Return the video clip with text

    @lru_cache(maxsize=128)
    def get_font(self, font_path, font_size):
        """Cache fonts to avoid reloading them multiple times"""
        if (font_path, font_size) not in self.font_cache:
            self.font_cache[(font_path, font_size)] = ImageFont.truetype(font_path, font_size)
        return self.font_cache[(font_path, font_size)]

    def create_text_image(self, text, font_path, font_size, max_width):
        # Create an image with the specified text
        initial_height = font_size * 3
        image = Image.new("RGBA", (max_width, initial_height), (0, 0, 0, 0))  # Create a transparent image

        # Use cached font loading
        font = self.get_font(font_path, font_size)
        draw = ImageDraw.Draw(image)  # Create a drawing context

        # Get the bounding box for the text
        _, _, w, h = draw.textbbox((0, 0), text, font=font)
        
        # Check if text is too wide for the image
        if w > max_width - 20:  # Leave some margin
            # Calculate how many words we can fit per line
            words = text.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                # Check width with added word
                test_line = " ".join(current_line + [word])
                _, _, test_w, _ = draw.textbbox((0, 0), test_line, font=font)
                
                if current_line and test_w > max_width - 40:  # If adding this word exceeds width
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            
            # Add the last line
            if current_line:
                lines.append(" ".join(current_line))
            
            # Create a taller image to accommodate multiple lines
            line_height = h * 1.2  # Add some space between lines
            new_height = int(line_height * len(lines) + font_size)
            multi_line_image = Image.new("RGBA", (max_width, new_height), (0, 0, 0, 0))
            multi_draw = ImageDraw.Draw(multi_line_image)
            
            # Draw each line centered
            for i, line in enumerate(lines):
                _, _, line_w, _ = multi_draw.textbbox((0, 0), line, font=font)
                y_pos = i * line_height
                multi_draw.text(
                    ((max_width - line_w) / 2, y_pos), 
                    line, 
                    font=font, 
                    fill="white", 
                    stroke_width=FONT_BORDER_WEIGHT, 
                    stroke_fill='black'
                )
            
            # Return the multi-line image with proper cropping
            return multi_line_image.crop((0, 0, max_width, new_height))
        else:
            # Original single-line text drawing
            draw.text(
                ((max_width - w) / 2, round(h * 0.2)), 
                text, 
                font=font, 
                fill="white", 
                stroke_width=FONT_BORDER_WEIGHT, 
                stroke_fill='black'
            )
            return image.crop((0, 0, max_width, round(h * 1.6)))  # Crop to fit the single line

def start_process(file_name, processes_status_dict, video_queue: multiprocessing.Queue):
    """
    Process a video file by applying transformations and saving the output.

    Args:
        file_name (str): The name of the video file to process.
        processes_status_dict (dict): A dictionary to track the status of processes.
        video_queue (multiprocessing.Queue): A queue to manage video processing tasks.
    """
    
    logging.info(f"Processing: {file_name}")  # Log the start of processing
    start_time = time.time()  # Record the start time

    # Get the current process identifier
    process_identifier = multiprocessing.current_process().pid

    # Mark the process as not finished in the status dictionary
    processes_status_dict[process_identifier] = False

    # Set lower resolution for faster processing of long videos
    input_path = os.path.join(INPUT_VIDEOS_DIR, file_name)
    
    # Check video duration before loading to decide on optimizations
    try:
        probe_result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        video_duration = float(probe_result.stdout.strip())
        
        # Apply different optimizations based on video length
        clip_params = {}
        if video_duration > 600:  # > 10 minutes
            clip_params = {
                'target_resolution': (854, 480),  # 480p for long videos
                'fps_target': 24
            }
        elif video_duration > 300:  # > 5 minutes
            clip_params = {
                'target_resolution': (1280, 720),  # 720p for medium videos
                'fps_target': 30
            }
        else:
            clip_params = {
                'target_resolution': None,  # Original resolution for short videos
                'fps_target': None  # Original FPS
            }
            
        logging.info(f"Video duration: {video_duration:.2f}s - Using optimized settings: {clip_params}")
    except Exception as e:
        logging.warning(f"Could not determine video duration: {e}")
        clip_params = {
            'target_resolution': None,
            'fps_target': None
        }
    
    # Load the input video file with optimized parameters
    try:
        input_video = VideoFileClip(
            input_path, 
            target_resolution=clip_params.get('target_resolution')  # Usa .get() para evitar KeyError
        )
        # If a target fps is specified, adjust the video clip's fps accordingly.
        if clip_params.get('fps_target') is not None:
            input_video = input_video.set_fps(clip_params.get('fps_target'))
    except Exception as e:
        logging.error(f"Error creando VideoFileClip: {e}")
        return  # Evita que el proceso continúe si hay un error
    
    # Process the video using a custom VideoCreation class
    output_video = VideoCreation(input_video).process()
    
    logging.info(f"Saving: {file_name}")  # Log the saving process

    # Define the output directory and calculate the end time for the subclip
    output_dir = os.path.join(OUTPUT_VIDEOS_DIR, file_name)
    end_time = round(((output_video.duration * 100 // output_video.fps) * output_video.fps / 100), 2)
    
    # Create a subclip of the output video
    output_video = output_video.subclip(t_end=end_time)

    # Optimize ffmpeg parameters for faster encoding
    ffmpeg_params = [
        '-preset', 'faster',  # Use faster preset for encoding speed
        '-tune', 'film',      # Optimize for film content
        '-movflags', '+faststart',  # Optimize for web streaming
        '-bf', '2',           # Use 2 B-frames for better compression
    ]
    
    if USE_GPU:
        # Check for hardware acceleration
        gpu_encoders = []
        
        # Try NVIDIA GPU acceleration
        try:
            nvidia_check = subprocess.run(
                ['ffmpeg', '-encoders'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if 'h264_nvenc' in nvidia_check.stdout:
                gpu_encoders.append('h264_nvenc')
        except:
            pass
            
        # Try Intel QuickSync
        try:
            intel_check = subprocess.run(
                ['ffmpeg', '-encoders'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if 'h264_qsv' in intel_check.stdout:
                gpu_encoders.append('h264_qsv')
        except:
            pass
            
        # Try AMD GPU acceleration
        try:
            amd_check = subprocess.run(
                ['ffmpeg', '-encoders'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if 'h264_amf' in amd_check.stdout:
                gpu_encoders.append('h264_amf')
        except:
            pass
            
        if gpu_encoders:
            encoder = gpu_encoders[0]
            logging.info(f"Using hardware acceleration with {encoder}")
            ffmpeg_params = ['-c:v', encoder] + ffmpeg_params
    
    # Attempt to save the output video, retrying up to 5 times on failure
    for pos in range(5):
        try:
            output_video.write_videofile(
                output_dir,
                codec="libx264",
                audio_codec="aac",
                fps=int(output_video.fps),
                threads=NUM_THREADS,
                preset='faster',
                verbose=False,
                logger=None,
                ffmpeg_params=ffmpeg_params
            )
            break  # Exit the loop if saving is successful
        except IOError:
            logging.warning(f"ERROR Saving: {file_name}. Trying again {pos + 1}/5")  # Log the error and retry
            time.sleep(1)  # Wait before retrying
    else:
        logging.error(f"ERROR Saving: {file_name}")  # Log if all attempts failed
    
    # Close the input and output video files to free resources
    input_video.close()
    output_video.close()

    # Log the runtime of the processing
    logging.info(f"Runtime: {round(time.time() - start_time, 2)} - {file_name}")
    
    # Mark the process as finished in the status dictionary
    processes_status_dict[process_identifier] = True


def delete_temp_folder():
    """
    Delete the temporary folder used for processing videos.
    """
    try:
        shutil.rmtree('cache')  # Remove the 'cache' directory and all its contents
        shutil.rmtree('temp')  # Remove the 'temp' directory and all its contents
    except (PermissionError, FileNotFoundError):
        pass  # Ignore permission errors if the folder cannot be deleted or is not found


def check_command(command):
    try:
        # Run the command and check if it is installed
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        return str(e)

def clone_repository():
    # Check for Git
    git_version = check_command(['git', '--version'])
    if not git_version:
        raise Exception("Git is not installed. Git must be installed to download model.")

    git_lfs_version = check_command(['git', 'lfs', 'version'])
    if not git_lfs_version:
        raise Exception("Git LFS is not installed. LFS is required to download model. Install Git LFS and try again.")
        
    repo_url = f'https://huggingface.co/openai/{MODEL_NAME}'
    
    logging.info(f"Cloning {repo_url}")
    # Run the git clone command
    subprocess.run(['git', 'clone', repo_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    logging.info(f"Cloned {repo_url}")


if __name__ == '__main__':
    # Clean up any temporary folders before starting
    delete_temp_folder()
    
    if not os.path.exists(MODEL_NAME):
        logging.warning(f'Model {MODEL_NAME} not found.')
        logging.info('Downloading model...')
        clone_repository()
        
    # Create a manager for shared data between processes
    manager = multiprocessing.Manager()
    processes_status_dict = manager.dict()  # Dictionary to track process statuses
    video_queue = multiprocessing.Queue()    # Queue to hold video file names

    # Create input and output directories if they don't exist
    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # List all video files in the input directory
    input_video_names = os.listdir(INPUT_VIDEOS_DIR)
    
    if not input_video_names:
        logging.warning("No videos found in the input directory. Please add videos to process.")
        print("No videos found in the input directory. Please add videos to process.")
    else:
        # Add video file names to the queue
        for name in input_video_names:
            video_queue.put(name)

        processes = {}  # Dictionary to store processes
        num_active_processes = 0  # Counter for active processes
        logging.info('STARTED')

        # Main loop to manage video processing
        while (video_queue.qsize() != 0) or (len(processes) != 0):
            # Check if we can start a new process
            if (num_active_processes < MAX_NUMBER_OF_PROCESSES) and (video_queue.qsize() > 0):
                file_name = video_queue.get()  # Get the next video file name from the queue

                # Create a new process for video processing
                p = multiprocessing.Process(target=start_process, args=(file_name, processes_status_dict, video_queue))
                p.start()  # Start the process
                processes[p.pid] = p  # Store the process in the dictionary
                num_active_processes += 1  # Increment the active process counter

            # Check for completed processes
            completed_pids = []
            for pid, complete in processes_status_dict.items():
                if complete:  # If the process is complete
                    completed_pids.append(pid)
            
            # Clean up completed processes
            for pid in completed_pids:
                if pid in processes:
                    processes[pid].join()  # Wait for the process to finish
                    del processes[pid]  # Remove the process from the dictionary
                    del processes_status_dict[pid]  # Remove the status from the dictionary
                    num_active_processes -= 1  # Decrement the active process counter
            
            # Sleep briefly to avoid high CPU usage in this monitoring loop
            time.sleep(0.1)

        # Clean up temporary folders after processing is complete
        delete_temp_folder()
        logging.info('MAIN PROCESS COMPLETE')