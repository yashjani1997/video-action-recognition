import subprocess
import os

def convert_to_mp4(input_path, output_path):
    """
    Converts any video format to MP4 using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-y",                 # overwrite if exists
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output_path
