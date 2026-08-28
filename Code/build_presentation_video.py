"""Builds a narrated MP4 video of the presentation deck: exports each slide to a
JPG via PowerPoint COM automation, synthesises narration audio per slide with
offline TTS (pyttsx3), renders each slide as a static video clip for the length of
its narration, and concatenates all clips into one master video.

Must be run with a Python interpreter that has pywin32, pyttsx3, opencv-python, and
imageio-ffmpeg installed, and requires PowerPoint installed locally (Windows COM
automation) since python-pptx cannot rasterise slides itself.
"""
import argparse
import os
import re
import shutil
import subprocess
import wave

import cv2
import imageio_ffmpeg
import pyttsx3
import win32com.client


def export_slides_to_images(pptx_path, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  # PowerPoint creates this folder itself on SaveAs
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    powerpoint.Visible = 1
    try:
        pres = powerpoint.Presentations.Open(pptx_path, WithWindow=False)
        pres.SaveAs(out_dir, 17)  # ppSaveAsJPG = 17 -> exports one JPG per slide
        pres.Close()
    finally:
        powerpoint.Quit()
    # PowerPoint names files "Slide1.JPG", "Slide2.JPG", ... in the target folder.


def parse_slide_narration(script_path):
    with open(script_path, "r", encoding="utf-8") as f:
        full_script = f.read()
    blocks = re.findall(
        r"\*\*\[Slide (\d+) . [^\]]+\]\*\*\n(.*?)(?=\n\n\*\*\[Slide |\Z)",
        full_script, re.DOTALL,
    )
    return [(int(n), text.strip().replace("\n", " ")) for n, text in blocks]


def build_video(base_dir, pptx_name, script_name, output_name, voice_rate=155):
    base_dir = os.path.abspath(base_dir)
    pptx_path = os.path.join(base_dir, pptx_name)
    script_path = os.path.join(base_dir, script_name)
    slides_img_dir = os.path.join(base_dir, "slides_img")
    temp_dir = os.path.join(base_dir, "temp_video_build")
    os.makedirs(temp_dir, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if os.path.isdir(slides_img_dir) and any(f.lower().endswith(".jpg") for f in os.listdir(slides_img_dir)):
        # PowerPoint COM automation needs an interactive desktop session; it hangs
        # indefinitely when this script itself runs detached/backgrounded. Slides
        # are exported separately in the foreground beforehand in that case, so
        # skip re-exporting (and re-hanging) here if images already exist.
        print(f"Found existing exported slides in {slides_img_dir}, skipping COM export.")
    else:
        print(f"Exporting slides from {pptx_name} to images...")
        export_slides_to_images(pptx_path, slides_img_dir)

    slide_blocks = parse_slide_narration(script_path)
    print(f"Found {len(slide_blocks)} slide narration blocks.")

    slide_clips = []
    for slide_num, narration_text in slide_blocks:
        print(f"\nProcessing Slide {slide_num}...")
        print(f"Text snippet: {narration_text[:80]}...")

        # A fresh engine per slide avoids a known pyttsx3/SAPI5 hang on Windows
        # where reusing one engine across repeated save_to_file+runAndWait calls
        # can deadlock after the first call (observed directly during this build).
        wav_path = os.path.join(temp_dir, f"slide_{slide_num}.wav")
        engine = pyttsx3.init()
        engine.setProperty("rate", voice_rate)
        engine.save_to_file(narration_text, wav_path)
        engine.runAndWait()
        engine.stop()
        del engine

        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate) + 0.5  # padding

        print(f"Audio duration: {duration:.2f} seconds")

        img_path = os.path.join(slides_img_dir, f"Slide{slide_num}.JPG")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing slide image: {img_path}")
        h, w, _ = img.shape

        fps = 30
        total_video_frames = int(duration * fps)

        silent_avi_path = os.path.join(temp_dir, f"slide_{slide_num}_silent.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(silent_avi_path, fourcc, fps, (w, h))
        for _ in range(total_video_frames):
            out.write(img)
        out.release()

        clip_mp4_path = os.path.join(temp_dir, f"slide_{slide_num}.mp4")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", silent_avi_path,
            "-i", wav_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            clip_mp4_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error encoding slide {slide_num}:", res.stderr)
        else:
            print(f"Slide {slide_num} MP4 created successfully.")
            slide_clips.append(clip_mp4_path)

    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for clip in slide_clips:
            f.write(f"file '{clip.replace(os.sep, '/')}'\n")

    final_video_path = os.path.join(base_dir, output_name)
    concat_cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        final_video_path,
    ]

    print("\nCombining all slide clips into master presentation video...")
    res = subprocess.run(concat_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("MASTER PRESENTATION VIDEO GENERATED SUCCESSFULLY!")
        print(f"File location: {final_video_path}")
        print(f"Video size: {os.path.getsize(final_video_path) / (1024*1024):.2f} MB")
    else:
        print("Error concatenating master video:", res.stderr)
        return

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    _default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "Supplementary")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=os.path.normpath(_default_base))
    parser.add_argument("--pptx", default="Presentation_FQL-GTA.pptx")
    parser.add_argument("--script", default="Presentation_Recording_Script.md")
    parser.add_argument("--output", default="FQL-GTA_Presentation_Video.mp4")
    args = parser.parse_args()
    build_video(args.base_dir, args.pptx, args.script, args.output)
