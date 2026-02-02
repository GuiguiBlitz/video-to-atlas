import gradio as gr
import cv2
import numpy as np
import math
import os

def create_atlas(video_path, strategy, num_frames_target, interval_seconds, output_width, atlas_columns):
    """
    Extracts frames and stitches them into a single Texture Atlas image.
    """
    if not video_path:
        return None, None

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_indices = []
    if strategy == "Total Frame Count":
        if num_frames_target > 1:
            step = total_video_frames / (num_frames_target - 1)
            frame_indices = [int(i * step) for i in range(num_frames_target)]
            frame_indices = [min(f, total_video_frames - 1) for f in frame_indices]
        else:
            frame_indices = [0]
    else: 
        step_frames = int(interval_seconds * fps)
        if step_frames < 1: step_frames = 1
        frame_indices = range(0, total_video_frames, step_frames)
        if num_frames_target > 0: 
            frame_indices = frame_indices[:int(num_frames_target)]

    extracted_images = []
    target_h, target_w = 0, 0
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if output_width > 0:
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_height = int(output_width * aspect_ratio)
            frame = cv2.resize(frame, (int(output_width), new_height))
            
        if i == 0:
            target_h, target_w = frame.shape[:2]
        elif frame.shape[:2] != (target_h, target_w):
             frame = cv2.resize(frame, (target_w, target_h))

        extracted_images.append(frame)
    cap.release()

    if not extracted_images:
        return None, None

    count = len(extracted_images)
    if atlas_columns == 0:
        cols = math.ceil(math.sqrt(count)) 
    else:
        cols = int(atlas_columns)
        
    rows = math.ceil(count / cols)
    
    atlas_h = rows * target_h
    atlas_w = cols * target_w
    
    atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(extracted_images):
        r = idx // cols
        c = idx % cols
        y = r * target_h
        x = c * target_w
        atlas[y:y+target_h, x:x+target_w] = img

    output_path = "atlas_output.png"
    cv2.imwrite(output_path, cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
    
    return atlas, output_path

with gr.Blocks(title="Atlas Generator") as demo:
    gr.Markdown("## 🎞️ Video to Texture Atlas")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video", sources=["upload"])
            
            with gr.Group():
                strategy = gr.Radio(["Total Frame Count", "Time Interval"], label="Mode", value="Total Frame Count")
                num_frames = gr.Slider(1, 100, 12, step=1, label="Frame Count / Limit")
                interval = gr.Number(1.0, label="Interval (sec)")
            
            with gr.Group():
                width_s = gr.Slider(0, 1024, 0, step=8, label="Tile Width (0 = Original)")
                col_s = gr.Slider(0, 16, 0, step=1, label="Columns (0 = Auto)")

            btn = gr.Button("Generate Atlas", variant="primary")

        with gr.Column():
            out_img = gr.Image(label="Preview", show_label=False)
            out_file = gr.File(label="Download PNG")

    btn.click(create_atlas, [video_input, strategy, num_frames, interval, width_s, col_s], [out_img, out_file])

if __name__ == "__main__":
    demo.launch()