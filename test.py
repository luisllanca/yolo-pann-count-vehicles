import os
import sys
import cv2
import json
import numpy as np
import librosa
import torch
import platform
from moviepy.editor import VideoFileClip
from panns_inference import SoundEventDetection, labels as pann_labels
from ultralytics import YOLO
from tqdm import tqdm
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PANN = r"C:\Users\gravi\panns_data\Cnn14_DecisionLevelMax.pth"
YOLO_MODEL_PATH = "yolo11s.pt"
class_map = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
target_classes = {"bicycle", "car", "motorcycle", "bus", "truck"}
CONF_THRESHOLD = 0.6
crossing_threshold = 10
GENERAL_INFO_JSON_PATH = "general_info.json"

def select_line(frame):
    pts = []
    def mouse_callback(event, x, y, flags, param):
        nonlocal pts, frame
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(pts) == 2:
                cv2.line(frame, pts[0], pts[1], (0, 0, 255), 2)
            cv2.imshow("Select Line", frame)
    cv2.imshow("Select Line", frame)
    cv2.setMouseCallback("Select Line", mouse_callback)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or len(pts) >= 2:
            break
    cv2.destroyWindow("Select Line")
    if len(pts) < 2:
        raise Exception("Not enough points")
    return pts[0], pts[1]

def point_line_distance(pt, l1, l2):
    x0, y0 = pt
    x1, y1 = l1
    x2, y2 = l2
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den if den != 0 else float('inf')

def process_audio_sed(video_path, checkpoint_path):
    clip = VideoFileClip(video_path)
    temp_audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(temp_audio_path, logger=None)
    sed = SoundEventDetection(checkpoint_path=checkpoint_path, device=DEVICE)
    audio, sr = librosa.load(temp_audio_path, sr=32000, mono=True)
    audio = audio[None, :]
    framewise_output = sed.inference(audio)[0]
    fps_audio = 32000 / 320
    os.remove(temp_audio_path)
    vehicle_label_indices = [i for i, label in enumerate(pann_labels) if label.lower() in target_classes]
    return framewise_output, fps_audio, vehicle_label_indices

def process_video_and_record_events(video_path, framewise_output, fps_audio, vehicle_label_indices, images_folder):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
    line_pt1, line_pt2 = select_line(first_frame.copy())
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    events = []
    triggered_centers = []
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Procesando video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 2)
        results = yolo_model.predict(frame, conf=CONF_THRESHOLD, iou=0.3, verbose=False)[0]
        t_frame = frame_idx / fps_video
        audio_idx = int(t_frame * fps_audio)
        if audio_idx >= framewise_output.shape[0]:
            audio_idx = framewise_output.shape[0] - 1
        audio_scores = framewise_output[audio_idx, :]
        target_scores = audio_scores[vehicle_label_indices]
        highest_prob = np.max(target_scores)
        index_in_target = np.argmax(target_scores)
        pann_class_index = vehicle_label_indices[index_in_target]
        pann_class = pann_labels[pann_class_index]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            if cls_id not in class_map or class_map[cls_id] not in target_classes:
                continue
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            dist = point_line_distance(center, line_pt1, line_pt2)
            if dist < crossing_threshold:
                duplicate = False
                for (prev_center, prev_frame) in triggered_centers:
                    if frame_idx - prev_frame < 10 and np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
                        duplicate = True
                        break
                if not duplicate:
                    crop = frame[y1:y2, x1:x2]
                    image_path = os.path.join(images_folder, f"{frame_idx}_{cls_id}.jpg")
                    cv2.imwrite(image_path, crop)
                    event = {
                        "video_id": os.path.basename(os.path.dirname(images_folder)),
                        "frame": frame_idx,
                        "timestamp": t_frame,
                        "yolo": {"bbox": [x1, y1, x2, y2], "class": class_map[cls_id], "prob": conf},
                        "pann": {"class": pann_class, "prob": float(highest_prob)},
                        "image": image_path
                    }
                    events.append(event)
                    triggered_centers.append((center, frame_idx))
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return events

def append_general_info(new_info, general_info_json_path):
    if os.path.exists(general_info_json_path):
        with open(general_info_json_path, "r") as f:
            general_info = json.load(f)
    else:
        general_info = []
    general_info.append(new_info)
    with open(general_info_json_path, "w") as f:
        json.dump(general_info, f, indent=4)

def main():
    if len(sys.argv) < 2:
        print("Uso: python script.py <ruta_del_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print("El archivo de video no existe.")
        sys.exit(1)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    video_id = f"{video_name}_{timestamp_str}"
    output_folder = os.path.join("outputs", video_id)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    events_json_path = os.path.join(output_folder, "detection_events.json")
    t1_pann = time.time()
    framewise_output, fps_audio, vehicle_label_indices = process_audio_sed(video_path, CHECKPOINT_PANN)
    t2_pann = time.time()
    t1_yolo = time.time()
    events = process_video_and_record_events(video_path, framewise_output, fps_audio, vehicle_label_indices, images_folder)
    t2_yolo = time.time()
    with open(events_json_path, "w") as f:
        json.dump(events, f, indent=4)
    video_clip = VideoFileClip(video_path)
    if DEVICE == "cuda":
        processor_model = torch.cuda.get_device_name(0)
    else:
        processor_model = platform.processor()
    general_info_record = {
        "video_id": video_id,
        "video": video_path,
        "duracion_video": video_clip.duration,
        "tiempo_pann": t2_pann - t1_pann,
        "tiempo_yolo": t2_yolo - t1_yolo,
        "caracteristica": DEVICE,
        "modelo_procesador": processor_model,
        "cantidad_eventos": len(events)
    }
    append_general_info(general_info_record, GENERAL_INFO_JSON_PATH)
    print(f"Procesamiento completado para el video {video_path}. Salidas en {output_folder}")

if __name__ == "__main__":
    main()
