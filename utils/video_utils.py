import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, fps, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = output_video_frames[0].shape[1], output_video_frames[0].shape[0]

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f"Video was saved to {output_video_path}")


