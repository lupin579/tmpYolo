import cv2
import torch
from concurrent.futures import ThreadPoolExecutor

def detection_task(model, frame):
    # 对帧进行检测
    results = model(frame)
    # 提取检测结果
    detections = results.pandas().xyxy[0]  # 获取检测结果为pandas DataFrame
    return detections, frame

def process_video(video_path, model_path, skip_frames=5):
    # 加载模型，指定在CPU上运行
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device='cpu')

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                # 异步提交检测任务
                future = executor.submit(detection_task, model, frame)
                detections, frame = future.result()  # 获取检测结果

                # 在帧上绘制边界框和标签
                for index, row in detections.iterrows():
                    cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
                                  (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
                    cv2.putText(frame, f"{row['name']} {row['confidence']:.2f}",
                                (int(row['xmin']), int(row['ymin']) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # 没有进行检测的帧可以在这里处理其他逻辑，例如简单地显示
                print(f'current frame {frame_count}')

            # 显示帧
            cv2.imshow('Video', frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "C:\\Users\\l2333\\Desktop\\test_datasets\\4783667-hd_2048_1080_25fps.mp4"  # 更新为你的视频文件路径
    model_path = "D:\\yoolo\\yolov5\\weights\\best.pt"  # 更新为你的模型权重文件路径
    process_video(video_path, model_path, skip_frames=20)
