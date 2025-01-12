from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import supervision as sv
import csv

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
FRAME_FOLDER = 'static/frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# 设置配置
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['FRAME_FOLDER'] = FRAME_FOLDER

# 加载 YOLO 模型
model = YOLO("models/best.pt")

# 类别映射表
class_names = {0: 'Vehicle'}

# 保存配置的全局变量
global_config = {
    "rois": [],
    "scale_factor": 0.0625,
    "confidence_threshold": 0.3,
    "iou_threshold": 0.7,
    "video_path": ""
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_config', methods=['POST'])
def set_config():
    """
    接收前端发送的检测框、比例和检测线信息
    """
    global global_config
    data = request.get_json()
    global_config.update(data)
    print("Received config:", global_config)
    return jsonify({"message": "Configuration received successfully!"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    上传视频，提取第一帧并返回图片路径和视频路径
    """
    file = request.files.get('file')
    url = request.form.get('url')
    input_path = None

    if file:
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
    elif url:
        response = requests.get(url, stream=True)
        filename = url.split("/")[-1]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        return jsonify({'error': 'No file or URL provided'})

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if ret:
        frame_path = os.path.join(app.config['FRAME_FOLDER'], f"{filename}.jpg")
        cv2.imwrite(frame_path, frame)
    else:
        return jsonify({'error': 'Failed to extract frame from video'})

    cap.release()
    return jsonify({"message": "Video uploaded successfully!", "frame_path": frame_path, "video_path": input_path})

def is_point_in_triangle(p, a, b, c):
    """
    判断点 p 是否在三角形 abc 内
    p, a, b, c: 坐标点 {'x': x, 'y': y}
    """
    def vector_cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    # 向量 p-a, p-b, p-c
    pa = (p['x'] - a['x'], p['y'] - a['y'])
    pb = (p['x'] - b['x'], p['y'] - b['y'])
    pc = (p['x'] - c['x'], p['y'] - c['y'])

    # 向量 a-b, b-c, c-a
    ab = (b['x'] - a['x'], b['y'] - a['y'])
    bc = (c['x'] - b['x'], c['y'] - b['y'])
    ca = (a['x'] - c['x'], a['y'] - c['y'])

    # 计算向量叉乘
    cross1 = vector_cross_product(pa, ab)
    cross2 = vector_cross_product(pb, bc)
    cross3 = vector_cross_product(pc, ca)

    # 判断点是否在三角形内
    return (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)

def is_point_in_quadrilateral(p, quad):
    """
    判断点 p 是否在四边形 quad 内
    p: 坐标点 {'x': x, 'y': y}
    quad: 四边形顶点 [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, {'x': x3, 'y': y3}, {'x': x4, 'y': y4}]
    """
    # 将四边形分成两个三角形
    a, b, c, d = quad
    return is_point_in_triangle(p, a, b, c) or is_point_in_triangle(p, a, c, d)

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    处理上传的视频，并应用四边形检测框
    """
    global global_config
    data = request.get_json()
    input_path = data.get('video_path')

    if not input_path:
        return jsonify({'error': 'No video path provided'})

    output_path = os.path.join(app.config['RESULT_FOLDER'], f"processed_{os.path.basename(input_path)}")

    # 处理视频，限制到用户定义的多个四边形 ROI
    cap = cv2.VideoCapture(input_path)
    writer = None

    # 获取配置
    rois = global_config["rois"]
    scale_factor = global_config["scale_factor"]
    confidence_threshold = global_config["confidence_threshold"]
    iou_threshold = global_config["iou_threshold"]
    vehicle_data = {}
    vehicle_count = 0
    vehicle_counts = []
    wrong_way_count = 0
    initial_directions = {}
    speed_threshold=120

    # 计算检测区域的最大和最小 y 坐标
    min_y = min(vertex['y'] for roi in rois for vertex in roi)
    max_y = max(vertex['y'] for roi in rois for vertex in roi)

    # 创建 ByteTrack 跟踪器
    tracker = sv.ByteTrack()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_vehicle_count = 0

        # 裁剪帧以仅包含检测区域
        detection_frame = frame[int(min_y):int(max_y), :, :]

        # 使用 YOLO 模型进行目标检测
        results = model(detection_frame, verbose=False, conf=confidence_threshold, iou=iou_threshold)[0]
        #print(results)
        # 提取检测结果
        detections = sv.Detections.from_ultralytics(results)

        # 更新 ByteTrack 跟踪器
        tracks = tracker.update_with_detections(detections) 
        print(tracks)
        used_labels_y = set()  # 集合用于存储已使用的y坐标

        # 绘制跟踪结果
        for index in range(len(tracks.class_id)):
            x1, y1, x2, y2 = tracks.xyxy[index]
            y1 += min_y
            y2 += min_y
            track_id = tracks.tracker_id[index]
            center = {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2}
            label = f'ID: {track_id}'
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label_y = int(y1) - 10
            while (label_y in used_labels_y):
                label_y -= 15  # 如果当前y坐标已被使用，向上移动一段距离
            used_labels_y.add(label_y)
            cv2.putText(frame, label, (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # 查找车辆所属的检测框
            roi_index = next((i for i, roi in enumerate(rois) if is_point_in_quadrilateral(center, roi)), None)

            if roi_index is None:
                continue

            if track_id in vehicle_data:
                last_position = vehicle_data[track_id]["last_position"]

                distance_moved = np.linalg.norm(np.array([center['x'], center['y']]) - np.array([last_position['x'], last_position['y']]))
                vehicle_data[track_id]["last_position"] = center
                speed = distance_moved * scale_factor * 30 * 3.6
                # 记录最大速度
                if speed > vehicle_data[track_id].get("max_speed", 0):
                    vehicle_data[track_id]["max_speed"] = speed
                    
                movement_direction = center['y'] - last_position['y']
                
                # 设置检测框的初始运动方向
                if roi_index not in initial_directions:
                    initial_directions[roi_index] = "up" if movement_direction < 0 else "down"

                # 利用检测框的初始运动方向来判断车辆是否逆行
                if (initial_directions[roi_index] == "up" and movement_direction > 0) or (initial_directions[roi_index] == "down" and movement_direction < 0):
                    vehicle_data[track_id]["direction"] = "wrong_way"
                    if not vehicle_data[track_id].get("counted_as_wrong_way", False):
                        wrong_way_count += 1
                        vehicle_data[track_id]["counted_as_wrong_way"] = True
                else:
                    vehicle_data[track_id]["direction"] = "correct_way"

                cv2.putText(frame, f"Speed: {speed:.2f} km/h", (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if vehicle_data[track_id]["direction"] == "wrong_way":
                    cv2.putText(frame, "Wrong Way!", (int(x1), int(y2) + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Correct Way", (int(x1), int(y2) + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 计算从上往下移动的车辆数目
                if last_position['y'] < (min_y + max_y) / 2 <= center['y']:
                    vehicle_count += 1
        
                # 计算从下往上移动的车辆数目
                if last_position['y'] > (min_y + max_y) / 2 >= center['y']:
                    vehicle_count += 1
                
                if speed > speed_threshold:
                    speed_frame_path = os.path.join(app.config['RESULT_FOLDER'], f"speed_{track_id}_y1_{y1}.jpg")
                    cv2.imwrite(speed_frame_path, frame)
            else:
                vehicle_data[track_id] = {"last_position": center}

            frame_vehicle_count += 1

        vehicle_counts.append(frame_vehicle_count)
        cv2.putText(frame, f"Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 绘制检测框
        for roi in rois:
            pts = np.array([[vertex['x'], vertex['y']] for vertex in roi], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)

    
    cap.release()
    writer.release()
    # 将视频转换为H.264编码
    h264_output_path = os.path.join(app.config['RESULT_FOLDER'], f"processed_{os.path.splitext(os.path.basename(input_path))[0]}_h264.mp4")
    ffmpeg_command = f"ffmpeg -i {output_path} -c:v libx264 -c:a aac -strict experimental {h264_output_path}"
    print(ffmpeg_command)
    os.system(ffmpeg_command)
        # 生成 CSV 文件
    csv_path = os.path.join(app.config['RESULT_FOLDER'], f"speeds_{os.path.splitext(os.path.basename(input_path))[0]}.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['track_id', 'max_speed']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for track_id, data in vehicle_data.items():
            writer.writerow({'track_id': track_id, 'max_speed': data.get('max_speed', 0)})


    return jsonify({
        "message": "Video processed successfully!",
        "video_path": f"/results/processed_{os.path.splitext(os.path.basename(input_path))[0]}_h264.mp4",  # 返回H.264编码的视频路径
        "csv_path": f"/results/speeds_{os.path.splitext(os.path.basename(input_path))[0]}.csv",  # 返回CSV文件路径
        "vehicle_counts": vehicle_counts,
        "wrong_way_count": wrong_way_count
    })
@app.route('/results/<filename>')
def serve_result(filename):
    """
    提供处理后视频的静态文件访问
    """
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/frames/<filename>')
def serve_frame(filename):
    """
    提供处理后视频的静态文件访问
    """
    return send_from_directory(app.config['FRAME_FOLDER'], filename)

@app.route('/js/<filename>')
def serve_js(filename):
    """
    提供js的静态文件访问
    """
    return send_from_directory('static/js', filename)

if __name__ == '__main__':
    app.run(debug=True)