from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import pyttsx3
import os
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from numpy import random


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

# @app.route('/team')
# def team():
#     return render_template('team.html')

@app.route('/service', methods=['GET', 'POST'])
def service():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            result_filename = detect(file_path)
            return render_template('service.html', filename=result_filename)
    return render_template('service.html')

def detect(image_path):
    weights, imgsz = 'C:/Real-Time-Traffic-Sign-Detection-main/Model/weights/best.pt', 640
    device = select_device('')
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()

    save_dir = Path(increment_path(Path("../Results"), exist_ok=True))
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    
    dataset = LoadImages(image_path, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    spoken_counts = {}

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.50, 0.45, agnostic=False)

        t2 = time_synchronized()

        for i, det in enumerate(pred):
            p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])

                for *xyxy, conf, cls in reversed(det):
                    traffic_sign_name = names[int(cls)]
                    if traffic_sign_name in spoken_counts and spoken_counts[traffic_sign_name] >= 2:
                        continue
                    engine.say(traffic_sign_name)
                    engine.runAndWait()
                    if traffic_sign_name in spoken_counts:
                        spoken_counts[traffic_sign_name] += 1
                    else:
                        spoken_counts[traffic_sign_name] = 1

                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            print(s)
            try:
                im0 = cv2.putText(im0, "FPS: %.2f" % (1/(t2-t1)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            except:
                pass

            result_filename = p.name
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            cv2.imwrite(result_path, im0)

    return result_filename

from werkzeug.utils import secure_filename
from detect import detect
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import pyttsx3
import os
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from numpy import random




UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)



@app.route('/about', methods=['GET', 'POST'])
def indx():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the video with YOLOv5
            detect(source=file_path, save_img=True, save_dir=RESULT_FOLDER)
            result_video_path = os.path.join(RESULT_FOLDER, filename)
            return redirect(url_for('result', filename=filename))
    return render_template('about.html')

@app.route('/result/<filename>')
def result(filename):
    result_video_path = os.path.join(RESULT_FOLDER, filename)
    return render_template('result.html', video_path=result_video_path)

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
