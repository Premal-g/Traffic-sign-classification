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



from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from detect import detect



UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
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
