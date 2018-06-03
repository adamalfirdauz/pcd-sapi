import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import pickle
import cv2
from scipy import stats
import numpy as np

UPLOAD_FOLDER = 'temp/'
RESULT_FOLDER = 'result/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_1 = cv2.CascadeClassifier('xml/1.xml')
class_2 = cv2.CascadeClassifier('xml/2.xml')
class_3 = cv2.CascadeClassifier('xml/3.xml')
class_4 = cv2.CascadeClassifier('xml/4.xml')
class_5 = cv2.CascadeClassifier('xml/5.xml')
class_6 = cv2.CascadeClassifier('xml/6.xml')
class_7 = cv2.CascadeClassifier('xml/7.xml')
class_8 = cv2.CascadeClassifier('xml/8.xml')


def patternDetection(filename, img):
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    class1 = class_1.detectMultiScale(gray, 1.3, 2)
    class2 = class_2.detectMultiScale(gray, 1.3, 1)
    class3 = class_3.detectMultiScale(gray, 1.3, 3)
    class4 = class_4.detectMultiScale(gray, 1.3, 1)
    class5 = class_5.detectMultiScale(gray, 1.3, 1)
    class6 = class_6.detectMultiScale(gray, 1.3, 3)
    class7 = class_7.detectMultiScale(gray, 1.3, 1)
    class8 = class_8.detectMultiScale(gray, 1.3, 1)
    for (x, y, w, h) in class1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 1', (x - w, y - h), font,
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)
    for (x, y, w, h) in class2:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 2', (x - w, y - h), font,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y, w, h) in class3:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 3', (x - w, y - h), font,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in class4:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 4', (x - w, y - h), font,
                    0.5, (255, 255, 0), 2, cv2.LINE_AA)
    for (x, y, w, h) in class5:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 5', (x - w, y - h), font,
                    0.5, (0, 255, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in class6:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 6', (x - w, y - h), font,
                    0.5, (255, 0, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in class7:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 7', (x - w, y - h),
                    font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    for (x, y, w, h) in class8:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Class 8', (x - w, y - h), font,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite('result/'+filename, img)
    return 'result/'+filename

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        class1 = class_1.detectMultiScale(gray, 1.3, 3)
        class2 = class_2.detectMultiScale(gray, 1.3, 3)
        class3 = class_3.detectMultiScale(gray, 1.3, 4)
        class4 = class_4.detectMultiScale(gray, 1.3, 3)
        class5 = class_5.detectMultiScale(gray, 1.3, 3)
        class6 = class_6.detectMultiScale(gray, 1.3, 4)
        class7 = class_7.detectMultiScale(gray, 1.3, 3)
        class8 = class_8.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in class1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 1', (x - w, y - h), font,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)
        for (x, y, w, h) in class2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 2', (x - w, y - h), font,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
        for (x, y, w, h) in class3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 3', (x - w, y - h), font,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
        for (x, y, w, h) in class4:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 4', (x - w, y - h), font,
                        0.5, (255, 255, 0), 2, cv2.LINE_AA)
        for (x, y, w, h) in class5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 5', (x - w, y - h), font,
                        0.5, (0, 255, 255), 2, cv2.LINE_AA)
        for (x, y, w, h) in class6:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 6', (x - w, y - h), font,
                        0.5, (255, 0, 255), 2, cv2.LINE_AA)
        for (x, y, w, h) in class7:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 7', (x - w, y - h),
                        font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        for (x, y, w, h) in class8:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Class 8', (x - w, y - h), font,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return index()



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread('temp/'+filename)
            link = patternDetection(filename, img)
            return render_template('home.html', filename=filename, berhasil="true", state=1)
    if request.method == 'GET':
        return render_template('home.html', state=0)


@app.route('/img/<filename>', methods=['GET'])
def show_file(filename):
    return send_from_directory('result/', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
