from asyncio import SendfileNotAvailableError
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from vgg import isFightingFace
from gan import createFightFace
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import io
from PIL import Image
import cv2
from flask import send_file


app = Flask(__name__)

# 画像のアップロード先ディレクトリ
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# vgg.pyとgan.pyから必要な関数をインポート
# from vgg import isFightingFace
# from gan import createFightFace

import os
os.makedirs('./uploads', exist_ok=True)
# モデルの保存パス
save_path = './saved_model'
# モデルを読み込む
loaded_model = tf.saved_model.load(save_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/judge', methods=['GET', 'POST'])
def judge():
    if request.method == 'POST':
        if 'upload_file' not in request.files:
            return redirect(request.url)
        file = request.files['upload_file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # 判定結果を取得
            result = isFightingFace(filepath)
            if isinstance(result, tuple):
                result = result[0]  # 例: タプルの最初の要素を使用する
            # resultをパーセンテージ形式で1から減算
            percentage_result = 1 - result
            percentage_result = "{:.1%}".format(percentage_result) #少数１桁まで表示
            return render_template('judge.html', result=percentage_result) #戦わない顔の割合で表示
    return render_template('judge.html', result="")


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        if 'upload_file' not in request.files:
            return redirect(request.url)
        file = request.files['upload_file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # 生成結果を取得
            generated_image = createFightFace(filepath)
            # 生成された画像のファイル名を設定
            generated_filename = f"generated_{filename}"
            generated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
            # 生成された画像を保存
            cv2.imwrite(generated_filepath, generated_image)
            return render_template('generate.html', generated_image=generated_filename)
            # 生成された画像を表示
            plt.imshow(generated_image) #選手名鑑風に表示させたい
            plt.axis('off')
            plt.show()

    return render_template('generate.html', generated_image="")

@app.route('/generated_image/<filename>')
def generated_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(image_path, mimetype='image/png')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
