####実行####
from keras.models import load_model
import pickle
import numpy as np
import cv2

#モデルとクラス名の読み込み
model = load_model('./kyoda.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classes = pickle.load(open('classes.sav', 'rb'))

def isFightingFace(image_path):
    # 画像の前処理
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img /= 255.0
    img = img[None, ...]
    # 予測の実行
    result = model.predict(img)

    np.set_printoptions(precision=3, suppress=True)
    result_percent = result * 100  # %表示

    # '戦う顔' クラスの予測結果を返す
    is_fighting_face = (result.argmax() == 0)
    confidence_score = result_percent[0][0]
    
    return is_fighting_face, confidence_score

# 画像を入力して関数をテスト
#fighting_face, confidence = isFightingFace('./static/image/hyuma.jpg')
#print("Is fighting face?: ", fighting_face)
#print("Confidence score: ", confidence)

