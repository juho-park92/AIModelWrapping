import tensorflow as tf
import numpy as np
import os
from django.shortcuts import render
from django.conf import settings
from django.urls import reverse
from django.http import HttpResponseRedirect

# 1. 모델 로드
model_path = "models-for-facial-expression-recognition-tensorflow2-default-v1/best_simple_model"  # .pb 파일이 있는 디렉토리 경로
model = tf.saved_model.load(model_path)

# 2. 서명(Signature) 함수 가져오기
# 보통 'serving_default'라는 서명으로 모델이 저장됩니다.
infer = model.signatures['serving_default']


# 3. 입력 데이터 준비
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(48, 48))  # 흑백 이미지로 로드
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 48, 48, 1)
    img_array = img_array.astype(np.float32)  # dtype을 float32로 변환
    img_array = img_array / 255.0  # 정규화
    return img_array


def home(request):
    return render(request, 'home.html')


def upload_and_recognition(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']

        # media 디렉토리가 없으면 생성
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 입력 데이터 준비
        input_data = preprocess_image(file_path)
        input_tensor = tf.convert_to_tensor(input_data)

        # 모델 예측
        predictions = infer(input_1=input_tensor)

        # 예측 결과 해석
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_label = np.argmax(predictions['dense_1'].numpy(), axis=-1)  # 확률이 가장 높은 인덱스 선택
        decoded_label = emotion_labels[predicted_label[0]]  # 해당 인덱스의 감정 레이블

        # 결과를 템플릿으로 전달
        context = {
            'image_url': settings.MEDIA_URL + uploaded_file.name,
            'predicted_emotion': decoded_label,
        }

        return render(request, 'upload.html', context)

    return render(request, 'upload.html')
