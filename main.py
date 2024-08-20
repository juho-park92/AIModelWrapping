import tensorflow as tf
import numpy as np

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


image_path = 'model_image_6.jpg'  # 사용할 이미지 경로
input_data = preprocess_image(image_path)

# 4. 모델 예측
# 입력 데이터를 TensorFlow 텐서로 변환
input_tensor = tf.convert_to_tensor(input_data)

# 'input_1'은 모델에 따라 다를 수 있습니다. 모델 서명을 확인해 적절한 이름을 사용하세요.
predictions = infer(input_1=input_tensor)

# 5. 예측 결과 해석
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
predicted_label = np.argmax(predictions['dense_1'].numpy(), axis=-1)  # 확률이 가장 높은 인덱스 선택
decoded_label = emotion_labels[predicted_label[0]]  # 해당 인덱스의 감정 레이블

print(f"Predicted Emotion: {decoded_label}")
