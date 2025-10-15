from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from deepface import DeepFace
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_age():
    try:
        if 'photo' not in request.files:
            return jsonify({'success': False, 'error': 'Выберите фото для анализа'})
        
        file = request.files['photo']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Файл не выбран'})
        
        if file.content_length > 20 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'Файл слишком большой. Максимум 20MB'})
        
        temp_path = "temp_analysis.jpg"
        file.save(temp_path)
        
        print("🔍 Запуск нейросети...")
        
        result = enhanced_age_analysis(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'success': False, 'error': 'Ошибка анализа'})

def enhanced_age_analysis(image_path):
    """Улучшенный анализ возраста без определения пола"""
    try:
        # Используем только возраст и эмоции
        analysis_results = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'emotion'],  # ТОЛЬКО ВОЗРАСТ И ЭМОЦИИ
            enforce_detection=False,
            detector_backend='opencv',
            silent=False
        )
        
        deepface_age = int(analysis_results[0]['age'])
        emotion = analysis_results[0]['dominant_emotion']
        
        print(f"🧠 DeepFace определил: {deepface_age} лет")
        
        # ДЕТАЛЬНЫЙ АНАЛИЗ ДЕТСКИХ ЧЕРТ
        child_features = analyze_child_features(image_path)
        
        # УМНАЯ КОРРЕКЦИЯ ВОЗРАСТА
        corrected_age = smart_age_correction(deepface_age, child_features)
        
        emotion_ru = translate_emotion(emotion)
        
        return {
            'success': True,
            'age': corrected_age,
            'emotion': emotion_ru,
            'category': get_age_category(corrected_age),
            'confidence': get_confidence_level(child_features['score']),
            'message': get_age_message(corrected_age, emotion_ru),
            'analysis': get_analysis_details(child_features),
            'original_age': deepface_age,
            'child_score': round(child_features['score'], 1)
        }
        
    except Exception as e:
        print(f"❌ Ошибка нейросети: {e}")
        return fallback_analysis(image_path)

def analyze_child_features(image_path):
    """Анализ детских характеристик лица"""
    img = cv2.imread(image_path)
    if img is None:
        return {'score': 0, 'features': {}}
    
    features = {}
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return {'score': 0, 'features': {}}
        
        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'score': 0, 'features': {}}
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. АНАЛИЗ ГЛАЗ (дети имеют большие глаза)
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        features['eyes_detected'] = len(eyes)
        
        if len(eyes) >= 2:
            eye_area = sum(ew * eh for (ex, ey, ew, eh) in eyes)
            face_area = w * h
            features['eye_face_ratio'] = eye_area / face_area
        else:
            features['eye_face_ratio'] = 0.15
        
        # 2. АНАЛИЗ ТЕКСТУРЫ КОЖИ
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        features['skin_smoothness'] = 1.0 - min(laplacian_var / 120.0, 1.0)  # Более чувствительный
        
        # 3. АНАЛИЗ ФОРМЫ ЛИЦА
        aspect_ratio = w / h
        features['face_roundness'] = 1.0 - min(abs(aspect_ratio - 0.85), 0.3) / 0.3
        
        # 4. ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ
        features['brightness'] = np.mean(gray_face) / 255.0
        features['contrast'] = gray_face.std() / 100.0
        
        # ВЫЧИСЛЯЕМ ОБЩИЙ SCORE ДЕТСКОСТИ
        child_score = calculate_child_score(features)
        
        print(f"📊 Анализ: глаза {features.get('eye_face_ratio', 0):.3f}, "
              f"гладкость {features.get('skin_smoothness', 0):.2f}, "
              f"округлость {features.get('face_roundness', 0):.2f}")
        
        return {'score': child_score, 'features': features}
        
    except Exception as e:
        print(f"❌ Ошибка анализа черт: {e}")
        return {'score': 0, 'features': {}}

def calculate_child_score(features):
    """Вычисление оценки детскости"""
    score = 0
    
    # Большие глаза (самый важный признак)
    eye_ratio = features.get('eye_face_ratio', 0.15)
    if eye_ratio > 0.25:
        score += 4.0
    elif eye_ratio > 0.20:
        score += 3.0
    elif eye_ratio > 0.16:
        score += 2.0
    elif eye_ratio > 0.13:
        score += 1.0
    
    # Гладкая кожа
    smoothness = features.get('skin_smoothness', 0.5)
    if smoothness > 0.85:
        score += 3.0
    elif smoothness > 0.70:
        score += 2.0
    elif smoothness > 0.55:
        score += 1.0
    
    # Круглое лицо
    roundness = features.get('face_roundness', 0.5)
    if roundness > 0.85:
        score += 2.0
    elif roundness > 0.70:
        score += 1.0
    
    # Яркость (дополнительный признак)
    brightness = features.get('brightness', 0.5)
    if brightness > 0.75:
        score += 0.5
    
    return min(score, 10.0)

def smart_age_correction(deepface_age, child_analysis):
    """Умная коррекция возраста на основе детских признаков"""
    child_score = child_analysis['score']
    
    print(f"🎯 Оценка детскости: {child_score:.1f}/10, DeepFace: {deepface_age} лет")
    
    # СИЛЬНЫЕ ДЕТСКИЕ ПРИЗНАКИ
    if child_score >= 8.0:
        if deepface_age > 12:
            return max(1, min(6, deepface_age - 20))  # Младенец/малыш
        else:
            return deepface_age
    
    # ЯВНЫЕ ДЕТСКИЕ ПРИЗНАКИ
    elif child_score >= 6.0:
        if deepface_age > 15:
            return max(4, min(10, deepface_age - 12))  # Ребенок
        else:
            return deepface_age
    
    # УМЕРЕННЫЕ ДЕТСКИЕ ПРИЗНАКИ
    elif child_score >= 4.0:
        if deepface_age > 18:
            return max(8, min(15, deepface_age - 8))  # Подросток
        else:
            return deepface_age
    
    # СЛАБЫЕ ДЕТСКИЕ ПРИЗНАКИ
    elif child_score >= 2.0:
        if deepface_age < 10:
            return min(20, deepface_age + 8)  # Молодой взрослый
        else:
            return deepface_age
    
    # ВЗРОСЛЫЕ ПРИЗНАКИ
    else:
        if deepface_age < 15:
            return min(30, deepface_age + 12)  # Взрослый
        else:
            return deepface_age

def get_confidence_level(child_score):
    """Определение уровня уверенности"""
    if child_score >= 7.0:
        return "очень высокая"
    elif child_score >= 5.0:
        return "высокая"
    elif child_score >= 3.0:
        return "средняя"
    else:
        return "базовая"

def get_age_message(age, emotion):
    """Сообщение в зависимости от возраста"""
    if age < 3:
        return f"Младенец! Эмоция: {emotion.lower()} 👶"
    elif age < 7:
        return f"Маленький ребенок! Эмоция: {emotion.lower()} 🧒"
    elif age < 13:
        return f"Ребенок! Эмоция: {emotion.lower()} 👦"
    elif age < 18:
        return f"Подросток! Эмоция: {emotion.lower()} 🎓"
    elif age < 30:
        return f"Молодой взрослый! Эмоция: {emotion.lower()} 💼"
    elif age < 50:
        return f"Взрослый! Эмоция: {emotion.lower()} 🔥"
    else:
        return f"Зрелый возраст! Эмоция: {emotion.lower()} 🎯"

def get_analysis_details(child_analysis):
    """Детали анализа"""
    features = child_analysis['features']
    score = child_analysis['score']
    
    details = []
    
    if features.get('eye_face_ratio', 0) > 0.2:
        details.append("большие глаза")
    elif features.get('eye_face_ratio', 0) > 0.15:
        details.append("средние глаза")
    else:
        details.append("маленькие глаза")
    
    if features.get('skin_smoothness', 0) > 0.7:
        details.append("гладкая кожа")
    elif features.get('skin_smoothness', 0) > 0.5:
        details.append("нормальная кожа")
    else:
        details.append("текстурированная кожа")
    
    if features.get('face_roundness', 0) > 0.7:
        details.append("округлое лицо")
    else:
        details.append("вытянутое лицо")
    
    return f"Анализ: {', '.join(details)} (детскость: {score:.1f}/10)"

def fallback_analysis(image_path):
    """Резервный анализ"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': 'Не удалось прочитать изображение'}
        
        child_features = analyze_child_features(image_path)
        
        # Определяем возраст на основе детских признаков
        if child_features['score'] >= 7.0:
            age = 3
        elif child_features['score'] >= 5.0:
            age = 8
        elif child_features['score'] >= 3.0:
            age = 14
        else:
            age = 28
        
        return {
            'success': True,
            'age': age,
            'emotion': "Нейтральный",
            'category': get_age_category(age),
            'confidence': 'резервный анализ',
            'message': f'Резервный анализ: {age} лет',
            'analysis': get_analysis_details(child_features)
        }
        
    except Exception as e:
        print(f"❌ Ошибка резервного анализа: {e}")
        return {'success': False, 'error': 'Не удалось проанализировать изображение'}

def translate_emotion(emotion):
    emotions_map = {
        'angry': 'Сердитый', 'disgust': 'Неприязнь', 'fear': 'Испуг', 
        'happy': 'Счастливый', 'sad': 'Грустный', 'surprise': 'Удивленный',
        'neutral': 'Нейтральный'
    }
    return emotions_map.get(emotion, emotion)

def get_age_category(age):
    if age < 3: return "Младенец"
    elif age < 7: return "Маленький ребенок"
    elif age < 13: return "Ребенок"
    elif age < 18: return "Подросток" 
    elif age < 30: return "Молодой взрослый"
    elif age < 50: return "Взрослый"
    else: return "Зрелый возраст"

if __name__ == '__main__':
    print("🚀 ЗАПУСК УЛУЧШЕННОЙ СИСТЕМЫ ОПРЕДЕЛЕНИЯ ВОЗРАСТА!")
    print("👶 Точный анализ детских черт: глаза, кожа, форма лица")
    print("🎯 Умная коррекция возрастов DeepFace")
    print("📊 Без определения пола - только возраст и эмоции")
    print("🌐 Откройте: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)