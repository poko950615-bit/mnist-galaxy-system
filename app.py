import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# 初始化 Flask，明確設定靜態檔案目錄為根目錄 '.'
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# 模型載入路徑優化
model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("✅ [MNIST GALAXY] 模型載入成功！")
except Exception as e:
    print(f"❌ [MNIST GALAXY] 模型載入失敗: {e}")

def advanced_preprocess(roi):
    """ 針對 MNIST 模型的影像預處理邏輯 """
    try:
        kernel = np.ones((2,2), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=1)
        h, w = roi.shape
        pad = int(max(h, w) * 0.45)
        roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        M = cv2.moments(roi_rs)
        if M["m00"] != 0:
            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            T = np.float32([[1, 0, 14-cx], [0, 1, 14-cy]])
            roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
        return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

@app.route('/')
def index():
    """ 確保首頁能正確加載 index.html """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ 辨識 API 主邏輯 """
    if model is None:
        return jsonify({'error': 'Server model is not available'}), 500
        
    try:
        req_data = request.get_json()
        if not req_data or 'image' not in req_data:
            return jsonify({'error': 'No image provided'}), 400
            
        # 解碼 Base64 圖片
        img_data = req_data['image']
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Image decoding failed'}), 400

        # 影像轉換與二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 120:  # 背景偵測自動反轉
            gray = 255 - gray

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 輪廓偵測與切割
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        cleaned_thresh = np.zeros_like(thresh)
        is_realtime = req_data.get('is_realtime', False)
        
        min_area = 500 if is_realtime else 150
        comps = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < min_area: continue
            if (w/h) > 2.5 or (w/h) < 0.15: continue
            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 由左至右排序
        comps.sort(key=lambda c: c[0])
        
        final_digits = ""
        details = []
        boxes = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # 處理連體字 (寬度過大時切割)
            if w > h * 1.3:
                proj = np.sum(roi, axis=0)
                split_x = np.argmin(proj[int(w*0.3):int(w*0.7)]) + int(w*0.3)
                sub_parts = [(roi[:, :split_x], x, y, split_x, h), 
                             (roi[:, split_x:], x + split_x, y, w - split_x, h)]
                for s_roi, sx, sy, sw, sh in sub_parts:
                    if s_roi.shape[1] < 5: continue
                    input_data = advanced_preprocess(s_roi)
                    if input_data is not None:
                        pred = model.predict(input_data, verbose=0)
                        d, c = int(np.argmax(pred)), float(np.max(pred))
                        final_digits += str(d)
                        details.append({"digit": d, "conf": f"{c*100:.1f}%"})
                        boxes.append({'x': int(sx), 'y': int(sy), 'w': int(sw), 'h': int(sh)})
                continue

            # 一般單數字辨識
            input_data = advanced_preprocess(roi)
            if input_data is not None:
                pred = model.predict(input_data, verbose=0)
                digit, conf = int(np.argmax(pred)), float(np.max(pred))
                
                # 即時模式下過濾低信心度結果，避免閃爍
                if is_realtime and conf < 0.8: continue
                    
                final_digits += str(digit)
                details.append({"digit": digit, "conf": f"{conf*100:.1f}%"})
                boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        return jsonify({
            'full_digit': final_digits,
            'details': details,
            'boxes': boxes
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 這是 Render 部署最重要的部分：監聽環境 Port
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
