import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# --- 1. 伺服器核心設定 (針對 Render 雲端環境) ---
# static_folder='.' 代表從根目錄找 index.html, style.css, script.js
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# 載入模型 - 確保 mnist_model.h5 在根目錄
MODEL_PATH = 'mnist_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MNIST Galaxy Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# --- 2. 頁面路由設定 ---
@app.route('/')
def index():
    # 確保打開網址就能看到你的銀河介面
    return send_from_directory(app.static_folder, 'index.html')

# --- 3. 完整移植：你的 p.py 進階影像處理邏輯 ---
def advanced_preprocess(roi):
    # 強化筆畫
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)
    
    h, w = roi.shape
    # 動態 Padding
    pad = int(max(h, w) * 0.45)
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 縮放至 28x28
    roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 質心校正 (提升辨識率的關鍵)
    M = cv2.moments(roi_rs)
    if M["m00"] != 0:
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        T = np.float32([[1, 0, 14-cx], [0, 1, 14-cy]])
        roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
        
    return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0

# --- 4. 辨識介面：整合多字切割與影像清洗 ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        data = req_data['image']
        is_realtime = req_data.get('is_realtime', False)
        
        # 解碼圖片
        img_bytes = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自動偵測背景反轉
        if np.mean(gray) > 120:
            gray = 255 - gray

        # 去噪與大津二值化
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 影像清洗機制
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        cleaned_thresh = np.zeros_like(thresh)
        h_img, w_img = gray.shape
        comps = []
        
        # 過濾門檻
        MIN_AREA = 500 if is_realtime else 150 
        
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < MIN_AREA: continue
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 or aspect_ratio < 0.15: continue
            if float(area) / (w * h) < 0.15: continue
            
            # 邊緣排除
            border = 8
            if x < border or y < border or (x+w) > (w_img-border) or (y+h) > (h_img-border):
                if area < 1000: continue 

            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 按 X 座標排序
        comps.sort(key=lambda c: c[0])
        final_res = ""
        details = []
        valid_boxes = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # 連體字切割
            if w > h * 1.3:
                proj = np.sum(roi, axis=0)
                split_x = np.argmin(proj[int(w*0.3):int(w*0.7)]) + int(w*0.3)
                sub_rois = [(roi[:, :split_x], x, y, split_x, h), (roi[:, split_x:], x + split_x, y, w - split_x, h)]
                for s_roi, sx, sy, sw, sh in sub_rois:
                    if s_roi.shape[1] < 5: continue
                    p = model.predict(advanced_preprocess(s_roi), verbose=0)
                    d, c = int(np.argmax(p)), float(np.max(p))
                    if c > 0.8:
                        final_res += str(d)
                        details.append({"digit": d, "conf": f"{c*100:.1f}%"})
                        valid_boxes.append({'x': int(sx), 'y': int(sy), 'w': int(sw), 'h': int(sh)})
                continue

            # 一般數字
            pred = model.predict(advanced_preprocess(roi), verbose=0)
            digit, conf = int(np.argmax(pred)), float(np.max(pred))
            
            if is_realtime and conf < 0.85:
                continue
                
            final_res += str(digit)
            details.append({"digit": digit, "conf": f"{conf*100:.1f}%"})
            valid_boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        return jsonify({
            'full_digit': final_res, 
            'details': details, 
            'boxes': valid_boxes
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render 必備的 Port 設定
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
