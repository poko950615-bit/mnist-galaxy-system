import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# --- 1. 伺服器核心設定 ---
# static_folder='.' 表示從根目錄讀取檔案
# static_url_path='' 確保 HTML 能直接抓取同層級的 style.css 與 script.js
app = Flask(__name__, static_folder='.', static_url_path='')
# 允許跨來源請求，確保前端能與後端通訊
CORS(app)

# 載入模型 - 請確保 mnist_model.h5 檔案位於 app.py 同層級
MODEL_PATH = 'mnist_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MNIST 模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {str(e)}")

# --- 2. 路由設定：確保網址打開即顯示銀河介面 ---
@app.route('/')
def index():
    # 明確回傳根目錄下的 index.html，解決 404 Not Found 問題
    return send_from_directory(app.static_folder, 'index.html')

# --- 3. 完整移植：影像預處理邏輯 (p.py 核心) ---
def advanced_preprocess(roi):
    """
    保留原始的高階預處理邏輯：包含筆畫強化、動態 Padding、質心校正
    """
    # 強化筆畫：稍微膨脹，確保辨識特徵明顯
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)
    
    h, w = roi.shape
    # 動態 Padding：保持數字比例，避免變形
    pad = int(max(h, w) * 0.45)
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 縮放至 MNIST 標準 28x28 尺寸
    roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 質心校正 (Centering)：將數字移動到畫面中央，這是提升辨識率的關鍵
    M = cv2.moments(roi_rs)
    if M["m00"] != 0:
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        T = np.float32([[1, 0, 14-cx], [0, 1, 14-cy]])
        roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
        
    return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0

# --- 4. 完整移植：辨識主邏輯 (包含清洗與連體字切割) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        data = req_data['image']
        is_realtime = req_data.get('is_realtime', False)
        
        # 解碼 Base64 圖片數據
        img_bytes = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 轉換灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自動背景反轉檢測：若背景過亮，則轉為黑底白字
        if np.mean(gray) > 120:
            gray = 255 - gray

        # 去噪與二值化處理
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 影像清洗機制：使用連通區域過濾噪音
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        cleaned_thresh = np.zeros_like(thresh)
        h_img, w_img = gray.shape
        comps = []
        valid_boxes = []

        # 設定面積過濾門檻
        MIN_AREA = 500 if is_realtime else 150 
        
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            
            # 1. 面積過小過濾
            if area < MIN_AREA: continue
            
            # 2. 長寬比檢查
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 or aspect_ratio < 0.15: continue
            
            # 3. Solidity (填滿率) 檢查
            rect_area = w * h
            if float(area) / rect_area < 0.15: continue

            # 4. 邊緣過濾
            border = 8
            if x < border or y < border or (x+w) > (w_img-border) or (y+h) > (h_img-border):
                if area < 1000: continue 

            # 通過檢查後，存入有效組件
            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 按照 X 座標排序，確保輸出數字順序正確
        comps.sort(key=lambda c: c[0])
        final_res = ""
        details = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # 連體字切割邏輯：若寬度過大，嘗試從中間切割辨識
            if w > h * 1.3:
                proj = np.sum(roi, axis=0)
                split_x = np.argmin(proj[int(w*0.3):int(w*0.7)]) + int(w*0.3)
                sub_rois = [roi[:, :split_x], roi[:, split_x:]]
                for s_roi in sub_rois:
                    if s_roi.shape[1] < 5: continue
                    p = model.predict(advanced_preprocess(s_roi), verbose=0)
                    d, c = int(np.argmax(p)), float(np.max(p))
                    if c > 0.8:
                        final_res += str(d)
                        details.append({"digit": d, "conf": f"{c*100:.1f}%"})
                continue

            # 一般數字辨識
            pred = model.predict(advanced_preprocess(roi), verbose=0)
            digit, conf = int(np.argmax(pred)), float(np.max(pred))
            
            # 信心度門檻
            if is_realtime and conf < 0.85:
                continue
                
            final_res += str(digit)
            details.append({"digit": digit, "conf": f"{conf*100:.1f}%"})
            valid_boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        # 回傳 JSON 結果
        return jsonify({
            'full_digit': final_res, 
            'details': details, 
            'boxes': valid_boxes
        })
    except Exception as e:
        # 回傳錯誤訊息，協助除錯
        return jsonify({'error': str(e)}), 500

# --- 5. 啟動設定 ---
if __name__ == '__main__':
    # 監聽 Render 環境變數提供的 Port，預設為 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
