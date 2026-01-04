import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# 修改：設定 static_folder 為目前目錄，以便讀取 HTML 檔案
app = Flask(__name__, static_folder='.')
# 允許來自任何網域的請求 (CORS)，這樣你的前端 HTML 才能呼叫它
CORS(app)

# 載入模型 - 請確保檔案就在 app.py 旁邊
MODEL_PATH = 'mnist_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# 新增方案 B 路由：讓 Render 網址直接顯示網頁介面
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def advanced_preprocess(roi):
    """
    保留原始 p.py 的高階預處理邏輯
    """
    # 強化筆畫：稍微膨脹，確保 7 的橫線與 1 有明顯區別
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)
    
    h, w = roi.shape
    # 動態 Padding，保持數字比例
    pad = int(max(h, w) * 0.45)
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 縮放至 28x28
    roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 質心校正，讓辨識更穩定
    M = cv2.moments(roi_rs)
    if M["m00"] != 0:
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        T = np.float32([[1, 0, 14-cx], [0, 1, 14-cy]])
        roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
        
    return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        data = req_data['image']
        is_realtime = req_data.get('is_realtime', False)
        
        # 解碼圖片 (Base64 -> OpenCV Image)
        img_bytes = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 背景反轉檢測
        if np.mean(gray) > 120:
            gray = 255 - gray

        # 去噪與二值化
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 影像清洗機制：偵測所有連通區域
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        cleaned_thresh = np.zeros_like(thresh)
        h_img, w_img = gray.shape
        comps = []
        valid_boxes = []

        # 設定過濾門檻
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

            # 4. 邊緣無效區過濾
            border = 8
            if x < border or y < border or (x+w) > (w_img-border) or (y+h) > (h_img-border):
                if area < 1000: continue 

            # 通過檢查，畫回清洗後的底圖
            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 按 X 座標排序
        comps.sort(key=lambda c: c[0])
        final_res = ""
        details = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # 連體字切割邏輯
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

            # 一般數字預測
            pred = model.predict(advanced_preprocess(roi), verbose=0)
            digit, conf = int(np.argmax(pred)), float(np.max(pred))
            
            # 信心度門檻
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
        return jsonify({'error': str(e)}), 500

# 這是 Render 部署最重要的部分：監聽環境變數提供的 Port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
