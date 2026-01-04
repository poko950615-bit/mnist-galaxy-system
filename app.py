import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# --- 1. 伺服器核心設定 (修正 Render 502/404 報錯) ---
# 設定 static_folder 為目前目錄，讓 index.html 可以被正確讀取
app = Flask(__name__, static_folder='.', static_url_path='')
# 啟用 CORS 確保前端 JavaScript 能與後端 API 通訊
CORS(app)

# 載入模型 - 確保 mnist_model.h5 檔案位於 app.py 同層級
MODEL_PATH = 'mnist_model.h5'
try:
    # 使用 tf.keras 載入模型
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MNIST 辨識模型已成功載入！")
except Exception as e:
    print(f"模型載入嚴重錯誤: {str(e)}")

# --- 2. 路由設定：網站首頁 ---
@app.route('/')
def index():
    # 當訪問根網址時，回傳 index.html
    return send_from_directory(app.static_folder, 'index.html')

# --- 3. 完整移植：影像預處理核心 (質心校正、筆畫強化) ---
def advanced_preprocess(roi):
    """
    保留原始最強辨識邏輯：質心位移校正、動態邊距填充
    """
    # 強化筆畫：稍微膨脹，避免筆劃斷裂
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)
    
    h, w = roi.shape
    # 動態邊距填充 (Padding)，保持數字比例不變形
    pad = int(max(h, w) * 0.45)
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 縮放至 MNIST 標準 28x28
    roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 質心校正 (Moments)：將數字重心移到正中央，提升精準度
    M = cv2.moments(roi_rs)
    if M["m00"] != 0:
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        T = np.float32([[1, 0, 14-cx], [0, 1, 14-cy]])
        roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
        
    # 正規化處理
    return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0

# --- 4. 辨識主路由：影像清洗、多字切割辨識 ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 接收前端 Base64 圖片
        req_data = request.json
        data = req_data.get('image')
        if not data:
            return jsonify({'error': 'No image data'}), 400
            
        is_realtime = req_data.get('is_realtime', False)
        
        # 解碼圖片
        img_bytes = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 轉灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自動背景檢測與反轉 (確保是黑底白字進入辨識)
        if np.mean(gray) > 120:
            gray = 255 - gray

        # 去噪與二值化
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 影像清洗：使用連通區域過濾雜訊
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        cleaned_thresh = np.zeros_like(thresh)
        h_img, w_img = gray.shape
        comps = []
        valid_boxes = []

        # 過濾門檻
        MIN_AREA = 500 if is_realtime else 150 
        
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            
            # 過濾太小的雜點
            if area < MIN_AREA: continue
            
            # 檢查比例與填滿率，排除非數字形狀
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 or aspect_ratio < 0.15: continue
            if float(area) / (w * h) < 0.15: continue
            
            # 邊緣排除
            border = 8
            if x < border or y < border or (x+w) > (w_img-border) or (y+h) > (h_img-border):
                if area < 1000: continue 

            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 按水平位置排序 (從左到右辨識)
        comps.sort(key=lambda c: c[0])
        final_res = ""
        details = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # --- 關鍵連體字切割邏輯 ---
            if w > h * 1.3:
                # 投影切割法
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

            # 一般情況：直接辨識
            pred = model.predict(advanced_preprocess(roi), verbose=0)
            digit, conf = int(np.argmax(pred)), float(np.max(pred))
            
            # 信心度過濾 (即時模式較嚴格)
            if is_realtime and conf < 0.85:
                continue
                
            final_res += str(digit)
            details.append({"digit": digit, "conf": f"{conf*100:.1f}%"})
            valid_boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        # 回傳預測結果
        return jsonify({
            'full_digit': final_res, 
            'details': details, 
            'boxes': valid_boxes
        })
    except Exception as e:
        # 將錯誤印出到 Render Logs 以供追蹤
        print(f"辨識處理出錯: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- 5. 啟動設定 ---
if __name__ == '__main__':
    # 監聽環境變數 Port (Render 必備)
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
