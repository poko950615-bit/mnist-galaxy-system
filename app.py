import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64

# --- 初始化 Flask 服務 ---
# 設定 static_folder 為當前目錄，確保 index.html 能被讀取
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- 模型載入邏輯 ---
model = None
try:
    # 使用絕對路徑確保在不同環境下都能讀取模型
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("✅ [系統訊息] MNIST 模型載入成功！")
except Exception as e:
    print(f"❌ [系統錯誤] 模型載入失敗: {e}")

def advanced_preprocess(roi):
    """ 
    移植自 p.py 的進階預處理邏輯 
    包含：膨脹、補邊、中心對齊與歸一化
    """
    try:
        # 輕微膨脹強化線條
        kernel = np.ones((2, 2), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=1)
        
        # 動態補邊 (Padding)
        h, w = roi.shape
        pad = int(max(h, w) * 0.45)
        roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        # 縮放到 28x28
        roi_rs = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 質心對齊 (矩分析)
        M = cv2.moments(roi_rs)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            T = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
            roi_rs = cv2.warpAffine(roi_rs, T, (28, 28))
            
        return roi_rs.reshape(1, 28, 28, 1).astype('float32') / 255.0
    except Exception as e:
        print(f"預處理異常: {e}")
        return None

# --- 路由設定 ---

@app.route('/')
def index():
    """ 讓 Render 訪問網址時能開啟 index.html """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ 
    完整的預測邏輯，包含連體字切割與即時模式處理 
    """
    if model is None:
        return jsonify({'error': '伺服器端模型尚未就緒'}), 500
        
    try:
        # 接收前端 Base64 資料
        req_data = request.get_json()
        if not req_data or 'image' not in req_data:
            return jsonify({'error': '未收到影像資料'}), 400
            
        data = req_data['image']
        is_realtime = req_data.get('is_realtime', False)
        
        # Base64 轉 CV2 影像
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': '影像解碼失敗'}), 400

        # --- 核心影像處理邏輯 (移植自 p.py) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 背景亮度檢查：如果背景太亮則反轉影像 (確保黑底白字)
        if np.mean(gray) > 120:
            gray = 255 - gray

        # 高斯模糊與大津演算法二值化
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 連通區域分析 (Connected Components)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        cleaned_thresh = np.zeros_like(thresh)
        comps = []
        
        # 過濾雜訊與非數字區域
        min_area = 500 if is_realtime else 150 
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < min_area: continue
            if (w / h) > 2.5 or (w / h) < 0.15: continue
            cleaned_thresh[labels == i] = 255
            comps.append((x, y, w, h))
        
        # 根據 X 座標排序 (確保由左至右辨識)
        comps.sort(key=lambda c: c[0])
        
        final_res = ""
        details = []
        boxes = []

        for (x, y, w, h) in comps:
            roi = cleaned_thresh[y:y+h, x:x+w]
            
            # --- 連體字處理邏輯 ---
            if w > h * 1.3:
                # 使用投影分析進行垂直切割
                proj = np.sum(roi, axis=0)
                split_x = np.argmin(proj[int(w*0.3):int(w*0.7)]) + int(w*0.3)
                
                # 分割為左右兩個部分
                sub_parts = [
                    (roi[:, :split_x], x, y, split_x, h), 
                    (roi[:, split_x:], x + split_x, y, w - split_x, h)
                ]
                
                for s_roi, sx, sy, sw, sh in sub_parts:
                    if s_roi.shape[1] < 5: continue
                    inp = advanced_preprocess(s_roi)
                    if inp is not None:
                        p = model.predict(inp, verbose=0)
                        d, c = int(np.argmax(p)), float(np.max(p))
                        final_res += str(d)
                        details.append({"digit": d, "conf": f"{c*100:.1f}%"})
                        boxes.append({'x': int(sx), 'y': int(sy), 'w': int(sw), 'h': int(sh)})
                continue

            # --- 單一數字辨識 ---
            inp = advanced_preprocess(roi)
            if inp is not None:
                pred = model.predict(inp, verbose=0)
                digit, conf = int(np.argmax(pred)), float(np.max(pred))
                
                # 即時模式下的信心度篩選
                if is_realtime and conf < 0.8: 
                    continue
                    
                final_res += str(digit)
                details.append({"digit": digit, "conf": f"{conf*100:.1f}%"})
                boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        # 返回辨識結果、詳細信心度以及座標框
        return jsonify({
            'full_digit': final_res, 
            'details': details, 
            'boxes': boxes
        })

    except Exception as e:
        print(f"❌ 辨識過程發生錯誤: {e}")
        return jsonify({'error': str(e)}), 500

# --- 啟動 Flask 服務 ---
if __name__ == '__main__':
    # Render 會自動分配 PORT 變數
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
