/**
 * 銀河主題手寫數字辨識系統 - 完整前端邏輯
 * 包含：繪圖、相機即時辨識、語音控制、雲端 API 串接
 */

// --- 關鍵配置：改為相對路徑以確保在 Render 部署環境穩定通訊 ---
const RENDER_URL = "/predict"; 

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('camera-feed');
const mainBox = document.getElementById('mainBox');
const camToggleBtn = document.getElementById('camToggleBtn');
const eraserBtn = document.getElementById('eraserBtn');
const fileInput = document.getElementById('fileInput');
const digitDisplay = document.getElementById('digit-display');
const confDetails = document.getElementById('conf-details');

const voiceBtn = document.getElementById('voiceBtn');
const voiceStatus = document.getElementById('voice-status');
let recognition = null;
let isVoiceActive = false;

let isDrawing = false;
let isEraser = false;
let cameraStream = null;
let realtimeInterval = null;
let lastX = 0;
let lastY = 0;

// 初始化系統
function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();

    // 添加銀河主題初始化效果
    addGalaxyEffects();
}

function addGalaxyEffects() {
    setTimeout(() => {
        if (!cameraStream) {
            // 在畫布角落添加一個小星星效果 (裝飾用)
            ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
            ctx.beginPath();
            ctx.arc(650, 20, 3, 0, Math.PI * 2);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(30, 300, 2, 0, Math.PI * 2);
            ctx.fill();

            updatePen();
        }
    }, 500);
}

function updatePen() {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (isEraser) {
        ctx.strokeStyle = "black";
        ctx.lineWidth = 40;
    } else {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 15;
    }
}

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    eraserBtn.classList.toggle('eraser-active', isEraser);
    updatePen();

    if (isEraser) {
        addVisualFeedback("#e74c3c");
    }
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    digitDisplay.innerText = "---";
    confDetails.innerText = "畫布已清空，銀河已淨空";

    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

function addVisualFeedback(color) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        const originalBoxShadow = btn.style.boxShadow;
        btn.style.boxShadow = `0 0 20px ${color}`;

        setTimeout(() => {
            btn.style.boxShadow = originalBoxShadow;
        }, 300);
    });
}

async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: 1280, height: 720 },
                audio: false
            });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📸</span> 關閉鏡頭';

            realtimeInterval = setInterval(() => {
                predictRealtime();
            }, 400);

            clearCanvas();
            addVisualFeedback("#9b59b6");
        } catch (err) {
            alert("鏡頭啟動失敗: " + err);
        }
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (realtimeInterval) clearInterval(realtimeInterval);
    video.style.display = "none";
    mainBox.classList.remove('cam-active');
    camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    init();
    addVisualFeedback("#34495e");
}

// 繪畫事件處理
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控支援
canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) startDrawing(e);
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    if (e.touches.length === 1) draw(e);
}, { passive: false });

canvas.addEventListener('touchend', stopDrawing);

function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    let x, y;
    if (e.type.includes('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    return { x, y };
}

function startDrawing(e) {
    if (e.cancelable) e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    lastX = x;
    lastY = y;
    if (!isEraser) addDrawingEffect(x, y);
}

function draw(e) {
    if (e.cancelable) e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    lastX = x;
    lastY = y;
    if (!isEraser) addDrawingEffect(x, y);
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) {
            setTimeout(() => predict(), 100); 
        }
    }
}

function addDrawingEffect(x, y) {
    const effect = document.createElement('div');
    effect.style.position = 'fixed';
    const rect = canvas.getBoundingClientRect();
    effect.style.left = (rect.left + x - 5) + 'px';
    effect.style.top = (rect.top + y - 5) + 'px';
    effect.style.width = '10px';
    effect.style.height = '10px';
    effect.style.borderRadius = '50%';
    effect.style.background = 'radial-gradient(circle, rgba(163, 217, 255, 0.8) 0%, transparent 70%)';
    effect.style.pointerEvents = 'none';
    effect.style.zIndex = '1000';
    document.body.appendChild(effect);
    setTimeout(() => effect.remove(), 500);
}

// --- API 串接區：即時辨識 ---
async function predictRealtime() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    const dataUrl = tempCanvas.toDataURL('image/png');

    try {
        const res = await fetch(RENDER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl, is_realtime: true })
        });
        const data = await res.json();

        if (cameraStream) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (data.boxes && data.details) {
                data.boxes.forEach((box, index) => {
                    ctx.strokeStyle = "#00FF00";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box.x, box.y, box.w, box.h);
                    const detectedDigit = data.details[index] ? data.details[index].digit : "";
                    ctx.fillStyle = "#00FF00";
                    ctx.font = "bold 24px Arial";
                    ctx.fillText(detectedDigit, box.x, box.y - 5);
                });
            }
            updatePen();
            digitDisplay.innerText = data.full_digit || "---";
            updateDetails(data);
        }
    } catch (err) {
        console.log("正在連接雲端銀河伺服器...");
    }
}

// --- API 串接區：手寫/上傳辨識 ---
async function predict() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    if (cameraStream) tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    try {
        digitDisplay.innerHTML = '<span class="pulse-icon">🌀</span>';
        const res = await fetch(RENDER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: tempCanvas.toDataURL('image/png'), is_realtime: false })
        });
        const data = await res.json();

        if (data.full_digit && data.full_digit !== "") {
            addVisualFeedback("#2ecc71");
            digitDisplay.innerText = data.full_digit;
            digitDisplay.style.transform = "scale(1.2)";
            setTimeout(() => { digitDisplay.style.transform = "scale(1)"; }, 300);
        } else {
            digitDisplay.innerText = "---";
        }
        updateDetails(data);
    } catch (err) {
        digitDisplay.innerText = "?";
        addVisualFeedback("#e74c3c");
    }
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data.details || data.details.length === 0) {
        html += "等待有效數字入鏡...";
    } else {
        data.details.forEach((item, i) => {
            const color = i % 2 === 0 ? "#a3d9ff" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

// 上傳功能
function triggerFile() {
    fileInput.click();
    addVisualFeedback("#3498db");
}

function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (cameraStream) stopCamera();
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const ratio = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.8;
            const w = img.width * ratio;
            const h = img.height * ratio;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            predict();
            addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 語音識別系統
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceBtn.style.display = 'none';
        return;
    }
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW';
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isVoiceActive = true;
        updateVoiceButton();
        voiceStatus.style.display = 'block';
        addVisualFeedback("#ff6b9d");
    };

    recognition.onend = () => {
        if (isVoiceActive) {
            try { recognition.start(); } catch (e) {}
        } else {
            updateVoiceButton();
            voiceStatus.style.display = 'none';
        }
    };

    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript.includes('清除') || transcript.includes('清空')) {
            clearCanvas();
        } else if (transcript.includes('開始') || transcript.includes('辨識')) {
            predict();
        } else if (transcript.includes('鏡頭') || transcript.includes('相機')) {
            toggleCamera();
        } else if (transcript.includes('橡皮擦')) {
            toggleEraser();
        } else {
            digitDisplay.innerText = transcript;
            confDetails.innerHTML = `<b>語音來源：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        }
    };
}

function updateVoiceButton() {
    if (isVoiceActive) {
        voiceBtn.innerHTML = '<span class="btn-icon">🎙️</span> 語音輸入：開啟';
        voiceBtn.classList.add('voice-active');
    } else {
        voiceBtn.innerHTML = '<span class="btn-icon">🎙️</span> 語音輸入：關閉';
        voiceBtn.classList.remove('voice-active');
    }
}

function toggleVoice() {
    if (!recognition) return;
    if (isVoiceActive) {
        isVoiceActive = false;
        recognition.stop();
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            stream.getTracks().forEach(track => track.stop());
            recognition.start();
        }).catch(() => alert("請開啟麥克風權限"));
    }
}

// 啟動
init();
