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

function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();

    // 添加銀河主題初始化效果
    addGalaxyEffects();
}

function addGalaxyEffects() {
    // 可以在畫布上添加一些初始效果（不影響辨識）
    setTimeout(() => {
        if (!cameraStream) {
            // 在畫布角落添加一個小星星效果
            ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
            ctx.beginPath();
            ctx.arc(650, 20, 3, 0, Math.PI * 2);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(30, 300, 2, 0, Math.PI * 2);
            ctx.fill();

            // 恢復畫筆設置
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

    // 添加銀河主題反饋
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

    // 添加視覺效果
    addVisualFeedback("#2ecc71");
    addGalaxyEffects();
}

function addVisualFeedback(color) {
    // 在按鈕上添加視覺反饋
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
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';

            realtimeInterval = setInterval(() => {
                predictRealtime();
            }, 400);

            clearCanvas();

            // 添加視覺效果
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

    // 添加視覺效果
    addVisualFeedback("#34495e");
}

// 繪畫事件處理
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控事件支援
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
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
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCanvasCoordinates(e);

    // 開始新的路徑
    ctx.beginPath();
    ctx.moveTo(x, y);

    lastX = x;
    lastY = y;

    // 添加繪畫視覺效果
    if (!isEraser) {
        addDrawingEffect(x, y);
    }
}

function draw(e) {
    e.preventDefault();

    if (!isDrawing) return;

    const { x, y } = getCanvasCoordinates(e);

    // 繪製線條
    ctx.lineTo(x, y);
    ctx.stroke();

    // 開始新的路徑，從當前點開始
    ctx.beginPath();
    ctx.moveTo(x, y);

    lastX = x;
    lastY = y;

    // 添加移動視覺效果
    if (!isEraser) {
        addDrawingEffect(x, y);
    }
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        if (!cameraStream) {
            setTimeout(() => predict(), 100); // 小延遲後進行辨識
        }
    }
}

function handleTouchStart(e) {
    if (e.touches.length === 1) {
        startDrawing(e);
    }
}

function handleTouchMove(e) {
    if (e.touches.length === 1) {
        draw(e);
    }
}

function addDrawingEffect(x, y) {
    // 在繪畫時添加視覺效果
    const effect = document.createElement('div');
    effect.style.position = 'fixed';
    effect.style.left = (x - 5) + 'px';
    effect.style.top = (y - 5) + 'px';
    effect.style.width = '10px';
    effect.style.height = '10px';
    effect.style.borderRadius = '50%';
    effect.style.background = 'radial-gradient(circle, rgba(163, 217, 255, 0.8) 0%, transparent 70%)';
    effect.style.pointerEvents = 'none';
    effect.style.zIndex = '1000';
    document.body.appendChild(effect);

    setTimeout(() => {
        effect.remove();
    }, 500);
}

// 修改後的即時辨識函式：在框框上顯示數字
async function predictRealtime() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    const dataUrl = tempCanvas.toDataURL('image/png');

    try {
        // 修改這裡：移除 localhost:5000，使用相對路徑
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl, is_realtime: true })
        });
        const data = await res.json();

        if (cameraStream) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (data.boxes && data.details) {
                data.boxes.forEach((box, index) => {
                    // 1. 畫綠色框框
                    ctx.strokeStyle = "#00FF00";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(box.x, box.y, box.w, box.h);

                    // 2. 畫辨識到的數字文字 (新增功能)
                    const detectedDigit = data.details[index] ? data.details[index].digit : "";
                    ctx.fillStyle = "#00FF00";
                    ctx.font = "bold 24px Arial";
                    // 將文字寫在框框左上角上方
                    ctx.fillText(detectedDigit, box.x, box.y - 5);
                });
            }
            updatePen();
            digitDisplay.innerText = data.full_digit || "---";
            updateDetails(data);
        }
    } catch (err) {
        console.log("即時辨識同步中...");
    }
}

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

            // 添加視覺效果
            addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function predict() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    if (cameraStream) tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    try {
        // 添加載入效果
        digitDisplay.innerHTML = '<span class="pulse-icon">🌠</span>';

        // 修改這裡：移除 localhost:5000，使用相對路徑
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: tempCanvas.toDataURL('image/png'), is_realtime: false })
        });
        const data = await res.json();

        // 添加辨識成功效果
        if (data.full_digit && data.full_digit !== "") {
            addVisualFeedback("#2ecc71");
            digitDisplay.innerText = data.full_digit;

            // 添加數字出現動畫
            digitDisplay.style.transform = "scale(1.2)";
            setTimeout(() => {
                digitDisplay.style.transform = "scale(1)";
            }, 300);
        } else {
            digitDisplay.innerText = "---";
        }

        updateDetails(data);
    } catch (err) {
        digitDisplay.innerText = "❌";
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

        // 添加視覺效果
        addVisualFeedback("#ff6b9d");
    };

    recognition.onend = () => {
        if (isVoiceActive) {
            try {
                recognition.start();
            } catch (e) {
                console.log("語音識別重啟失敗:", e);
                // 如果重啟失敗，將語音狀態設為關閉
                isVoiceActive = false;
                updateVoiceButton();
                voiceStatus.style.display = 'none';
            }
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

            // 添加視覺效果
            addVisualFeedback("#ff6b9d");
        }
    };

    recognition.onerror = (event) => {
        console.log("語音識別錯誤:", event.error);
        if (event.error === 'not-allowed' || event.error === 'audio-capture') {
            alert("請允許瀏覽器使用麥克風權限");
            isVoiceActive = false;
            updateVoiceButton();
            voiceStatus.style.display = 'none';
        }
    };
}

// 更新語音按鈕狀態
function updateVoiceButton() {
    if (isVoiceActive) {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：開啟';
        voiceBtn.classList.add('voice-active');
    } else {
        voiceBtn.innerHTML = '<span class="btn-icon">🌌</span> 語音輸入：關閉';
        voiceBtn.classList.remove('voice-active');
    }
}

function toggleVoice() {
    if (!recognition) {
        alert("您的瀏覽器不支援語音識別功能");
        return;
    }

    if (isVoiceActive) {
        isVoiceActive = false;
        recognition.stop();
        updateVoiceButton();
        voiceStatus.style.display = 'none';
        addVisualFeedback("#34495e");
    } else {
        try {
            // 請求麥克風權限
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    // 停止音訊串流以避免佔用麥克風
                    stream.getTracks().forEach(track => track.stop());

                    // 啟動語音識別
                    recognition.start();
                    updateVoiceButton();
                    addVisualFeedback("#ff6b9d");
                })
                .catch(err => {
                    console.log("麥克風權限錯誤:", err);
                    alert("請允許使用麥克風以啟用語音輸入功能");
                });
        } catch (e) {
            console.log("語音識別啟動錯誤:", e);
            alert("無法啟動語音識別功能");
        }
    }
}

// 初始化
init();
