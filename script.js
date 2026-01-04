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

let recognition = null, isVoiceActive = false, isDrawing = false, isEraser = false;
let cameraStream = null, realtimeInterval = null, lastX = 0, lastY = 0;

function init() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePen();
    initSpeechRecognition();
    addGalaxyEffects();
}

function addGalaxyEffects() {
    setTimeout(() => {
        if (!cameraStream) {
            ctx.fillStyle = "rgba(163, 217, 255, 0.3)";
            ctx.beginPath(); ctx.arc(650, 20, 3, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(30, 300, 2, 0, Math.PI * 2); ctx.fill();
            updatePen();
        }
    }, 500);
}

function updatePen() {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (isEraser) { ctx.strokeStyle = "black"; ctx.lineWidth = 40; } 
    else { ctx.strokeStyle = "white"; ctx.lineWidth = 15; }
}

function toggleEraser() {
    isEraser = !isEraser;
    eraserBtn.innerText = isEraser ? "橡皮擦：開啟" : "橡皮擦：關閉";
    eraserBtn.classList.toggle('eraser-active', isEraser);
    updatePen();
    if (isEraser) addVisualFeedback("#e74c3c");
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!cameraStream) { ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height); }
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
        setTimeout(() => { btn.style.boxShadow = originalBoxShadow; }, 300);
    });
}

async function toggleCamera() {
    if (cameraStream) { stopCamera(); } 
    else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: 1280, height: 720 },
                audio: false
            });
            video.srcObject = cameraStream;
            video.style.display = "block";
            mainBox.classList.add('cam-active');
            camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 關閉鏡頭';
            realtimeInterval = setInterval(() => { predictRealtime(); }, 400);
            clearCanvas();
            addVisualFeedback("#9b59b6");
        } catch (err) { alert("鏡頭啟動失敗: " + err); }
    }
}

function stopCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach(track => track.stop()); cameraStream = null; }
    if (realtimeInterval) clearInterval(realtimeInterval);
    video.style.display = "none";
    mainBox.classList.remove('cam-active');
    camToggleBtn.innerHTML = '<span class="btn-icon">📷</span> 開啟鏡頭';
    init();
    addVisualFeedback("#34495e");
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', stopDrawing);

function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
}

function startDrawing(e) {
    isDrawing = true; const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath(); ctx.moveTo(x, y);
    if (!isEraser) addDrawingEffect(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y);
    if (!isEraser) addDrawingEffect(x, y);
}

function stopDrawing() {
    if (isDrawing) { isDrawing = false; ctx.beginPath(); if (!cameraStream) setTimeout(() => predict(), 100); }
}

function addDrawingEffect(x, y) {
    const effect = document.createElement('div');
    effect.style.cssText = `position:fixed;left:${x-5}px;top:${y-5}px;width:10px;height:10px;border-radius:50%;background:radial-gradient(circle,rgba(163,217,255,0.8) 0%,transparent 70%);pointer-events:none;z-index:1000;`;
    document.body.appendChild(effect);
    setTimeout(() => effect.remove(), 500);
}

async function predictRealtime() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: tempCanvas.toDataURL('image/png'), is_realtime: true })
        });
        const data = await res.json();
        if (cameraStream) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (data.boxes) {
                data.boxes.forEach((box, i) => {
                    ctx.strokeStyle = "#00FF00"; ctx.lineWidth = 3; ctx.strokeRect(box.x, box.y, box.w, box.h);
                    ctx.fillStyle = "#00FF00"; ctx.font = "bold 24px Arial";
                    ctx.fillText(data.details[i].digit, box.x, box.y - 5);
                });
            }
            updatePen();
            digitDisplay.innerText = data.full_digit || "---";
            updateDetails(data);
        }
    } catch (err) { console.log("即時辨識同步中..."); }
}

async function predict() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');
    if (cameraStream) tCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    tCtx.drawImage(canvas, 0, 0);

    try {
        digitDisplay.innerHTML = '<span class="pulse-icon">🌠</span>';
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: tempCanvas.toDataURL('image/png'), is_realtime: false })
        });
        const data = await res.json();
        if (data.full_digit) {
            addVisualFeedback("#2ecc71");
            digitDisplay.innerText = data.full_digit;
            digitDisplay.style.transform = "scale(1.2)";
            setTimeout(() => { digitDisplay.style.transform = "scale(1)"; }, 300);
        } else { digitDisplay.innerText = "---"; }
        updateDetails(data);
    } catch (err) { digitDisplay.innerText = "❌"; addVisualFeedback("#e74c3c"); }
}

function updateDetails(data) {
    let html = "<b>詳細辨識資訊：</b><br>";
    if (!data.details || data.details.length === 0) { html += "等待有效數字入鏡..."; } 
    else {
        data.details.forEach((item, i) => {
            const color = i % 2 === 0 ? "#a3d9ff" : "#ff6b9d";
            html += `數字 ${i + 1}: <b style="color:${color}">${item.digit}</b> (信心度: ${item.conf})<br>`;
        });
    }
    confDetails.innerHTML = html;
}

function triggerFile() { fileInput.click(); addVisualFeedback("#3498db"); }
function handleFile(event) {
    const file = event.target.files[0]; if (!file) return;
    if (cameraStream) stopCamera();
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            ctx.fillStyle = "black"; ctx.fillRect(0, 0, canvas.width, canvas.height);
            const ratio = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.8;
            const w = img.width * ratio, h = img.height * ratio;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            predict(); addVisualFeedback("#3498db");
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { voiceBtn.style.display = 'none'; return; }
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-TW'; recognition.continuous = true;
    recognition.onstart = () => { isVoiceActive = true; updateVoiceButton(); voiceStatus.style.display = 'block'; addVisualFeedback("#ff6b9d"); };
    recognition.onend = () => { if (isVoiceActive) recognition.start(); else voiceStatus.style.display = 'none'; };
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript.includes('清除') || transcript.includes('清空')) clearCanvas();
        else if (transcript.includes('開始') || transcript.includes('辨識')) predict();
        else if (transcript.includes('鏡頭') || transcript.includes('相機')) toggleCamera();
        else if (transcript.includes('橡皮擦')) toggleEraser();
        else {
            digitDisplay.innerText = transcript;
            confDetails.innerHTML = `<b>語音來源：</b><span style="color:#ff6b9d">${transcript}</span>`;
            addVisualFeedback("#ff6b9d");
        }
    };
}

function toggleVoice() {
    if (isVoiceActive) { isVoiceActive = false; recognition.stop(); updateVoiceButton(); } 
    else {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            stream.getTracks().forEach(t => t.stop());
            recognition.start(); updateVoiceButton();
        }).catch(() => alert("請開啟麥克風權限"));
    }
}

function updateVoiceButton() {
    voiceBtn.innerHTML = isVoiceActive ? '<span class="btn-icon">🌌</span> 語音輸入：開啟' : '<span class="btn-icon">🌌</span> 語音輸入：關閉';
    voiceBtn.classList.toggle('voice-active', isVoiceActive);
}

init();
