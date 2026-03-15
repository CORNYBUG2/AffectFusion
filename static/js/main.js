// ===================== main.js =====================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultDiv = document.getElementById("result");
const waveform = document.getElementById("waveform");

let mediaStream, micStream, mediaRecorder, audioChunks = [];
let audioCtx, analyser, dataArray;
let isRecording = false;

function getSupportedMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav"
  ];
  for (const type of types) if (MediaRecorder.isTypeSupported(type)) return type;
  return "";
}

// ===== Start Media =====
async function start() {
  try {
    console.log("Requesting camera + mic access...");
    const camStream = await navigator.mediaDevices.getUserMedia({ video: true });
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    video.srcObject = camStream;
    console.log("Media access granted.");

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    drawWaveform();

    const mime = getSupportedMimeType();
    console.log("Using MIME:", mime);
    mediaRecorder = new MediaRecorder(micStream, { mimeType: mime });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => { isRecording = false; };

    console.log("Starting periodic capture loop...");
    setInterval(captureAndSend, 4000);
  } catch (err) {
    console.error("Media access error:", err);
    resultDiv.innerText = "Error accessing camera/mic";
  }
}

// ===== Waveform =====
function drawWaveform() {
  requestAnimationFrame(drawWaveform);
  if (!analyser) return;
  analyser.getByteTimeDomainData(dataArray);
  const wctx = waveform.getContext("2d");
  wctx.fillStyle = "#111";
  wctx.fillRect(0, 0, waveform.width, waveform.height);
  wctx.lineWidth = 2;
  wctx.strokeStyle = "#00ff00";
  wctx.beginPath();
  const slice = waveform.width / dataArray.length;
  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * waveform.height) / 2;
    if (i === 0) wctx.moveTo(0, y);
    else wctx.lineTo(i * slice, y);
  }
  wctx.stroke();
}

// ===== Capture + Send =====
async function captureAndSend() {
  if (isRecording || !mediaRecorder) return;
  console.log("captureAndSend triggered");
  isRecording = true;

  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/png");

    audioChunks = [];
    console.log("Starting recorder...");
    mediaRecorder.start();
    await new Promise(r => setTimeout(r, 1000));
    if (mediaRecorder.state !== "inactive") mediaRecorder.stop();

    await new Promise(resolve => (mediaRecorder.onstop = resolve));

    console.log("Recording stopped, got chunks:", audioChunks.length);
    const webmBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
    const wavBlob = await webmToWav(webmBlob);

    console.log("Sending to backend...");
    const form = new FormData();
    form.append("frame", dataURL);
    form.append("audio", wavBlob, "clip.wav");

    const resp = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: form
    });

    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();

    if (data.error) resultDiv.innerText = "Error: " + data.error;
    else resultDiv.innerText = `Predicted: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`;
    console.log("Prediction:", data);
  } catch (err) {
    console.error("Capture/send error:", err);
    resultDiv.innerText = "Request failed";
  } finally {
    isRecording = false;
  }
}

// ===== Convert WebM → WAV =====
async function webmToWav(webmBlob) {
  const arrayBuffer = await webmBlob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const wavBuffer = audioBufferToWav(audioBuffer);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function audioBufferToWav(buffer) {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const out = new ArrayBuffer(length);
  const view = new DataView(out);
  const channels = [];
  let pos = 0;
  writeUTFBytes(view, pos, "RIFF"); pos += 4;
  view.setUint32(pos, length - 8, true); pos += 4;
  writeUTFBytes(view, pos, "WAVE"); pos += 4;
  writeUTFBytes(view, pos, "fmt "); pos += 4;
  view.setUint32(pos, 16, true); pos += 4;
  view.setUint16(pos, 1, true); pos += 2;
  view.setUint16(pos, numOfChan, true); pos += 2;
  view.setUint32(pos, buffer.sampleRate, true); pos += 4;
  view.setUint32(pos, buffer.sampleRate * numOfChan * 2, true); pos += 4;
  view.setUint16(pos, numOfChan * 2, true); pos += 2;
  view.setUint16(pos, 16, true); pos += 2;
  writeUTFBytes(view, pos, "data"); pos += 4;
  view.setUint32(pos, length - pos - 4, true); pos += 4;
  for (let i = 0; i < numOfChan; i++) channels.push(buffer.getChannelData(i));
  let offset = 0;
  while (offset < buffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      let sample = Math.max(-1, Math.min(1, channels[i][offset]));
      view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      pos += 2;
    }
    offset++;
  }
  return out;
}

function writeUTFBytes(view, offset, string) {
  for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
}

// ===== Silence favicon error =====
const link = document.createElement("link");
link.rel = "icon";
link.href = "data:,";
document.head.appendChild(link);

// ===== Start =====
start();
