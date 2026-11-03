let dataset = null;
let session = null;

const $ = (id) => document.getElementById(id);

async function init() {
  try {
    const response = await fetch("demo-data.json");
    dataset = await response.json();
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    session = await ort.InferenceSession.create("https://vercel-demo-gqf7mx8p9-chidvigamer-5480s-projects.vercel.app/swallow_binary.onnx", { executionProviders: ["wasm"] });
    $("modelStatus").textContent = "Model ready";
    populateMetrics();
    setSample(0);
  } catch (error) {
    console.error(error);
    $("modelStatus").textContent = "Load failed";
    $("decisionBanner").textContent = "Model could not load";
  }
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function normalize(sample) {
  return sample.map((channel) => {
    let peak = 0;
    for (const value of channel) peak = Math.max(peak, Math.abs(value));
    const denom = Math.max(peak, 1e-8);
    return channel.map((value) => value / denom);
  });
}

async function predict(sample) {
  const normalized = normalize(sample);
  const data = Float32Array.from(normalized.flat());
  const input = new ort.Tensor("float32", data, [1, 4, 3200]);
  const result = await session.run({ audio: input });
  const probability = sigmoid(Number(result.logit.data[0]));
  return { probability, className: probability >= 0.5 ? "Swallow" : "NonSwallow" };
}

function setSample(index) {
  const row = dataset.samples.find((item) => item.index === Number(index));
  if (!row) return;
  document.querySelectorAll(".sample-btn").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.index) === row.index);
  });
  $("subject").textContent = row.subject;
  $("stem").textContent = row.stem;
  $("truth").textContent = row.label.replace("_", "-");
  $("row").textContent = row.index;
  $("rawSource").textContent = row.raw_exists ? "Verified in source folder" : "Not found";
  drawWaveform(row.samples);
  runPrediction(row);
}

async function runPrediction(row) {
  $("decisionBanner").textContent = "Running inference…";
  $("decisionBanner").className = "decision-banner";
  $("match").textContent = "Comparing with annotation…";
  $("match").className = "match";
  const result = await predict(row.samples);
  const probability = result.probability;
  const prediction = result.className;
  $("prediction").textContent = prediction;
  $("probability").textContent = probability.toFixed(3);
  $("groundTruth").textContent = row.label.replace("_", "-");
  if (prediction === "Swallow") {
    $("decisionBanner").textContent = "SWALLOW DETECTED";
    $("decisionBanner").className = "decision-banner swallow";
  } else {
    $("decisionBanner").textContent = "NON-SWALLOW DETECTED";
    $("decisionBanner").className = "decision-banner non";
  }
  const truth = row.label === "swallow" ? "Swallow" : "NonSwallow";
  const matches = prediction === truth;
  $("match").textContent = matches ? "✓ Prediction matches the dataset annotation" : "✗ Prediction does not match the dataset annotation";
  $("match").className = matches ? "match ok" : "match bad";
}

function drawWaveform(sample) {
  const canvas = $("waveform");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 6; i++) {
    const y = (height / 6) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  const colors = ["#54d6ff", "#65e6ad", "#b5a8ff", "#ff9f6e"];
  const step = Math.max(1, Math.floor(sample[0].length / width));
  sample.forEach((channel, channelIndex) => {
    ctx.strokeStyle = colors[channelIndex];
    ctx.globalAlpha = 0.72;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const idx = Math.min(channel.length - 1, x * step);
      const y = height / 2 - channel[idx] * (height * 0.34);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function populateMetrics() {
  const m = dataset.metrics;
  $("accuracy").textContent = `${(m.accuracy * 100).toFixed(2)}%`;
  $("recall").textContent = `${(m.recall_swallow * 100).toFixed(2)}%`;
  $("specificity").textContent = `${(m.specificity_non_swallow * 100).toFixed(2)}%`;
  $("validationN").textContent = Number(m.n_validation).toLocaleString();
}

document.querySelectorAll(".sample-btn").forEach((button) => {
  button.addEventListener("click", () => setSample(button.dataset.index));
});

window.addEventListener("resize", () => {
  if (!dataset) return;
  const active = document.querySelector(".sample-btn.active");
  if (!active) return;
  const row = dataset.samples.find((item) => item.index === Number(active.dataset.index));
  if (row) drawWaveform(row.samples);
});

init();
