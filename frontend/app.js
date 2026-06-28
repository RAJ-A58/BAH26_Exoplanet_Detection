const state = {
  mode: "synthetic",
  busy: false,
};

const form = document.querySelector("#predictForm");
const modeTabs = document.querySelectorAll(".mode-tab");
const panes = document.querySelectorAll(".mode-pane");
const modelStatus = document.querySelector("#modelStatus");
const verdict = document.querySelector("#verdict");
const confidenceFill = document.querySelector("#confidenceFill");
const confidenceText = document.querySelector("#confidenceText");
const metrics = document.querySelector("#metrics");
const huntList = document.querySelector("#huntList");
const runButton = document.querySelector(".run-button");

function setMode(mode) {
  state.mode = mode;
  modeTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  panes.forEach((pane) => pane.classList.toggle("hidden", pane.dataset.pane !== mode));
  const isSynthetic = mode === "synthetic";
  document.querySelector("#periodSource").disabled = isSynthetic;
  document.querySelector("#maxIterations").disabled = isSynthetic;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

async function buildPayload() {
  const payload = {
    mode: state.mode,
    periodSource: document.querySelector("#periodSource").value,
    maxIterations: Number(document.querySelector("#maxIterations").value),
    period: Number(document.querySelector("#period").value),
    t0: Number(document.querySelector("#t0").value),
  };

  if (state.mode === "target") {
    payload.target = document.querySelector("#target").value.trim() || "Kepler-10";
  }

  if (state.mode === "csv") {
    const file = document.querySelector("#csvFile").files[0];
    if (!file) {
      throw new Error("Choose a CSV file first.");
    }
    payload.fileName = file.name;
    payload.csvText = await readFile(file);
  }

  return payload;
}

function drawChart(canvasId, points, color) {
  const canvas = document.querySelector(canvasId);
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const background = ctx.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#050913");
  background.addColorStop(0.52, "#02050a");
  background.addColorStop(1, "#10151d");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(0, 231, 255, 0.11)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 8; i += 1) {
    const y = (height / 8) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < 12; i += 1) {
    const x = (width / 12) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  if (!points || points.length < 2) {
    ctx.fillStyle = "rgba(217, 255, 63, 0.16)";
    for (let i = 0; i < 70; i += 1) {
      const x = (i * 37) % width;
      const y = (i * 53) % height;
      ctx.fillRect(x, y, 2, 2);
    }
    return;
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(0.000001, max - min);
  const pad = 24;
  const coordinates = points.map((point, index) => {
    const x = pad + (index / (points.length - 1)) * (width - pad * 2);
    const y = pad + (1 - (point - min) / span) * (height - pad * 2);
    return [x, y];
  });

  const fill = ctx.createLinearGradient(0, pad, 0, height - pad);
  fill.addColorStop(0, color.replace(")", ", 0.34)").replace("rgb", "rgba"));
  fill.addColorStop(1, "rgba(255, 47, 179, 0.03)");
  ctx.beginPath();
  coordinates.forEach(([x, y], index) => {
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.lineTo(width - pad, height - pad);
  ctx.lineTo(pad, height - pad);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  coordinates.forEach(([x, y], index) => {
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.strokeStyle = "rgba(217, 255, 63, 0.62)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(width / 2, pad);
  ctx.lineTo(width / 2, height - pad);
  ctx.stroke();

  const centerWindow = coordinates.slice(
    Math.max(0, Math.floor(coordinates.length / 2) - 4),
    Math.min(coordinates.length, Math.floor(coordinates.length / 2) + 5),
  );
  ctx.fillStyle = "#d9ff3f";
  centerWindow.forEach(([x, y]) => {
    ctx.fillRect(x - 2, y - 2, 4, 4);
  });
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(6) : "-";
}

function renderResult(result) {
  const iterations = result.iterations || [];
  const firstDetection = iterations.find((item) => item.predictedClass === 1) || iterations[0];
  if (!firstDetection) {
    return;
  }

  const confidence = firstDetection.prediction || 0;
  const detected = firstDetection.predictedClass === 1;
  verdict.textContent = detected ? "Planet detected" : "No planet detected";
  verdict.style.color = detected ? "var(--lime)" : "var(--danger)";
  confidenceFill.style.height = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  confidenceText.textContent = formatPercent(confidence);

  metrics.innerHTML = `
    <div><span>Source</span><strong>${result.source || "-"}</strong></div>
    <div><span>Period</span><strong>${formatNumber(firstDetection.period)}</strong></div>
    <div><span>t0</span><strong>${formatNumber(firstDetection.t0)}</strong></div>
  `;

  drawChart("#globalChart", firstDetection.globalView, "rgb(217, 255, 63)");
  drawChart("#localChart", firstDetection.localView, "rgb(0, 231, 255)");

  huntList.innerHTML = iterations
    .map((item) => {
      const label = item.predictedClass ? "Detected" : "Rejected";
      const truth = item.knownLabel === null || item.knownLabel === undefined ? "" : `Known label: ${item.knownLabel}`;
      return `
        <div class="hunt-item">
          <strong>#${item.planetIndex}</strong>
          <span>${label}<br><small>Period ${formatNumber(item.period)} ${truth}</small></span>
          <strong>${formatPercent(item.prediction || 0)}</strong>
        </div>
      `;
    })
    .join("");
}

async function runDetection(event) {
  event.preventDefault();
  if (state.busy) {
    return;
  }

  state.busy = true;
  runButton.disabled = true;
  document.body.classList.add("is-running");
  modelStatus.textContent = "Running model";
  verdict.textContent = "Reading light curve";

  try {
    const payload = await buildPayload();
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    renderResult(data.result);
    modelStatus.textContent = "Complete";
  } catch (error) {
    verdict.textContent = "Run failed";
    verdict.style.color = "var(--danger)";
    modelStatus.textContent = error.message;
  } finally {
    state.busy = false;
    runButton.disabled = false;
    document.body.classList.remove("is-running");
  }
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});
form.addEventListener("submit", runDetection);
setMode("synthetic");
drawChart("#globalChart", [], "rgb(217, 255, 63)");
drawChart("#localChart", [], "rgb(0, 231, 255)");
window.addEventListener("resize", () => {
  drawChart("#globalChart", [], "rgb(217, 255, 63)");
  drawChart("#localChart", [], "rgb(0, 231, 255)");
});
