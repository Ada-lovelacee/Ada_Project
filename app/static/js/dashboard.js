let serverChart;
let testChart;
let lastArtifactsKey = "";
let loadInFlight = false;

const statusConfig = {
  not_started: { label: "Not Started", className: "status-not-started" },
  running: { label: "Running", className: "status-running" },
  completed: { label: "Completed", className: "status-completed" },
  idle: { label: "Idle", className: "status-idle" },
  reporting: { label: "Reported", className: "status-reporting" },
  waiting: { label: "Waiting", className: "status-waiting" },
};

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${(number * 100).toFixed(2)}%`;
}

function formatFloat(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return number.toFixed(digits);
}

function formatInteger(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${Math.trunc(number)}`;
}

function getStatusInfo(status) {
  return statusConfig[status] || { label: status, className: "status-idle" };
}

function buildStatusBadge(status) {
  const statusInfo = getStatusInfo(status);
  return `<span class="status-badge ${statusInfo.className}">${statusInfo.label}</span>`;
}

function renderClients(clients) {
  const tbody = document.getElementById("clientTableBody");
  if (!clients.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center text-secondary py-4">No client result files detected yet</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = clients
    .map(
      (client) => `
        <tr>
          <td class="fw-semibold">${client.name}</td>
          <td>${buildStatusBadge(client.status)}</td>
          <td>${client.last_round ?? "--"}</td>
          <td>${formatFloat(client.last_loss)}</td>
          <td>${formatPercent(client.last_accuracy)}</td>
          <td>${formatPercent(client.last_auc)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderServerChart(rounds) {
  const ctx = document.getElementById("serverChart");
  const labels = rounds.map((item) => `Round ${item.round_no}`);

  if (!serverChart) {
    serverChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Global Accuracy",
            data: [],
            borderColor: "#2d9cdb",
            backgroundColor: "rgba(45, 156, 219, 0.16)",
            yAxisID: "y",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Global AUC",
            data: [],
            borderColor: "#1d4e89",
            backgroundColor: "rgba(29, 78, 137, 0.12)",
            yAxisID: "y",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Global Loss",
            data: [],
            borderColor: "#f2c14e",
            backgroundColor: "rgba(242, 193, 78, 0.14)",
            yAxisID: "y1",
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: {
            position: "left",
            title: { display: true, text: "Accuracy / AUC (%)" },
            grid: { color: "rgba(16, 37, 66, 0.08)" },
          },
          y1: {
            position: "right",
            title: { display: true, text: "Loss" },
            grid: { drawOnChartArea: false },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
          },
        },
      },
    });
  }

  serverChart.data.labels = labels;
  serverChart.data.datasets[0].data = rounds.map((item) => Number((item.global_accuracy * 100).toFixed(2)));
  serverChart.data.datasets[1].data = rounds.map((item) => Number((item.global_auc * 100).toFixed(2)));
  serverChart.data.datasets[2].data = rounds.map((item) => item.global_loss);
  serverChart.update("none");
}

function renderTestChart(testMetrics) {
  const ctx = document.getElementById("testChart");
  const labels = testMetrics.map((item) => item.location);

  if (!testChart) {
    testChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "Accuracy",
            data: [],
            backgroundColor: "#2d9cdb",
            borderRadius: 10,
          },
          {
            label: "AUC",
            data: [],
            backgroundColor: "#1d4e89",
            borderRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: "Percent (%)" },
            grid: { color: "rgba(16, 37, 66, 0.08)" },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
          },
        },
      },
    });
  }

  testChart.data.labels = labels;
  testChart.data.datasets[0].data = testMetrics.map((item) => Number((item.acc * 100).toFixed(2)));
  testChart.data.datasets[1].data = testMetrics.map((item) => Number((item.auc * 100).toFixed(2)));
  testChart.update("none");
}

function renderArtifacts(artifacts) {
  const gallery = document.getElementById("artifactGallery");
  const artifactsKey = artifacts
    .map((artifact) => `${artifact.filename}|${artifact.updated_at}|${artifact.size_kb}`)
    .join(";");

  if (artifactsKey === lastArtifactsKey) {
    return;
  }

  lastArtifactsKey = artifactsKey;

  if (!artifacts.length) {
    gallery.innerHTML = `
      <div class="col-12">
        <div class="artifact-empty">
          No PNG artifacts found yet. Once the training scripts generate charts, they will appear here automatically.
        </div>
      </div>
    `;
    return;
  }

  gallery.innerHTML = artifacts
    .map(
      (artifact) => `
        <div class="col-md-6 col-xl-4">
          <div class="artifact-card">
            <div class="artifact-card-body">
              <img class="artifact-image" src="/artifacts/${encodeURIComponent(artifact.filename)}" alt="${artifact.name}" loading="lazy">
            </div>
            <div class="artifact-card-footer">
              <div class="fw-semibold">${artifact.name}</div>
              <div class="text-secondary small">${artifact.updated_at} | ${artifact.size_kb} KB</div>
            </div>
          </div>
        </div>
      `,
    )
    .join("");
}

function renderOverview(data) {
  const taskStatus = getStatusInfo(data.task.status);

  document.getElementById("taskName").textContent = data.task.name;
  document.getElementById("modelNameHero").textContent = data.task.model_name;
  document.getElementById("alphaValueHero").textContent = data.task.alpha;
  document.getElementById("datasetFileHero").textContent = data.task.dataset_file;

  document.getElementById("taskStatusText").textContent = taskStatus.label;
  document.getElementById("globalAccuracy").textContent = formatPercent(data.task.global_accuracy);
  document.getElementById("globalAuc").textContent = formatPercent(data.task.global_auc);
  document.getElementById("roundProgressText").textContent = `${formatInteger(data.task.round_count)}/${formatInteger(data.task.total_rounds)}`;

  const statusBadge = document.getElementById("taskStatusBadge");
  statusBadge.className = `status-badge ${taskStatus.className}`;
  statusBadge.textContent = taskStatus.label;

  document.getElementById("modelName").textContent = data.task.model_name;
  document.getElementById("roundCount").textContent = data.task.round_count;
  document.getElementById("totalRounds").textContent = data.task.total_rounds;
  document.getElementById("globalLoss").textContent = formatFloat(data.task.global_loss);
  document.getElementById("reportedClients").textContent = `${data.summary.reported_clients}/${data.summary.total_clients}`;
  document.getElementById("datasetFile").textContent = data.task.dataset_file;
  document.getElementById("resultsDir").textContent = data.task.results_dir;

  const progressBar = document.getElementById("taskProgress");
  progressBar.style.width = `${data.task.progress}%`;
  progressBar.textContent = `${data.task.progress}%`;

  renderClients(data.clients);
  renderServerChart(data.rounds);
  renderTestChart(data.test_metrics);
  renderArtifacts(data.artifacts);
}

async function loadOverview() {
  if (loadInFlight) {
    return;
  }

  loadInFlight = true;
  try {
    const response = await fetch("/api/overview", { cache: "no-store" });
    const data = await response.json();
    renderOverview(data);
  } catch (error) {
    console.error("Failed to load dashboard overview.", error);
  } finally {
    loadInFlight = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadOverview();

  const refreshButton = document.getElementById("refreshBtn");
  if (refreshButton) {
    refreshButton.addEventListener("click", loadOverview);
  }

  const refreshMs = Number(window.dashboardRefreshMs) || 5000;
  window.setInterval(loadOverview, refreshMs);
});
