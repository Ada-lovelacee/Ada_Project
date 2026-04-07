let roundChart;
let contributionChart;

const statusClassMap = {
  online: "status-online",
  training: "status-training",
  offline: "status-offline",
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function buildStatusBadge(status) {
  const labelMap = {
    online: "在线",
    training: "训练中",
    offline: "离线",
  };
  const className = statusClassMap[status] || "status-online";
  return `<span class="status-badge ${className}">${labelMap[status] || status}</span>`;
}

function renderClients(clients) {
  const tbody = document.getElementById("clientTableBody");
  tbody.innerHTML = clients
    .map(
      (client) => `
        <tr>
          <td class="fw-semibold">${client.name}</td>
          <td>${client.location}</td>
          <td>${buildStatusBadge(client.status)}</td>
          <td>${client.data_size.toLocaleString()}</td>
          <td>${formatPercent(client.last_accuracy)}</td>
          <td>${(client.contribution_score * 100).toFixed(0)}%</td>
        </tr>
      `,
    )
    .join("");
}

function renderRoundChart(rounds) {
  const ctx = document.getElementById("roundChart");
  const labels = rounds.map((item) => `第${item.round_no}轮`);
  const accuracyData = rounds.map((item) => Number((item.global_accuracy * 100).toFixed(2)));
  const lossData = rounds.map((item) => item.global_loss);

  if (roundChart) {
    roundChart.destroy();
  }

  roundChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "全局准确率 (%)",
          data: accuracyData,
          borderColor: "#2d9cdb",
          backgroundColor: "rgba(45, 156, 219, 0.16)",
          yAxisID: "y",
          tension: 0.3,
          fill: true,
        },
        {
          label: "全局损失",
          data: lossData,
          borderColor: "#f2c14e",
          backgroundColor: "rgba(242, 193, 78, 0.16)",
          yAxisID: "y1",
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          position: "left",
          grid: { color: "rgba(16, 37, 66, 0.08)" },
        },
        y1: {
          position: "right",
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

function renderContributionChart(clients) {
  const ctx = document.getElementById("contributionChart");
  if (contributionChart) {
    contributionChart.destroy();
  }

  contributionChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: clients.map((item) => item.name),
      datasets: [
        {
          data: clients.map((item) => item.contribution_score),
          backgroundColor: ["#2d9cdb", "#1d4e89", "#f2c14e", "#4d6c8e"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function renderOverview(data) {
  document.getElementById("taskName").textContent = data.task.name;
  document.getElementById("globalAccuracy").textContent = formatPercent(data.task.global_accuracy);
  document.getElementById("onlineClients").textContent = `${data.summary.online_clients}/${data.summary.total_clients}`;
  document.getElementById("avgLatency").textContent = data.summary.avg_latency;
  document.getElementById("taskStatus").textContent = data.task.status;
  document.getElementById("modelName").textContent = data.task.model_name;
  document.getElementById("roundCount").textContent = data.summary.round_count;
  document.getElementById("globalLoss").textContent = data.task.global_loss.toFixed(3);

  const progressBar = document.getElementById("taskProgress");
  progressBar.style.width = `${data.task.progress}%`;
  progressBar.textContent = `${data.task.progress}%`;

  renderClients(data.clients);
  renderRoundChart(data.rounds);
  renderContributionChart(data.clients);
}

async function loadOverview() {
  const response = await fetch("/api/overview");
  const data = await response.json();
  renderOverview(data);
}

async function simulateRound() {
  const button = document.getElementById("simulateBtn");
  button.disabled = true;
  button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>正在生成';

  try {
    await fetch("/api/simulate", { method: "POST" });
    await loadOverview();
  } finally {
    button.disabled = false;
    button.innerHTML = '<i class="bi bi-play-circle me-2"></i>模拟下一轮训练';
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadOverview();

  const button = document.getElementById("simulateBtn");
  if (button) {
    button.addEventListener("click", simulateRound);
  }
});
