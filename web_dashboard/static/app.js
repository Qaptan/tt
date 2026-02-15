const ui = {
  healthPill: document.getElementById("health-pill"),
  accountSelect: document.getElementById("account-select"),
  scheduleCycles: document.getElementById("schedule-cycles"),
  uptimeValue: document.getElementById("uptime-value"),
  accountsTotalValue: document.getElementById("accounts-total-value"),
  accountsActiveValue: document.getElementById("accounts-active-value"),
  statsWrap: document.getElementById("stats-table-wrap"),
  responseConsole: document.getElementById("response-console"),
  configConsole: document.getElementById("config-console"),
  eventFeed: document.getElementById("event-feed"),
  btnInit: document.getElementById("btn-init"),
  btnRun: document.getElementById("btn-run"),
  btnAnalyze: document.getElementById("btn-analyze"),
  btnRetrain: document.getElementById("btn-retrain"),
  btnSchedule: document.getElementById("btn-schedule"),
  btnInstallCron: document.getElementById("btn-install-cron"),
  btnRefresh: document.getElementById("btn-refresh"),
  btnRefreshLogs: document.getElementById("btn-refresh-logs"),
  accLogin: document.getElementById("acc-login"),
  accPassword: document.getElementById("acc-password"),
  btnAccountSave: document.getElementById("btn-account-save"),
  btnAccountClear: document.getElementById("btn-account-clear"),
  btnAccountDelete: document.getElementById("btn-account-delete"),
};

const eventMemory = [];
let renderedLogKeys = new Set();
let accountCache = [];

async function apiGet(path) {
  const response = await fetch(path);
  const json = await response.json();
  if (!response.ok) {
    throw new Error(json.message || `Istek basarisiz: ${response.status}`);
  }
  return json;
}

async function apiPost(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const json = await response.json();
  if (!response.ok) {
    throw new Error(json.message || `Istek basarisiz: ${response.status}`);
  }
  return json;
}

function writeConsole(data) {
  ui.responseConsole.textContent = JSON.stringify(data, null, 2);
}

function writeConfig(data) {
  ui.configConsole.textContent = JSON.stringify(data, null, 2);
}

function setPill(status, text) {
  ui.healthPill.textContent = text;
  ui.healthPill.classList.remove("ok", "warn");
  ui.healthPill.classList.add(status === "ok" ? "ok" : "warn");
}

function addEvent(title, detail, tone = "ok") {
  eventMemory.unshift({
    title,
    detail,
    tone,
    timestamp: new Date().toLocaleTimeString("tr-TR"),
  });
  if (eventMemory.length > 60) {
    eventMemory.pop();
  }
  renderEvents();
}

function renderEvents() {
  ui.eventFeed.innerHTML = "";
  for (const item of eventMemory.slice(0, 24)) {
    const li = document.createElement("li");
    li.className = `event-item ${item.tone === "warn" ? "warn" : ""}`;
    li.innerHTML = `
      <strong>${escapeHtml(item.title)}</strong>
      <span>${escapeHtml(item.detail)}</span>
      <time>${escapeHtml(item.timestamp)}</time>
    `;
    ui.eventFeed.appendChild(li);
  }
}

function renderAccounts(accounts) {
  accountCache = Array.isArray(accounts) ? accounts : [];
  const previous = ui.accountSelect.value;
  ui.accountSelect.innerHTML = `<option value="">Tum Aktif Hesaplar</option>`;
  for (const account of accountCache) {
    const option = document.createElement("option");
    option.value = account.name;
    const login = account.username_or_email || account.name;
    option.textContent = login;
    ui.accountSelect.appendChild(option);
  }
  if (previous) {
    ui.accountSelect.value = previous;
  }
}

function findAccount(name) {
  if (!name) return null;
  return accountCache.find((item) => item.name === name) || null;
}

function setAccountForm(account) {
  if (!account) {
    clearAccountForm();
    return;
  }
  ui.accLogin.value = account.username_or_email || account.name || "";
  ui.accPassword.value = "";
}

function clearAccountForm() {
  ui.accLogin.value = "";
  ui.accPassword.value = "";
}

function slugifyLogin(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

function collectAccountForm() {
  const username_or_email = ui.accLogin.value.trim();
  const password = ui.accPassword.value;
  const name = slugifyLogin(username_or_email);
  // Legacy backend compatibility: keep `name` in payload as fallback.
  return { username_or_email, password, name };
}

function renderStats(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    ui.statsWrap.innerHTML = "<p>Henuz istatistik yok.</p>";
    return;
  }

  const maxUploaded = Math.max(...rows.map((r) => Number(r.uploaded || 0)), 1);
  const table = document.createElement("table");
  table.className = "stats-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Hesap</th>
        <th>Uretilen</th>
        <th>Yuklenen</th>
        <th>Basarisiz</th>
        <th>Kuyruk</th>
        <th>Bekleyen Metrik</th>
        <th>Yukleme Hizi</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map((row) => {
          const uploaded = Number(row.uploaded || 0);
          const width = Math.round((uploaded / maxUploaded) * 100);
          return `
            <tr>
              <td>${escapeHtml(String(row.account))}</td>
              <td>${row.generated}</td>
              <td>${uploaded}</td>
              <td>${row.failed_uploads}</td>
              <td>${row.queue_pending}</td>
              <td>${row.metrics_pending}</td>
              <td>
                <div class="meter">
                  <div class="meter-fill" style="width: ${width}%"></div>
                </div>
              </td>
            </tr>
          `;
        })
        .join("")}
    </tbody>
  `;
  ui.statsWrap.innerHTML = "";
  ui.statsWrap.appendChild(table);
}

function selectedAccountPayload() {
  const account = ui.accountSelect.value.trim();
  return account ? { account } : {};
}

function schedulePayload() {
  const raw = Number(ui.scheduleCycles.value || 1);
  const cycles = Number.isFinite(raw) ? Math.max(1, Math.min(500, Math.floor(raw))) : 1;
  ui.scheduleCycles.value = String(cycles);
  return { cycles };
}

function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins} dk ${secs} sn`;
}

async function refreshCore() {
  const accountQuery = ui.accountSelect.value
    ? `?account=${encodeURIComponent(ui.accountSelect.value)}`
    : "";

  const [healthRes, accountRes, statsRes] = await Promise.all([
    apiGet("/api/health"),
    apiGet("/api/accounts"),
    apiGet(`/api/stats${accountQuery}`),
  ]);

  const health = healthRes.data || {};
  setPill("ok", "Sistem cevrimici");
  ui.uptimeValue.textContent = formatDuration(Number(health.uptime_seconds || 0));
  ui.accountsTotalValue.textContent = String(health.accounts_total || 0);
  ui.accountsActiveValue.textContent = String(health.accounts_active || 0);
  renderAccounts(accountRes.data || []);
  renderStats(statsRes.data || []);

  try {
    const cfgRes = await apiGet("/api/config");
    writeConfig(cfgRes.data || {});
  } catch (error) {
    writeConfig({
      bilgi: "Konfigurasyon ozeti bu sunucu surumunde desteklenmiyor.",
      hata: String(error),
    });
  }
}

async function refreshLogs() {
  const logsRes = await apiGet("/api/logs?lines=60");
  const logs = logsRes.data || [];

  for (const line of logs) {
    const level = String(line.level || "INFO").toUpperCase();
    const message = String(line.message || "log");
    const timestamp = String(line.timestamp || "");
    const key = `${timestamp}|${level}|${message}`;
    if (renderedLogKeys.has(key)) {
      continue;
    }
    renderedLogKeys.add(key);
    if (renderedLogKeys.size > 1000) {
      renderedLogKeys = new Set(Array.from(renderedLogKeys).slice(-500));
    }

    const tone = level === "ERROR" || level === "WARNING" ? "warn" : "ok";
    addEvent(level, message, tone);
  }
}

function setLoading(button, on) {
  if (!button) return;
  button.disabled = on;
  button.style.opacity = on ? "0.7" : "1";
}

async function runAction(label, endpoint, payload = {}) {
  try {
    const response = await apiPost(endpoint, payload);
    writeConsole(response);
    addEvent(label, "Islem basariyla tamamlandi.");
    await refreshCore();
    await refreshLogs();
    return response;
  } catch (error) {
    const message = String(error);
    writeConsole({ status: "error", message });
    addEvent(label, message, "warn");
    setPill("warn", "Dikkat gerekli");
    throw error;
  }
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function bootstrap() {
  try {
    await refreshCore();
    await refreshLogs();
    addEvent("Panel", "Kontrol paneli hazir.");
  } catch (error) {
    const message = String(error);
    setPill("warn", "Baslangic hatasi");
    writeConsole({ status: "error", message });
    addEvent("Baslangic", message, "warn");
  }
}

ui.btnInit.addEventListener("click", async () => {
  setLoading(ui.btnInit, true);
  try {
    await runAction("Ilk kurulum", "/api/init");
  } finally {
    setLoading(ui.btnInit, false);
  }
});

ui.btnRun.addEventListener("click", async () => {
  setLoading(ui.btnRun, true);
  try {
    await runAction("Dongu calistir", "/api/run", selectedAccountPayload());
  } finally {
    setLoading(ui.btnRun, false);
  }
});

ui.btnAnalyze.addEventListener("click", async () => {
  setLoading(ui.btnAnalyze, true);
  try {
    await runAction("Analiz", "/api/analyze", selectedAccountPayload());
  } finally {
    setLoading(ui.btnAnalyze, false);
  }
});

ui.btnRetrain.addEventListener("click", async () => {
  setLoading(ui.btnRetrain, true);
  try {
    await runAction("Model egitimi", "/api/retrain", selectedAccountPayload());
  } finally {
    setLoading(ui.btnRetrain, false);
  }
});

ui.btnSchedule.addEventListener("click", async () => {
  setLoading(ui.btnSchedule, true);
  try {
    const payload = schedulePayload();
    await runAction("Zamanlayici dongusu", "/api/schedule", payload);
  } finally {
    setLoading(ui.btnSchedule, false);
  }
});

ui.btnInstallCron.addEventListener("click", async () => {
  setLoading(ui.btnInstallCron, true);
  try {
    await runAction("Cron kurulumu", "/api/schedule", { install_cron: true });
  } finally {
    setLoading(ui.btnInstallCron, false);
  }
});

ui.btnRefresh.addEventListener("click", async () => {
  try {
    await refreshCore();
    addEvent("Yenile", "Istatistikler guncellendi.");
  } catch (error) {
    addEvent("Yenile", String(error), "warn");
  }
});

ui.btnRefreshLogs.addEventListener("click", async () => {
  try {
    await refreshLogs();
    addEvent("Log", "Log akisi guncellendi.");
  } catch (error) {
    addEvent("Log", String(error), "warn");
  }
});

ui.accountSelect.addEventListener("change", async () => {
  try {
    const selected = ui.accountSelect.value || "";
    const account = findAccount(selected);
    if (account) {
      setAccountForm(account);
    }
    await refreshCore();
    addEvent("Hesap kapsami", selected || "Tum aktif hesaplar");
  } catch (error) {
    addEvent("Hesap kapsami", String(error), "warn");
  }
});

ui.btnAccountSave.addEventListener("click", async () => {
  setLoading(ui.btnAccountSave, true);
  try {
    const payload = collectAccountForm();
    if (!payload.username_or_email) {
      throw new Error("Kullanici adi veya e-posta bos olamaz");
    }
    if (!payload.password) {
      throw new Error("Sifre bos olamaz");
    }
    await runAction("Hesap kaydet", "/api/accounts/upsert", payload);
  } finally {
    setLoading(ui.btnAccountSave, false);
  }
});

ui.btnAccountDelete.addEventListener("click", async () => {
  setLoading(ui.btnAccountDelete, true);
  try {
    const username_or_email = ui.accLogin.value.trim();
    if (!username_or_email) {
      throw new Error("Silmek icin kullanici adi veya e-posta girin");
    }
    await runAction("Hesap sil", "/api/accounts/delete", {
      username_or_email,
      name: slugifyLogin(username_or_email),
    });
    clearAccountForm();
    ui.accountSelect.value = "";
  } finally {
    setLoading(ui.btnAccountDelete, false);
  }
});

ui.btnAccountClear.addEventListener("click", () => {
  clearAccountForm();
  addEvent("Form", "Hesap formu temizlendi.");
});

setInterval(() => {
  refreshCore().catch((error) => addEvent("Oto yenileme", String(error), "warn"));
}, 25000);

setInterval(() => {
  refreshLogs().catch((error) => addEvent("Log yenileme", String(error), "warn"));
}, 15000);

bootstrap();
