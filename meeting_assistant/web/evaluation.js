const $ = (selector) => document.querySelector(selector);

const sourceLabels = {
  local: '本地 llama.cpp',
  remote: '遠端 gpt-6-luna',
  external: '外部會議整理',
};

let externalReferences = [];

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#039;',
}[char]));

function formatDate(value, fallback = '時間未提供') {
  if (!value) return fallback;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return fallback;
  try {
    return date.toLocaleString('zh-TW', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  } catch (_) {
    return date.toISOString();
  }
}

function responseError(status, body) {
  const compactBody = String(body || '').replace(/\s+/g, ' ').trim();
  let detail = '';
  if (body) {
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.error === 'string') detail = parsed.error;
    } catch (_) {
      // A reverse proxy may return an HTML error page.
    }
  }
  detail = detail || (compactBody && !/<\/?(?:html|body|title)\b/i.test(compactBody)
    ? compactBody.slice(0, 240)
    : '伺服器沒有提供可讀取的錯誤內容');
  if (/CGULLM_API_KEY/i.test(detail)) {
    return '評估失敗：後端未設定 CGULLM_API_KEY。請在啟動服務的環境設定 API key，並重新啟動後端。';
  }
  return `評估請求失敗（HTTP ${status}）：${detail}`;
}

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(`無法連線至後端服務：${error?.message || '網路連線失敗'}`);
  }
  const body = await response.text();
  if (!response.ok) throw new Error(responseError(response.status, body));
  if (!body.trim()) throw new Error('後端回應是空的，請查看服務 log。');
  try {
    return JSON.parse(body);
  } catch (_) {
    throw new Error(`後端回應不是有效 JSON（HTTP ${response.status}），請查看服務 log。`);
  }
}

function selectedMeetingId() {
  return $('#meeting-select').value;
}

function selectedExternalId() {
  return $('#external-reference').value || null;
}

function updateRunState() {
  const hasMeeting = Boolean(selectedMeetingId());
  const useRemote = $('#include-remote').checked;
  const useExternal = Boolean(selectedExternalId());
  const valid = hasMeeting && (useRemote || useExternal);
  $('#run').disabled = !valid;
  if (!hasMeeting) {
    $('#status').textContent = '請先選擇會議。';
  } else if (!valid) {
    $('#status').textContent = '本地摘要固定納入；請至少選擇遠端模型或一份外部整理。';
  } else {
    const count = 1 + (useRemote ? 1 : 0) + (useExternal ? 1 : 0);
    $('#status').textContent = `本次將比較 ${count} 個來源：本地摘要固定納入。`;
  }
}

function renderReferenceOptions() {
  const select = $('#external-reference');
  const current = select.value;
  select.innerHTML = '<option value="">不使用外部整理</option>' + externalReferences.map((reference) => (
    `<option value="${esc(reference.id)}">${esc(reference.title || reference.filename)} · ${esc(reference.filename)}</option>`
  )).join('');
  select.disabled = !selectedMeetingId() || !externalReferences.length;
  if (externalReferences.some((reference) => String(reference.id) === String(current))) select.value = current;
  updateRunState();
}

async function loadReferences() {
  const meetingId = selectedMeetingId();
  externalReferences = meetingId
    ? await api(`/api/evaluation-references?meeting_id=${encodeURIComponent(meetingId)}`)
    : [];
  renderReferenceOptions();
  $('#external-file').disabled = !meetingId;
  $('#import-external').disabled = !meetingId || !$('#external-file').files.length;
  $('#external-import-status').textContent = meetingId
    ? (externalReferences.length ? `已匯入 ${externalReferences.length} 份外部整理，可選擇其中一份。` : '尚未匯入外部整理。')
    : '請先選擇會議。';
}

function scoreHtml(score) {
  const metrics = [
    ['整體正確性', 'overall_accuracy'],
    ['事實正確性', 'factual_accuracy'],
    ['內容涵蓋度', 'coverage'],
    ['待辦與決議', 'action_decision_accuracy'],
  ];
  const metricHtml = metrics.map(([label, key]) => {
    const value = Math.max(0, Math.min(100, Number(score?.[key] || 0)));
    return `<div class="metric"><div class="metric-label"><span>${label}</span><strong>${value}</strong></div><div class="track"><div class="fill" style="width:${value}%"></div></div></div>`;
  }).join('');
  const issues = Array.isArray(score?.issues) ? score.issues : [];
  return `${metricHtml}<h4>評審指出</h4>${issues.length
    ? `<ul class="issues">${issues.map((item) => `<li>${esc(item)}</li>`).join('')}</ul>`
    : '<p class="muted">未指出明顯錯誤</p>'}`;
}

function summaryHtml(value) {
  const actions = Array.isArray(value?.actions) ? value.actions : [];
  const decisions = Array.isArray(value?.decisions) ? value.decisions : [];
  const keyPoints = Array.isArray(value?.key_points) ? value.key_points : [];
  const source = value?.source_filename ? `<p class="source-file">來源：${esc(value.source_filename)}</p>` : '';
  return `${source}<h4>摘要</h4><div>${esc(value?.summary || '（無摘要）')}</div>${keyPoints.length ? `<h4>重點</h4><ul>${keyPoints.slice(0, 12).map((item) => `<li>${esc(item)}</li>`).join('')}</ul>` : ''}<h4>待辦</h4>${actions.length
    ? `<ul>${actions.map((item) => `<li>${esc(item?.item)}｜${esc(item?.owner || '未指派')}｜${esc(item?.due_date || '未提供期限')}</li>`).join('')}</ul>`
    : '<p class="muted">無</p>'}<h4>決議</h4>${decisions.length
    ? `<ul>${decisions.map((item) => `<li>${esc(item?.decision)}${item?.context ? `｜${esc(item.context)}` : ''}</li>`).join('')}</ul>`
    : '<p class="muted">無</p>'}`;
}

function render(result) {
  $('#result').hidden = false;
  $('#result-meta').textContent = `${result.model || '遠端評審模型'} · ${formatDate(result.created_at)}`;
  $('#winner').textContent = result.winner === 'tie'
    ? '評估結果相近'
    : `${sourceLabels[result.winner] || result.winner} 分數較高`;

  const sources = Array.isArray(result.sources) && result.sources.length
    ? result.sources
    : Object.keys(result.scores || {});
  $('#score-grid').innerHTML = sources.map((source) => `<article class="score-card source-${esc(source)}"><div class="card-head"><span class="model-dot ${source === 'local' ? 'local-dot' : source === 'external' ? 'external-dot' : 'remote-dot'}"></span><h3>${esc(sourceLabels[source] || source)}</h3></div>${scoreHtml(result.scores?.[source])}</article>`).join('');
  $('#summary-grid').innerHTML = sources.map((source) => `<article class="summary-card source-${esc(source)}"><p class="overline">${esc(sourceLabels[source] || source).toUpperCase()}</p><h3>${esc(sourceLabels[source] || source)}</h3><div>${summaryHtml(result[source])}</div></article>`).join('');
  $('#reason').textContent = result.reason || '評審未提供說明';
}

async function loadHistory() {
  const id = selectedMeetingId();
  if (!id) {
    $('#history-list').innerHTML = '<p class="empty">請先選擇會議</p>';
    return;
  }
  const rows = await api(`/api/evaluations?meeting_id=${encodeURIComponent(id)}`);
  if (!Array.isArray(rows)) throw new Error('評估紀錄回應格式錯誤。');
  $('#history-list').innerHTML = rows.length
    ? rows.map((row, index) => {
      const sources = Array.isArray(row.sources) && row.sources.length ? row.sources : Object.keys(row.scores || {});
      const scores = sources.map((source) => `${esc(sourceLabels[source] || source)} ${Number(row.scores?.[source]?.overall_accuracy || 0)}`).join(' · ');
      return `<div class="history-row" data-index="${index}"><span>${formatDate(row.created_at)}</span><strong>${scores}</strong></div>`;
    }).join('')
    : '<p class="empty">尚無評估紀錄</p>';
  document.querySelectorAll('.history-row').forEach((row) => {
    row.onclick = () => render(rows[Number(row.dataset.index)]);
  });
}

async function init() {
  const meetings = await api('/api/meetings');
  if (!Array.isArray(meetings)) throw new Error('會議清單回應格式錯誤。');
  $('#meeting-select').innerHTML = '<option value="">選擇一場會議</option>' + meetings.map((meeting) => (
    `<option value="${esc(meeting.id)}">${esc(meeting.title)} · ${formatDate(meeting.created_at)}</option>`
  )).join('');
  $('#meeting-select').onchange = () => Promise.all([loadReferences(), loadHistory()]).catch((error) => {
    $('#status').textContent = error.message;
  });
  $('#include-remote').onchange = updateRunState;
  $('#external-reference').onchange = updateRunState;
  $('#external-file').onchange = () => {
    const file = $('#external-file').files[0];
    $('#external-file-name').textContent = file ? `${file.name}（${Math.ceil(file.size / 1024)} KB）` : '尚未選擇檔案';
    $('#import-external').disabled = !selectedMeetingId() || !file;
  };
  await loadReferences();
  await loadHistory();
  updateRunState();
}

$('#import-external').onclick = async () => {
  const file = $('#external-file').files[0];
  const meetingId = selectedMeetingId();
  if (!meetingId || !file) return;
  if (file.size > 2 * 1024 * 1024) {
    $('#external-import-status').textContent = '檔案不可超過 2 MB。';
    return;
  }
  const button = $('#import-external');
  button.disabled = true;
  $('#external-import-status').textContent = '正在匯入外部整理…';
  try {
    const content = await file.text();
    await api('/api/evaluation-references', {
      method: 'POST',
      body: JSON.stringify({ meeting_id: meetingId, filename: file.name, content }),
    });
    $('#external-import-status').textContent = '外部整理已匯入，可在評估來源中選擇。';
    $('#external-file').value = '';
    $('#external-file-name').textContent = '尚未選擇檔案';
    await loadReferences();
  } catch (error) {
    $('#external-import-status').textContent = error.message;
  } finally {
    button.disabled = !selectedMeetingId() || !$('#external-file').files.length;
  }
};

$('#run').onclick = async () => {
  const meetingId = selectedMeetingId();
  const includeRemote = $('#include-remote').checked;
  const externalReferenceId = selectedExternalId();
  if (!meetingId || (!includeRemote && !externalReferenceId)) {
    updateRunState();
    return;
  }
  const button = $('#run');
  button.disabled = true;
  button.textContent = '評估中…';
  $('#status').textContent = '正在以所選來源進行摘要比較與評分，請保持此頁開啟。';
  try {
    const result = await api('/api/evaluations', {
      method: 'POST',
      body: JSON.stringify({ meeting_id: meetingId, include_remote: includeRemote, external_reference_id: externalReferenceId }),
    });
    render(result);
    $('#status').textContent = '評估完成';
    await loadHistory();
  } catch (error) {
    $('#status').textContent = error.message;
  } finally {
    button.textContent = '開始評估';
    updateRunState();
  }
};

init().catch((error) => {
  $('#status').textContent = error.message;
});
