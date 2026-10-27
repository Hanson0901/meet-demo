const $ = (selector) => document.querySelector(selector);

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
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
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
      // A reverse proxy may return an HTML error page. Do not expose it as HTML.
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

async function api(url, options) {
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
  return `<h4>摘要</h4><div>${esc(value?.summary || '（無摘要）')}</div><h4>待辦</h4>${actions.length
    ? `<ul>${actions.map((item) => `<li>${esc(item?.item)}｜${esc(item?.owner || '未指派')}｜${esc(item?.due_date || '未提供期限')}</li>`).join('')}</ul>`
    : '<p class="muted">無</p>'}<h4>決議</h4>${decisions.length
    ? `<ul>${decisions.map((item) => `<li>${esc(item?.decision)}${item?.context ? `｜${esc(item.context)}` : ''}</li>`).join('')}</ul>`
    : '<p class="muted">無</p>'}`;
}

function render(result) {
  $('#result').hidden = false;
  $('#result-meta').textContent = `${result.model || '遠端模型'} · ${formatDate(result.created_at)}`;
  $('#winner').textContent = result.winner === 'tie'
    ? '評估結果相近'
    : `${result.winner === 'local' ? '本地 llama.cpp' : '遠端 gpt-6-luna'} 分數較高`;
  $('#local-score').innerHTML = scoreHtml(result.scores?.local);
  $('#remote-score').innerHTML = scoreHtml(result.scores?.remote);
  $('#reason').textContent = result.reason || '評審未提供說明';
  $('#local-summary').innerHTML = summaryHtml(result.local);
  $('#remote-summary').innerHTML = summaryHtml(result.remote);
}

async function loadHistory() {
  const id = $('#meeting-select').value;
  if (!id) {
    $('#history-list').innerHTML = '<p class="empty">請先選擇會議</p>';
    return;
  }
  const rows = await api(`/api/evaluations?meeting_id=${encodeURIComponent(id)}`);
  if (!Array.isArray(rows)) throw new Error('評估紀錄回應格式錯誤。');
  $('#history-list').innerHTML = rows.length
    ? rows.map((row, index) => `<div class="history-row" data-index="${index}"><span>${formatDate(row.created_at)}</span><strong>本地 ${Number(row.scores?.local?.overall_accuracy || 0)} · 遠端 ${Number(row.scores?.remote?.overall_accuracy || 0)}</strong></div>`).join('')
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
  $('#meeting-select').onchange = () => loadHistory().catch((error) => {
    $('#status').textContent = error.message;
  });
  await loadHistory();
}

$('#run').onclick = async () => {
  const meetingId = $('#meeting-select').value;
  if (!meetingId) {
    $('#status').textContent = '請先選擇會議';
    return;
  }
  const button = $('#run');
  button.disabled = true;
  button.textContent = '遠端 API 評估中…';
  $('#status').textContent = '正在以遠端模型產生摘要並評審既有本地結果，請保持此頁開啟。';
  try {
    const result = await api('/api/evaluations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ meeting_id: meetingId }),
    });
    render(result);
    $('#status').textContent = '評估完成';
    await loadHistory();
  } catch (error) {
    $('#status').textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = '呼叫遠端模型進行比較';
  }
};

init().catch((error) => {
  $('#status').textContent = error.message;
});
