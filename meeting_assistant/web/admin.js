const $ = (selector) => document.querySelector(selector);
const token = () => localStorage.getItem('meeting-admin-token') || '';
const api = async (url, options = {}) => {
  options.headers = {
    'Content-Type': 'application/json',
    'X-Admin-Token': token(),
    ...(options.headers || {}),
  };
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '操作失敗');
  return data;
};
const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char]));
const labels = {
<<<<<<< Updated upstream
  meetings: '會議記錄', speakers: '說話人', utterances: '逐字稿段落', actions: '待辦事項',
=======
  meetings: '會議記錄', speakers: '會議說話人', speaker_profiles: '已確認聲紋', utterances: '逐字稿段落', actions: '待辦事項',
>>>>>>> Stashed changes
  decisions: '會議決議', follow_ups: '前次待辦追蹤', logs: '處理記錄', evaluations: '模型評估',
  threads: '會議串', rag_content: '內文向量', rag_summaries: '摘要向量', rag_actions: '待辦向量',
  analysis_checkpoints: '摘要恢復檢查點',
};
const tokenSettingInputs = {
  direct_max_tokens: '#direct-max-tokens',
  chunk_tokens: '#chunk-tokens',
  map_max_tokens: '#map-max-tokens',
  reduce_input_tokens: '#reduce-input-tokens',
  reduce_max_tokens: '#reduce-max-tokens',
  final_max_tokens: '#final-max-tokens',
};

let contextPollTimer = null;

async function load() {
  try {
    const [stats, meetings] = await Promise.all([api('/api/admin/stats'), api('/api/meetings')]);
    $('#stats').innerHTML = Object.entries(stats).map(([key, value]) => `<div class="stat-card"><strong>${value}</strong><small>${labels[key] || escapeHtml(key)}</small></div>`).join('');
    $('#meetings').innerHTML = meetings.length ? meetings.map((meeting) => `
      <div class="meeting-row"><div><strong class="meeting-title">${escapeHtml(meeting.title)}</strong><small class="meeting-id">${escapeHtml(meeting.id)}</small></div><small class="meeting-date">${escapeHtml(new Date(meeting.meeting_time || meeting.created_at).toLocaleString())}</small><button class="delete-button" data-id="${escapeHtml(meeting.id)}">刪除</button></div>`).join('') : '<p class="empty-state">目前沒有會議記錄。</p>';
    document.querySelectorAll('.delete-button').forEach((button) => { button.onclick = () => removeMeeting(button.dataset.id); });
  } catch (error) {
    $('#message').textContent = error.message;
  }
  await loadContext();
}

async function loadContext() {
  try {
    const settings = await api('/api/admin/config');
    renderContext(settings);
    renderTokenSettings(settings);
  } catch (error) {
    $('#context-status').textContent = error.message;
    $('#context-status').className = 'status-message is-error';
    $('#token-settings-status').textContent = error.message;
    $('#token-settings-status').className = 'status-message is-error';
  }
}

function scheduleContextPoll() {
  clearTimeout(contextPollTimer);
  contextPollTimer = setTimeout(loadContext, 1500);
}

function renderContext(settings) {
  const job = settings.job || {};
  const pending = job.status === 'queued' || job.status === 'reloading';
  const contextTokens = pending ? job.context_tokens : settings.context_tokens;
  const input = $('#context-tokens');
  const button = $('#apply-context');
  const status = $('#context-status');
  input.min = settings.min_context_tokens;
  input.max = settings.max_context_tokens;
  if (document.activeElement !== input || pending) input.value = contextTokens;
  input.disabled = pending;
  button.disabled = pending;

  if (pending) {
    status.textContent = `${job.stage || '正在套用設定'}（${contextTokens} tokens）；模型載入可能需要一些時間。`;
    status.className = 'status-message is-busy';
    scheduleContextPoll();
  } else if (job.status === 'failed') {
    status.textContent = job.error || job.stage || 'context 調整失敗';
    status.className = 'status-message is-error';
  } else if (job.status === 'completed') {
    status.textContent = job.stage;
    status.className = 'status-message is-success';
  } else if (settings.model_loaded) {
    status.textContent = `模型目前以 ${settings.context_tokens} tokens 載入。`;
    status.className = 'status-message';
  } else {
    status.textContent = `目前設定 ${settings.context_tokens} tokens；模型尚未載入，會在下次摘要時套用。`;
    status.className = 'status-message';
  }
}

function renderTokenSettings(settings) {
  const values = settings.token_settings || {};
  const specs = settings.token_setting_specs || {};
  for (const [key, selector] of Object.entries(tokenSettingInputs)) {
    const input = $(selector);
    const spec = specs[key];
    if (!input || !spec) continue;
    input.min = spec.min;
    input.max = spec.max;
    if (document.activeElement !== input) input.value = values[key] ?? spec.default;
  }
}

async function applyContext() {
  const contextTokens = Number($('#context-tokens').value);
  if (!Number.isInteger(contextTokens) || contextTokens < Number($('#context-tokens').min) || contextTokens > Number($('#context-tokens').max)) {
    $('#context-status').textContent = `請輸入 ${$('#context-tokens').min} 到 ${$('#context-tokens').max} 之間的整數。`;
    $('#context-status').className = 'status-message is-error';
    return;
  }
  $('#apply-context').disabled = true;
  $('#context-status').textContent = '正在排入 context 調整…';
  $('#context-status').className = 'status-message is-busy';
  try {
    await api('/api/admin/config', {
      method: 'PUT',
      body: JSON.stringify({ context_tokens: contextTokens }),
    });
    await loadContext();
  } catch (error) {
    $('#apply-context').disabled = false;
    $('#context-status').textContent = error.message;
    $('#context-status').className = 'status-message is-error';
  }
}

async function applyTokenSettings() {
  const updates = {};
  for (const [key, selector] of Object.entries(tokenSettingInputs)) {
    const input = $(selector);
    const value = Number(input.value);
    if (!Number.isInteger(value) || value < Number(input.min) || value > Number(input.max)) {
      $('#token-settings-status').textContent = `${input.closest('label').querySelector('span').textContent}請輸入 ${input.min} 到 ${input.max} 之間的整數。`;
      $('#token-settings-status').className = 'status-message is-error';
      input.focus();
      return;
    }
    updates[key] = value;
  }

  const button = $('#apply-token-settings');
  button.disabled = true;
  $('#token-settings-status').textContent = '正在保存摘要 token 設定…';
  $('#token-settings-status').className = 'status-message is-busy';
  try {
    await api('/api/admin/config', {
      method: 'PUT',
      body: JSON.stringify({ token_settings: updates }),
    });
    $('#token-settings-status').textContent = '設定已保存，下一次摘要時生效。';
    $('#token-settings-status').className = 'status-message is-success';
    await loadContext();
  } catch (error) {
    $('#token-settings-status').textContent = error.message;
    $('#token-settings-status').className = 'status-message is-error';
  } finally {
    button.disabled = false;
  }
}

async function removeMeeting(id) {
  if (!window.confirm('確定刪除此會議及其逐字稿、RAG 索引、待辦、決議與評估紀錄？')) return;
  try { await api(`/api/admin/meetings/${encodeURIComponent(id)}`, { method: 'DELETE' }); $('#message').textContent = '會議及關聯資料已刪除'; await load(); }
  catch (error) { $('#message').textContent = error.message; }
}

$('#token').value = token();
$('#save-token').onclick = () => { localStorage.setItem('meeting-admin-token', $('#token').value); $('#message').textContent = 'Token 已儲存'; load(); };
$('#apply-context').onclick = applyContext;
$('#apply-token-settings').onclick = applyTokenSettings;
$('#purge').onclick = async () => {
  if (!window.confirm('確定清除全部資料？此操作無法復原。')) return;
  if (window.prompt('請輸入 DELETE_ALL 確認') !== 'DELETE_ALL') return;
  try { const result = await api('/api/admin/purge', { method: 'POST' }); $('#message').textContent = `已清除 ${result.deleted_meetings} 場會議與全部索引`; await load(); }
  catch (error) { $('#message').textContent = error.message; }
};
load();
