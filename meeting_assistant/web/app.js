const $ = (selector) => document.querySelector(selector);
let currentMeetingId = null;
const transcriptionPolls = new Set();
const analysisPolls = new Map();
const speakerPolls = new Map();
const activeJobStatuses = new Set(['queued', 'running', 'reloading']);
const currentMeetingStorageKey = 'meeting-assistant.current-meeting-id';
const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const api = async (url, options) => {
  const response = await fetch(url, options);
  const raw = await response.text();
  let data = null;
  if (raw.trim()) {
    try {
      data = JSON.parse(raw);
    } catch (_error) {
      const status = response.status ? `HTTP ${response.status}` : '伺服器';
      throw new Error(`${status} 回應不是有效 JSON，可能是服務未啟動或反向代理錯誤。`);
    }
  }
  if (!response.ok) throw new Error(data?.error || `請求失敗（HTTP ${response.status}）`);
  if (data === null) throw new Error('伺服器回應是空的，請確認服務是否仍在執行。');
  return data;
};
const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char]));
const formatDate = (value) => {
  if (!value) return '時間未設定';
  const text = String(value).trim();
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return `時間格式不明：${text}`;
  try {
    return date.toLocaleString('zh-TW', { dateStyle: 'medium', timeStyle: 'short' });
  } catch (_error) {
    return date.toLocaleString('zh-TW');
  }
};
const formatTime = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const formatTimestamp = (value) => value ? formatDate(value) : '—';
const formatElapsed = (value) => {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round((seconds - minutes * 60) * 10) / 10;
  if (minutes < 60) return `${minutes} 分 ${remainder.toFixed(1)} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小時 ${minutes % 60} 分 ${remainder.toFixed(1)} 秒`;
};
const processStatusLabels = {
  idle: '尚未開始', queued: '排隊中', running: '執行中', reloading: '套用設定中',
  completed: '已完成', failed: '失敗', canceled: '已中止',
};

function renderProcessLogs(logs) {
  const lines = (logs || []).map((log) => {
    const level = String(log.level || 'INFO').toLowerCase();
    return `${formatTimestamp(log.timestamp)} [${log.module || 'SYSTEM'}] ${level} ${log.message || ''}`;
  });
  const liveLogs = $('#live-logs');
  const shouldStick = liveLogs.scrollTop + liveLogs.clientHeight >= liveLogs.scrollHeight - 24;
  liveLogs.textContent = lines.join('\n') || '尚無處理記錄';
  if (shouldStick) liveLogs.scrollTop = liveLogs.scrollHeight;
  $('#logs').textContent = lines.join('\n') || '尚無處理記錄';
}

function renderProcessSteps(steps, status) {
  if (!steps?.length) {
    $('#process-steps').innerHTML = '<p class="empty-list">尚未產生步驟記錄</p>';
    return;
  }
  $('#process-steps').innerHTML = steps.map((step) => {
    const isActive = !step.finished_at;
    const isError = status === 'failed' && isActive;
    const classes = ['step-row', isActive ? 'is-active' : 'is-complete', isError ? 'is-error' : ''].filter(Boolean).join(' ');
    const timing = `開始 ${formatTimestamp(step.started_at)} · 結束 ${formatTimestamp(step.finished_at)} · ${formatElapsed(step.elapsed_seconds)}`;
    return `<div class="${classes}"><span class="step-marker" aria-hidden="true"></span><div class="step-copy"><strong>${escapeHtml(step.name || '未命名步驟')}</strong><small>${escapeHtml(timing)}</small></div></div>`;
  }).join('');
}

function renderProcessJob(job, type = 'analysis') {
  if (!job) return;
  const logs = job.logs || [];
  const hasActivity = job.status !== 'idle' || logs.length > 0 || (job.steps || []).length > 0 || Boolean(job.model_output);
  $('#process-panel').hidden = !hasActivity;
  if (!hasActivity) return;

  const status = job.status || 'idle';
  const statusElement = $('#process-status');
  statusElement.textContent = processStatusLabels[status] || status;
  statusElement.className = `process-status is-${escapeHtml(status)}`;
  $('#process-stage').textContent = job.stage || (type === 'transcription' ? '語音辨識' : '模型處理');
  $('#process-started').textContent = formatTimestamp(job.started_at);
  $('#process-finished').textContent = formatTimestamp(job.finished_at);
  $('#process-elapsed').textContent = formatElapsed(job.elapsed_seconds);
  renderProcessSteps(job.steps || [], status);

  const modelOutput = job.model_output || '';
  const outputElement = $('#model-output');
  const shouldStick = outputElement.scrollTop + outputElement.clientHeight >= outputElement.scrollHeight - 24;
  outputElement.textContent = modelOutput || (job.model_output_error ? `模型輸出失敗：${job.model_output_error}` : '等待模型輸出…');
  if (shouldStick) outputElement.scrollTop = outputElement.scrollHeight;
  const modelMeta = [];
  if (job.model_stage) modelMeta.push(job.model_stage);
  if (job.model_output_chars) modelMeta.push(`${job.model_output_chars} 字`);
  if (job.model_elapsed_seconds != null) modelMeta.push(formatElapsed(job.model_elapsed_seconds));
  if (job.model_output_complete) modelMeta.push('本階段完成');
  else if (job.model_stage) modelMeta.push('生成中');
  $('#model-output-meta').textContent = modelMeta.length ? ` · ${modelMeta.join(' · ')}` : '';
  renderProcessLogs(logs);
}

async function loadProcessJob(meetingId) {
  const endpoints = [
    ['analysis', `/api/meetings/${encodeURIComponent(meetingId)}/analysis-status`],
    ['speaker', `/api/meetings/${encodeURIComponent(meetingId)}/speaker-suggestions-status`],
    ['transcription', `/api/meetings/${encodeURIComponent(meetingId)}/transcription-status`],
  ];
  const results = await Promise.all(endpoints.map(async ([type, url]) => {
    try { return { type, job: await api(url) }; } catch (_error) { return null; }
  }));
  const available = results.filter(Boolean);
  const active = available.find(({ job }) => ['queued', 'running', 'reloading'].includes(job.status));
  const withSteps = available.find(({ job }) => job.status !== 'idle' && ((job.steps || []).length || job.model_output));
  const selected = active || withSteps || available.find(({ job }) => job.status !== 'idle') || available[0];
  if (selected) {
    renderProcessJob(selected.job, selected.type);
    if (activeJobStatuses.has(selected.job.status)) {
      if (selected.type === 'analysis') void watchAnalysis(meetingId);
      if (selected.type === 'speaker') void watchSpeakerSuggestions(meetingId);
      if (selected.type === 'transcription') void watchTranscription(meetingId);
    }
  }
  return selected?.job || null;
}

async function loadMeetings() {
  const meetings = await api('/api/meetings');
  $('#meeting-list').innerHTML = meetings.length ? meetings.map((meeting) => `
    <button class="meeting-item ${meeting.id === currentMeetingId ? 'selected' : ''}" data-id="${escapeHtml(meeting.id)}">
      <span class="meeting-item-icon"><svg viewBox="0 0 24 24"><path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/></svg></span>
      <span class="meeting-item-copy"><strong title="${escapeHtml(meeting.title)}">${escapeHtml(meeting.title)}</strong><small>${escapeHtml(formatDate(meeting.meeting_time || meeting.created_at))}</small></span>
      <svg class="chevron" viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></svg>
    </button>`).join('') : '<p class="empty-list">還沒有會議記錄</p>';
  document.querySelectorAll('.meeting-item').forEach((button) => { button.onclick = () => loadMeeting(button.dataset.id); });
  return meetings;
}

function renderSpeakers(speakers) {
  $('#speaker-list').innerHTML = speakers.length ? speakers.map((speaker) => `
    <div class="speaker-card"><label class="speaker-edit"><span class="speaker-id">${escapeHtml(speaker.speaker_id)}</span><input data-speaker="${escapeHtml(speaker.speaker_id)}" value="${escapeHtml(speaker.display_name || speaker.suggested_name || '')}" placeholder="輸入或確認姓名"></label><small class="speaker-hint" title="${escapeHtml(speaker.suggestion_evidence || '')}">${speakerHint(speaker)}</small></div>`).join('') : '<p class="empty-list">沒有辨識到說話人</p>';
}

function speakerHint(speaker) {
  if (speaker.name_source === 'voiceprint' && speaker.display_name) {
    const score = Number.isFinite(Number(speaker.match_score)) ? ` · 相似度 ${(Number(speaker.match_score) * 100).toFixed(0)}%` : '';
    return `已套用過往確認的聲紋${score}`;
  }
  if (speaker.name_source === 'confirmed') return speaker.has_voiceprint ? '姓名已確認；聲紋已保存在本機' : '姓名已確認';
  const labels = {
    self_introduction: '逐字稿自我介紹候選，請核對',
    addressed_by_name: '依對話稱呼推測，請核對',
    local_llm: '本地模型候選，請核對',
    voiceprint_candidate: '相似聲紋候選未達自動套用門檻，請核對',
  };
  if (speaker.suggested_name) {
    const source = labels[speaker.suggestion_source] || '姓名候選，請核對';
    const value = speaker.suggestion_source === 'voiceprint_candidate'
      ? Number(speaker.match_score) : Number(speaker.suggestion_confidence);
    const metric = speaker.suggestion_source === 'voiceprint_candidate' ? '相似度' : '信心';
    const score = Number.isFinite(value) && value > 0 ? ` · ${metric} ${(value * 100).toFixed(0)}%` : '';
    return `${source}${score}`;
  }
  if (speaker.has_voiceprint) return '已有聲紋樣本，姓名尚未確認';
  return '尚無可比對聲紋或姓名候選；請手動確認';
}

function renderTranscript(rows) {
  $('#utterance-count').textContent = `${rows.length} 段`;
  $('#transcript-list').innerHTML = rows.length ? rows.map((row) => `
    <div class="utterance"><span class="time">${formatTime(row.start_time)}–${formatTime(row.end_time)}</span><span class="speaker">${escapeHtml(row.speaker_name || row.speaker_id)}</span><span class="utterance-text">${escapeHtml(row.normalized_text)}</span></div>`).join('') : '<p class="empty-list">尚無逐字稿</p>';
}

function renderThread(topic, meetings) {
  const panel = $('#thread-panel');
  panel.hidden = !topic || meetings.length === 0;
  if (panel.hidden) return;
  $('#thread-title').textContent = topic.thread_name || topic.display_name;
  $('#thread-count').textContent = `${meetings.length + 1} 場會議`;
  $('#related-meetings').innerHTML = meetings.map((meeting) => `
    <button class="related-meeting" data-id="${escapeHtml(meeting.id)}"><span><strong>${escapeHtml(meeting.title)}</strong><small>${escapeHtml(formatDate(meeting.meeting_time))}</small></span><svg viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></svg></button>`).join('');
  document.querySelectorAll('.related-meeting').forEach((button) => { button.onclick = () => loadMeeting(button.dataset.id); });
}

function renderActions(actions) {
  $('#action-count').textContent = `${actions.length}`;
  $('#actions').innerHTML = actions.length ? actions.map((action) => `
    <article class="action-item"><span class="task-check" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg></span><div><strong>${escapeHtml(action.item)}</strong><small>${escapeHtml(action.owner || '尚未指派')} · ${escapeHtml(action.due_date || '未設定期限')}</small></div></article>`).join('') : '<p class="empty-list">目前沒有待辦事項</p>';
}

function renderDecisions(decisions) {
  $('#decisions').innerHTML = decisions.length ? decisions.map((decision) => `
    <article class="decision-item"><span class="decision-dot"></span><div><strong>${escapeHtml(decision.decision)}</strong>${decision.context ? `<p>${escapeHtml(decision.context)}</p>` : ''}</div></article>`).join('') : '<p class="empty-list">目前沒有會議決議</p>';
}

function renderFollowUps(followUps) {
  const panel = $('#followup-panel');
  panel.hidden = !followUps.length;
  if (panel.hidden) return;
  const statusText = { completed: '已完成', in_progress: '進行中', not_mentioned: '本次未提及' };
  $('#follow-ups').innerHTML = followUps.map((item) => `
    <article class="followup-item"><div class="followup-top"><span class="followup-source">${escapeHtml(item.previous_title || '前次會議')}</span><span class="status-chip ${escapeHtml(item.status)}">${escapeHtml(statusText[item.status] || '待確認')}</span></div><strong>${escapeHtml(item.item)}</strong><p>${escapeHtml(item.conclusion)}</p></article>`).join('');
}

async function loadMeeting(meetingId) {
  const previousMeetingId = currentMeetingId;
  const data = await api(`/api/meetings/${encodeURIComponent(meetingId)}`);
  currentMeetingId = meetingId;
  try { window.localStorage.setItem(currentMeetingStorageKey, meetingId); } catch (_error) { /* storage may be disabled */ }
  $('#empty').hidden = true;
  $('#search-results').hidden = true;
  $('#meeting-view').hidden = false;
  $('#meeting-title').textContent = data.meeting.title;
  $('#meeting-date').textContent = formatDate(data.meeting.meeting_time || data.meeting.created_at);
  $('#topic-chip').hidden = !data.topic;
  $('#topic-chip').textContent = data.topic?.display_name || '';
  $('#summary').textContent = data.meeting.summary || '摘要尚未產生';
  renderSpeakers(data.speakers || []);
  if (previousMeetingId !== meetingId) $('#speaker-status').textContent = '';
  renderTranscript(data.transcript || []);
  renderActions(data.actions || []);
  renderDecisions(data.decisions || []);
  renderFollowUps(data.follow_ups || []);
  renderThread(data.topic, data.related_meetings || []);
  renderProcessLogs(data.logs || []);
  await loadProcessJob(meetingId);
  await loadMeetings();
}

async function watchTranscription(meetingId) {
  if (transcriptionPolls.has(meetingId)) return;
  transcriptionPolls.add(meetingId);
  try {
    while (true) {
      const job = await api(`/api/meetings/${encodeURIComponent(meetingId)}/transcription-status`);
      if (currentMeetingId === meetingId) renderProcessJob(job, 'transcription');
      if (job.status === 'completed') {
        if (currentMeetingId === meetingId) {
          $('#upload-status').textContent = `語音辨識完成，共 ${job.utterance_count || 0} 段逐字稿。請確認說話人姓名後再產生摘要。`;
          await loadMeeting(meetingId);
        }
        return;
      }
      if (job.status === 'failed') {
        if (currentMeetingId === meetingId) {
          $('#upload-status').textContent = `語音辨識失敗：${job.error || '請查看處理記錄。'}`;
          await loadMeeting(meetingId);
        }
        return;
      }
      if (job.status === 'canceled') {
        if (currentMeetingId === meetingId) {
          $('#upload-status').textContent = `語音辨識已中止：目前保留 ${job.utterance_count || 0} 段已寫入逐字稿。`;
          await loadMeeting(meetingId);
        }
        return;
      }
      if (job.status === 'idle') return;
      if (currentMeetingId === meetingId) $('#upload-status').textContent = job.stage || '語音辨識中…';
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
    }
  } catch (error) {
    if (currentMeetingId === meetingId) $('#upload-status').textContent = `無法取得語音辨識狀態：${error.message}`;
  } finally {
    transcriptionPolls.delete(meetingId);
  }
}

async function watchAnalysis(meetingId, button = null) {
  const existing = analysisPolls.get(meetingId);
  if (existing) return existing;
  const promise = (async () => {
    while (true) {
      const job = await api(`/api/meetings/${encodeURIComponent(meetingId)}/analysis-status`);
      if (currentMeetingId === meetingId) {
        renderProcessJob(job, 'analysis');
        if (button) button.querySelector('span').textContent = job.stage || '分段摘要進行中…';
      }
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'canceled') {
        if (!button && currentMeetingId === meetingId) await loadMeeting(meetingId);
        return job;
      }
      if (job.status === 'idle') return job;
      await wait(800);
    }
  })();
  analysisPolls.set(meetingId, promise);
  try {
    return await promise;
  } finally {
    analysisPolls.delete(meetingId);
  }
}

async function watchSpeakerSuggestions(meetingId, button = null) {
  const existing = speakerPolls.get(meetingId);
  if (existing) return existing;
  const promise = (async () => {
    while (true) {
      const job = await api(`/api/meetings/${encodeURIComponent(meetingId)}/speaker-suggestions-status`);
      if (currentMeetingId === meetingId) {
        renderProcessJob(job, 'speaker');
        if (button) $('#speaker-status').textContent = job.stage || '本地模型分析中…';
      }
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'canceled') {
        if (!button && currentMeetingId === meetingId) await loadMeeting(meetingId);
        return job;
      }
      if (job.status === 'idle') return job;
      await wait(1000);
    }
  })();
  speakerPolls.set(meetingId, promise);
  try {
    return await promise;
  } finally {
    speakerPolls.delete(meetingId);
  }
}

$('#upload-form').onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  const file = $('#audio').files?.[0];
  if (!file) {
    $('#upload-status').textContent = '請先選擇音訊檔案。';
    return;
  }
  button.disabled = true;
  $('#upload-status').textContent = '正在上傳音訊…';
  try {
    $('#started-at').value = new Date().toISOString();
    const result = await api('/api/transcribe', { method: 'POST', body: new FormData(event.currentTarget) });
    $('#upload-status').textContent = '音訊已上傳，語音辨識已在背景開始…';
    await loadMeeting(result.meeting_id);
    watchTranscription(result.meeting_id);
  } catch (error) {
    $('#upload-status').textContent = error.message;
  } finally {
    button.disabled = false;
  }
};

$('#audio').onchange = (event) => { $('#file-name').textContent = event.target.files[0]?.name || 'WAV、MP3、M4A 等格式'; };
$('#refresh').onclick = () => loadMeetings();
$('#save-speakers').onclick = async () => {
  if (!currentMeetingId) return;
  const mapping = {};
  document.querySelectorAll('[data-speaker]').forEach((input) => { mapping[input.dataset.speaker] = input.value; });
  const button = $('#save-speakers');
  button.disabled = true;
  try {
    await api(`/api/meetings/${currentMeetingId}/speakers`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ speakers: mapping }) });
    $('#speaker-status').textContent = '姓名已儲存並同步到逐字稿';
    await loadMeeting(currentMeetingId);
  } catch (error) { $('#speaker-status').textContent = error.message; }
  finally { button.disabled = false; }
};

$('#suggest-speakers').onclick = async () => {
  if (!currentMeetingId) return;
  const meetingId = currentMeetingId;
  const button = $('#suggest-speakers');
  button.disabled = true;
  $('#speaker-status').textContent = '本地模型正在尋找自我介紹與稱呼線索；此操作不會送到遠端服務。';
  try {
    await api(`/api/meetings/${encodeURIComponent(meetingId)}/speaker-suggestions`, { method: 'POST' });
    const job = await watchSpeakerSuggestions(meetingId, button);
    if (job.status === 'failed') throw new Error(job.error || '本地模型推測失敗');
    if (job.status === 'idle') throw new Error('找不到推測工作狀態，請重新整理後再試。');
    if (currentMeetingId !== meetingId) return;
    const data = job.result || {};
    renderSpeakers(data.speakers || []);
    const count = (data.suggestions || []).length;
    $('#speaker-status').textContent = count
      ? `找到 ${count} 個姓名候選；請核對輸入欄後按「確認姓名」，才會保存。`
      : '沒有足夠明確的姓名線索；請手動輸入，模型不會強行猜測。';
  } catch (error) {
    if (currentMeetingId === meetingId) $('#speaker-status').textContent = `姓名推測失敗：${error.message}`;
  } finally {
    button.disabled = false;
  }
};

$('#analyze').onclick = async () => {
  if (!currentMeetingId) return;
  const meetingId = currentMeetingId;
  const button = $('#analyze');
  button.disabled = true;
  button.querySelector('span').textContent = '排隊等待摘要…';
  try {
    await api(`/api/meetings/${encodeURIComponent(meetingId)}/analyze`, { method: 'POST' });
    const job = await watchAnalysis(meetingId, button);
    if (job.status === 'failed') throw new Error(job.error || '摘要生成失敗');
    if (job.status === 'canceled') throw new Error(job.error || '摘要已中止');
    if (job.status === 'idle') throw new Error('找不到摘要工作狀態，服務可能已重啟；請重新整理頁面後再試。');
    if (currentMeetingId === meetingId) await loadMeeting(meetingId);
    else await loadMeetings();
  }
  catch (error) { window.alert(error.message); }
  finally { button.disabled = false; button.querySelector('span').textContent = '產生摘要與追蹤'; }
};

$('#export').onclick = () => { if (currentMeetingId) window.location.href = `/api/meetings/${currentMeetingId}`; };
$('#search-form').onsubmit = async (event) => {
  event.preventDefault();
  const query = $('#search').value.trim();
  if (!query) return;
  const type = $('#search-type').value;
  try {
    const rows = await api(`/api/search?q=${encodeURIComponent(query)}&type=${encodeURIComponent(type)}`);
    $('#meeting-view').hidden = true;
    $('#empty').hidden = true;
    $('#search-results').hidden = false;
    $('#result-count').textContent = `${rows.length} 筆`;
    const labels = { content: '會議內文', summary: '會議總結', action: '待辦事項' };
    $('#results').innerHTML = rows.length ? rows.map((row) => `
      <button class="result-row" data-id="${escapeHtml(row.meeting_id)}"><span class="result-type">${labels[row.category] || '會議記憶'}</span><span class="result-main"><strong>${escapeHtml(row.title)}</strong><span>${escapeHtml(row.content)}</span><small>${escapeHtml(formatDate(row.meeting_time))} · 相似度 ${(row.score * 100).toFixed(0)}%</small></span><svg viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></svg></button>`).join('') : '<p class="empty-list">找不到相關記憶，試試不同關鍵字。</p>';
    document.querySelectorAll('.result-row').forEach((button) => { button.onclick = () => loadMeeting(button.dataset.id); });
  } catch (error) { $('#result-count').textContent = error.message; }
};

api('/health').then(() => { $('#health').classList.add('online'); $('#health').lastChild.textContent = '系統已連線'; }).catch(() => { $('#health').lastChild.textContent = '服務未連線'; });

async function findActiveMeeting(meetings) {
  const states = await Promise.all(meetings.map(async (meeting) => {
    const suffixes = ['analysis-status', 'transcription-status', 'speaker-suggestions-status'];
    const jobs = await Promise.all(suffixes.map(async (suffix) => {
      try { return await api(`/api/meetings/${encodeURIComponent(meeting.id)}/${suffix}`); } catch (_error) { return null; }
    }));
    return jobs.some((job) => job && activeJobStatuses.has(job.status)) ? meeting : null;
  }));
  return states.find(Boolean) || null;
}

(async () => {
  try {
    const meetings = await loadMeetings();
    let rememberedId = null;
    try { rememberedId = window.localStorage.getItem(currentMeetingStorageKey); } catch (_error) { /* storage may be disabled */ }
    const remembered = meetings.find((meeting) => meeting.id === rememberedId);
    const active = await findActiveMeeting(meetings);
    const target = active || remembered;
    if (target) await loadMeeting(target.id);
  } catch (_error) {
    $('#meeting-list').innerHTML = '<p class="empty-list">無法載入會議記錄</p>';
  }
})();
