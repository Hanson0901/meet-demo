const $ = (selector) => document.querySelector(selector);
let currentMeetingId = null;

const api = async (url, options) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '請求失敗');
  return data;
};
const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char]));
const formatDate = (value) => value ? new Date(value).toLocaleString('zh-TW', { dateStyle: 'medium', timeStyle: 'short' }) : '時間未設定';
const formatTime = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;

async function loadMeetings() {
  const meetings = await api('/api/meetings');
  $('#meeting-list').innerHTML = meetings.length ? meetings.map((meeting) => `
    <button class="meeting-item ${meeting.id === currentMeetingId ? 'selected' : ''}" data-id="${escapeHtml(meeting.id)}">
      <span class="meeting-item-icon"><svg viewBox="0 0 24 24"><path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/></svg></span>
      <span class="meeting-item-copy"><strong>${escapeHtml(meeting.title)}</strong><small>${escapeHtml(formatDate(meeting.meeting_time || meeting.created_at))}</small></span>
      <svg class="chevron" viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></svg>
    </button>`).join('') : '<p class="empty-list">還沒有會議記錄</p>';
  document.querySelectorAll('.meeting-item').forEach((button) => { button.onclick = () => loadMeeting(button.dataset.id); });
}

function renderSpeakers(speakers) {
  $('#speaker-list').innerHTML = speakers.length ? speakers.map((speaker) => `
    <label class="speaker-edit"><span class="speaker-id">${escapeHtml(speaker.speaker_id)}</span><input data-speaker="${escapeHtml(speaker.speaker_id)}" value="${escapeHtml(speaker.display_name || '')}" placeholder="輸入姓名"></label>`).join('') : '<p class="empty-list">沒有辨識到說話人</p>';
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
  const data = await api(`/api/meetings/${encodeURIComponent(meetingId)}`);
  currentMeetingId = meetingId;
  $('#empty').hidden = true;
  $('#search-results').hidden = true;
  $('#meeting-view').hidden = false;
  $('#meeting-title').textContent = data.meeting.title;
  $('#meeting-date').textContent = formatDate(data.meeting.meeting_time || data.meeting.created_at);
  $('#topic-chip').hidden = !data.topic;
  $('#topic-chip').textContent = data.topic?.display_name || '';
  $('#summary').textContent = data.meeting.summary || '摘要尚未產生';
  renderSpeakers(data.speakers || []);
  renderTranscript(data.transcript || []);
  renderActions(data.actions || []);
  renderDecisions(data.decisions || []);
  renderFollowUps(data.follow_ups || []);
  renderThread(data.topic, data.related_meetings || []);
  $('#logs').textContent = (data.logs || []).map((log) => `[${log.module}] ${log.level.toLowerCase()} ${log.message}`).join('\n') || '尚無處理記錄';
  await loadMeetings();
}

$('#upload-form').onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  $('#upload-status').textContent = '正在辨識音訊，請稍候…';
  try {
    $('#started-at').value = new Date().toISOString();
    const result = await api('/api/transcribe', { method: 'POST', body: new FormData(event.currentTarget) });
    $('#upload-status').textContent = '逐字稿已建立；完成摘要後會自動產生會議主題。';
    await loadMeeting(result.meeting_id);
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

$('#analyze').onclick = async () => {
  if (!currentMeetingId) return;
  const button = $('#analyze');
  button.disabled = true;
  button.querySelector('span').textContent = '正在整理主題與待辦…';
  try { await api(`/api/meetings/${currentMeetingId}/analyze`, { method: 'POST' }); await loadMeeting(currentMeetingId); }
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
loadMeetings().catch(() => { $('#meeting-list').innerHTML = '<p class="empty-list">無法載入會議記錄</p>'; });
