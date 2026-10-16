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
      <span class="meeting-item-copy"><strong title="${escapeHtml(meeting.title)}">${escapeHtml(meeting.title)}</strong><small>${escapeHtml(formatDate(meeting.meeting_time || meeting.created_at))}</small></span>
      <svg class="chevron" viewBox="0 0 24 24"><path d="m9 5 7 7-7 7"/></svg>
    </button>`).join('') : '<p class="empty-list">還沒有會議記錄</p>';
  document.querySelectorAll('.meeting-item').forEach((button) => { button.onclick = () => loadMeeting(button.dataset.id); });
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

$('#suggest-speakers').onclick = async () => {
  if (!currentMeetingId) return;
  const meetingId = currentMeetingId;
  const button = $('#suggest-speakers');
  button.disabled = true;
  $('#speaker-status').textContent = '本地模型正在尋找自我介紹與稱呼線索；此操作不會送到遠端服務。';
  try {
    await api(`/api/meetings/${encodeURIComponent(meetingId)}/speaker-suggestions`, { method: 'POST' });
    let job;
    while (true) {
      job = await api(`/api/meetings/${encodeURIComponent(meetingId)}/speaker-suggestions-status`);
      if (job.status === 'completed') break;
      if (job.status === 'failed') throw new Error(job.error || '本地模型推測失敗');
      if (job.status === 'idle') throw new Error('找不到推測工作狀態，請重新整理後再試。');
      if (currentMeetingId === meetingId) $('#speaker-status').textContent = job.stage || '本地模型分析中…';
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
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
    while (true) {
      const job = await api(`/api/meetings/${encodeURIComponent(meetingId)}/analysis-status`);
      if (job.status === 'completed') break;
      if (job.status === 'failed') throw new Error(job.error || '摘要生成失敗');
      if (job.status === 'idle') throw new Error('找不到摘要工作狀態，服務可能已重啟；請重新整理頁面後再試。');
      button.querySelector('span').textContent = job.stage || '分段摘要進行中…';
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
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
loadMeetings().catch(() => { $('#meeting-list').innerHTML = '<p class="empty-list">無法載入會議記錄</p>'; });
