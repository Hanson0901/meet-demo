const $ = (selector) => document.querySelector(selector);
const token = () => localStorage.getItem('meeting-admin-token') || '';
const api = async (url, options = {}) => {
  options.headers = { 'X-Admin-Token': token(), ...(options.headers || {}) };
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '操作失敗');
  return data;
};
const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char]));
const labels = { meetings: '會議記錄', speakers: '說話人', utterances: '逐字稿段落', actions: '待辦事項', decisions: '會議決議', follow_ups: '前次待辦追蹤', logs: '處理記錄', evaluations: '模型評估', threads: '會議串', rag_content: '內文向量', rag_summaries: '摘要向量', rag_actions: '待辦向量' };

async function load() {
  try {
    const [stats, meetings] = await Promise.all([api('/api/admin/stats'), api('/api/meetings')]);
    $('#stats').innerHTML = Object.entries(stats).map(([key, value]) => `<div class="stat-card"><strong>${value}</strong><small>${labels[key] || escapeHtml(key)}</small></div>`).join('');
    $('#meetings').innerHTML = meetings.length ? meetings.map((meeting) => `
      <div class="meeting-row"><div><strong class="meeting-title">${escapeHtml(meeting.title)}</strong><small class="meeting-id">${escapeHtml(meeting.id)}</small></div><small class="meeting-date">${escapeHtml(new Date(meeting.meeting_time || meeting.created_at).toLocaleString())}</small><button class="delete-button" data-id="${escapeHtml(meeting.id)}">刪除</button></div>`).join('') : '<p class="empty-state">目前沒有會議記錄。</p>';
    document.querySelectorAll('.delete-button').forEach((button) => { button.onclick = () => removeMeeting(button.dataset.id); });
  } catch (error) { $('#message').textContent = error.message; }
}

async function removeMeeting(id) {
  if (!window.confirm('確定刪除此會議及其逐字稿、RAG 索引、待辦、決議與評估紀錄？')) return;
  try { await api(`/api/admin/meetings/${encodeURIComponent(id)}`, { method: 'DELETE' }); $('#message').textContent = '會議及關聯資料已刪除'; await load(); }
  catch (error) { $('#message').textContent = error.message; }
}

$('#token').value = token();
$('#save-token').onclick = () => { localStorage.setItem('meeting-admin-token', $('#token').value); $('#message').textContent = 'Token 已儲存'; load(); };
$('#purge').onclick = async () => {
  if (!window.confirm('確定清除全部資料？此操作無法復原。')) return;
  if (window.prompt('請輸入 DELETE_ALL 確認') !== 'DELETE_ALL') return;
  try { const result = await api('/api/admin/purge', { method: 'POST' }); $('#message').textContent = `已清除 ${result.deleted_meetings} 場會議與全部索引`; await load(); }
  catch (error) { $('#message').textContent = error.message; }
};
load();
