# 工具層

## 職責
提供給系統與使用者呼叫的標準工具，讓會議資料能被處理、轉換與輸出。

## 建議 tools
- `transcribe_meeting(audio)`
- `convert_traditional(text)`
- `summarize_meeting(meeting_id)`
- `extract_actions(meeting_id)`
- `extract_decisions(meeting_id)`
- `query_meeting_memory(query)`
- `export_minutes(meeting_id, format)`

## 工具原則
- 每個 tool 單一職責
- 每個 tool 回傳結構化資料
- 每個 tool 都要有錯誤碼與 log

## 轉繁體工具
`convert_traditional(text)` 建議獨立成純文字處理模組，避免和 ASR 或 LLM 耦合。

