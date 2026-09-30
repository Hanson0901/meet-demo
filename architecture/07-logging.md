# 即時 Log 層

## 職責
讓所有模組都能即時輸出可追蹤的處理狀態，方便除錯與監控。

## Log 規格
所有 log 統一使用前綴標示模組，例如：
- `[ASR]`
- `[DIAR]`
- `[NORM]`
- `[LLM]`
- `[DB]`
- `[UI]`

## 範例
```text
[ASR] 00:01:23.120-00:01:27.880 speaker=2 text=...
[DIAR] speaker=2 mapped=Alice
[NORM] zh-cn -> zh-tw done
[LLM] summary chunk=3 generated
[DB] utterance saved id=184
```

## 設計要求
- 即時輸出，不等整批結束才顯示
- 每筆 log 都可回溯到會議片段
- 錯誤與警告需分級

