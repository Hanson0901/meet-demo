# Meeting Assistant 目前工作流

本文依目前程式實作整理，描述網頁會議記錄從音訊上傳到檢索、模型評估與管理的流程。

## 一眼看懂

```mermaid
flowchart TD
    A[網頁選擇音訊並按「開始整理」] --> B[記下 started_at 並上傳音訊]
    B --> C[建立暫名會議記錄]
    C --> D[切分音訊並執行 FunASR]
    D --> E[保存逐字稿與說話人]
    E --> F[使用者編輯並確認人名]
    F --> G[按「產生摘要與追蹤」]
    G --> H[RAG 檢索相關舊會議與待辦]
    H --> I[本地 llama.cpp 產生主題、摘要、待辦與追蹤]
    I --> J{摘要與主題均有效？}
    J -->|是| K[以主題命名並關聯會議串]
    K --> L[保存分析結果並建立三類向量索引]
    J -->|否| M[保留「整理中 · 日期時間」暫名]
    L --> N[網頁顯示與 RAG 搜尋]
    M --> N
    N --> O[模型評估：Responses API 產生遠端摘要並評審]
    N --> P[管理頁：刪除單場或清除資料庫]
```

## 會議建立與語音辨識

1. 使用者選擇音訊檔並按「開始整理」。瀏覽器記錄當下時間為 ISO `started_at`，以 multipart 表單送到 `POST /api/transcribe`。
2. 後端將音訊存至 `data/uploads/`，建立 UUID 會議記錄。初始標題為「整理中 · YYYY-MM-DD HH:mm」；`created_at` 與 `meeting_time` 先使用這次開始時間。
3. `AudioIngest` 將音訊切成片段，FunASR adapter 辨識逐字稿與說話人；辨識文字及繁體化文字一起保存至 SQLite。
4. 網頁顯示逐字稿與說話人。使用者可編輯姓名，按「確認人名」呼叫 `PUT /api/meetings/{meeting_id}/speakers`；逐字稿顯示時會以確認後的姓名取代 speaker ID。

語音辨識與 LLM 分成兩個操作：上傳完成只建立逐字稿，不會自動開始 llama.cpp 分析。

## 摘要、會議日期與自動命名

使用者按「產生摘要與追蹤」後，網頁呼叫 `POST /api/meetings/{meeting_id}/analyze`：

1. 後端讀取逐字稿，使用本地 llama.cpp 分析。模型依內容產生會議主題、摘要、待辦、決議，以及可能的會議日期時間。
2. 如果逐字稿明確提到會議日期或時間，模型應回傳 ISO 格式；否則使用建立記錄時保存的 `started_at`。無法解析的日期也會退回開始時間。
3. 摘要與主題都有效時，以主題更新會議名稱。若摘要或主題未成功產生，則保留「整理中 · 日期時間」暫名；模型呼叫失敗時網頁會收到錯誤，不會以錯誤結果覆蓋既有會議記錄。
4. 新主題會作為 topic key，並連結符合條件的舊會議。多場相關會議共用一個 thread；有多場會議時，模型可為會議串提供合併後的主軸名稱。
5. 保存摘要、待辦、決議與前次待辦追蹤結果，成功完成摘要後再建立三類 RAG 索引。

## RAG 檢索與前次待辦追蹤

RAG 資料目前存放於同一個 SQLite 資料庫的三張表中，並非三個獨立資料庫：

| 資料表 | 索引內容 | 用途 |
| --- | --- | --- |
| `rag_content` | 分段逐字稿 | 檢索會議內文 |
| `rag_summaries` | 會議摘要 | 檢索摘要 |
| `rag_actions` | 待辦事項 | 檢索待辦與期限、負責人 |

索引以生成的會議主題作為 `topic_key`，並保留對應 `meeting_id`。目前向量為無外部依賴的 512 維 hashed n-gram，搜尋以 cosine 相似度排序；它是輕量本機檢索，不等同於神經網路 embedding。

產生新摘要前，系統會用本次逐字稿檢索舊會議摘要、內文與待辦，將候選會議及其待辦交給本地模型。模型只應評估檢索資料中確實相關的舊會議，並把待辦標記為已完成、進行中或本次未提及，附上簡短結論。未明確關聯的候選不會自動加入會議串。

首頁提供「所有記憶、會議內文、會議總結、待辦事項」檢索選項。資料庫啟動時也會替有既有摘要、但尚未建立索引的舊會議補建 RAG 索引。

## 會議頁面與 API

| 用途 | 路徑 | 方法 |
| --- | --- | --- |
| 首頁與會議列表 | `/`、`/api/meetings` | GET |
| 上傳音訊與辨識 | `/api/transcribe` | POST |
| 確認說話人姓名 | `/api/meetings/{id}/speakers` | PUT |
| 產生摘要與追蹤 | `/api/meetings/{id}/analyze` | POST |
| 讀取完整會議記錄 | `/api/meetings/{id}` | GET |
| 搜尋 RAG 記憶 | `/api/search?q=...&type=content\|summary\|action\|all` | GET |
| 健康檢查 | `/health` | GET |

完整會議記錄頁會顯示會議名稱與日期、摘要、說話人、逐字稿、待辦、決議、前次待辦追蹤、所屬會議串及處理記錄。資料預設存於 `data/meetings.sqlite3`，上傳音訊存於 `data/uploads/`。

## 遠端模型評估

1. 先在會議頁完成一次本地 llama.cpp 摘要；評估頁不會重新執行本地模型。
2. 開啟 `/evaluate` 並選擇會議，按評估按鈕後呼叫 `POST /api/evaluations`。
3. 遠端 Responses API（預設 `gpt-6-luna`）產生另一份摘要；評審模型以原始逐字稿為準，匿名比較兩份結果的事實正確性、涵蓋度及待辦／決議準確性。
4. 評估結果保存於 `evaluations`，可於頁面查看歷史結果。評分是模型估計，不能視為人工標註的客觀正確率。

設定使用 `CGULLM_API_KEY`、`CGULLM_BASE_URL` 與 `CGULLM_MODEL` 環境變數。API key 只由後端讀取。

## 管理與部署

- 管理頁：`/admin`。可查看資料統計、刪除單場會議及相關索引，或清除全部資料。
- 若設定 `MEETING_ADMIN_TOKEN`，管理變更需透過 `X-Admin-Token` 驗證；未設定時不會要求 token。
- Web server 預設監聽 `0.0.0.0:8080`。在本機或區網可用主機 IP 加上 `:8080` 開啟；Nginx 可將 `meet.hyang.icu` 反向代理至本機 8080。
- 目前以 Python 3.10 環境在背景啟動時，可使用：

  ```bash
  cd ~/meet-demo
  nohup /home/cgu-csie/miniconda3/envs/py3_10/bin/python -m meeting_assistant.cli web --host 0.0.0.0 --port 8080 >> nohup.out 2>&1 < /dev/null &
  ```

  日誌會追加至 `nohup.out`；啟動後可查看 `/health` 確認服務狀態。
