# Meeting Assistant

依 `architecture/` 文件建立的模組化會議助手。ASR 與本地 llama.cpp 模型延遲載入；OpenCC 負責將辨識文字正規化為台灣繁體中文。

## 使用

```bash
python -m meeting_assistant.cli transcribe recording.wav
python -m meeting_assistant.cli web
```

Web server 預設監聽 `0.0.0.0:8080`，區網裝置請使用主機的實際 IP 開啟，例如 `http://192.168.1.20:8080`；也可用 `--host` 與 `--port` 覆寫。

實際 FunASR 請使用 Python 3.10 環境啟動：

```bash
conda activate py3_10
python -m pip install -e ".[asr,llm]"
python -m meeting_assistant.cli web
```

流程分成兩個步驟：「開始整理」會建立會議並辨識音訊，之後按「產生摘要與追蹤」才呼叫 llama.cpp。設定 GGUF 模型後啟動：

```bash
# 若 GGUF 放在 ~/ 或專案 models/，程式會自動尋找；也可明確指定：
export LLAMA_CPP_MODEL=/home/cgu-csie/qwen3-4b-instruct-2507-q8_0.gguf
conda activate py3_10
python -m meeting_assistant.cli web
```

辨識文字會以 OpenCC `s2twp` 轉成台灣繁體中文後顯示及索引，資料庫仍保留原始 ASR 文字供核對。說話者人名可由逐字稿中的自我介紹／稱呼線索先提供候選，也可用「本地模型推測姓名」按鈕讓 llama.cpp 分析對話線索；候選不會自動確認，編輯後按「確認姓名」才會同步更新逐字稿並建立聲紋姓名資料。

啟用預設的 FunASR `cam++` 說話者模型後，系統會取每位說話者的 CAM++ embedding，在本機 SQLite 中比對之前已確認的聲紋。相似度達門檻且沒有近似的第二候選時，會直接套用先前確認的姓名；不確定時只顯示候選，不會自動改名。比對門檻可用 `FUNASR_SPK_MATCH_THRESHOLD` 調整（預設 `0.74`，範圍 `0.50`–`0.99`）。短於約 1.5 秒的說話樣本不建立聲紋，以免將不可靠片段當成身份依據。聲紋向量僅保存在本機資料庫，不會送到遠端；刪除最後一場關聯會議或管理頁清除資料時，孤立的聲紋也會刪除。

姓名推測按鈕與會議摘要共用本地模型，但採獨立工作並避免與摘要同時佔用模型；它不會在 ASR 時自動執行，也不會呼叫遠端 API。

本地摘要依輸入長度採兩種方式：短逐字稿可一次分析；較長逐字稿會以 GGUF tokenizer 按發言切段，逐段抽取事實、原文證據、決議與待辦，再在上下文預算內分批合併，最後生成正式摘要。若模型輸出遭截斷、JSON 不完整或合併內容仍超出上下文，系統會中止該次保存，不把部分結果當成完整摘要。長摘要在背景執行，網頁會輪詢處理階段，因此不必讓一次 HTTP 請求等待整段推理。

摘要階段預設參數如下，可依模型能力調整；上下文長度不要設得超過 GGUF 模型實際支援值。

| 環境變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `LLAMA_CPP_CONTEXT` | `16384` | llama.cpp 上下文長度（依本機記憶體實測調整） |
| `LLAMA_CPP_DIRECT_MAX_TOKENS` | `2400` | 低於此完整 prompt token 數時採單次摘要 |
| `LLAMA_CPP_CHUNK_TOKENS` | `1800` | 長逐字稿每段目標 token 數 |
| `LLAMA_CPP_MAP_MAX_TOKENS` | `3072` | 每段事實抽取的輸出上限 |
| `LLAMA_CPP_REDUCE_INPUT_TOKENS` | `4096` | 每批分層合併的輸入目標，會依 context 再限制 |
| `LLAMA_CPP_REDUCE_MAX_TOKENS` | `4096` | 分層合併的輸出上限 |
| `LLAMA_CPP_MAX_TOKENS` | `4096` | 最終摘要輸出上限 |

每段輸入目標預設 1800 tokens，以降低單段抽取輸出被截斷的機率。合併階段預設每批最多約 4096 input tokens；輸出遭截斷時會自動把批次再拆小重試。逐段與合併結果會逐批寫入 SQLite，失敗或服務重啟後再次按「產生摘要」會跳過已完成的同一批次；正式摘要仍只會在最後完成後保存。成功保存摘要後會清除該場檢查點。上述預設值可用同名環境變數覆蓋；所有階段仍會依目前 context 檢查輸入、預留輸出與安全空間。

管理頁 `/admin` 可在 8192–16384 間調整本地模型 context，也可調整直接摘要門檻、每段輸入／輸出、合併批次輸入／輸出及最終摘要輸出 token。設定會保存在 SQLite 的應用設定中，重啟後仍保留，無須重啟 Web 服務；token 設定自下一次分析生效，正在執行的分析沿用開始時的設定。context 調整則會在背景套用，已載入的 llama.cpp 模型會重新載入；若會議摘要正在執行，會等摘要完成後才重載。管理頁已保存的設定優先於對應環境變數。

本地摘要依輸入長度採兩種方式：短逐字稿可一次分析；較長逐字稿會以 GGUF tokenizer 按發言切段，逐段抽取事實、原文證據、決議與待辦，再在上下文預算內分批合併，最後生成正式摘要。若模型輸出遭截斷、JSON 不完整或合併內容仍超出上下文，系統會中止該次保存，不把部分結果當成完整摘要。長摘要在背景執行，網頁會輪詢處理階段，因此不必讓一次 HTTP 請求等待整段推理。

摘要階段預設參數如下，可依模型能力調整；上下文長度不要設得超過 GGUF 模型實際支援值。

| 環境變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `LLAMA_CPP_CONTEXT` | `16384` | llama.cpp 上下文長度（依本機記憶體實測調整） |
| `LLAMA_CPP_DIRECT_MAX_TOKENS` | `2400` | 低於此完整 prompt token 數時採單次摘要 |
| `LLAMA_CPP_CHUNK_TOKENS` | `1800` | 長逐字稿每段目標 token 數 |
| `LLAMA_CPP_MAP_MAX_TOKENS` | `3072` | 每段事實抽取的輸出上限 |
| `LLAMA_CPP_REDUCE_INPUT_TOKENS` | `4096` | 每批分層合併的輸入目標，會依 context 再限制 |
| `LLAMA_CPP_REDUCE_MAX_TOKENS` | `4096` | 分層合併的輸出上限 |
| `LLAMA_CPP_MAX_TOKENS` | `4096` | 最終摘要輸出上限 |

每段輸入目標預設 1800 tokens，以降低單段抽取輸出被截斷的機率。合併階段預設每批最多約 4096 input tokens；輸出遭截斷時會自動把批次再拆小重試。逐段與合併結果會逐批寫入 SQLite，失敗或服務重啟後再次按「產生摘要」會跳過已完成的同一批次；正式摘要仍只會在最後完成後保存。成功保存摘要後會清除該場檢查點。上述預設值可用同名環境變數覆蓋；所有階段仍會依目前 context 檢查輸入、預留輸出與安全空間。

管理頁 `/admin` 可在 8192–16384 間調整本地模型 context，也可調整直接摘要門檻、每段輸入／輸出、合併批次輸入／輸出及最終摘要輸出 token。設定會保存在 SQLite 的應用設定中，重啟後仍保留，無須重啟 Web 服務；token 設定自下一次分析生效，正在執行的分析沿用開始時的設定。context 調整則會在背景套用，已載入的 llama.cpp 模型會重新載入；若會議摘要正在執行，會等摘要完成後才重載。管理頁已保存的設定優先於對應環境變數。

新會議在按下「開始整理」時記錄開始時間，這會作為未能從逐字稿判定會議日期時的預設日期時間。網頁不需要手動命名；摘要成功產生主題後會以主題重新命名，在此之前使用「整理中 · 日期時間」暫名。摘要會索引逐字稿、摘要與待辦三類資料，並以主題名稱關聯向量；相關會議會組成會議串，摘要也會追蹤舊會議待辦是否已完成。

RAG 使用標準函式庫實作的 hashed n-gram 向量，無需下載額外 embedding 模型。這適合本機輕量檢索，但語意品質不及專用 embedding 模型。資料表為 `rag_content`、`rag_summaries`、`rag_actions`；啟動時會替有既有摘要的舊會議補建索引，不會刪除原會議資料。

管理員頁面位於 `http://localhost:8080/admin`，可刪除單場會議或清除全部資料。若服務不是只在可信任的內網使用，請設定管理員 token：

```bash
export MEETING_ADMIN_TOKEN='請換成長且隨機的字串'
python -m meeting_assistant.cli web
```

Web API：`POST /api/transcribe` 支援網頁 multipart 上傳，或 JSON 傳入 `{"audio":"/path/file.wav","started_at":"ISO 日期時間"}`；`POST /api/meetings/{meeting_id}/analyze` 排入背景摘要工作，搭配 `GET /api/meetings/{meeting_id}/analysis-status` 查詢進度；`GET /api/search?q=關鍵字&type=content|summary|action|all`；`GET /api/meetings/{meeting_id}` 讀取完整記錄。

管理頁模型設定使用 `GET /api/admin/config` 查詢目前 context、摘要 token 設定和套用狀態。`PUT /api/admin/config` 傳入 `{"context_tokens":16384}` 排入 context 調整工作；傳入 `{"token_settings":{"chunk_tokens":1800,"map_max_tokens":3072,"reduce_input_tokens":4096,"reduce_max_tokens":4096,"final_max_tokens":4096,"direct_max_tokens":2400}}` 可保存摘要 token 設定。兩個端點都使用 `X-Admin-Token` 驗證。

## 模型評估

先在會議頁執行一次本地 llama.cpp 分析，再開啟 `http://localhost:8080/evaluate`。評估頁會重用資料庫裡已保存的本地摘要、待辦與決議，只透過遠端 `gpt-6-luna` Responses API 產生遠端摘要並評審比較，不會重新執行本地模型。結果會保存供回看。遠端 API key 的環境變數名稱是 `CGULLM_API_KEY`。在 Raspberry Pi 編輯 `~/.bashrc`，加入：

```bash
export CGULLM_API_KEY='在此填入 API key'
export CGULLM_BASE_URL='https://air.cgu.edu.tw/cgullmapi/v1'
export CGULLM_MODEL='gpt-6-luna'
```

儲存後執行 `source ~/.bashrc`，再重啟 Meeting Assistant。金鑰只由後端讀取，不會送到瀏覽器。評審分數是模型估計值，不等於人工標註的客觀正確率。

## Nginx 反向代理

Nginx 範例設定位於 `deploy/nginx-meeting-assistant.conf`，為已有 Let's Encrypt 憑證的 `meet.hyang.icu` 設定 HTTP 轉 HTTPS，並反向代理到本機 8080。以獨立虛擬主機安裝，避免與其他網域的 SSL 憑證設定衝突：

```bash
sudo cp deploy/nginx-meeting-assistant.conf /etc/nginx/sites-available/meet-hyang-icu
sudo ln -s /etc/nginx/sites-available/meet-hyang-icu /etc/nginx/sites-enabled/meet-hyang-icu
sudo nginx -t
sudo systemctl reload nginx
```

啟用前，確認舊的 Nginx 設定沒有重複宣告 `server_name meet.hyang.icu`。保留 `meet-helper.hyang.icu` 自己的 server block 與憑證路徑。憑證到期時間與自動更新由 Certbot 管理。
