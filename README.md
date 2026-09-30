# Meeting Assistant

依 `architecture/` 八份文件建立的模組化會議助手。核心只使用 Python 標準函式庫；FunASR 與 Gemma adapter 可在部署時替換，不會把模型依賴耦合到資料庫或 UI。

## 使用

```bash
python -m meeting_assistant.cli transcribe recording.wav
python -m meeting_assistant.cli web
```

Web server 預設監聽 `0.0.0.0:8080`，區網裝置請使用主機的實際 IP 開啟，例如 `http://192.168.1.20:8080`；也可用 `--host` 與 `--port` 覆寫。

實際 FunASR 請使用 Python 3.10 環境啟動：

```bash
conda activate py3_10
python -m meeting_assistant.cli web
```

流程分成兩個步驟：「開始整理」會建立會議並辨識音訊，之後按「產生摘要與追蹤」才呼叫 llama.cpp。設定 GGUF 模型後啟動：

```bash
# 若 GGUF 放在 ~/ 或專案 models/，程式會自動尋找；也可明確指定：
export LLAMA_CPP_MODEL=/home/cgu-csie/qwen3-4b-instruct-2507-q8_0.gguf
conda activate py3_10
python -m meeting_assistant.cli web
```

辨識完成後可在網頁編輯 speaker 人名，按「確認人名」會同步更新逐字稿；確認後按「產生摘要與追蹤」執行 llama.cpp 分析。

新會議在按下「開始整理」時記錄開始時間，這會作為未能從逐字稿判定會議日期時的預設日期時間。網頁不需要手動命名；摘要成功產生主題後會以主題重新命名，在此之前使用「整理中 · 日期時間」暫名。摘要會索引逐字稿、摘要與待辦三類資料，並以主題名稱關聯向量；相關會議會組成會議串，摘要也會追蹤舊會議待辦是否已完成。

RAG 使用標準函式庫實作的 hashed n-gram 向量，無需下載額外 embedding 模型。這適合本機輕量檢索，但語意品質不及專用 embedding 模型。資料表為 `rag_content`、`rag_summaries`、`rag_actions`；啟動時會替有既有摘要的舊會議補建索引，不會刪除原會議資料。

管理員頁面位於 `http://localhost:8080/admin`，可刪除單場會議或清除全部資料。若服務不是只在可信任的內網使用，請設定管理員 token：

```bash
export MEETING_ADMIN_TOKEN='請換成長且隨機的字串'
python -m meeting_assistant.cli web
```

Web API：`POST /api/transcribe` 支援網頁 multipart 上傳，或 JSON 傳入 `{"audio":"/path/file.wav","started_at":"ISO 日期時間"}`；`POST /api/meetings/{meeting_id}/analyze` 產生摘要；`GET /api/search?q=關鍵字&type=content|summary|action|all`；`GET /api/meetings/{meeting_id}` 讀取完整記錄。

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
