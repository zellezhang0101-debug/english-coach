# 我的专属 C1 英语教练（Flask 最小示例）

这是一个**最简单的 Python Flask 网页应用**，只有一个首页，标题是「我的专属 C1 英语教练」。

下面的步骤会尽量用「小白友好」的方式说明。

---

## 1. 准备环境（只做一次）

1. **确认你有 Python 3.9+**

   在终端里输入：

   ```bash
   python3 --version
   ```

   看到的版本号需要是 3.9、3.10、3.11 或以上。

2. **在当前项目创建虚拟环境（推荐）**

   ```bash
   cd "/Users/zelle01.zhang/English learning"
   python3 -m venv .venv
   ```

3. **激活虚拟环境**

   ```bash
   source .venv/bin/activate
   ```

   激活成功后，命令行前面一般会多出一个 `(.venv)` 前缀。

---

## 2. 安装依赖

在终端、并且已经 `cd` 到项目目录后，运行：

```bash
pip install -r requirements.txt
```

这一步会安装 Flask、用于访问 DeepSeek 的 OpenAI SDK，以及 Gemini 的 SDK。

---

## 3. 运行 Flask 应用

同样在项目目录下运行：

```bash
python app.py
```

如果一切正常，你会看到类似：

```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

这时用浏览器打开：

```text
http://127.0.0.1:5000
```

你就能看到标题为「我的专属 C1 英语教练」的首页啦。

---

## 5. 生产环境部署（推荐 Docker）

### 5.1 准备环境变量

1) 复制示例文件：

```bash
cd "/Users/zelle01.zhang/English learning"
cp .env.example .env
```

2) 编辑 `.env`，至少填上：

- `FLASK_SECRET_KEY`（建议 32+ 位随机字符串）
- 以及你要用的 `GEMINI_API_KEY` / `DEEPSEEK_API_KEY`（可只填一个）

### 5.2 用 Docker Compose 启动（生产 WSGI：Gunicorn）

```bash
cd "/Users/zelle01.zhang/English learning"
docker compose up -d --build
```

启动后访问：

```text
http://服务器IP:8000
```

### 5.3 数据持久化

应用会把 `vocab_book.json`、`daily_practice.json` 等数据写到容器的 `/data`（由 compose 的 `app_data` volume 持久化）。

### 5.4 更新上线

```bash
cd "/Users/zelle01.zhang/English learning"
docker compose up -d --build
```

---

## 6. 部署到云平台（Render / Fly.io / 阿里云）

### 6.1 Render（最省事）

- **服务类型**：Web Service（Docker）
- **端口**：Render 会提供 `PORT` 环境变量，本项目镜像已支持自动监听 `$PORT`。
- **环境变量**（在 Render 控制台设置）：
  - `FLASK_SECRET_KEY`（必填）
  - `GEMINI_API_KEY` / `DEEPSEEK_API_KEY`（按需）
  - `DATA_DIR=/data`（建议）
- **数据持久化**：
  - 如果你希望生词本/历史记录不丢，需要在 Render 开启 Persistent Disk，并挂载到 `/data`

### 6.2 Fly.io（更灵活）

- 用 Fly CLI 直接从 `Dockerfile` 部署
- 需要在 Fly 的 secrets 里设置：
  - `FLASK_SECRET_KEY`
  - `GEMINI_API_KEY` / `DEEPSEEK_API_KEY`
  - （可选）`DATA_DIR=/data` 并配置 volume 做持久化

### 6.3 阿里云（两种常见方式）

- **ECS（最通用）**：在一台 ECS 上装 Docker，然后用本项目的 `docker-compose.yml` 跑（和本地几乎一样）
- **ACK/SAE**：同样用 `Dockerfile` 构建镜像后部署，环境变量与 `DATA_DIR` 逻辑一致

---

## 3.1（可选）本地运行 Ming-UniAudio（用于 Gemini 失败时自动降级）

本项目已支持在 Gemini 无法调用时，自动降级到 **Ming-UniAudio**（通过一个本地/远程 HTTP 服务）。

### A. 本地先跑一个 Ming-UniAudio 服务（建议在有 GPU 的机器）

1. 克隆 Ming-UniAudio（建议放到项目同级目录，或任意目录都行）：

```bash
cd "/Users/zelle01.zhang"
git clone https://github.com/inclusionAI/Ming-UniAudio.git
```

2. 按 Ming-UniAudio 官方 README 准备环境 + 下载模型权重（模型很大，且通常需要 GPU）。

3. 在本项目安装一个轻量的 API server 依赖（只给 `ming_uniaudio_server.py` 用）：

```bash
cd "/Users/zelle01.zhang/English learning"
.venv/bin/pip install -r requirements_ming_uniaudio_server.txt
```

4. 启动 Ming-UniAudio HTTP 服务（默认端口 8001）：

```bash
export MING_UNIAUDIO_REPO_PATH="/Users/zelle01.zhang/Ming-UniAudio"
export MING_UNIAUDIO_MODEL_PATH="/path/to/inclusionAI/Ming-UniAudio-16B-A3B"
export MING_UNIAUDIO_DEVICE="cuda:0"

# TTS 需要 prompt wav + prompt text（请你准备一个示例音频和对应文本）
export MING_UNIAUDIO_PROMPT_WAV="/path/to/prompt.wav"
export MING_UNIAUDIO_PROMPT_TEXT="This is the transcript of the prompt audio."

cd "/Users/zelle01.zhang/English learning"
.venv/bin/uvicorn ming_uniaudio_server:APP --host 127.0.0.1 --port 8001
```

5. 在运行主 Flask 应用前，设置 fallback 地址：

```bash
export MING_UNIAUDIO_URL="http://127.0.0.1:8001"
```

然后重启 Flask（`Ctrl+C` 后重新运行 `python app.py`）。

### B. 后续迁移到远程服务

把 `ming_uniaudio_server.py` 同样部署到远程 GPU 机器上（`--host 0.0.0.0` 并配好安全策略/反代），然后把本机 Flask 的：

```bash
export MING_UNIAUDIO_URL="http://你的远程地址:8001"
```

即可完成迁移，不需要改 Flask 代码。

---

## 4. 使用「阅读生成」功能

现在首页已经变成一个**阅读生成器**，并且可以选择用 DeepSeek 或 Gemini 生成文章：

- 在输入框里输入你感兴趣的主题（可以是中文或英文，例如 `太空探索` 或 `space exploration`）
- 点击「生成文章」
- 稍等几秒，就会出现一篇约 300 词的英文短文
- 文中会有 5 个用 `<mark>` 高亮的 B2–C1 级别词汇
- 文章下方会列出这 5 个词的**中文释义**

### 4.1 配置 DeepSeek API 密钥（可选）

要让大模型正常工作，你需要有一个 DeepSeek 的 API Key，并把它设置到环境变量里。

1. 在浏览器打开并登录（需要自己注册账号）：

   `https://platform.deepseek.com/`

2. 在网站里创建一个 API Key（注意保存好）。

3. 回到终端，在运行应用之前先执行（把 `YOUR_API_KEY_HERE` 换成你自己的）：

   ```bash
   export DEEPSEEK_API_KEY="YOUR_API_KEY_HERE"
   ```

   只要当前终端窗口还开着，这个设置就有效。

4. 然后再运行：

   ```bash
   python app.py
   ```

如果你在页面上选择 DeepSeek 却生成失败，页面底部会提示你检查 `DEEPSEEK_API_KEY`。

### 4.2 配置 Gemini API 密钥（可选）

如果你想用 Gemini 生成文章，需要有一个 Gemini 的 API Key（Google 官方 Gemini API）。

1. 在浏览器打开并登录（需要自己注册账号）：

   `https://ai.google.dev/`

2. 按页面指引创建 API Key。

3. 回到终端，在运行应用之前先执行（把 `YOUR_GEMINI_KEY_HERE` 换成你自己的）：

   ```bash
   export GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
   ```

4. 然后再运行（或重启）应用：

   ```bash
   python app.py
   ```

在首页选择 Gemini，然后生成文章即可。

---

## 5. 生词本（新增）

当你在文章里查词后，可以点击「添加到生词本」，单词会被保存到本地文件 `vocab_book.json`。

生词本页面地址：

```text
http://127.0.0.1:5000/vocab
```

它会展示你保存过的 **单词、中文释义、以及当时捕获的原文句子**。

---

## 6. 历史文章管理

每次在首页成功生成一篇文章后，会自动保存到本地历史（`article_history.json`）。

- **历史文章列表**：`http://127.0.0.1:5000/history`
- 支持按 **日期**、**主题关键词**、**难度**（如 C1、B2）筛选，方便回顾。
- 点击某篇的「查看」可进入该篇的完整内容（正文 + 高亮词汇释义）。

---

## 6.1 阅读推荐（每日 3 篇）

- **阅读推荐页**：`http://127.0.0.1:5000/reading`
- 每日自动从三个来源各爬取 1 篇英文文章：**21voa.com**、**i21st.cn**、**Buzzing.cc（hn.buzzing.cc）**。
- 每篇文章会经 AI 标注 B2–C1 词汇并生成释义，支持朗读、查词、加入生词本，并自动写入历史文章。

---

## 7. 下一步可以做什么？

- **慢慢加功能**：例如加一个简单的「单词卡片」功能、口语句子模板、听力材料链接等。
- **练习英文阅读**：可以尝试阅读 Flask 官方文档的入门部分，把看不懂的地方记下来问 AI。

如果你愿意，我也可以一步步带你：

- 改成多页面结构（首页 / 口语 / 写作等）
- 加上简单的用户数据（例如记录你每天学习了多久）
- 部署到网上，让手机也能访问
