from flask import Flask, render_template_string, request, jsonify, Response, session
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import json
import os
import traceback
import io
import re
import wave
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, urljoin, urlparse
import tempfile
import time
import base64
import threading
from threading import Lock

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
try:
    # When running behind a reverse proxy (Caddy/Nginx), this helps Flask build correct URLs.
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
except Exception:
    pass

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MING_UNIAUDIO_URL = (os.getenv("MING_UNIAUDIO_URL") or "").strip().rstrip("/")
MING_UNIAUDIO_TIMEOUT = float(os.getenv("MING_UNIAUDIO_TIMEOUT") or "90")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "sk-placeholder",
    base_url="https://api.deepseek.com",
)

APP_DIR = os.path.dirname(__file__)
DATA_DIR = (os.getenv("DATA_DIR") or APP_DIR).strip()
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    # If DATA_DIR is invalid/unwritable, fall back to app directory.
    DATA_DIR = APP_DIR

VOCAB_BOOK_PATH = os.path.join(DATA_DIR, "vocab_book.json")
ARTICLE_HISTORY_PATH = os.path.join(DATA_DIR, "article_history.json")
RECOMMEND_DATA_PATH = os.path.join(DATA_DIR, "reading_recommendations.json")
DAILY_PRACTICE_PATH = os.path.join(DATA_DIR, "daily_practice.json")

TZ_BEIJING = timezone(timedelta(hours=8))


def _today_date():
    """今日日期（北京时间），用于每日缓存 key。"""
    return datetime.now(TZ_BEIJING).strftime("%Y-%m-%d")


CRAWL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
}


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>我的专属 B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e5e7eb;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
      }
      .card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 16px;
        padding: 28px 24px;
        max-width: 720px;
        width: 100%;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
      }
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: #5eead4;
        font-size: 12px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
      }
      .dot {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: #2dd4bf;
      }
      h1 {
        margin: 18px 0 8px;
        font-size: 26px;
        line-height: 1.25;
      }
      p {
        margin: 0;
        color: #9ca3af;
        font-size: 14px;
        line-height: 1.6;
      }
      .hint {
        font-size: 12px;
        color: #6b7280;
        margin-top: 14px;
      }
      code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        background: rgba(15, 23, 42, 0.7);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
      }
      .article-body p.pronunciation-highlight {
        background: rgba(250, 204, 21, 0.2);
        border-left: 4px solid #facc15;
        padding-left: 10px;
        margin-left: -10px;
        border-radius: 0 6px 6px 0;
      }
      .form-section {
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px dashed rgba(148, 163, 184, 0.6);
      }
      label {
        display: block;
        font-size: 13px;
        color: #e5e7eb;
        margin-bottom: 6px;
      }
      input[type="text"] {
        width: 100%;
        padding: 9px 11px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background: rgba(15, 23, 42, 0.7);
        color: #e5e7eb;
        font-size: 14px;
        outline: none;
      }
      input[type="text"]:focus {
        border-color: #22c55e;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4);
      }
      .button-row {
        margin-top: 14px;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }
      button.button-primary,
      a.button-secondary,
      a.button-link {
        text-decoration: none;
        padding: 9px 17px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: none;
        cursor: pointer;
      }
      button.button-primary {
        background: linear-gradient(135deg, #22c55e, #22d3ee);
        color: #020617;
        box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
        transition: transform 0.1s ease, box-shadow 0.1s ease, filter 0.1s ease;
      }
      button.button-primary:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
        box-shadow: 0 18px 40px rgba(34, 197, 94, 0.5);
      }
      a.button-secondary {
        background: rgba(15, 23, 42, 0.8);
        color: #e5e7eb;
        border: 1px solid rgba(148, 163, 184, 0.6);
      }
      a.button-link {
        padding-left: 0;
        padding-right: 0;
        color: #9ca3af;
        font-weight: 400;
      }
      .error {
        margin-top: 10px;
        font-size: 13px;
        color: #fecaca;
      }
      .article-section {
        margin-top: 22px;
        padding: 14px 12px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.35);
      }
      .article-section h2 {
        margin-top: 0;
        margin-bottom: 6px;
        font-size: 18px;
      }
      .article-body {
        font-size: 14px;
        line-height: 1.7;
        color: #e5e7eb;
      }
      .article-body mark {
        background: rgba(250, 204, 21, 0.2);
        color: #facc15;
        padding: 0 2px;
        border-radius: 3px;
      }
      .vocab-section {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px dashed rgba(148, 163, 184, 0.6);
      }
      .vocab-list {
        margin: 8px 0 0;
        padding-left: 16px;
        font-size: 13px;
        color: #e5e7eb;
      }
      .vocab-list li span.word {
        font-weight: 600;
        color: #facc15;
      }
      .vocab-list li span.meaning {
        color: #9ca3af;
      }
      .small {
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <main class="card">
      <div class="badge">
        <span class="dot"></span>
        <span>B1/B2/C1 English Coach</span>
        <span style="margin-left: 8px; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600;">当前难度 {{ difficulty|default('C1') }}</span>
      </div>
      <h1>每日练习</h1>
      <p>根据<strong>当前难度</strong>推荐一篇今日文章（优先爬取，无则 AI 生成）。完成朗读、总结与 3 个开放题练习。</p>
      <p class="small" style="margin-top: 6px;">
        <a class="button-link" href="/vocab">生词本</a>
        &nbsp;·&nbsp;
        <a class="button-link" href="/history">历史练习</a>
        &nbsp;·&nbsp;
        <a class="button-link" href="/speaking">发音练习</a>
        &nbsp;·&nbsp;
        <a class="button-link" href="/reading">阅读推荐</a>
      </p>
      <div class="provider-bar" style="margin-top: 12px; padding: 10px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型（全站统一）：</span>
        <button type="button" class="provider-btn {% if provider == 'gemini' %}active{% endif %}" data-provider="gemini" onclick="setProvider('gemini')" style="margin-left: 10px; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee){% else %}transparent{% endif %}; color: {% if provider == 'gemini' %}#020617{% else %}#e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" class="provider-btn {% if provider == 'deepseek' %}active{% endif %}" data-provider="deepseek" onclick="setProvider('deepseek')" style="margin-left: 6px; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee){% else %}transparent{% endif %}; color: {% if provider == 'deepseek' %}#020617{% else %}#e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 16px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
        <span class="small" style="margin-left: 8px; color: #6b7280;">生成文章、口语等均使用所选难度</span>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>

      <section class="form-section">
        <form method="post" action="/daily-practice" id="daily-practice-form">
          <label for="topic">可选：若无同难度爬取文章，AI 将按此主题生成（留空则用默认主题）</label>
          <input
            type="text"
            id="topic"
            name="topic"
            placeholder="例如：space exploration"
            value="{{ topic|default('') }}"
          />
          <div class="button-row">
            <button type="submit" class="button-primary" id="get-practice-btn">获取今日练习</button>
            <a href="/daily-practice" target="_blank" class="button-secondary" style="padding: 9px 17px;">直接打开今日练习</a>
          </div>
        </form>
      </section>
      <script>
      document.getElementById("daily-practice-form").addEventListener("submit", function(e) {
        e.preventDefault();
        var w = window.open("/daily-practice/loading", "daily_practice_win", "noopener");
        if (w) { this.target = "daily_practice_win"; this.submit(); }
        else { this.target = "_blank"; this.submit(); }
      });
      </script>

      <p class="hint">点击「获取今日练习」将生成并在新页面打开；若今日已有缓存，可点击「直接打开今日练习」。</p>
    </main>
  </body>
</html>
"""

# 今日练习独立页面模板（文章、朗读、查词、三项练习）
DAILY_PRACTICE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>今日练习 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 720px; width: 100%; margin: 0 auto; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      .article-body p.pronunciation-highlight { background: rgba(250, 204, 21, 0.2); border-left: 4px solid #facc15; padding-left: 10px; margin-left: -10px; border-radius: 0 6px 6px 0; }
      .article-section { margin-top: 0; padding: 14px 12px; border-radius: 12px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.35); }
      .article-section h2 { margin-top: 0; margin-bottom: 6px; font-size: 18px; }
      .article-body { font-size: 14px; line-height: 1.7; color: #e5e7eb; }
      .article-body mark { background: rgba(250, 204, 21, 0.2); color: #facc15; padding: 0 2px; border-radius: 3px; }
      .article-body p.tts-current {
        background: rgba(34, 211, 238, 0.14);
        border-left: 4px solid rgba(34, 211, 238, 0.95);
        padding-left: 10px;
        margin-left: -10px;
        border-radius: 0 6px 6px 0;
      }
      .vocab-section { margin-top: 14px; padding-top: 10px; border-top: 1px dashed rgba(148, 163, 184, 0.6); }
      .vocab-list { margin: 8px 0 0; padding-left: 16px; font-size: 13px; color: #e5e7eb; }
      .vocab-list li span.word { font-weight: 600; color: #facc15; }
      .vocab-list li span.meaning { color: #9ca3af; }
      .small { font-size: 12px; }
      button.button-primary, .btn { padding: 9px 17px; border-radius: 999px; font-weight: 600; font-size: 13px; border: none; cursor: pointer; }
      button.button-primary { background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; }
      button.button-primary.tts-playing { position: relative; overflow: hidden; }
      button.button-primary.tts-playing::after {
        content: "";
        position: absolute;
        top: 0;
        left: -35%;
        width: 35%;
        height: 100%;
        background: rgba(255, 255, 255, 0.22);
        transform: skewX(-18deg);
        animation: ttsSheen 1.1s linear infinite;
        pointer-events: none;
      }
      @keyframes ttsSheen {
        from { left: -35%; }
        to { left: 120%; }
      }
      .btn { background: rgba(148, 163, 184, 0.3); color: #e5e7eb; }
      .btn:disabled { opacity: 0.6; cursor: not-allowed; }
      a { color: #5eead4; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .error { font-size: 13px; color: #fecaca; margin-top: 10px; }
      .article-topbar {
        position: sticky;
        top: 0;
        z-index: 20;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: -14px -12px 10px;
        padding: 10px 12px;
        background: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px 12px 0 0;
      }
      .article-title { margin: 0; font-size: 18px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
      .article-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
      .article-actions .hint { color: #9ca3af; }
      .article-actions button.button-primary { padding: 6px 12px; font-size: 12px; }
      @media (max-width: 520px) {
        .article-topbar { flex-direction: column; align-items: flex-start; }
        .article-actions { justify-content: flex-start; width: 100%; }
      }
      @media (max-width: 520px) {
        body { padding: 10px; }
        .card { padding: 22px 16px; }
        .article-section { padding: 12px 10px; }
        .exercise-block { padding: 12px !important; }
        button.button-primary, .btn { padding: 10px 14px; font-size: 13px; }
      }
    </style>
  </head>
  <body>
    <main class="card">
      <p><a href="/">← 返回首页</a></p>
      {% if error %}
        <div class="error">{{ error|e }}</div>
      {% elif article_html %}
        <section class="article-section">
          <div class="article-topbar">
            <h2 class="article-title">今日阅读 <span style="display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600; margin-left: 6px;">{{ (article_difficulty|default(difficulty)|default('C1'))|e }}</span></h2>
            <div class="article-actions">
              <button id="tts-btn" type="button" class="button-primary" onclick="speakArticle()">全文朗读</button>
              <button type="button" class="button-primary" onclick="lookupSelection()">查选中单词</button>
              <button id="tap-lookup-btn" type="button" class="btn" style="padding: 6px 12px; font-size: 12px;" onclick="toggleTapLookup()">点词查词：关</button>
              <span class="small hint">先选中单词再点查词</span>
            </div>
          </div>
          <div class="article-body" id="article-body" data-article-b64="{{ article_html_b64|e }}"></div>
          <audio id="tts-audio" controls style="width: 100%; margin-top: 10px; display: none;"></audio>
          <div id="lookup-result" style="margin-top: 12px; padding: 10px; border-radius: 10px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.35); display: none;">
            <div id="lookup-word" style="font-weight: 600; color: #e5e7eb;"></div>
            <div id="lookup-phonetic" class="small" style="color: #9ca3af; margin-top: 2px;"></div>
            <div id="lookup-meaning" class="small" style="margin-top: 6px; color: #e5e7eb;"></div>
            <div id="lookup-example-en" class="small" style="margin-top: 8px; color: #9ca3af;"></div>
            <div id="lookup-example-zh" class="small" style="margin-top: 2px; color: #9ca3af;"></div>
            <div style="margin-top: 10px;">
              <button id="add-vocab-btn" type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px;" onclick="addToVocab()">添加到生词本</button>
              <span id="add-vocab-status" class="small" style="margin-left: 6px; color: #9ca3af;"></span>
            </div>
          </div>

          <div class="exercise-block" style="margin-top: 20px; padding: 14px; border-radius: 12px; background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.3);">
            <h3 style="margin: 0 0 10px; font-size: 15px;">（1）发音练习</h3>
            <p class="small" style="color: #94a3b8;">随机选一段话，朗读后提交可获得断句/重音/连读分析。</p>
            <div id="pronunciation-segment" class="small" style="padding: 10px; border-radius: 8px; background: rgba(15,23,42,0.8); margin: 8px 0; color: #cbd5e1; font-style: italic;">点击「换一段」随机选取</div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
              <button type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px;" onclick="pickPronunciationSegment()">换一段</button>
              <button type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px;" id="pronunciation-tts-btn" onclick="playPronunciationRef()" disabled>播放参考朗读</button>
              <button type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px;" id="pronunciation-record-btn" onclick="togglePronunciationRecord()">开始录音</button>
              <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px; display: none;" id="pronunciation-stop-btn" onclick="stopPronunciationRecord()">停止录音</button>
              <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="pronunciation-submit-btn" onclick="submitPronunciation()" disabled>提交模仿</button>
            </div>
            <div id="pronunciation-result" style="margin-top: 10px; padding: 10px; border-radius: 8px; background: rgba(15,23,42,0.9); display: none;">
              <div id="pronunciation-transcript" class="small" style="color: #e5e7eb;"></div>
              <div id="pronunciation-feedback" class="small" style="color: #e5e7eb; white-space: pre-wrap; margin-top: 6px;"></div>
            </div>
          </div>

          <div class="exercise-block" style="margin-top: 14px; padding: 14px; border-radius: 12px; background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.3);">
            <h3 style="margin: 0 0 10px; font-size: 15px;">（2）总结全文</h3>
            <p class="small" style="color: #94a3b8;">用英语录音总结文章要点，提交后获得发音与表达反馈。</p>
            <button type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px; margin-right: 8px;" id="summary-record-btn" onclick="toggleSummaryRecord()">开始录音</button>
            <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px; display: none;" id="summary-stop-btn" onclick="stopSummaryRecord()">停止录音</button>
            <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="summary-submit-btn" onclick="submitSummary()" disabled>提交</button>
            <div id="summary-result" style="margin-top: 10px; padding: 10px; border-radius: 8px; background: rgba(15,23,42,0.9); display: none;">
              <div id="summary-transcript" class="small" style="color: #e5e7eb;"></div>
              <div id="summary-feedback" class="small" style="color: #e5e7eb; white-space: pre-wrap; margin-top: 6px;"></div>
            </div>
          </div>

          <div class="exercise-block" style="margin-top: 14px; padding: 14px; border-radius: 12px; background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.3);">
            <h3 style="margin: 0 0 10px; font-size: 15px;">（3）开放性问题</h3>
            <p class="small" style="color: #94a3b8;">AI 生成 3 个与文章相关的问题，逐题录音作答。</p>
            <button type="button" class="button-primary" style="padding: 7px 14px; font-size: 12px; margin-bottom: 10px;" id="load-questions-btn" onclick="loadOpenQuestions()">生成 3 个问题</button>
            <div id="open-questions-list"></div>
          </div>

          {% if vocab_list %}
            <div class="vocab-section">
              <p class="small">高亮词汇及中文释义：</p>
              <ul class="vocab-list">
                {% for item in vocab_list %}
                  <li><span class="word">{{ item.word|e }}</span><span class="meaning"> — {{ item.meaning_zh|e }}</span></li>
                {% endfor %}
              </ul>
            </div>
          {% endif %}
        </section>
      {% else %}
        <p class="small" style="color: #9ca3af;">今日暂无练习，请从首页点击「获取今日练习」生成。</p>
      {% endif %}
    </main>
    <script>
      (function(){var el=document.getElementById("article-body");if(el&&el.dataset.articleB64){try{el.innerHTML=atob(el.dataset.articleB64);}catch(e){el.innerHTML='<p class="small" style="color:#9ca3af;">加载失败</p>';}}})();
      let lastLookup = null;
      let lastContextSentence = "";

      let pronunciationSegmentText = "";
      let pronunciationChunks = [];
      let pronunciationRecorder = null;
      let summaryChunks = [];
      let summaryRecorder = null;
      let openQuestionsList = [];
      let openQuestionRecorder = null;
      let openQuestionChunks = [];
      let openQuestionIdx = -1;

      function pickPronunciationSegment() {
        const articleEl = document.querySelector(".article-body");
        if (!articleEl) return;
        articleEl.querySelectorAll("p.pronunciation-highlight").forEach(function(el) { el.classList.remove("pronunciation-highlight"); });
        const paras = articleEl.querySelectorAll("p");
        if (!paras.length) {
          pronunciationSegmentText = (articleEl.innerText || "").trim().slice(0, 400);
        } else {
          const p = paras[Math.floor(Math.random() * paras.length)];
          pronunciationSegmentText = (p.innerText || "").trim();
          p.classList.add("pronunciation-highlight");
          p.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        if (!pronunciationSegmentText) {
          document.getElementById("pronunciation-segment").textContent = "无可用段落，请换一段。";
          return;
        }
        document.getElementById("pronunciation-segment").textContent = pronunciationSegmentText;
        document.getElementById("pronunciation-tts-btn").disabled = false;
        document.getElementById("pronunciation-submit-btn").disabled = false;
        document.getElementById("pronunciation-result").style.display = "none";
      }

      async function playPronunciationRef() {
        if (!pronunciationSegmentText) return;
        const btn = document.getElementById("pronunciation-tts-btn");
        btn.disabled = true;
        try {
          const r = await fetch("/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: pronunciationSegmentText }) });
          if (!r.ok) { const d = await r.json().catch(function(){}); alert(d.error || "播放失败"); btn.disabled = false; return; }
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          audio.onended = function() { URL.revokeObjectURL(url); btn.disabled = false; };
          await audio.play();
        } catch (e) { alert("播放失败"); btn.disabled = false; }
      }

      function togglePronunciationRecord() {
        if (pronunciationRecorder && pronunciationRecorder.state === "recording") { pronunciationRecorder.stop(); return; }
        if (!pronunciationSegmentText) { pickPronunciationSegment(); if (!pronunciationSegmentText) { alert("请先点击「换一段」选取要朗读的段落。"); return; } }
        pronunciationChunks = [];
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
          pronunciationRecorder = new MediaRecorder(stream);
          pronunciationRecorder.ondataavailable = function(e) { if (e.data.size) pronunciationChunks.push(e.data); };
          pronunciationRecorder.onstop = function() { stream.getTracks().forEach(function(t) { t.stop(); }); document.getElementById("pronunciation-record-btn").style.display = "inline-block"; document.getElementById("pronunciation-record-btn").textContent = "开始录音"; document.getElementById("pronunciation-stop-btn").style.display = "none"; };
          pronunciationRecorder.start();
          document.getElementById("pronunciation-record-btn").textContent = "停止录音";
          document.getElementById("pronunciation-stop-btn").style.display = "inline-block";
        }).catch(function() { alert("无法使用麦克风。"); });
      }
      function stopPronunciationRecord() { if (pronunciationRecorder && pronunciationRecorder.state === "recording") pronunciationRecorder.stop(); }

      async function submitPronunciation() {
        if (!pronunciationSegmentText || !pronunciationChunks.length) { alert("请先选取段落并完成录音后再提交。"); return; }
        var form = new FormData();
        form.append("text", pronunciationSegmentText);
        form.append("audio", new Blob(pronunciationChunks, { type: "audio/webm" }), "pronunciation.webm");
        var resultEl = document.getElementById("pronunciation-result");
        var transEl = document.getElementById("pronunciation-transcript");
        var feedEl = document.getElementById("pronunciation-feedback");
        resultEl.style.display = "block";
        transEl.textContent = ""; feedEl.textContent = "提交中…";
        try {
          var r = await fetch("/imitation/feedback", { method: "POST", body: form });
          var d = await r.json().catch(function() { return {}; });
          if (d.error) { feedEl.textContent = d.error; return; }
          transEl.textContent = d.transcript || "（无）";
          feedEl.textContent = d.feedback || "";
        } catch (e) { feedEl.textContent = "网络错误，请重试。"; }
      }

      function toggleSummaryRecord() {
        if (summaryRecorder && summaryRecorder.state === "recording") { summaryRecorder.stop(); return; }
        summaryChunks = [];
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
          summaryRecorder = new MediaRecorder(stream);
          summaryRecorder.ondataavailable = function(e) { if (e.data.size) summaryChunks.push(e.data); };
          summaryRecorder.onstop = function() { stream.getTracks().forEach(function(t) { t.stop(); }); document.getElementById("summary-record-btn").style.display = "inline-block"; document.getElementById("summary-stop-btn").style.display = "none"; document.getElementById("summary-record-btn").textContent = "开始录音"; };
          summaryRecorder.start();
          document.getElementById("summary-record-btn").textContent = "停止录音";
          document.getElementById("summary-stop-btn").style.display = "inline-block";
        }).catch(function() { alert("无法使用麦克风。"); });
      }
      function stopSummaryRecord() { if (summaryRecorder && summaryRecorder.state === "recording") summaryRecorder.stop(); }

      async function submitSummary() {
        if (!summaryChunks.length) { alert("请先完成录音后再提交。"); return; }
        var topic = "Summarize the main points of the article you just read in English.";
        var form = new FormData();
        form.append("topic", topic);
        form.append("audio", new Blob(summaryChunks, { type: "audio/webm" }), "summary.webm");
        var resultEl = document.getElementById("summary-result");
        var transEl = document.getElementById("summary-transcript");
        var feedEl = document.getElementById("summary-feedback");
        resultEl.style.display = "block";
        transEl.textContent = ""; feedEl.textContent = "提交中…";
        try {
          var r = await fetch("/speaking/feedback", { method: "POST", body: form });
          var d = await r.json().catch(function() { return {}; });
          if (d.error) { feedEl.textContent = d.error; return; }
          transEl.textContent = d.transcript || "（无）";
          feedEl.textContent = d.feedback || "";
        } catch (e) { feedEl.textContent = "网络错误，请重试。"; }
      }

      async function loadOpenQuestions() {
        var articleEl = document.querySelector(".article-body");
        var text = articleEl ? (articleEl.innerText || "").trim() : "";
        if (!text || text.length < 50) { alert("请先获取今日练习文章。"); return; }
        var btn = document.getElementById("load-questions-btn");
        btn.disabled = true;
        btn.textContent = "生成中…";
        try {
          var r = await fetch("/daily-practice/questions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ article: text, difficulty: "{{ difficulty|default('C1') }}" }) });
          var d = await r.json().catch(function() { return {}; });
          openQuestionsList = d.topics || [];
          if (d.error) { alert(d.error); btn.disabled = false; btn.textContent = "生成 3 个问题"; return; }
          var listEl = document.getElementById("open-questions-list");
          listEl.innerHTML = "";
          openQuestionsList.forEach(function(q, i) {
            var div = document.createElement("div");
            div.style.marginTop = "12px";
            div.style.padding = "10px";
            div.style.borderRadius = "8px";
            div.style.background = "rgba(15,23,42,0.8)";
            var qEsc = (q || "").replace(/</g, "&lt;").replace(/"/g, "&quot;");
            div.innerHTML =
              '<div class="small" style="color:#94a3b8; margin-bottom:6px;">问题 ' + (i + 1) + '</div>' +
              '<div class="small" style="color:#e5e7eb; margin-bottom:8px;">' + qEsc + '</div>' +
              '<button type="button" class="button-primary" style="padding:6px 12px; font-size:12px; margin-right:8px;" id="oq-record-' + i + '" onclick="startOpenQuestionRecord(' + i + ')">开始录音</button>' +
              '<button type="button" class="btn" style="padding:6px 12px; font-size:12px; display:none;" id="oq-stop-' + i + '" onclick="stopOpenQuestionRecord()">停止</button>' +
              '<button type="button" class="btn" style="padding:6px 12px; font-size:12px;" id="oq-submit-' + i + '" onclick="submitOpenQuestion(' + i + ')" disabled>提交</button>' +
              '<div id="oq-result-' + i + '" style="margin-top:8px; display:none;">' +
                '<div id="oq-transcript-' + i + '" class="small"></div>' +
                '<div id="oq-feedback-' + i + '" class="small" style="white-space:pre-wrap; margin-top:4px;"></div>' +
              '</div>';
            listEl.appendChild(div);
          });
        } catch (e) { alert("网络错误"); }
        btn.disabled = false;
        btn.textContent = "生成 3 个问题";
      }

      function startOpenQuestionRecord(idx) {
        if (openQuestionRecorder && openQuestionRecorder.state === "recording") openQuestionRecorder.stop();
        openQuestionChunks = [];
        openQuestionIdx = idx;
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
          openQuestionRecorder = new MediaRecorder(stream);
          openQuestionRecorder.ondataavailable = function(e) { if (e.data.size) openQuestionChunks.push(e.data); };
          openQuestionRecorder.onstop = function() { stream.getTracks().forEach(function(t) { t.stop(); }); document.getElementById("oq-record-" + idx).style.display = "inline-block"; document.getElementById("oq-stop-" + idx).style.display = "none"; };
          openQuestionRecorder.start();
          document.getElementById("oq-record-" + idx).style.display = "none";
          document.getElementById("oq-stop-" + idx).style.display = "inline-block";
        }).catch(function() { alert("无法使用麦克风"); });
      }
      function stopOpenQuestionRecord() { if (openQuestionRecorder && openQuestionRecorder.state === "recording") openQuestionRecorder.stop(); }

      async function submitOpenQuestion(idx) {
        if (openQuestionIdx !== idx || !openQuestionChunks.length) { alert("请先对该问题完成录音后再提交。"); return; }
        var q = openQuestionsList[idx];
        var articleEl = document.querySelector(".article-body");
        var articleText = articleEl ? (articleEl.innerText || "").trim() : "";
        var form = new FormData();
        form.append("topic", q);
        form.append("article", articleText);
        form.append("audio", new Blob(openQuestionChunks, { type: "audio/webm" }), "answer.webm");
        var transEl = document.getElementById("oq-transcript-" + idx);
        var feedEl = document.getElementById("oq-feedback-" + idx);
        var resultEl = document.getElementById("oq-result-" + idx);
        resultEl.style.display = "block";
        transEl.textContent = ""; feedEl.textContent = "提交中…";
        try {
          var r = await fetch("/read-to-speak/feedback", { method: "POST", body: form });
          var d = await r.json().catch(function() { return {}; });
          if (d.error) { feedEl.textContent = d.error; return; }
          transEl.textContent = d.transcript || "（无）";
          feedEl.textContent = d.feedback || "";
        } catch (e) { feedEl.textContent = "网络错误，请重试。"; }
      }

      let ttsUrl = null;
      let ttsGenerating = false;
      let ttsTimeline = null; // { starts:number[], ends:number[], paras:Element[] }

      function _ensureArticleInjected() {
        var el = document.getElementById("article-body");
        if (!el) return;
        if ((el.innerText || "").trim()) return;
        if (el.dataset && el.dataset.articleB64) {
          try { el.innerHTML = atob(el.dataset.articleB64); } catch (e) {}
        }
      }

      function _clearTtsHighlight() {
        var el = document.getElementById("article-body");
        if (!el) return;
        el.querySelectorAll("p.tts-current").forEach(function(p){ p.classList.remove("tts-current"); });
      }

      function _buildTtsTimeline(durationSec) {
        var el = document.getElementById("article-body");
        if (!el) return null;
        var paras = Array.prototype.slice.call(el.querySelectorAll("p"));
        if (!paras.length || !durationSec || !isFinite(durationSec) || durationSec <= 0) return null;
        var lens = paras.map(function(p){ return ((p.innerText || p.textContent || "").trim().length) || 0; });
        var total = lens.reduce(function(a,b){ return a + b; }, 0) || 1;
        var starts = [];
        var ends = [];
        var acc = 0;
        for (var i=0; i<paras.length; i++) {
          var w = lens[i] / total;
          var seg = w * durationSec;
          starts.push(acc);
          acc += seg;
          ends.push(acc);
        }
        // 保底：最后一个 end 对齐 duration
        ends[ends.length - 1] = durationSec;
        return { starts: starts, ends: ends, paras: paras };
      }

      function _updateTtsHighlight(t) {
        if (!ttsTimeline) return;
        var idx = -1;
        for (var i=0; i<ttsTimeline.starts.length; i++) {
          if (t >= ttsTimeline.starts[i] && t < ttsTimeline.ends[i]) { idx = i; break; }
        }
        if (idx < 0) return;
        for (var j=0; j<ttsTimeline.paras.length; j++) {
          if (j === idx) ttsTimeline.paras[j].classList.add("tts-current");
          else ttsTimeline.paras[j].classList.remove("tts-current");
        }
      }

      function _setTtsBtnState(state) {
        // state: idle | generating | playing | paused
        var btn = document.getElementById("tts-btn");
        if (!btn) return;
        btn.classList.remove("tts-playing");
        if (state === "idle") { btn.disabled = false; btn.textContent = "全文朗读"; return; }
        if (state === "generating") { btn.disabled = true; btn.textContent = "生成中…"; return; }
        if (state === "playing") { btn.disabled = false; btn.classList.add("tts-playing"); btn.textContent = "暂停朗读"; return; }
        if (state === "paused") { btn.disabled = false; btn.textContent = "继续朗读"; return; }
      }

      (function _initTtsAudio() {
        var audio = document.getElementById("tts-audio");
        if (!audio) return;
        audio.addEventListener("loadedmetadata", function() {
          try { ttsTimeline = _buildTtsTimeline(audio.duration || 0); } catch (e) { ttsTimeline = null; }
        });
        audio.addEventListener("timeupdate", function() {
          try { _updateTtsHighlight(audio.currentTime || 0); } catch (e) {}
        });
        audio.addEventListener("play", function() { _setTtsBtnState("playing"); });
        audio.addEventListener("pause", function() {
          // ended 也会触发 pause，避免覆盖 ended 的状态
          if (audio.ended) return;
          _setTtsBtnState("paused");
        });
        audio.addEventListener("ended", function() {
          _setTtsBtnState("idle");
          _clearTtsHighlight();
        });
      })();

      async function speakArticle() {
        _ensureArticleInjected();
        var btn = document.getElementById("tts-btn");
        var audio = document.getElementById("tts-audio");
        var articleEl = document.getElementById("article-body");
        var text = articleEl ? (articleEl.innerText || "").trim() : "";
        if (!text) { alert("没有找到文章内容，无法朗读。"); return; }
        if (!audio) { alert("音频组件不可用。"); return; }

        // 若已有音频：点击切换播放/暂停
        if (audio.src) {
          try {
            if (!audio.paused && !audio.ended) { audio.pause(); return; }
            await audio.play(); return;
          } catch (e) {
            // 继续走生成逻辑（兜底）
          }
        }

        if (ttsGenerating) return;
        ttsGenerating = true;
        _setTtsBtnState("generating");
        _clearTtsHighlight();

        try {
          var resp = await fetch("/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
          });
          if (!resp.ok) {
            var data = await resp.json().catch(function() { return {}; });
            alert(data.error || "生成音频失败，请稍后再试。");
            _setTtsBtnState("idle");
            return;
          }
          var blob = await resp.blob();
          if (ttsUrl) { try { URL.revokeObjectURL(ttsUrl); } catch (e) {} }
          ttsUrl = URL.createObjectURL(blob);
          audio.src = ttsUrl;
          audio.style.display = "block";
          // duration 需要 loadedmetadata 后才稳定；先尝试构建一次，后续事件会再建
          try { ttsTimeline = null; } catch (e) {}
          await audio.play();
        } catch (e) {
          alert("网络或服务器错误，请稍后再试。");
          _setTtsBtnState("idle");
        } finally {
          ttsGenerating = false;
        }
      }

      async function lookupSelection() {
        const sel = window.getSelection ? window.getSelection().toString().trim() : "";
        if (!sel) {
          alert("请先在上面的文章中选中一个英文单词（双击或拖动选择）。");
          return;
        }
        let query = sel.replace(/\\s+/g, " ").trim();
        query = query.replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, "");
        if (!query) {
          alert("看起来没有选中有效的英文单词或短语，请再试一次。");
          return;
        }
        if (query.length > 80) query = query.slice(0, 80);

        // 尝试从文章中抓取“原文句子”（优先取选中的段落文本）
        let ctxEl = null;
        try {
          const selection = window.getSelection ? window.getSelection() : null;
          const anchor = selection && selection.anchorNode ? selection.anchorNode : null;
          let el = anchor && anchor.parentElement ? anchor.parentElement : null;
          while (el && el !== document.body && !el.classList.contains("article-body") && el.tagName !== "P") {
            el = el.parentElement;
          }
          if (el && el.tagName === "P") ctxEl = el;
        } catch (e) {}

        return lookupWord(query, ctxEl);
      }

      async function lookupWord(word, ctxEl) {
        word = (word || "").replace(/\\s+/g, " ").trim().replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, "");
        if (!word) return;
        if (word.length > 80) word = word.slice(0, 80);

        const resultBox = document.getElementById("lookup-result");
        const wEl = document.getElementById("lookup-word");
        const pEl = document.getElementById("lookup-phonetic");
        const mEl = document.getElementById("lookup-meaning");
        const eEn = document.getElementById("lookup-example-en");
        const eZh = document.getElementById("lookup-example-zh");
        const addBtn = document.getElementById("add-vocab-btn");
        const addStatus = document.getElementById("add-vocab-status");

        wEl.textContent = "正在查询：" + word + " …";
        pEl.textContent = "";
        mEl.textContent = "";
        eEn.textContent = "";
        eZh.textContent = "";
        addStatus.textContent = "";
        lastLookup = null;
        lastContextSentence = "";
        resultBox.style.display = "block";

        try {
          if (ctxEl && ctxEl.tagName === "P") {
            lastContextSentence = (ctxEl.innerText || "").trim();
          } else {
            const articleEl = document.querySelector(".article-body");
            lastContextSentence = articleEl ? (articleEl.innerText || "").trim() : "";
          }
        } catch (e) {}

        try {
          const resp = await fetch("/lookup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ word: word, provider: {{ provider|tojson }} })
          });
          const data = await resp.json();
          if (data.error) {
            wEl.textContent = "查询失败";
            mEl.textContent = data.error;
            return;
          }
          wEl.textContent = data.word || word;
          pEl.textContent = data.phonetic ? "/" + data.phonetic + "/" : "";
          mEl.textContent = data.meaning_zh || "";
          eEn.textContent = data.example_en ? "例句: " + data.example_en : "";
          eZh.textContent = data.example_zh ? "译文: " + data.example_zh : "";
          lastLookup = data;
          addBtn.disabled = false;
        } catch (e) {
          wEl.textContent = "查询失败";
          mEl.textContent = "网络或服务器错误，请稍后再试。";
        }
      }

      let tapLookupEnabled = false;
      function toggleTapLookup(force) {
        const btn = document.getElementById("tap-lookup-btn");
        tapLookupEnabled = (typeof force === "boolean") ? force : !tapLookupEnabled;
        if (btn) btn.textContent = "点词查词：" + (tapLookupEnabled ? "开" : "关");
        const article = document.getElementById("article-body");
        if (article) article.style.cursor = tapLookupEnabled ? "pointer" : "";
      }

      function _isWordChar(ch) { return /[A-Za-z'-]/.test(ch || ""); }
      function _extractWordAround(text, offset) {
        if (!text) return "";
        let i = Math.max(0, Math.min(offset || 0, text.length));
        let l = i, r = i;
        while (l > 0 && _isWordChar(text[l - 1])) l--;
        while (r < text.length && _isWordChar(text[r])) r++;
        let w = text.slice(l, r);
        w = w.replace(/^[^A-Za-z]+|[^A-Za-z]+$/g, "");
        return w;
      }

      function _wordAtPoint(x, y) {
        const doc = document;
        let node = null, offset = 0;
        if (doc.caretPositionFromPoint) {
          const pos = doc.caretPositionFromPoint(x, y);
          node = pos ? pos.offsetNode : null;
          offset = pos ? pos.offset : 0;
        } else if (doc.caretRangeFromPoint) {
          const range = doc.caretRangeFromPoint(x, y);
          node = range ? range.startContainer : null;
          offset = range ? range.startOffset : 0;
        }
        if (!node) return { word: "", ctxEl: null };

        let textNode = node.nodeType === 3 ? node : null;
        if (!textNode && node.nodeType === 1) {
          for (let i = 0; i < node.childNodes.length; i++) {
            if (node.childNodes[i].nodeType === 3) { textNode = node.childNodes[i]; break; }
          }
        }
        const text = textNode ? (textNode.nodeValue || "") : "";
        const word = _extractWordAround(text, offset);

        let el = (textNode && textNode.parentElement) ? textNode.parentElement : (node.parentElement || null);
        while (el && el !== document.body && !el.classList.contains("article-body") && el.tagName !== "P") el = el.parentElement;
        return { word: word, ctxEl: (el && el.tagName === "P") ? el : null };
      }

      (function initTapLookup() {
        const isTouch = ("ontouchstart" in window) || (navigator.maxTouchPoints && navigator.maxTouchPoints > 0);
        if (isTouch) toggleTapLookup(true);
        const article = document.getElementById("article-body");
        if (!article) return;
        article.addEventListener("click", function(e) {
          if (!tapLookupEnabled) return;
          // If user has a selection (common on mobile for phrases), prefer it.
          try {
            const sel = window.getSelection ? window.getSelection().toString().trim() : "";
            if (sel && sel.replace(/\\s+/g, " ").trim().split(" ").length >= 2) {
              let q = sel.replace(/\\s+/g, " ").trim().replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, "");
              if (q) {
                // best-effort context paragraph
                let el = e.target && e.target.closest ? e.target.closest("p") : null;
                lookupWord(q, el);
                return;
              }
            }
          } catch (err) {}
          const t = e.target;
          if (t && t.closest && t.closest("a")) return;
          const res = _wordAtPoint(e.clientX, e.clientY);
          if (res && res.word) lookupWord(res.word, res.ctxEl);
        });
      })();

      async function addToVocab() {
        const btn = document.getElementById("add-vocab-btn");
        const status = document.getElementById("add-vocab-status");
        if (!lastLookup || !lastLookup.word) {
          status.textContent = "请先查词。";
          return;
        }
        btn.disabled = true;
        status.textContent = "保存中…";
        try {
          const resp = await fetch("/vocab/add", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              word: lastLookup.word,
              meaning_zh: lastLookup.meaning_zh || "",
              sentence_en: lastContextSentence || lastLookup.example_en || "",
              phonetic: lastLookup.phonetic || "",
              provider: {{ provider|tojson }}
            })
          });
          const data = await resp.json().catch(function() { return {}; });
          if (!resp.ok || data.error) {
            status.textContent = data.error || "保存失败，请稍后再试。";
            return;
          }
          status.textContent = "已保存。去生词本看看 →";
        } catch (e) {
          status.textContent = "网络或服务器错误，请稍后再试。";
        } finally {
          btn.disabled = false;
        }
      }

    </script>
  </body>
</html>
"""


def _parse_model_json(content: str):
    """去掉可能的 ```json 代码块外壳，并解析 JSON。"""
    content_stripped = (content or "").strip()
    if content_stripped.startswith("```"):
        lines = content_stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content_stripped = "\n".join(lines).strip()
    data = json.loads(content_stripped)
    return data


def _strip_html(html: str) -> str:
    # very small helper: remove tags and collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_script_tags(html: str) -> str:
    """移除 HTML 中的 <script> 标签，并转义 </script> 文本，防止注入导致页面 JS 解析错误。"""
    if not html:
        return ""
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all("script"):
                tag.decompose()
            html = str(soup)
        except Exception:
            pass
    html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<script[^>]*>[\s\S]*", "", html, flags=re.IGNORECASE)
    html = html.replace("</script>", "&lt;/script&gt;").replace("</SCRIPT>", "&lt;/script&gt;")
    return html


def _pcm_to_wav_bytes(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _load_vocab_book():
    try:
        with open(VOCAB_BOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except FileNotFoundError:
        return []
    except Exception:
        traceback.print_exc()
    return []


def _save_vocab_book(items):
    tmp_path = VOCAB_BOOK_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, VOCAB_BOOK_PATH)


def _load_article_history():
    try:
        with open(ARTICLE_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except FileNotFoundError:
        return []
    except Exception:
        traceback.print_exc()
    return []


def _save_article_history(items):
    tmp_path = ARTICLE_HISTORY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, ARTICLE_HISTORY_PATH)


def _filter_articles(items, date_filter=None, topic_filter=None, difficulty_filter=None):
    """按日期、主题关键词、难度筛选。"""
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        created = (it.get("created_at") or "")[:10]  # YYYY-MM-DD
        if date_filter and created != date_filter:
            continue
        topic = (it.get("topic") or "").lower()
        if topic_filter and topic_filter.lower() not in topic:
            continue
        diff = (it.get("difficulty") or "").strip()
        if difficulty_filter and diff != difficulty_filter:
            continue
        out.append(it)
    return out


# ---------- 阅读推荐：爬取 21voa / i21st 并做 AI 标注 ----------

def _load_recommend_data():
    try:
        with open(RECOMMEND_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception:
        traceback.print_exc()
    return {"used_urls": [], "daily": {}, "daily_assigned": {}}


def _save_recommend_data(data):
    if not isinstance(data, dict):
        data = {"used_urls": [], "daily": {}, "daily_assigned": {}}
    tmp_path = RECOMMEND_DATA_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, RECOMMEND_DATA_PATH)


def _load_daily_practice():
    """每日练习缓存：key = "YYYY-MM-DD|B1" 等，value = 单篇文章结构。文件不存在时返回空结构。"""
    try:
        with open(DAILY_PRACTICE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "by_key" in data:
                return data
    except FileNotFoundError:
        return {"by_key": {}}
    except Exception:
        traceback.print_exc()
    return {"by_key": {}}


def _save_daily_practice(data):
    if not isinstance(data, dict):
        data = {"by_key": {}}
    data.setdefault("by_key", {})
    tmp_path = DAILY_PRACTICE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DAILY_PRACTICE_PATH)


def get_daily_practice_article(difficulty: str, provider: str, topic_for_fallback: str = None):
    """获取当日练习：从每日 0 点缓存的 3 篇中选一篇最新且未读的；若无则 AI 生成。"""
    today = _today_date()
    diff = (difficulty or "C1").strip().upper()
    if diff not in ("B1", "B2", "C1"):
        diff = "C1"
    key = f"{today}|{diff}"

    # 1. 从今日缓存池中选一篇未读的（优先同难度）
    rec_data = _load_recommend_data()
    rec_data.setdefault("daily", {})
    rec_data.setdefault("daily_assigned", {})
    assigned = set(rec_data["daily_assigned"].get(today, []))
    pool = rec_data["daily"].get(today, [])

    def _assign_and_return(a):
        url = a.get("url") or ""
        a = dict(a)
        a["article_html"] = _strip_script_tags(a.get("article_html", ""))
        assigned.add(url)
        rec_data["daily_assigned"][today] = list(assigned)
        _save_recommend_data(rec_data)
        dp_data = _load_daily_practice()
        dp_data.setdefault("by_key", {})
        dp_data["by_key"][key] = a
        _save_daily_practice(dp_data)
        history_items = _load_article_history()
        history_items.append({
            "id": datetime.now(timezone.utc).isoformat(),
            "topic": f"[每日练习] {a.get('title', '')}",
            "difficulty": a.get("difficulty", diff),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "article_html": a["article_html"],
            "vocab_list": a.get("vocab_list", []),
            "provider": provider,
        })
        _save_article_history(history_items)
        return a

    for a in pool:
        url = a.get("url") or ""
        if url in assigned:
            continue
        if (a.get("difficulty") or "").strip() == diff:
            return _assign_and_return(a)
    for a in pool:
        url = a.get("url") or ""
        if url in assigned:
            continue
        return _assign_and_return(a)

    # 2. 池空或已全读过，AI 生成
    topic = (topic_for_fallback or "").strip() or "technology and society"
    try:
        article_html, vocab_list = generate_reading(topic, provider, diff)
        plain_text = _strip_html(article_html or "")
        assessed = _assess_article_cefr(plain_text, provider) if plain_text.strip() else diff
        article_html = _strip_script_tags(article_html)
        created_at = datetime.now(timezone.utc).isoformat()
        article = {
            "title": topic,
            "article_html": article_html,
            "vocab_list": vocab_list,
            "difficulty": assessed,
        }
        dp_data = _load_daily_practice()
        dp_data.setdefault("by_key", {})
        dp_data["by_key"][key] = article
        _save_daily_practice(dp_data)
        history_items = _load_article_history()
        history_items.append({
            "id": created_at,
            "topic": f"[每日练习] {topic}",
            "difficulty": assessed,
            "created_at": created_at,
            "article_html": article_html,
            "vocab_list": vocab_list,
            "provider": provider,
        })
        _save_article_history(history_items)
        return article
    except Exception:
        traceback.print_exc()
        return None


def _crawl_21voa_one(used_urls):
    """从 21voa.com 抓取一篇未使用过的文章（优先「最近更新」里的 /special_english/ 链接）。返回 {url, title, text} 或 None。"""
    if not requests or not BeautifulSoup:
        return None
    base = "https://www.21voa.com"
    try:
        r = requests.get(base + "/", headers=CRAWL_HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        # 优先抓「最近更新」里的单篇链接：/special_english/xxx-数字.html
        links = []
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#") or "javascript" in href.lower():
                continue
            full = urljoin(base, href)
            if "21voa.com" not in urlparse(full).netloc:
                continue
            if full in used_urls:
                continue
            if full == base or full.rstrip("/") == base:
                continue
            # 单篇文章在 /special_english/ 下且带数字 id（如 -93397.html）
            if "/special_english/" in full and ".html" in full and re.search(r"-\d+\.html", full):
                links.append((full, (a.get_text() or "").strip()[:200]))
        if not links:
            for a in soup.find_all("a", href=True):
                href = (a.get("href") or "").strip()
                full = urljoin(base, href)
                if "21voa.com" not in urlparse(full).netloc or full in used_urls:
                    continue
                if ".html" in full and full not in [u for u, _ in links]:
                    links.append((full, (a.get_text() or "").strip()[:200]))
        for url, _ in links[:20]:
            try:
                r2 = requests.get(url, headers=CRAWL_HEADERS, timeout=12)
                r2.encoding = r2.apparent_encoding or "utf-8"
                s2 = BeautifulSoup(r2.text, "html.parser")
                title = ""
                if s2.title:
                    title = (s2.title.get_text() or "").strip()
                for tag in s2.find_all(["h1", "h2"]):
                    t = (tag.get_text() or "").strip()
                    if len(t) > 10:
                        title = t
                        break
                paras = []
                for cand in [s2.find(id="content"), s2.find(id="article"), s2.find("article")]:
                    if cand:
                        for p in cand.find_all("p"):
                            t = (p.get_text() or "").strip()
                            if len(t) > 20 and not t.startswith("http"):
                                paras.append(t)
                if not paras:
                    for div in s2.find_all(["div", "article", "section"]):
                        cls = " ".join(div.get("class", [])).lower()
                        if "content" in cls or "article" in cls or "post" in cls or "story" in cls or "entry" in cls:
                            for p in div.find_all("p"):
                                t = (p.get_text() or "").strip()
                                if len(t) > 20 and not t.startswith("http"):
                                    paras.append(t)
                if not paras:
                    for p in s2.find_all("p"):
                        t = (p.get_text() or "").strip()
                        if len(t) > 40 and not t.startswith("http"):
                            paras.append(t)
                text = "\n\n".join(paras[:50])
                if len(text) < 150:
                    continue
                return {"url": url, "title": title or "VOA Article", "text": text[:15000]}
            except Exception:
                continue
    except Exception as e:
        traceback.print_exc()
    return None


def _crawl_i21st_one(used_urls):
    """从 i21st.cn 抓取一篇未使用过的文章。返回 {url, title, text} 或 None。"""
    if not requests or not BeautifulSoup:
        return None
    base = "https://www.i21st.cn"
    try:
        r = requests.get(base + "/", headers=CRAWL_HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if "/article/" not in href or "_1.html" not in href and ".html" not in href:
                continue
            full = urljoin(base, href)
            if "i21st.cn" not in urlparse(full).netloc:
                continue
            if full in used_urls:
                continue
            links.append((full, (a.get_text() or "").strip()[:200]))
        for url, _ in links[:15]:
            try:
                r2 = requests.get(url, headers=CRAWL_HEADERS, timeout=12)
                r2.encoding = r2.apparent_encoding or "utf-8"
                s2 = BeautifulSoup(r2.text, "html.parser")
                title = ""
                for tag in s2.find_all(["h1", "h2", "title"]):
                    t = (tag.get_text() or "").strip()
                    if len(t) > 5:
                        title = t
                        break
                paras = []
                for div in s2.find_all(["div", "article"]):
                    cls = " ".join(div.get("class", [])).lower()
                    if "content" in cls or "article" in cls or "text" in cls or "post" in cls:
                        for p in div.find_all("p"):
                            t = (p.get_text() or "").strip()
                            if len(t) > 20:
                                paras.append(t)
                if not paras:
                    for p in s2.find_all("p"):
                        t = (p.get_text() or "").strip()
                        if len(t) > 30:
                            paras.append(t)
                text = "\n\n".join(paras[:50])
                if len(text) < 150:
                    continue
                return {"url": url, "title": title or "21st Century", "text": text[:15000]}
            except Exception:
                continue
    except Exception as e:
        traceback.print_exc()
    return None


def _crawl_buzzing_one(used_urls):
    """从 Buzzing.cc（hn.buzzing.cc）抓取一篇：取一条外链并抓取原文。返回 {url, title, text} 或 None。"""
    if not requests or not BeautifulSoup:
        return None
    list_url = "https://hn.buzzing.cc/"
    try:
        r = requests.get(list_url, headers=CRAWL_HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        seen = set()
        candidates = []
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#") or "javascript" in href.lower():
                continue
            full = urljoin(list_url, href)
            parsed = urlparse(full)
            netloc = (parsed.netloc or "").lower()
            if "buzzing.cc" in netloc or "ycombinator.com" in netloc:
                continue
            if full in used_urls or full in seen:
                continue
            seen.add(full)
            title = (a.get_text() or "").strip()[:300]
            if not title or len(title) < 5:
                continue
            candidates.append((full, title))
        for url, title in candidates[:25]:
            try:
                r2 = requests.get(url, headers=CRAWL_HEADERS, timeout=14)
                r2.encoding = r2.apparent_encoding or "utf-8"
                s2 = BeautifulSoup(r2.text, "html.parser")
                page_title = ""
                if s2.title:
                    page_title = (s2.title.get_text() or "").strip()
                for tag in s2.find_all(["h1", "h2"]):
                    t = (tag.get_text() or "").strip()
                    if len(t) > 10 and not t.startswith("http"):
                        page_title = t
                        break
                display_title = page_title or title
                paras = []
                for cand in [s2.find(id="content"), s2.find(id="article"), s2.find("article"), s2.find(class_=re.compile(r"post|entry|story|content", re.I))]:
                    if cand:
                        for p in cand.find_all("p"):
                            t = (p.get_text() or "").strip()
                            if len(t) > 25 and not t.startswith("http"):
                                paras.append(t)
                if not paras:
                    for div in s2.find_all(["div", "article", "section"]):
                        cls = " ".join(div.get("class", [])).lower()
                        if "content" in cls or "article" in cls or "post" in cls or "entry" in cls:
                            for p in div.find_all("p"):
                                t = (p.get_text() or "").strip()
                                if len(t) > 25 and not t.startswith("http"):
                                    paras.append(t)
                if not paras:
                    for p in s2.find_all("p"):
                        t = (p.get_text() or "").strip()
                        if len(t) > 50 and not t.startswith("http"):
                            paras.append(t)
                text = "\n\n".join(paras[:50])
                if len(text) < 200:
                    continue
                return {"url": url, "title": display_title or "Buzzing Pick", "text": text[:15000]}
            except Exception:
                continue
    except Exception:
        traceback.print_exc()
    return None


def _assess_article_cefr(article_content: str, provider: str) -> str:
    """根据 CEFR 标准（词汇、句式、概念与逻辑）评估文章难度，返回 B1/B2/C1。"""
    text = (_strip_html(article_content) if article_content else "")[:8000]
    if not text.strip():
        return "C1"
    prompt = """你现在是一名专业的 CEFR (欧洲语言共同参考标准) 评估专家。你的任务是分析下方提供的文章内容，并根据 B1、B2、C1 的标准对其进行难度定义。

1. 评估维度 (Scoring Metrics)：

Vocabulary (词汇):
- B1: 主要是常用词（A1-B1），涉及日常话题。
- B2: 包含中级话题词汇（如：Environment, Technology），有少量习语。
- C1: 包含大量低频词、学术词汇、隐喻及地道搭配（Collocations）。

Sentence Structure (句式):
- B1: 以简单句和基本从句（because, that）为主。
- B2: 句子较长，包含多种复合从句和被动语态。
- C1: 结构复杂，包含倒装、插入语、长难句和微妙的衔接方式。

Concept & Logic (概念与逻辑):
- B1: 具体的、事实性的描述。
- B2: 涉及抽象观点、因果论证。
- C1: 深度社论、哲学探讨或具有复杂反讽/隐喻的专业评论。

2. 判定逻辑 (Decision Logic)：
如果词汇和句式不匹配（例如词汇简单但句式极难），请以较高的一方为准，以确保用户的挑战性。

3. 输出要求 (Output Format)：
请直接返回一个 JSON 对象，严禁包含任何多余的解释文字。JSON 必须包含键 "cefr_level"，取值为 "B1"、"B2" 或 "C1" 之一。

待处理内容：
""" + text
    try:
        if provider == "deepseek" and DEEPSEEK_API_KEY:
            completion = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a CEFR assessment expert. Output only a single JSON object with key cefr_level (value B1, B2, or C1)."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = completion.choices[0].message.content
        elif GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            content = response.text or ""
        else:
            return "C1"
        data = _parse_model_json(content.strip())
        level = (data.get("cefr_level") or data.get("difficulty") or "").strip().upper()
        if level in ("B1", "B2", "C1"):
            return level
    except Exception:
        traceback.print_exc()
    return "C1"


def _enrich_article_with_vocab(plain_text: str, provider: str, difficulty: str = "C1"):
    """对已有文章做与 CEFR 难度一致的词汇标注，返回 article_html 与 vocab_list。"""
    vocab_band = {"B1": "A2–B1", "B2": "B1–B2", "C1": "B2–C1"}.get(difficulty.upper(), "B2–C1")
    level_desc = {
        "B1": "B1 (intermediate) level: common words, daily topics.",
        "B2": "B2 (upper-intermediate) level: mid-level topic vocabulary, some idioms.",
        "C1": "C1 (advanced) level: low-frequency, academic, collocations.",
    }
    level_guide = level_desc.get(difficulty.upper(), level_desc["C1"])
    system_prompt = f"You are an English teacher preparing reading materials for {difficulty.upper()}-level learners (first language Chinese). Use the same CEFR criteria: vocabulary appropriate to this level."
    user_prompt = f"""Below is an existing English article. Your task:

1) Keep the article content as-is, but wrap exactly FIVE {vocab_band} level vocabulary words in <mark>...</mark>. Choose words that are useful for this level: {level_guide}
2) Output the full article as HTML: use <p>...</p> for each paragraph. No other changes.
3) Provide a vocabulary list for those five words with Chinese meanings.

Output format: a SINGLE JSON object only, no extra text.
- "article_html": string, the article with <mark>word</mark> for the five words, paragraphs in <p>...</p>
- "vocab": array of 5 objects: {{ "word": "...", "meaning_zh": "..." }}

Article:
{plain_text[:12000]}
"""
    if provider == "deepseek":
        completion = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = completion.choices[0].message.content
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt + "\n\n" + user_prompt,
        )
        content = response.text
    else:
        raise ValueError("Unsupported provider")
    data = _parse_model_json(content)
    article_html = data.get("article_html", "")
    vocab_list = data.get("vocab", [])
    normalized = []
    for item in vocab_list:
        if not isinstance(item, dict):
            continue
        w, m = item.get("word"), item.get("meaning_zh")
        if w and m:
            normalized.append({"word": w, "meaning_zh": m})
    return article_html, normalized


def _add_one_article(source: str, crawler, used: list, provider: str, articles: list):
    """从指定站点抓取一篇，先按 CEFR 评估难度再按该难度做词汇标注，成功则追加到 articles。"""
    raw = crawler(used)
    if not raw:
        return False
    url, title, text = raw["url"], raw["title"], raw["text"]
    if url in used:
        return False
    try:
        difficulty = _assess_article_cefr(text, provider)
        article_html, vocab_list = _enrich_article_with_vocab(text, provider, difficulty)
        if not article_html:
            article_html = "<p>" + text.replace("\n\n", "</p><p>")[:8000] + "</p>"
        article_html = _strip_script_tags(article_html)
        articles.append({
            "title": title,
            "url": url,
            "source": source,
            "article_html": article_html,
            "vocab_list": vocab_list,
            "difficulty": difficulty,
        })
        used.append(url)
        return True
    except Exception:
        traceback.print_exc()
        return False


def _pool_has_enough_per_difficulty(data: dict) -> bool:
    """爬取库中若每个难度(B1/B2/C1)的文章都大于1篇则返回 True，当日可不爬取。"""
    daily = data.get("daily") or {}
    count = {"B1": 0, "B2": 0, "C1": 0}
    for date_articles in daily.values():
        if not isinstance(date_articles, list):
            continue
        for a in date_articles:
            if not isinstance(a, dict):
                continue
            d = (a.get("difficulty") or "").strip().upper()
            if d in count:
                count[d] += 1
    return count["B1"] > 1 and count["B2"] > 1 and count["C1"] > 1


def _crawl_daily_pool(provider: str = "gemini"):
    """爬取 3 篇文章并缓存到 daily 池，不写入历史。供每日 0 点定时任务调用。"""
    today = _today_date()
    data = _load_recommend_data()
    data.setdefault("used_urls", [])
    data.setdefault("daily", {})
    # 若每个难度已有 >1 篇，则当日不爬取，节省调用
    if _pool_has_enough_per_difficulty(data):
        return data["daily"].get(today, [])
    used = list(data["used_urls"])
    if today in data["daily"] and len(data["daily"][today]) >= 3:
        return data["daily"][today]
    articles = []
    _add_one_article("21voa", _crawl_21voa_one, used, provider, articles)
    _add_one_article("i21st", _crawl_i21st_one, used, provider, articles)
    _add_one_article("buzzing", _crawl_buzzing_one, used, provider, articles)
    data["used_urls"] = used
    data["daily"][today] = articles
    _save_recommend_data(data)
    return articles


def get_today_recommendations(provider: str):
    """获取今日阅读推荐：最多 3 篇（21voa、i21st、buzzing 各 1 篇），不重复，并与 AI 生成文章能力相同（高亮+释义）。"""
    today = _today_date()
    data = _load_recommend_data()
    data.setdefault("used_urls", [])
    data.setdefault("daily", {})
    used = list(data["used_urls"])
    daily = data["daily"]
    if today in daily and len(daily[today]) >= 3:
        return daily[today]
    articles = []
    _add_one_article("21voa", _crawl_21voa_one, used, provider, articles)
    _add_one_article("i21st", _crawl_i21st_one, used, provider, articles)
    _add_one_article("buzzing", _crawl_buzzing_one, used, provider, articles)
    data["used_urls"] = used
    data["daily"][today] = articles
    _save_recommend_data(data)
    if articles:
        history_items = _load_article_history()
        for a in articles:
            created_at = datetime.now(timezone.utc).isoformat()
            history_items.append({
                "id": created_at,
                "topic": f"[推荐] {a['title']}",
                "difficulty": a.get("difficulty", "C1"),
                "created_at": created_at,
                "article_html": a["article_html"],
                "vocab_list": a["vocab_list"],
                "provider": provider,
            })
        _save_article_history(history_items)
    return data["daily"].get(today, articles)


def generate_reading(topic: str, provider: str, difficulty: str = "C1"):
    """调用指定大模型生成阅读文章和词汇表。difficulty 为 B1/B2/C1。"""
    level_desc = {
        "B1": ("B1 (intermediate) level learners", "A2–B1", "simple, clear sentences and everyday vocabulary; avoid complex structures."),
        "B2": ("B2 (upper-intermediate) level learners", "B1–B2", "natural style with some varied structures and mid-level vocabulary."),
        "C1": ("C1 (advanced) level learners", "B2–C1", "sophisticated style suitable for advanced learners; use varied and precise vocabulary."),
    }
    desc, vocab_band, style = level_desc.get(difficulty.upper(), level_desc["C1"])
    system_prompt = (
        f"You are an English teacher creating reading passages for {desc} "
        "whose first language is Chinese."
    )
    user_prompt = f"""
Please write a short reading passage about the topic below, for {desc}:

Topic: {topic}

Requirements:
- Around 270–330 English words.
- {style.capitalize()}
- Use exactly FIVE {vocab_band} level vocabulary words that are useful for this level.
- In the article HTML, highlight these five words by wrapping each of them in <mark>...</mark>.
- Do NOT explain the words inside the passage.

Then provide a vocabulary list for these five words with their Chinese meanings.

VERY IMPORTANT OUTPUT FORMAT:
- Respond with a SINGLE JSON object.
- Keys:
  - "article_html": string, HTML fragment of the passage, paragraphs separated with <p>...</p>, using <mark>highlighted_word</mark> for the five words.
  - "vocab": an array of exactly 5 objects, each with:
      - "word": the English word (same as in the article, without <mark>)
      - "meaning_zh": a short Chinese explanation
- Do NOT include any extra commentary or text outside the JSON.
"""

    if provider == "deepseek":
        completion = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
        )
        content = completion.choices[0].message.content
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=GEMINI_API_KEY)
        full_prompt = system_prompt + "\n\n" + user_prompt
        # 使用 Gemini 2.5 Flash 文本模型
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        content = response.text
    else:
        raise ValueError("Unsupported provider")

    try:
        data = _parse_model_json(content)
    except Exception:
        # 打印原始返回，方便调试 JSON 解析问题
        print("\n=== MODEL RAW RESPONSE START ===")
        print(content)
        print("=== MODEL RAW RESPONSE END ===\n")
        raise

    article_html = data.get("article_html", "")
    vocab_list = data.get("vocab", [])

    # 确保词汇表结构统一
    normalized_vocab = []
    for item in vocab_list:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        meaning = item.get("meaning_zh")
        if word and meaning:
            normalized_vocab.append({"word": word, "meaning_zh": meaning})

    return article_html, normalized_vocab


def lookup_word(word: str, provider: str):
    """调用大模型查询单词/短语释义、音标和例句。"""
    system_prompt = (
        "You are an English-Chinese bilingual dictionary specialized for ESL learners."
    )
    user_prompt = f"""
Please provide a concise dictionary-style explanation for the English word or phrase below.

Query: {word}

Output requirements:
- Explain the most common meaning suitable for intermediate-advanced ESL learners (B2–C1).
- Provide one natural English example sentence using this word in context.
- Provide a natural-sounding Chinese translation of that sentence.

VERY IMPORTANT OUTPUT FORMAT:
- Respond with a SINGLE JSON object, no extra commentary, no backticks.
- Keys:
  - "word": the queried word or phrase (string)
  - "phonetic": English phonetic transcription in IPA or a simple learner-friendly style, WITHOUT surrounding slashes (string). If the query is a phrase, you may leave it empty.
  - "meaning_zh": short Chinese explanation of the core meaning (string)
  - "example_en": one English example sentence (string)
  - "example_zh": the Chinese translation of the example sentence (string)
"""

    if provider == "deepseek":
        completion = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = completion.choices[0].message.content
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=GEMINI_API_KEY)
        full_prompt = system_prompt + "\n\n" + user_prompt
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        content = response.text
    else:
        raise ValueError("Unsupported provider for lookup")

    data = _parse_model_json(content)

    return {
        "word": data.get("word", word),
        "phonetic": data.get("phonetic", ""),
        "meaning_zh": data.get("meaning_zh", ""),
        "example_en": data.get("example_en", ""),
        "example_zh": data.get("example_zh", ""),
    }


def _get_collocations(word: str, meaning_zh: str, provider: str):
    """AI 生成该词的高频搭配（词块），返回 [{"phrase": "...", "meaning_zh": "..."}, ...]。"""
    prompt = f"""For the English word "{word}" (Chinese meaning: {meaning_zh or 'not provided'}).
Generate 5 to 8 high-frequency collocations or lexical chunks (词块) that native speakers often use with this word.
Each item should be a natural phrase or chunk containing the word, plus a very short Chinese explanation.

Output ONLY a JSON object with a single key "collocations", value is an array of objects:
- "phrase": the English collocation/chunk (e.g. "have a significant impact on", "impact on society")
- "meaning_zh": short Chinese meaning of the phrase

No extra text. Example: {{"collocations": [{{"phrase": "have an impact on", "meaning_zh": "对…有影响"}}, ...]}}"""
    try:
        if provider == "deepseek" and DEEPSEEK_API_KEY:
            completion = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert in English collocations and lexical chunks for ESL learners. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message.content
        elif GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            content = response.text or ""
        else:
            return []
        data = _parse_model_json(content)
        raw = data.get("collocations") or data.get("collocation") or []
        out = []
        for x in raw if isinstance(raw, list) else []:
            if isinstance(x, dict) and x.get("phrase"):
                out.append({
                    "phrase": (x.get("phrase") or "").strip()[:200],
                    "meaning_zh": (x.get("meaning_zh") or "").strip()[:150],
                })
        return out[:10]
    except Exception:
        traceback.print_exc()
        return []


def _current_provider():
    """当前会话选用的大模型：优先 session，其次请求体/查询参数。"""
    p = session.get("provider")
    if p in ("gemini", "deepseek"):
        return p
    p = (request.form.get("provider") or "").strip() or (request.args.get("provider") or "").strip()
    if p in ("gemini", "deepseek"):
        return p
    try:
        j = request.get_json(silent=True) or {}
        p = (j.get("provider") or "").strip()
        if p in ("gemini", "deepseek"):
            return p
    except Exception:
        pass
    return "gemini"


def _is_gemini_rate_limited(err: Exception) -> bool:
    try:
        return isinstance(err, ClientError) and getattr(err, "code", None) == 429
    except Exception:
        return False


def _ming_uniaudio_enabled() -> bool:
    return bool(MING_UNIAUDIO_URL)


def _ming_uniaudio_tts(text: str, *, lang: str = "en") -> bytes:
    if not _ming_uniaudio_enabled():
        raise RuntimeError("MING_UNIAUDIO_URL is not set")
    if not requests:
        raise RuntimeError("requests is not available")
    r = requests.post(
        MING_UNIAUDIO_URL + "/tts",
        json={"text": text, "lang": lang},
        timeout=MING_UNIAUDIO_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Ming-UniAudio TTS failed: {r.status_code} {r.text[:200]}")
    return r.content


def _guess_audio_mime(path: str) -> str:
    p = (path or "").lower()
    if p.endswith(".wav"):
        return "audio/wav"
    if p.endswith(".mp3"):
        return "audio/mp3"
    return "audio/webm"


def _ming_uniaudio_asr(audio_path: str) -> str:
    if not _ming_uniaudio_enabled():
        raise RuntimeError("MING_UNIAUDIO_URL is not set")
    if not requests:
        raise RuntimeError("requests is not available")
    mime = _guess_audio_mime(audio_path)
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path) or "audio", f, mime)}
        r = requests.post(
            MING_UNIAUDIO_URL + "/asr",
            files=files,
            timeout=MING_UNIAUDIO_TIMEOUT,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Ming-UniAudio ASR failed: {r.status_code} {r.text[:200]}")
    data = r.json() if r.headers.get("content-type", "").lower().startswith("application/json") else {}
    text = (data.get("text") or data.get("transcript") or "").strip()
    if not text:
        raise RuntimeError("Ming-UniAudio ASR returned empty transcript")
    return text


def _llm_text_generate(prompt: str, *, provider: str) -> str:
    """文本生成（不含音频）。优先按 provider；若不可用则尽量回退。"""
    p = (provider or "").strip().lower()
    if p == "deepseek" and DEEPSEEK_API_KEY:
        completion = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Output only the requested content."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return (completion.choices[0].message.content or "").strip()
    # fallback to Gemini text
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (resp.text or "").strip()
    raise RuntimeError("No text model available (need DEEPSEEK_API_KEY or GEMINI_API_KEY)")


def _speaking_feedback_from_transcript(transcript: str, topic: str, difficulty: str, *, provider: str) -> tuple:
    diff = (difficulty or "C1").strip().upper()
    if diff not in ("B1", "B2", "C1"):
        diff = "C1"
    t = (transcript or "").strip()[:4000]
    prompt = f"""You are an English speaking examiner. This practice was set for {diff} level. The candidate was asked this discussion topic: "{topic}"

Below is the TRANSCRIPT of what the candidate said:
{t}

Please provide FEEDBACK in Chinese, in three clear sections:
- 发音问题：(pronunciation issues, or 无 if none notable)
- 语法问题：(grammar issues, or 无 if none notable)
- 可优化的说法：(suggestions for better wording or expressions)

Then give CEFR assessment using the same criteria as elsewhere (Vocabulary, Sentence structure, Concept & logic). If dimensions disagree, take the HIGHER level.
At the very end, add exactly one line on its own line: CEFR_LEVEL: B1 or CEFR_LEVEL: B2 or CEFR_LEVEL: C1

Output format:
发音问题：
[content]

语法问题：
[content]

可优化的说法：
[content]

CEFR_LEVEL: [B1 or B2 or C1]
"""
    text = _llm_text_generate(prompt, provider=provider)
    cefr_level = ""
    for level in ("C1", "B2", "B1"):
        if f"CEFR_LEVEL: {level}" in text or f"CEFR_LEVEL:{level}" in text:
            cefr_level = level
            break
    if not cefr_level and re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I):
        cefr_level = re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I).group(1).upper()
    return t, text, (cefr_level or "B2")


def _imitation_feedback_from_transcript(transcript: str, reference_text: str, *, provider: str) -> tuple:
    t = (transcript or "").strip()[:4000]
    ref = (reference_text or "").strip()[:3000]
    prompt = f"""You are an English pronunciation and prosody coach. The learner has imitated the following reference text.

REFERENCE TEXT:
\"\"\"{ref}\"\"\"

LEARNER TRANSCRIPT:
\"\"\"{t}\"\"\"

Give detailed feedback in Chinese in three sections:
- 断句与停顿
- 重音
- 连读与省音

Be specific: quote the word or phrase where the deviation occurs and suggest how to improve.

Output format:
断句与停顿：
[content]

重音：
[content]

连读与省音：
[content]
"""
    text = _llm_text_generate(prompt, provider=provider)
    return t, text


def _read_to_speak_feedback_from_transcript(transcript: str, topic: str, article: str, difficulty: str, *, provider: str) -> tuple:
    diff = (difficulty or "C1").strip().upper()
    if diff not in ("B1", "B2", "C1"):
        diff = "C1"
    t = (transcript or "").strip()[:4000]
    article_excerpt = (_strip_html(article) if article else "")[:2500]
    prompt = f"""You are an English speaking examiner. The candidate has just read an article and is answering a discussion topic based on it.
Article excerpt (for context): {article_excerpt}
Topic: \"{topic}\"
Practice level: {diff}

Below is the TRANSCRIPT of what the candidate said:
{t}

Give FEEDBACK in Chinese in FOUR sections:
- 发音问题
- 语法问题
- 可优化的说法
- Upgrade my vocabulary（词汇升级）: suggest higher-level alternatives for simpler words/phrases they used; if already advanced, say 无或较少，可保持.

Then add CEFR assessment line at the end: CEFR_LEVEL: B1/B2/C1

Output format:
发音问题：
[content]

语法问题：
[content]

可优化的说法：
[content]

Upgrade my vocabulary（词汇升级）：
[content]

CEFR_LEVEL: [B1 or B2 or C1]
"""
    text = _llm_text_generate(prompt, provider=provider)
    cefr_level = ""
    for level in ("C1", "B2", "B1"):
        if f"CEFR_LEVEL: {level}" in text or f"CEFR_LEVEL:{level}" in text:
            cefr_level = level
            break
    if not cefr_level and re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I):
        cefr_level = re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I).group(1).upper()
    return t, text, (cefr_level or "B2")


def _current_difficulty():
    """当前会话选用的学习难度（B1/B2/C1）：优先 session，其次请求。"""
    d = session.get("difficulty")
    if d in ("B1", "B2", "C1"):
        return d
    d = (request.form.get("difficulty") or request.args.get("difficulty") or "").strip().upper()
    if d in ("B1", "B2", "C1"):
        return d
    try:
        j = request.get_json(silent=True) or {}
        d = (j.get("difficulty") or "").strip().upper()
        if d in ("B1", "B2", "C1"):
            return d
    except Exception:
        pass
    return "C1"


@app.route("/set-provider", methods=["POST"])
def set_provider():
    """设置当前会话使用的大模型（统一模型选择）。"""
    try:
        data = request.get_json(silent=True) or {}
        p = (data.get("provider") or "").strip().lower()
        if p in ("gemini", "deepseek"):
            session["provider"] = p
            return jsonify({"ok": True, "provider": p})
    except Exception:
        pass
    return jsonify({"error": "无效的 provider。"}), 400


@app.route("/set-difficulty", methods=["POST"])
def set_difficulty():
    """设置当前会话学习难度（全站统一）。"""
    try:
        data = request.get_json(silent=True) or {}
        d = (data.get("difficulty") or "").strip().upper()
        if d in ("B1", "B2", "C1"):
            session["difficulty"] = d
            return jsonify({"ok": True, "difficulty": d})
    except Exception:
        pass
    return jsonify({"error": "无效的 difficulty。"}), 400


@app.route("/favicon.ico")
def favicon():
    """避免浏览器请求 /favicon.ico 时返回 404。"""
    return "", 204


@app.route("/", methods=["GET"])
def index():
    """首页：仅展示表单，点击「获取今日练习」后在新页面打开练习。"""
    provider = session.get("provider", "gemini")
    difficulty = session.get("difficulty", "C1")
    if difficulty not in ("B1", "B2", "C1"):
        difficulty = "C1"
    return render_template_string(
        TEMPLATE,
        topic="",
        provider=provider,
        difficulty=difficulty,
    )


LOADING_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>正在生成 - 今日练习</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 16px; }
    .card { background: rgba(15,23,42,0.9); border-radius: 16px; padding: 40px; max-width: 400px; text-align: center; }
    .spinner { width: 40px; height: 40px; border: 3px solid rgba(148,163,184,0.3); border-top-color: #5eead4; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 20px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    p { color: #9ca3af; font-size: 14px; }
    .err { color: #fecaca; font-size: 13px; margin-top: 12px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="spinner"></div>
    <p>正在生成今日练习，请稍候…</p>
    <p style="font-size: 12px; margin-top: 8px;">AI 生成约 15～30 秒，生成完成后将自动跳转</p>
    <p id="status" style="font-size: 12px; margin-top: 12px; color: #94a3b8;"></p>
    <p id="err" class="err" style="display:none;"></p>
  </div>
  <script>
  (function(){
    var status = document.getElementById("status");
    var errEl = document.getElementById("err");
    function start() {
      // Ensure generation is started even if form submit/popup failed.
      fetch("/daily-practice/start", { method: "POST" })
        .then(function(){ /* ignore; polling will reflect actual state */ })
        .catch(function(){ /* ignore */ });
    }
    function poll() {
      status.textContent = "正在检查…";
      fetch("/daily-practice/article").then(function(r) {
        if (r.ok) {
          status.textContent = "生成完成，正在跳转…";
          window.location.href = "/daily-practice";
          return;
        }
        if (r.status === 500) {
          r.json().then(function(d){
            status.textContent = "";
            errEl.style.display = "block";
            errEl.textContent = (d && d.error) ? d.error : "生成失败，请稍后重试。";
          }).catch(function(){
            status.textContent = "";
            errEl.style.display = "block";
            errEl.textContent = "生成失败，请稍后重试。";
          });
          return;
        }
        status.textContent = (r.status === 202) ? "生成中… 2 秒后自动刷新" : "尚未就绪，2 秒后重试";
        setTimeout(poll, 2000);
      }).catch(function(e) {
        status.textContent = "";
        errEl.style.display = "block";
        errEl.textContent = "请求失败，请刷新重试";
      });
    }
    start();
    setTimeout(poll, 1200);
  })();
  </script>
</body>
</html>
"""


_DAILY_PRACTICE_GEN_LOCK = Lock()
_DAILY_PRACTICE_GEN = {}  # key -> {"state": "running"|"done", "error": str, "started_at": float}


def _daily_practice_gen_state(key: str) -> dict:
    with _DAILY_PRACTICE_GEN_LOCK:
        return dict(_DAILY_PRACTICE_GEN.get(key) or {})


def _ensure_daily_practice_generation(difficulty: str, provider: str, topic: str = "") -> str:
    """确保当日当难度的练习生成线程已启动（用于 loading 页兜底启动）。"""
    diff = (difficulty or "C1").strip().upper()
    if diff not in ("B1", "B2", "C1"):
        diff = "C1"
    today = _today_date()
    key = f"{today}|{diff}"

    # If already generated, nothing to do.
    dp = _load_daily_practice()
    if key in dp.get("by_key", {}):
        return key

    # Provider fallback if key missing.
    p = (provider or "gemini").strip().lower()
    if p == "deepseek" and not DEEPSEEK_API_KEY:
        p = "gemini"
    if p == "gemini" and not GEMINI_API_KEY and DEEPSEEK_API_KEY:
        p = "deepseek"

    with _DAILY_PRACTICE_GEN_LOCK:
        st = _DAILY_PRACTICE_GEN.get(key) or {}
        if st.get("state") == "running":
            return key
        _DAILY_PRACTICE_GEN[key] = {"state": "running", "error": "", "started_at": time.time()}

    def _run():
        try:
            art = get_daily_practice_article(diff, p, topic_for_fallback=topic or "")
            if not art:
                raise RuntimeError("生成失败：未获取到文章内容。")
        except Exception as e:
            traceback.print_exc()
            with _DAILY_PRACTICE_GEN_LOCK:
                _DAILY_PRACTICE_GEN[key] = {
                    "state": "done",
                    "error": str(e) or "生成失败",
                    "started_at": _DAILY_PRACTICE_GEN.get(key, {}).get("started_at", time.time()),
                }
            return
        with _DAILY_PRACTICE_GEN_LOCK:
            _DAILY_PRACTICE_GEN[key] = {
                "state": "done",
                "error": "",
                "started_at": _DAILY_PRACTICE_GEN.get(key, {}).get("started_at", time.time()),
            }

    threading.Thread(target=_run, daemon=True).start()
    return key


@app.route("/daily-practice/start", methods=["POST"])
def daily_practice_start():
    """启动当日练习生成（幂等）。用于 loading 页兜底启动，避免跨窗口提交失败导致一直 404。"""
    try:
        payload = request.get_json(silent=True) or {}
        topic = (payload.get("topic") or "").strip()
    except Exception:
        topic = ""
    key = _ensure_daily_practice_generation(_current_difficulty(), _current_provider(), topic=topic)
    st = _daily_practice_gen_state(key)
    if st.get("state") == "done" and st.get("error"):
        return jsonify({"status": "failed", "error": st.get("error")}), 500
    if st.get("state") == "done":
        return jsonify({"status": "done"}), 200
    return jsonify({"status": "generating"}), 202


@app.route("/daily-practice/loading")
def daily_practice_loading():
    """加载页：点击获取今日练习后立即显示，等待 POST 响应。"""
    # In production, pop-up or cross-window form submit may fail; ensure generation is started here too.
    try:
        _ensure_daily_practice_generation(_current_difficulty(), _current_provider(), topic="")
    except Exception:
        traceback.print_exc()
    return LOADING_TEMPLATE


@app.route("/daily-practice/article", methods=["GET"])
def daily_practice_article_api():
    """API：返回当日练习文章 HTML，供前端 fetch 注入，避免文章内容进入主 HTML 导致脚本解析错误。"""
    difficulty = _current_difficulty()
    if difficulty not in ("B1", "B2", "C1"):
        difficulty = "C1"
    today = _today_date()
    key = f"{today}|{difficulty}"
    data = _load_daily_practice()
    if key not in data.get("by_key", {}):
        st = _daily_practice_gen_state(key)
        if st.get("state") == "running":
            return jsonify({"status": "generating"}), 202
        if st.get("state") == "done" and st.get("error"):
            return jsonify({"error": st.get("error")}), 500
        return jsonify({"error": "今日暂无练习"}), 404
    article = data["by_key"][key]
    article_html = _strip_script_tags(article.get("article_html", ""))
    if not article_html:
        return jsonify({"error": "文章为空"}), 404
    return jsonify({
        "article_html": article_html,
        "article_difficulty": article.get("difficulty", difficulty),
    })


def _pool_has_unread_for_today(today: str, difficulty: str) -> bool:
    """今日缓存池中是否有未读且与难度匹配或任意的文章。"""
    rec = _load_recommend_data()
    assigned = set(rec.get("daily_assigned", {}).get(today, []))
    pool = rec.get("daily", {}).get(today, [])
    for a in pool:
        if (a.get("url") or "") in assigned:
            continue
        if (a.get("difficulty") or "").strip() == difficulty:
            return True
    for a in pool:
        if (a.get("url") or "") in assigned:
            continue
        return True
    return False


@app.route("/daily-practice", methods=["GET", "POST"])
def daily_practice_page():
    """今日练习独立页面：展示文章与三项练习。"""
    topic = ""
    article_html = ""
    vocab_list = []
    provider = _current_provider()
    difficulty = _current_difficulty()
    if difficulty not in ("B1", "B2", "C1"):
        difficulty = "C1"
    article_difficulty = difficulty
    error = ""
    today = _today_date()
    key = f"{today}|{difficulty}"

    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        if provider == "deepseek" and not DEEPSEEK_API_KEY:
            error = "尚未配置 DEEPSEEK_API_KEY，请先在终端里设置后再试。"
        elif provider == "gemini" and not GEMINI_API_KEY:
            error = "尚未配置 GEMINI_API_KEY，请先在终端里设置后再试。"
        else:
            # 已有缓存则直接展示
            dp = _load_daily_practice()
            if key in dp.get("by_key", {}):
                article = dp["by_key"][key]
                article_html = article.get("article_html", "")
                vocab_list = article.get("vocab_list", [])
                article_difficulty = article.get("difficulty", difficulty)
            # 池中有未读则同步取（很快），否则后台生成并返回“请稍候”页
            elif _pool_has_unread_for_today(today, difficulty):
                try:
                    article = get_daily_practice_article(difficulty, provider, topic)
                    if article:
                        article_html = article.get("article_html", "")
                        vocab_list = article.get("vocab_list", [])
                        article_difficulty = article.get("difficulty", difficulty)
                    else:
                        error = "获取今日练习失败，请稍后再试。"
                except Exception:
                    traceback.print_exc()
                    error = "获取今日练习时出错，请检查 API Key 或稍后再试。"
            else:
                # 需 AI 生成：后台执行，立即返回加载页，前端轮询
                import threading
                def _generate():
                    try:
                        get_daily_practice_article(difficulty, provider, topic)
                    except Exception:
                        traceback.print_exc()
                threading.Thread(target=_generate, daemon=True).start()
                return LOADING_TEMPLATE
    else:
        # GET: 从缓存读取当日该难度的练习
        data = _load_daily_practice()
        if key in data.get("by_key", {}):
            article = data["by_key"][key]
            article_html = article.get("article_html", "")
            vocab_list = article.get("vocab_list", [])
            article_difficulty = article.get("difficulty", difficulty)

    if not article_html:
        article_difficulty = difficulty
    article_html = _strip_script_tags(article_html)
    article_html_b64 = base64.b64encode(article_html.encode("utf-8")).decode("ascii") if article_html else ""
    return render_template_string(
        DAILY_PRACTICE_TEMPLATE,
        topic=topic,
        article_html=article_html,
        article_html_b64=article_html_b64 or "",
        vocab_list=vocab_list,
        error=error,
        provider=provider,
        difficulty=difficulty,
        article_difficulty=article_difficulty,
    )


@app.route("/lookup", methods=["POST"])
def lookup():
    payload = request.get_json(silent=True) or {}
    word = (payload.get("word") or "").strip()
    provider = (payload.get("provider") or "").strip() or _current_provider()
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        provider = "gemini"

    if not word:
        return jsonify({"error": "缺少单词参数。"}), 400

    # 如果 DeepSeek 未配置，自动回退到 Gemini
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        provider = "gemini"

    try:
        result = lookup_word(word, provider)
        return jsonify(result)
    except ClientError as e:
        if getattr(e, "code", None) == 429:
            return jsonify({
                "error": "Gemini 免费额度已用尽或请求过于频繁（免费版约 20 次/日）。请稍后再试，或查看用量与计费：https://ai.google.dev/gemini-api/docs/rate-limits"
            }), 429
        traceback.print_exc()
        return jsonify({"error": "查询单词时出错，请稍后再试。"}), 500
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "查询单词时出错，请稍后再试。"}), 500


@app.route("/tts", methods=["POST"])
def tts():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少要朗读的文本。"}), 400

    # 简单限制长度，避免一次生成太长导致失败/太慢
    if len(text) > 4000:
        text = text[:4000]

    try:
        # 1) Prefer Gemini TTS when available and provider is gemini
        if _current_provider() == "gemini" and GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                        )
                    ),
                ),
            )
            pcm = response.candidates[0].content.parts[0].inline_data.data
            wav_bytes = _pcm_to_wav_bytes(pcm)
            return Response(wav_bytes, mimetype="audio/wav")

        # 2) Fallback: Ming-UniAudio
        if _ming_uniaudio_enabled():
            wav = _ming_uniaudio_tts(text, lang="en")
            return Response(wav, mimetype="audio/wav")

        # 3) No available backend
        if _current_provider() == "gemini" and not GEMINI_API_KEY:
            return jsonify({"error": "尚未配置 GEMINI_API_KEY，且未配置 MING_UNIAUDIO_URL，无法朗读。"}), 400
        return jsonify({"error": "当前模型不支持朗读，且未配置 MING_UNIAUDIO_URL。"}), 400
    except ClientError as e:
        # Gemini may be rate-limited; try Ming-UniAudio as fallback.
        if _ming_uniaudio_enabled():
            try:
                wav = _ming_uniaudio_tts(text, lang="en")
                return Response(wav, mimetype="audio/wav")
            except Exception:
                traceback.print_exc()
        if _is_gemini_rate_limited(e):
            return jsonify({"error": "Gemini 朗读额度/频率受限，且 Ming-UniAudio 未成功调用。请稍后再试或检查 MING_UNIAUDIO_URL。"}), 429
        traceback.print_exc()
        return jsonify({"error": "生成音频时出错，请稍后再试。"}), 500
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "生成音频时出错，请稍后再试。"}), 500


VOCAB_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>生词本 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e5e7eb;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
      }
      .card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 16px;
        padding: 28px 24px;
        max-width: 900px;
        width: 100%;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
      }
      h1 { margin: 0 0 10px; font-size: 22px; }
      p { margin: 0; color: #9ca3af; font-size: 13px; line-height: 1.6; }
      a { color: #9ca3af; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .list { margin-top: 14px; display: grid; gap: 10px; }
      .item {
        padding: 12px 12px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.35);
      }
      .word { font-weight: 700; color: #facc15; }
      .phonetic { font-size: 13px; color: #9ca3af; margin-top: 2px; }
      .meaning { color: #e5e7eb; margin-top: 6px; font-size: 13px; }
      .vocab-actions { margin-top: 8px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
      .btn-vocab { padding: 4px 10px; border-radius: 8px; font-size: 12px; border: none; cursor: pointer; }
      .btn-speak { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.5); }
      .btn-speak:hover { background: rgba(34, 197, 94, 0.35); }
      .btn-speak:disabled { opacity: 0.6; cursor: not-allowed; }
      .sentence {
        margin-top: 8px;
        color: #9ca3af;
        font-size: 13px;
        border-left: 3px solid rgba(250, 204, 21, 0.35);
        padding-left: 10px;
      }
      .collocations { margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(148, 163, 184, 0.4); }
      .collocations-title { font-size: 12px; font-weight: 600; color: #5eead4; margin-bottom: 6px; }
      .collocations-list { margin: 0; padding-left: 18px; font-size: 13px; }
      .collocations-list li { margin: 4px 0; color: #e5e7eb; }
      .collocations-list .phrase { color: #facc15; font-weight: 500; }
      .collocations-list .meaning { color: #9ca3af; margin-left: 4px; }
      .btn-colloc { padding: 4px 10px; border-radius: 8px; font-size: 12px; border: 1px solid rgba(94, 234, 212, 0.5); background: rgba(94, 234, 212, 0.15); color: #5eead4; cursor: pointer; }
      .btn-colloc:hover { background: rgba(94, 234, 212, 0.25); }
      .btn-colloc:disabled { opacity: 0.6; cursor: not-allowed; }
      .colloc-status { font-size: 12px; color: #9ca3af; margin-left: 8px; }
      .small { font-size: 12px; color: #6b7280; margin-top: 10px; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>生词本 <span style="display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600; margin-left: 6px;">词汇随文章难度 B1/B2/C1</span></h1>
      <p>
        <a href="/">← 返回首页</a>
        &nbsp;·&nbsp;
        当前共 {{ items|length }} 个单词
      </p>
      <div class="provider-bar" style="margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>

      {% if not items %}
        <p class="small">还没有保存任何单词。回到首页查词后点「添加到生词本」即可。</p>
      {% else %}
        <div class="list">
          {% for it in items %}
            <div class="item">
              <div class="word">{{ it.word }}</div>
              {% if it.phonetic %}
                <div class="phonetic">/ {{ it.phonetic }} /</div>
              {% endif %}
              <div class="meaning">{{ it.meaning_zh }}</div>
              <div class="vocab-actions">
                <button type="button" class="btn-vocab btn-speak" data-word="{{ it.word|e }}" data-sentence="{{ it.sentence_en|default('')|e }}" onclick="speakVocab(this)">朗读</button>
              </div>
              {% if it.sentence_en %}
                <div class="sentence">{{ it.sentence_en }}</div>
              {% endif %}
              {% if it.collocations %}
                <div class="collocations">
                  <div class="collocations-title">高频搭配 (Collocations / 词块)</div>
                  <ul class="collocations-list">
                    {% for c in it.collocations %}
                      <li><span class="phrase">{{ c.phrase|default('') }}</span><span class="meaning">— {{ c.meaning_zh|default('') }}</span></li>
                    {% endfor %}
                  </ul>
                </div>
              {% else %}
                <div class="collocations">
                  <button type="button" class="btn-colloc" onclick="generateCollocations(this, '{{ it.word|e }}')">生成高频搭配</button>
                  <span class="colloc-status" id="status-{{ it.word|replace(' ', '-')|e }}"></span>
                </div>
              {% endif %}
              {% if it.created_at %}
                <div class="small">保存时间：{{ it.created_at }}</div>
              {% endif %}
            </div>
          {% endfor %}
        </div>
      {% endif %}
    </main>
    <script>
      async function speakVocab(btn) {
        var word = (btn.dataset && btn.dataset.word) ? btn.dataset.word : '';
        var sentence = (btn.dataset && btn.dataset.sentence) ? btn.dataset.sentence : '';
        var text = word + (sentence ? ' . ' + sentence : '');
        if (!text.trim()) return;
        btn.disabled = true;
        try {
          var r = await fetch('/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) });
          if (!r.ok) { var d = await r.json().catch(function() {}); alert(d.error || '朗读失败'); return; }
          var blob = await r.blob();
          var url = URL.createObjectURL(blob);
          var audio = new Audio(url);
          await audio.play();
        } catch (e) { alert('朗读失败'); }
        finally { btn.disabled = false; }
      }
      async function generateCollocations(btn, word) {
        if (!word) return;
        var id = 'status-' + word.replace(/\\s+/g, '-');
        var statusEl = document.getElementById(id);
        if (statusEl) statusEl.textContent = '生成中…';
        btn.disabled = true;
        try {
          var r = await fetch('/vocab/collocations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ word: word }) });
          var d = await r.json().catch(function() {});
          if (r.ok && d.ok) { if (statusEl) statusEl.textContent = '已生成'; window.location.reload(); }
          else { if (statusEl) statusEl.textContent = d.error || '生成失败'; btn.disabled = false; }
        } catch (e) { if (statusEl) statusEl.textContent = '请求失败'; btn.disabled = false; }
      }
    </script>
  </body>
</html>
"""


@app.route("/vocab", methods=["GET"])
def vocab_page():
    items = _load_vocab_book()
    items = list(reversed(items))
    return render_template_string(VOCAB_TEMPLATE, items=items, provider=session.get("provider", "gemini"), difficulty=session.get("difficulty", "C1"))


@app.route("/vocab/add", methods=["POST"])
def vocab_add():
    payload = request.get_json(silent=True) or {}
    word = (payload.get("word") or "").strip()
    meaning_zh = (payload.get("meaning_zh") or "").strip()
    sentence_en = (payload.get("sentence_en") or "").strip()
    phonetic = (payload.get("phonetic") or "").strip()
    provider = (payload.get("provider") or "").strip() or _current_provider()

    if not word:
        return jsonify({"error": "缺少 word。"}), 400

    items = _load_vocab_book()
    word_key = word.lower()
    target = None
    updated = False

    for it in items:
        if isinstance(it, dict) and (it.get("word") or "").lower() == word_key:
            if meaning_zh:
                it["meaning_zh"] = meaning_zh
            if sentence_en:
                it["sentence_en"] = sentence_en
            if phonetic:
                it["phonetic"] = phonetic
            it["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            target = it
            break

    if not updated:
        target = {
            "word": word,
            "meaning_zh": meaning_zh,
            "sentence_en": sentence_en,
            "phonetic": phonetic or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        items.append(target)

    # 若无音标则自动获取并保存
    if target and not (target.get("phonetic") or "").strip():
        try:
            lookup_result = lookup_word(word, provider)
            target["phonetic"] = (lookup_result.get("phonetic") or "").strip()
        except Exception:
            traceback.print_exc()

    # 若无高频搭配则自动生成（词块 / Collocations）
    if target and not target.get("collocations"):
        target["collocations"] = _get_collocations(
            word, target.get("meaning_zh") or meaning_zh, provider
        )

    _save_vocab_book(items)
    return jsonify({"ok": True, "updated": updated})


@app.route("/vocab/collocations", methods=["POST"])
def vocab_collocations():
    """为生词本中已有单词生成高频搭配（无则生成，有则覆盖）。"""
    payload = request.get_json(silent=True) or {}
    word = (payload.get("word") or "").strip()
    provider = (payload.get("provider") or "").strip() or _current_provider()
    if not word:
        return jsonify({"error": "缺少 word。"}), 400
    items = _load_vocab_book()
    word_key = word.lower()
    for it in items:
        if isinstance(it, dict) and (it.get("word") or "").lower() == word_key:
            meaning_zh = (it.get("meaning_zh") or "").strip()
            it["collocations"] = _get_collocations(word, meaning_zh, provider)
            it["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_vocab_book(items)
            return jsonify({"ok": True, "collocations": it["collocations"]})
    return jsonify({"error": "生词本中未找到该词。"}), 404


RECOMMEND_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>阅读推荐 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 720px; width: 100%; margin: 0 auto 20px; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      h1 { margin: 0 0 10px; font-size: 22px; }
      .article-section { margin-top: 22px; padding: 14px 12px; border-radius: 12px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.35); }
      .article-section h2 { margin: 0 0 6px; font-size: 18px; }
      .article-body { font-size: 14px; line-height: 1.7; color: #e5e7eb; }
      .article-body mark { background: rgba(250, 204, 21, 0.2); color: #facc15; padding: 0 2px; border-radius: 3px; }
      .vocab-section { margin-top: 14px; padding-top: 10px; border-top: 1px dashed rgba(148, 163, 184, 0.6); }
      .vocab-list { margin: 8px 0 0; padding-left: 16px; font-size: 13px; color: #e5e7eb; }
      .vocab-list li span.word { font-weight: 600; color: #facc15; }
      .vocab-list li span.meaning { color: #9ca3af; }
      .btn { padding: 7px 14px; border-radius: 999px; font-weight: 600; font-size: 12px; border: none; cursor: pointer; background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; margin-right: 8px; margin-top: 8px; }
      .btn:disabled { opacity: 0.7; }
      .lookup-box { margin-top: 12px; padding: 10px; border-radius: 10px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.35); display: none; }
      .lookup-box.show { display: block; }
      .small { font-size: 12px; color: #9ca3af; }
      a { color: #5eead4; text-decoration: none; }
      a:hover { text-decoration: underline; }
      #loading { margin-top: 12px; color: #9ca3af; }
      .error { color: #fecaca; font-size: 13px; margin-top: 10px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>今日阅读推荐 <span style="display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600; margin-left: 6px;">每篇已标 CEFR 难度 B1/B2/C1</span></h1>
      <p><a href="/">← 返回首页</a></p>
      <div class="provider-bar" style="margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>
      <p class="small">每日从 <a href="https://www.21voa.com/" target="_blank" rel="noreferrer">21voa</a>、<a href="https://www.i21st.cn/" target="_blank" rel="noreferrer">21世纪报</a>、<a href="https://www.buzzing.cc/" target="_blank" rel="noreferrer">Buzzing</a> 各选 1 篇，经 AI 按<strong>同一套 CEFR 标准</strong>（词汇、句式、概念与逻辑）评估难度并标注对应级别词汇与释义（朗读、查词、加入生词本）。</p>
    </div>

    <div id="loading" class="card" style="display: none;">正在拉取今日推荐…（首次会爬取并标注，请稍候）</div>
    <div id="error" class="card error" style="display: none;"></div>
    <div id="articles"></div>

    <script>
      const provider = "{{ provider }}";
      (async function() {
        const loading = document.getElementById("loading");
        const errEl = document.getElementById("error");
        const container = document.getElementById("articles");
        loading.style.display = "block";
        errEl.style.display = "none";
        try {
          const r = await fetch("/reading/data?provider=" + encodeURIComponent(provider));
          const d = await r.json();
          loading.style.display = "none";
          if (d.error) { errEl.textContent = d.error; errEl.style.display = "block"; return; }
          const articles = d.articles || [];
          if (!articles.length) { errEl.textContent = "今日暂无推荐，或爬取失败，请改日再试。"; errEl.style.display = "block"; return; }
          articles.forEach((a, i) => {
            const card = document.createElement("div");
            card.className = "card";
            card.innerHTML = `
              <section class="article-section" data-idx="${i}">
                <h2>${escapeHtml(a.title)} <span style="padding: 2px 6px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600; margin-left: 6px;">${(a.difficulty || 'C1')}</span></h2>
                <p class="small"><a href="${escapeHtml(a.url)}" target="_blank" rel="noreferrer">${escapeHtml(a.source)} · 原文链接</a></p>
                <div class="article-body">${a.article_html}</div>
                <div style="margin-top:10px;">
                  <button type="button" class="btn" onclick="speakArticle(${i})">朗读</button>
                  <button type="button" class="btn" onclick="lookupInArticle(${i})">查选中单词</button>
                </div>
                <div class="lookup-box" id="lookup-${i}">
                  <div id="lookup-word-${i}" style="font-weight:600;"></div>
                  <div id="lookup-phonetic-${i}" class="small"></div>
                  <div id="lookup-meaning-${i}" class="small"></div>
                  <div id="lookup-example-en-${i}" class="small"></div>
                  <div id="lookup-example-zh-${i}" class="small"></div>
                  <button type="button" class="btn" onclick="addToVocabFrom(${i})">添加到生词本</button>
                </div>
                ${(a.vocab_list && a.vocab_list.length) ? `
                <div class="vocab-section">
                  <p class="small">高亮词汇及中文释义：</p>
                  <ul class="vocab-list">
                    ${a.vocab_list.map(v => '<li><span class="word">' + escapeHtml(v.word) + '</span><span class="meaning"> — ' + escapeHtml(v.meaning_zh) + '</span></li>').join('')}
                  </ul>
                </div>
                ` : ''}
                <div class="imitation-section" style="margin-top: 14px; padding: 12px; border-radius: 10px; background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.25);">
                  <p class="small" style="color: #94a3b8; margin-bottom: 8px;"><strong>断句模仿</strong>：选中本段文字后点「刷新选中」再录音，AI 分析断句/重音/连读。</p>
                  <div style="margin-bottom: 8px;"><span class="small" style="color: #9ca3af;">当前选中：</span><span id="imitation-selected-text-${i}" class="small" style="color: #cbd5e1; font-style: italic;">（请在此文章中拖动选中一段）</span></div>
                  <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" onclick="refreshImitationSelection(${i})">刷新选中</button>
                    <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-record-btn-${i}" onclick="toggleImitationRecord(${i})">开始模仿录音</button>
                    <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px; display: none;" id="imitation-stop-btn-${i}" onclick="stopImitationRecord()">停止录音</button>
                    <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-submit-btn-${i}" onclick="submitImitation(${i})">提交模仿</button>
                    <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" onclick="playImitationReference(${i})">播放参考朗读</button>
                  </div>
                  <div id="imitation-result-${i}" style="margin-top: 12px; padding: 10px; border-radius: 8px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.35); display: none;">
                    <div class="small" style="color: #94a3b8; margin-bottom: 4px;">识别结果：</div>
                    <div id="imitation-transcript-${i}" class="small" style="color: #e5e7eb; margin-bottom: 10px;"></div>
                    <div class="small" style="color: #94a3b8; margin-bottom: 4px;">偏差分析：</div>
                    <div id="imitation-feedback-${i}" class="small" style="color: #e5e7eb; white-space: pre-wrap;"></div>
                  </div>
                </div>
              </section>
            `;
            container.appendChild(card);
          });
        } catch (e) {
          loading.style.display = "none";
          errEl.textContent = "加载失败，请稍后再试。";
          errEl.style.display = "block";
        }
      })();
      function escapeHtml(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
      let lastLookup = null;
      let lastContextSentence = '';
      let lastLookupIdx = -1;
      async function speakArticle(idx) {
        const section = document.querySelector('.article-section[data-idx="'+idx+'"]');
        const body = section ? section.querySelector('.article-body') : null;
        const text = body ? body.innerText.trim() : '';
        if (!text) { alert('无内容可朗读'); return; }
        try {
          const r = await fetch("/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.slice(0, 4000) }) });
          if (!r.ok) { const d = await r.json().catch(()=>{}); alert(d.error || '朗读失败'); return; }
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          await audio.play();
        } catch (e) { alert('朗读失败'); }
      }
      async function lookupInArticle(idx) {
        const sel = window.getSelection ? window.getSelection().toString().trim() : '';
        if (!sel) { alert('请先选中一个英文单词'); return; }
        let word = sel.split(/\\s+/)[0].replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, '');
        if (!word) { alert('请选中有效英文单词'); return; }
        const box = document.getElementById('lookup-' + idx);
        const wEl = document.getElementById('lookup-word-' + idx);
        box.classList.add('show');
        wEl.textContent = '查询中…';
        try {
          const r = await fetch('/lookup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ word, provider }) });
          const d = await r.json();
          lastLookup = d; lastLookupIdx = idx;
          const section = document.querySelector('.article-section[data-idx="'+idx+'"]');
          const body = section ? section.querySelector('.article-body') : null;
          lastContextSentence = body ? body.innerText.trim() : '';
          if (d.error) { wEl.textContent = '查询失败'; document.getElementById('lookup-meaning-' + idx).textContent = d.error; return; }
          document.getElementById('lookup-word-' + idx).textContent = d.word || word;
          document.getElementById('lookup-phonetic-' + idx).textContent = d.phonetic ? '/' + d.phonetic + '/' : '';
          document.getElementById('lookup-meaning-' + idx).textContent = d.meaning_zh || '';
          document.getElementById('lookup-example-en-' + idx).textContent = d.example_en ? '例句: ' + d.example_en : '';
          document.getElementById('lookup-example-zh-' + idx).textContent = d.example_zh ? '译文: ' + d.example_zh : '';
        } catch (e) { wEl.textContent = '查询失败'; }
      }
      async function addToVocabFrom(idx) {
        if (!lastLookup || lastLookupIdx !== idx) { alert('请先在该文章中查词'); return; }
        try {
          const r = await fetch('/vocab/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ word: lastLookup.word, meaning_zh: lastLookup.meaning_zh || '', sentence_en: lastContextSentence || lastLookup.example_en || '', phonetic: lastLookup.phonetic || '', provider }) });
          const d = await r.json().catch(()=>{});
          if (r.ok && !d.error) alert('已加入生词本'); else alert(d.error || '保存失败');
        } catch (e) { alert('保存失败'); }
      }
      let imitationSelectedText = '';
      let imitationCardIdx = -1;
      let imitationChunks = [];
      let imitationMediaRecorder = null;
      function refreshImitationSelection(idx) {
        const sel = window.getSelection ? window.getSelection().toString().trim() : '';
        const section = document.querySelector('.article-section[data-idx="'+idx+'"]');
        const body = section ? section.querySelector('.article-body') : null;
        if (!body) return;
        const anchor = window.getSelection() && window.getSelection().anchorNode;
        const inBody = anchor && body.contains(anchor);
        if (sel && inBody) { imitationSelectedText = sel; imitationCardIdx = idx; }
        const span = document.getElementById('imitation-selected-text-'+idx);
        if (span) span.textContent = (imitationCardIdx === idx && imitationSelectedText) ? ('\"'+imitationSelectedText.slice(0,80)+(imitationSelectedText.length>80?'…':'')+'\"') : '（请在此文章中拖动选中一段）';
      }
      function toggleImitationRecord(idx) {
        if (imitationMediaRecorder && imitationMediaRecorder.state === 'recording') { imitationMediaRecorder.stop(); return; }
        if (!imitationSelectedText || imitationCardIdx !== idx) { refreshImitationSelection(idx); if (!imitationSelectedText || imitationCardIdx !== idx) { alert('请先在此文章中选中要模仿的段落，再点「刷新选中」。'); return; } }
        imitationChunks = [];
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
          imitationMediaRecorder = new MediaRecorder(stream);
          imitationMediaRecorder.ondataavailable = function(e) { if (e.data.size) imitationChunks.push(e.data); };
          imitationMediaRecorder.onstop = function() {
            stream.getTracks().forEach(function(t) { t.stop(); });
            for (let i = 0; i < 10; i++) {
              const rb = document.getElementById('imitation-record-btn-'+i); const sb = document.getElementById('imitation-stop-btn-'+i);
              if (rb) rb.style.display = 'inline-block'; if (sb) sb.style.display = 'none';
            }
          };
          imitationMediaRecorder.start();
          const rb = document.getElementById('imitation-record-btn-'+idx); const sb = document.getElementById('imitation-stop-btn-'+idx);
          if (rb) rb.style.display = 'none'; if (sb) sb.style.display = 'inline-block';
        }).catch(function() { alert('无法访问麦克风，请检查权限。'); });
      }
      function stopImitationRecord() {
        if (imitationMediaRecorder && imitationMediaRecorder.state === 'recording') imitationMediaRecorder.stop();
        for (let i = 0; i < 10; i++) {
          const rb = document.getElementById('imitation-record-btn-'+i); const sb = document.getElementById('imitation-stop-btn-'+i);
          if (rb) rb.style.display = 'inline-block'; if (sb) sb.style.display = 'none';
        }
      }
      async function submitImitation(idx) {
        if (imitationCardIdx !== idx || !imitationSelectedText) { refreshImitationSelection(idx); if (imitationCardIdx !== idx || !imitationSelectedText) { alert('请先选中要模仿的段落并完成录音后再提交。'); return; } }
        if (!imitationChunks.length) { alert('请先点击「开始模仿录音」并录一段音后再提交。'); return; }
        const btn = document.getElementById('imitation-submit-btn-'+idx);
        if (btn) btn.disabled = true;
        const blob = new Blob(imitationChunks, { type: 'audio/webm' });
        const form = new FormData();
        form.append('text', imitationSelectedText);
        form.append('audio', blob, 'imitation.webm');
        try {
          const resp = await fetch('/imitation/feedback', { method: 'POST', body: form });
          const data = await resp.json().catch(function() { return {}; });
          const resultEl = document.getElementById('imitation-result-'+idx);
          const transEl = document.getElementById('imitation-transcript-'+idx);
          const feedEl = document.getElementById('imitation-feedback-'+idx);
          if (!resp.ok || data.error) { if (transEl) transEl.textContent = ''; if (feedEl) feedEl.textContent = data.error || '分析失败，请稍后再试。'; if (resultEl) resultEl.style.display = 'block'; }
          else { if (transEl) transEl.textContent = data.transcript || '（无）'; if (feedEl) feedEl.textContent = data.feedback || ''; if (resultEl) resultEl.style.display = 'block'; }
        } catch (e) { const feedEl = document.getElementById('imitation-feedback-'+idx); if (feedEl) feedEl.textContent = '网络或服务器错误，请稍后再试。'; const resultEl = document.getElementById('imitation-result-'+idx); if (resultEl) resultEl.style.display = 'block'; }
        if (btn) btn.disabled = false;
      }
      async function playImitationReference(idx) {
        if (imitationCardIdx !== idx || !imitationSelectedText) refreshImitationSelection(idx);
        if (imitationCardIdx !== idx || !imitationSelectedText) { alert('请先在此文章中选中要模仿的段落。'); return; }
        try {
          const resp = await fetch('/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: imitationSelectedText }) });
          if (!resp.ok) { const d = await resp.json().catch(()=>{}); alert(d.error || '生成参考朗读失败'); return; }
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          audio.onended = function() { URL.revokeObjectURL(url); };
          await audio.play();
        } catch (e) { alert('播放失败'); }
      }
    </script>
  </body>
</html>
"""


@app.route("/reading", methods=["GET"])
def reading_page():
    provider = (request.args.get("provider") or "").strip() or _current_provider()
    return render_template_string(RECOMMEND_TEMPLATE, provider=provider, difficulty=session.get("difficulty", "C1"))


@app.route("/reading/data", methods=["GET"])
def reading_data():
    provider = (request.args.get("provider") or "").strip() or _current_provider()
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        provider = "gemini"
    if provider == "gemini" and not GEMINI_API_KEY:
        return jsonify({"error": "请先配置 GEMINI_API_KEY 以使用阅读推荐。"}), 400
    try:
        articles = get_today_recommendations(provider)
        for a in articles:
            if isinstance(a, dict) and "article_html" in a:
                a["article_html"] = _strip_script_tags(a.get("article_html", ""))
        return jsonify({"articles": articles})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e) or "获取推荐失败，请稍后再试。"}), 500


HISTORY_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>历史练习 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 900px; width: 100%; margin: 0 auto; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      h1 { margin: 0 0 10px; font-size: 22px; }
      a { color: #9ca3af; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .button-link { color: #5eead4; }
      .filter { margin-top: 14px; padding: 12px; border-radius: 10px; background: rgba(15,23,42,0.75); border: 1px solid rgba(148,163,184,0.35); }
      .filter label { display: inline-block; margin-right: 8px; font-size: 13px; color: #9ca3af; }
      .filter input, .filter select { padding: 6px 10px; border-radius: 6px; border: 1px solid rgba(148,163,184,0.6); background: rgba(15,23,42,0.8); color: #e5e7eb; font-size: 13px; margin-right: 12px; margin-bottom: 8px; }
      .filter button { padding: 6px 14px; border-radius: 999px; background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; font-weight: 600; font-size: 13px; border: none; cursor: pointer; }
      .list { margin-top: 16px; display: grid; gap: 10px; }
      .item { padding: 12px 14px; border-radius: 12px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.35); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
      .item-title { font-weight: 600; color: #e5e7eb; }
      .item-meta { font-size: 12px; color: #9ca3af; }
      .item a { color: #5eead4; }
      .small { font-size: 12px; color: #6b7280; margin-top: 10px; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>历史练习 <span style="display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600; margin-left: 6px;">已标难度 B1/B2/C1</span></h1>
      <p>
        <a href="/">← 返回首页</a>
        &nbsp;·&nbsp;
        共 {{ total }} 篇（当前筛选后 {{ items|length }} 篇）
      </p>
      <div class="provider-bar" style="margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>

      <form class="filter" method="get" action="/history">
        <label>日期：</label>
        <input type="date" name="date" value="{{ request_date|default('') }}" />
        <label>主题关键词：</label>
        <input type="text" name="topic" value="{{ request_topic|default('') }}" placeholder="输入主题关键词" style="min-width: 120px;" />
        <label>难度：</label>
        <select name="difficulty">
          <option value="">全部</option>
          <option value="C1" {{ 'selected' if request_difficulty == 'C1' else '' }}>C1</option>
          <option value="B2" {{ 'selected' if request_difficulty == 'B2' else '' }}>B2</option>
          <option value="B1" {{ 'selected' if request_difficulty == 'B1' else '' }}>B1</option>
        </select>
        <button type="submit">筛选</button>
      </form>

      {% if not items %}
        <p class="small">暂无文章，或当前筛选无结果。在首页生成文章后会自动出现在这里。</p>
      {% else %}
        <div class="list">
          {% for it in items %}
            <div class="item">
              <div>
                <div class="item-title">{{ it.topic }}</div>
                <div class="item-meta">{{ it.created_at[:10] }} · <span style="padding: 2px 6px; border-radius: 999px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-size: 11px; font-weight: 600;">{{ it.difficulty }}</span> · {{ it.provider }}</div>
              </div>
              <a href="{{ it.view_url }}">查看</a>
            </div>
          {% endfor %}
        </div>
      {% endif %}
    </main>
  </body>
</html>
"""


HISTORY_VIEW_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>回顾 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 720px; width: 100%; margin: 0 auto; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      h1 { margin: 0 0 10px; font-size: 20px; }
      .article-body { font-size: 14px; line-height: 1.7; color: #e5e7eb; margin-top: 12px; }
      .article-body mark { background: rgba(250, 204, 21, 0.2); color: #facc15; padding: 0 2px; border-radius: 3px; }
      .vocab-section { margin-top: 14px; padding-top: 10px; border-top: 1px dashed rgba(148, 163, 184, 0.6); }
      .vocab-list { margin: 8px 0 0; padding-left: 16px; font-size: 13px; color: #e5e7eb; }
      .vocab-list li span.word { font-weight: 600; color: #facc15; }
      .vocab-list li span.meaning { color: #9ca3af; }
      .small { font-size: 12px; color: #9ca3af; }
      a { color: #5eead4; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .btn { padding: 7px 14px; border-radius: 999px; font-weight: 600; font-size: 12px; border: none; cursor: pointer; background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; margin-right: 8px; margin-top: 8px; }
      .btn:disabled { opacity: 0.7; cursor: not-allowed; }
      .lookup-result { margin-top: 12px; padding: 10px; border-radius: 10px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.35); display: none; }
      .lookup-result.show { display: block; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>{{ article.topic }} <span style="display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(250, 204, 21, 0.25); color: #facc15; font-size: 12px; font-weight: 600; margin-left: 6px;">{{ article.difficulty }}</span></h1>
      <p class="small">{{ article.created_at[:10] }} · 内容难度 {{ article.difficulty }}</p>
      <p><a href="/history">← 返回历史练习</a> · <a href="/vocab">生词本</a></p>
      <div class="provider-bar" style="margin-top: 8px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>

      <div class="article-body">
        {{ article.article_html|safe }}
      </div>

      <div style="margin-top: 14px;">
        <button id="history-tts-btn" type="button" class="btn" onclick="speakArticle()">朗读</button>
        <span class="small" style="margin-left: 6px; color: #9ca3af;">点击后生成音频并播放（Gemini TTS）</span>
      </div>
      <audio id="history-tts-audio" controls style="width: 100%; margin-top: 10px; display: none;"></audio>

      <div style="margin-top: 10px;">
        <button type="button" class="btn" onclick="lookupSelection()">查选中单词</button>
        <span class="small" style="margin-left: 6px; color: #9ca3af;">先选中文中单词再点击</span>
      </div>
      <div id="history-lookup-result" class="lookup-result">
        <div id="history-lookup-word" style="font-weight: 600; color: #e5e7eb;"></div>
        <div id="history-lookup-phonetic" class="small" style="color: #9ca3af; margin-top: 2px;"></div>
        <div id="history-lookup-meaning" class="small" style="margin-top: 6px; color: #e5e7eb;"></div>
        <div id="history-lookup-example-en" class="small" style="margin-top: 8px; color: #9ca3af;"></div>
        <div id="history-lookup-example-zh" class="small" style="margin-top: 2px; color: #9ca3af;"></div>
        <div style="margin-top: 10px;">
          <button id="history-add-vocab-btn" type="button" class="btn" onclick="addToVocab()">添加到生词本</button>
          <span id="history-add-vocab-status" class="small" style="margin-left: 6px; color: #9ca3af;"></span>
        </div>
      </div>

      <div class="imitation-section" style="margin-top: 18px; padding: 12px; border-radius: 10px; background: rgba(30,41,59,0.6); border: 1px solid rgba(148,163,184,0.25);">
        <p class="small" style="color: #94a3b8; margin-bottom: 8px;"><strong>断句模仿</strong>：选中一段文字后录音模仿，AI 将分析断句、重音、连读的偏差。</p>
        <div style="margin-bottom: 8px;">
          <span class="small" style="color: #9ca3af;">当前选中：</span>
          <span id="imitation-selected-text" class="small" style="color: #cbd5e1; font-style: italic;">（请在文章中拖动选中一段内容）</span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
          <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-refresh-btn" onclick="refreshImitationSelection()">刷新选中</button>
          <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-record-btn" onclick="toggleImitationRecord()">开始模仿录音</button>
          <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px; display: none;" id="imitation-stop-btn" onclick="stopImitationRecord()">停止录音</button>
          <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-submit-btn" onclick="submitImitation()">提交模仿</button>
          <button type="button" class="btn" style="padding: 7px 14px; font-size: 12px;" id="imitation-tts-btn" onclick="playImitationReference()">播放参考朗读</button>
        </div>
        <div id="imitation-result" style="margin-top: 12px; padding: 10px; border-radius: 8px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.35); display: none;">
          <div class="small" style="color: #94a3b8; margin-bottom: 4px;">识别结果：</div>
          <div id="imitation-transcript" class="small" style="color: #e5e7eb; margin-bottom: 10px;"></div>
          <div class="small" style="color: #94a3b8; margin-bottom: 4px;">偏差分析：</div>
          <div id="imitation-feedback" class="small" style="color: #e5e7eb; white-space: pre-wrap;"></div>
        </div>
      </div>

      {% if article.vocab_list %}
        <div class="vocab-section">
          <p class="small">高亮词汇及中文释义：</p>
          <ul class="vocab-list">
            {% for item in article.vocab_list %}
              <li><span class="word">{{ item.word }}</span><span class="meaning"> — {{ item.meaning_zh }}</span></li>
            {% endfor %}
          </ul>
        </div>
      {% endif %}
    </main>
    <script>
      var currentProvider = "{{ provider|default('gemini') }}";
      var lastLookup = null;
      var lastContextSentence = "";

      async function speakArticle() {
        var btn = document.getElementById("history-tts-btn");
        var audio = document.getElementById("history-tts-audio");
        var articleEl = document.querySelector(".article-body");
        var text = articleEl ? (articleEl.innerText || "").trim() : "";
        if (!text) { alert("没有找到文章内容，无法朗读。"); return; }
        btn.disabled = true;
        var oldText = btn.textContent;
        btn.textContent = "生成中…";
        try {
          var resp = await fetch("/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text }) });
          if (!resp.ok) {
            var data = await resp.json().catch(function() {}); alert(data.error || "生成音频失败"); return;
          }
          var blob = await resp.blob();
          var url = URL.createObjectURL(blob);
          audio.src = url;
          audio.style.display = "block";
          await audio.play();
        } catch (e) { alert("网络或服务器错误"); } finally { btn.disabled = false; btn.textContent = oldText; }
      }

      async function lookupSelection() {
        var sel = window.getSelection ? window.getSelection().toString().trim() : "";
        if (!sel) { alert("请先在文章中选中一个英文单词。"); return; }
        var word = sel.split(/\\s+/)[0] || "";
        word = word.replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, "");
        if (!word) { alert("未选中有效英文单词"); return; }
        var box = document.getElementById("history-lookup-result");
        var wEl = document.getElementById("history-lookup-word");
        var pEl = document.getElementById("history-lookup-phonetic");
        var mEl = document.getElementById("history-lookup-meaning");
        var eEn = document.getElementById("history-lookup-example-en");
        var eZh = document.getElementById("history-lookup-example-zh");
        wEl.textContent = "正在查询：" + word + " …";
        pEl.textContent = ""; mEl.textContent = ""; eEn.textContent = ""; eZh.textContent = "";
        document.getElementById("history-add-vocab-status").textContent = "";
        lastLookup = null; lastContextSentence = "";
        box.className = "lookup-result show";
        try {
          var selection = window.getSelection ? window.getSelection() : null;
          var anchor = selection && selection.anchorNode ? selection.anchorNode : null;
          var el = anchor && anchor.parentElement ? anchor.parentElement : null;
          while (el && el !== document.body && !el.classList.contains("article-body") && el.tagName !== "P") { el = el.parentElement; }
          if (el && el.tagName === "P") lastContextSentence = (el.innerText || "").trim();
          else { var articleEl = document.querySelector(".article-body"); lastContextSentence = articleEl ? (articleEl.innerText || "").trim() : ""; }
        } catch (e) {}
        try {
          var resp = await fetch("/lookup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ word: word, provider: currentProvider }) });
          var data = await resp.json();
          if (data.error) { wEl.textContent = "查询失败"; mEl.textContent = data.error; return; }
          wEl.textContent = data.word || word;
          pEl.textContent = data.phonetic ? "/" + data.phonetic + "/" : "";
          mEl.textContent = data.meaning_zh || "";
          eEn.textContent = data.example_en ? "例句: " + data.example_en : "";
          eZh.textContent = data.example_zh ? "译文: " + data.example_zh : "";
          lastLookup = data;
          document.getElementById("history-add-vocab-btn").disabled = false;
        } catch (e) { wEl.textContent = "查询失败"; mEl.textContent = "网络或服务器错误"; }
      }

      async function addToVocab() {
        var btn = document.getElementById("history-add-vocab-btn");
        var status = document.getElementById("history-add-vocab-status");
        if (!lastLookup || !lastLookup.word) { status.textContent = "请先查词"; return; }
        btn.disabled = true;
        status.textContent = "保存中…";
        try {
          var resp = await fetch("/vocab/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ word: lastLookup.word, meaning_zh: lastLookup.meaning_zh || "", sentence_en: lastContextSentence || lastLookup.example_en || "", phonetic: lastLookup.phonetic || "", provider: currentProvider }) });
          var data = await resp.json().catch(function() {});
          if (!resp.ok || data.error) { status.textContent = data.error || "保存失败"; return; }
          status.textContent = "已保存。去生词本看看 →";
        } catch (e) { status.textContent = "网络或服务器错误"; } finally { btn.disabled = false; }
      }

      var imitationSelectedText = "";
      var imitationMediaRecorder = null;
      var imitationChunks = [];
      function refreshImitationSelection() {
        var sel = window.getSelection ? window.getSelection().toString().trim() : "";
        var articleEl = document.querySelector(".article-body");
        var anchor = window.getSelection && window.getSelection().anchorNode;
        if (!articleEl) imitationSelectedText = "";
        else if (sel && anchor && articleEl.contains(anchor)) imitationSelectedText = sel;
        var span = document.getElementById("imitation-selected-text");
        if (span) span.textContent = imitationSelectedText ? ("\\\"" + imitationSelectedText.slice(0, 80) + (imitationSelectedText.length > 80 ? "…" : "") + "\\\"") : "（请在文章中拖动选中一段内容）";
      }
      function toggleImitationRecord() {
        try {
          if (imitationMediaRecorder && imitationMediaRecorder.state === "recording") { imitationMediaRecorder.stop(); return; }
          if (!imitationSelectedText) { refreshImitationSelection(); if (!imitationSelectedText) { alert("请先在文章中选中要模仿的段落，再点击「刷新选中」，然后开始录音。"); return; } }
          imitationChunks = [];
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { alert("您的浏览器不支持录音，请使用 Chrome/Edge 并允许麦克风权限。"); return; }
          navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
            imitationMediaRecorder = new MediaRecorder(stream);
            imitationMediaRecorder.ondataavailable = function(e) { if (e.data.size) imitationChunks.push(e.data); };
            imitationMediaRecorder.onstop = function() {
              stream.getTracks().forEach(function(t) { t.stop(); });
              var rb = document.getElementById("imitation-record-btn"); var sb = document.getElementById("imitation-stop-btn");
              if (rb) rb.style.display = "inline-block"; if (sb) sb.style.display = "none";
            };
            imitationMediaRecorder.start();
            var rb = document.getElementById("imitation-record-btn"); var sb = document.getElementById("imitation-stop-btn");
            if (rb) rb.style.display = "none"; if (sb) sb.style.display = "inline-block";
          }).catch(function() { alert("无法访问麦克风，请检查权限。"); });
        } catch (err) { alert("录音出错，请重试。"); }
      }
      function stopImitationRecord() {
        if (imitationMediaRecorder && imitationMediaRecorder.state === "recording") imitationMediaRecorder.stop();
        document.getElementById("imitation-record-btn").style.display = "inline-block";
        document.getElementById("imitation-stop-btn").style.display = "none";
      }
      async function submitImitation() {
        if (!imitationSelectedText) { refreshImitationSelection(); if (!imitationSelectedText) { alert("请先选中要模仿的段落并完成录音后再提交。"); return; } }
        if (!imitationChunks.length) { alert("请先点击「开始模仿录音」并录一段音后再提交。"); return; }
        var btn = document.getElementById("imitation-submit-btn");
        btn.disabled = true;
        var blob = new Blob(imitationChunks, { type: "audio/webm" });
        var form = new FormData();
        form.append("text", imitationSelectedText);
        form.append("audio", blob, "imitation.webm");
        try {
          var resp = await fetch("/imitation/feedback", { method: "POST", body: form });
          var data = await resp.json().catch(function() { return {}; });
          var resultEl = document.getElementById("imitation-result");
          var transEl = document.getElementById("imitation-transcript");
          var feedEl = document.getElementById("imitation-feedback");
          if (!resp.ok || data.error) { transEl.textContent = ""; feedEl.textContent = data.error || "分析失败，请稍后再试。"; resultEl.style.display = "block"; }
          else { transEl.textContent = data.transcript || "（无）"; feedEl.textContent = data.feedback || ""; resultEl.style.display = "block"; }
        } catch (e) { document.getElementById("imitation-feedback").textContent = "网络或服务器错误，请稍后再试。"; document.getElementById("imitation-result").style.display = "block"; }
        btn.disabled = false;
      }
      async function playImitationReference() {
        if (!imitationSelectedText) { refreshImitationSelection(); if (!imitationSelectedText) { alert("请先在文章中选中要模仿的段落。"); return; } }
        var btn = document.getElementById("imitation-tts-btn");
        btn.disabled = true;
        try {
          var resp = await fetch("/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: imitationSelectedText }) });
          if (!resp.ok) { var d = await resp.json().catch(function() { return {}; }); alert(d.error || "生成参考朗读失败。"); return; }
          var audioBlob = await resp.blob();
          var url = URL.createObjectURL(audioBlob);
          var audio = new Audio(url);
          audio.onended = function() { URL.revokeObjectURL(url); };
          await audio.play();
        } catch (e) { alert("播放失败，请稍后再试。"); }
        btn.disabled = false;
      }
    </script>
  </body>
</html>
"""


@app.route("/history", methods=["GET"])
def history_page():
    items = _load_article_history()
    total = len(items)
    request_date = (request.args.get("date") or "").strip()
    request_topic = (request.args.get("topic") or "").strip()
    request_difficulty = (request.args.get("difficulty") or "").strip()

    filtered = _filter_articles(items, date_filter=request_date or None, topic_filter=request_topic or None, difficulty_filter=request_difficulty or None)
    # 最新在前
    filtered = list(reversed(filtered))
    for it in filtered:
        it["view_url"] = "/history/view?id=" + quote((it.get("id") or it.get("created_at") or ""), safe="")

    return render_template_string(
        HISTORY_TEMPLATE,
        items=filtered,
        total=total,
        request_date=request_date,
        request_topic=request_topic,
        request_difficulty=request_difficulty,
        provider=session.get("provider", "gemini"),
        difficulty=session.get("difficulty", "C1"),
    )


@app.route("/history/view", methods=["GET"])
def history_view():
    article_id = (request.args.get("id") or "").strip()
    if not article_id:
        return "<p>缺少文章 id。</p><a href='/history'>返回历史练习</a>", 400

    items = _load_article_history()
    article = None
    for it in items:
        if isinstance(it, dict) and (it.get("id") or it.get("created_at")) == article_id:
            article = it
            break

    if not article:
        return "<p>未找到该文章。</p><a href='/history'>返回历史练习</a>", 404

    if isinstance(article, dict) and "article_html" in article:
        article = dict(article)
        article["article_html"] = _strip_script_tags(article.get("article_html", ""))
    return render_template_string(HISTORY_VIEW_TEMPLATE, article=article, provider=session.get("provider", "gemini"), difficulty=session.get("difficulty", "C1"))


# ---------- 发音练习（原口语练习） ----------

SPEAKING_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>发音练习 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 720px; width: 100%; margin: 0 auto; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      h1 { margin: 0 0 10px; font-size: 22px; }
      a { color: #5eead4; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .topic-box { margin: 16px 0; padding: 14px; border-radius: 12px; background: rgba(15,23,42,0.75); border: 1px solid rgba(148,163,184,0.35); font-size: 15px; line-height: 1.5; }
      .btn { padding: 10px 18px; border-radius: 999px; font-weight: 600; font-size: 14px; border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
      .btn-primary { background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; }
      .btn-primary:hover:not(:disabled) { filter: brightness(1.05); transform: translateY(-1px); }
      .btn-primary:disabled { opacity: 0.7; cursor: not-allowed; }
      .btn-danger { background: #dc2626; color: #fff; }
      .section { margin-top: 18px; }
      .section h3 { font-size: 15px; margin: 0 0 6px; color: #e5e7eb; }
      .section pre, .section .text { font-size: 13px; line-height: 1.6; color: #9ca3af; white-space: pre-wrap; word-break: break-word; background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; }
      .small { font-size: 12px; color: #6b7280; margin-top: 8px; }
      #status { margin-top: 8px; font-size: 13px; color: #9ca3af; }
      .cefr-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.05em; background: rgba(250, 204, 21, 0.25); color: #facc15; }
      .cefr-badge-user { background: rgba(94, 234, 212, 0.25); color: #5eead4; }
      .cefr-result-box { margin-bottom: 14px; padding: 10px 12px; border-radius: 10px; background: rgba(15,23,42,0.8); border: 1px solid rgba(94, 234, 212, 0.4); }
      .cefr-result-label { font-size: 13px; color: #9ca3af; margin-right: 8px; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>发音练习 <span id="page-cefr-badge" class="cefr-badge">本话题难度 C1</span></h1>
      <p><a href="/">← 返回首页</a></p>
      <div class="provider-bar" style="margin-top: 8px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
        <span class="small" style="margin-left: 8px; color: #6b7280;">（口语需 Gemini，难度全站统一）</span>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>
      <p class="small">点击开始后，AI 将根据当前难度给出讨论话题并给出发音、语法与表达优化反馈及口语水平评估（CEFR）。</p>

      <div class="section">
        <button id="start-btn" type="button" class="btn btn-primary" onclick="startPractice()">开始练习</button>
        <div id="status"></div>
      </div>

      <div id="topic-section" class="section" style="display: none;">
        <h3>本话题（<span id="topic-difficulty-label">C1</span> 讨论题）</h3>
        <div id="topic-text" class="topic-box"></div>
        <p class="small">请用麦克风作答，然后点击「停止并提交」。</p>
        <button id="record-btn" type="button" class="btn btn-primary" onclick="toggleRecord()">开始录音</button>
        <button id="submit-btn" type="button" class="btn btn-primary" style="display: none; margin-left: 8px;" onclick="submitAudio()" disabled>停止并提交</button>
        <div id="record-status" class="small" style="margin-top: 6px;"></div>
      </div>

      <div id="result-section" class="section" style="display: none;">
        <div id="cefr-result-box" class="cefr-result-box" style="display: none;">
          <span class="cefr-result-label">你的口语水平评估：</span>
          <span id="cefr-level-badge" class="cefr-badge cefr-badge-user"></span>
        </div>
        <h3>识别文字</h3>
        <div id="transcript" class="text"></div>
        <h3 style="margin-top: 14px;">AI 反馈</h3>
        <div id="feedback" class="text"></div>
        <p class="small" style="margin-top: 10px;"><button type="button" class="btn btn-primary" onclick="startPractice()">再练一题</button></p>
      </div>
    </main>
    <script>
      let currentTopic = "";
      let mediaRecorder = null;
      let audioChunks = [];

      let currentDifficulty = "{{ difficulty|default('C1') }}";
      async function startPractice() {
        document.getElementById("page-cefr-badge").textContent = "本话题难度 " + currentDifficulty;
        document.getElementById("topic-difficulty-label").textContent = currentDifficulty;
        document.getElementById("result-section").style.display = "none";
        document.getElementById("topic-section").style.display = "none";
        document.getElementById("start-btn").disabled = true;
        document.getElementById("status").textContent = "正在获取话题…";
        try {
          const r = await fetch("/speaking/topic", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: currentDifficulty }) });
          const d = await r.json();
          if (d.error) { document.getElementById("status").textContent = d.error; return; }
          currentTopic = d.topic || "";
          document.getElementById("topic-text").textContent = currentTopic;
          document.getElementById("topic-section").style.display = "block";
          document.getElementById("status").textContent = "";
          document.getElementById("record-btn").textContent = "开始录音";
          document.getElementById("submit-btn").style.display = "none";
          document.getElementById("submit-btn").disabled = true;
          document.getElementById("record-status").textContent = "";
        } catch (e) {
          document.getElementById("status").textContent = "网络错误，请重试。";
        } finally {
          document.getElementById("start-btn").disabled = false;
        }
      }

      function toggleRecord() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          mediaRecorder.stop();
          return;
        }
        audioChunks = [];
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = e => { if (e.data.size) audioChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            document.getElementById("record-btn").textContent = "开始录音";
            document.getElementById("submit-btn").style.display = "inline-flex";
            document.getElementById("submit-btn").disabled = false;
            document.getElementById("record-status").textContent = "已停止，可点击「停止并提交」上传。";
          };
          mediaRecorder.start();
          document.getElementById("record-btn").textContent = "停止录音";
          document.getElementById("record-btn").classList.add("btn-danger");
          document.getElementById("record-status").textContent = "正在录音…";
        }).catch(() => {
          document.getElementById("record-status").textContent = "无法使用麦克风，请允许权限或使用 HTTPS。";
        });
      }

      async function submitAudio() {
        if (!audioChunks.length || !currentTopic) return;
        const btn = document.getElementById("submit-btn");
        btn.disabled = true;
        document.getElementById("record-status").textContent = "上传并分析中…";
        const blob = new Blob(audioChunks, { type: "audio/webm" });
        const form = new FormData();
        form.append("topic", currentTopic);
        form.append("difficulty", currentDifficulty);
        form.append("audio", blob, "speech.webm");
        try {
          const r = await fetch("/speaking/feedback", { method: "POST", body: form });
          const d = await r.json();
          if (d.error) {
            document.getElementById("record-status").textContent = d.error;
            btn.disabled = false;
            return;
          }
          document.getElementById("transcript").textContent = d.transcript || "（无识别文字）";
          document.getElementById("feedback").textContent = d.feedback || "";
          var cefrBox = document.getElementById("cefr-result-box");
          var cefrBadge = document.getElementById("cefr-level-badge");
          if (d.cefr_level) {
            cefrBadge.textContent = d.cefr_level;
            cefrBox.style.display = "block";
          } else {
            cefrBox.style.display = "none";
          }
          document.getElementById("result-section").style.display = "block";
          document.getElementById("record-status").textContent = "";
        } catch (e) {
          document.getElementById("record-status").textContent = "上传失败，请重试。";
          btn.disabled = false;
        }
      }
    </script>
  </body>
</html>
"""


def _speaking_get_topic(difficulty: str = "C1"):
    """让 AI 生成指定难度（B1/B2/C1）的口语讨论话题（英文一句），与 CEFR 评估体系一致。"""
    # 与 _assess_article_cefr 相同的三维度：词汇、句式、概念与逻辑
    level_guide = {
        "B1": (
            "Vocabulary: common words (A1–B1), daily topics. "
            "Sentence: simple sentences and basic clauses (because, that). "
            "Concept: concrete, factual description."
        ),
        "B2": (
            "Vocabulary: mid-level topic vocab (e.g. environment, technology), some idioms. "
            "Sentence: longer sentences, compound clauses, passive voice. "
            "Concept: abstract ideas, cause-effect argumentation."
        ),
        "C1": (
            "Vocabulary: low-frequency words, academic terms, metaphor, collocations. "
            "Sentence: complex structures, inversion, parenthesis, long nuanced sentences. "
            "Concept: editorial depth, philosophical or professional commentary, irony/metaphor."
        ),
    }
    guide = level_guide.get(difficulty.upper(), level_guide["C1"])
    prompt = f"""You are an English speaking examiner. The candidate's practice level is {difficulty.upper()} (CEFR).

Give exactly ONE discussion topic that fits this level. The topic should be answerable in a way that matches:
{guide}

Output ONLY one sentence in English, no number, no explanation, no quotation marks."""
    if not GEMINI_API_KEY:
        raise RuntimeError("需要 GEMINI_API_KEY 才能出题")
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = (response.text or "").strip()
    return text if text else "Discuss the impact of technology on work-life balance."


def _speaking_get_feedback(audio_path: str, topic: str, difficulty: str = "C1") -> tuple:
    """用 Gemini 分析音频：转写 + 发音/语法/可优化说法反馈 + CEFR 等级。返回 (transcript, feedback, cefr_level)。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("需要 GEMINI_API_KEY 才能分析语音")
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    suffix = (audio_path or "").lower()
    if suffix.endswith(".wav"):
        mime = "audio/wav"
    elif suffix.endswith(".mp3"):
        mime = "audio/mp3"
    else:
        mime = "audio/webm"
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)
    diff = difficulty.upper() if difficulty else "C1"
    if diff not in ("B1", "B2", "C1"):
        diff = "C1"
    prompt = f"""You are an English speaking examiner. This practice was set for {diff} level. The candidate was asked this discussion topic: "{topic}"

Listen to the attached audio of their spoken response. Then:

1) TRANSCRIPT: Write exactly what the candidate said in English (transcribe accurately).

2) FEEDBACK in Chinese, in three clear sections:
   - 发音问题：(pronunciation issues, or 无 if none notable)
   - 语法问题：(grammar issues, or 无 if none notable)
   - 可优化的说法：(suggestions for better wording or expressions)

3) CEFR assessment: Use the same CEFR criteria as for reading (Vocabulary, Sentence structure, Concept & logic).
   - B1: Vocabulary = common words, daily topics; Sentence = simple and basic clauses; Concept = concrete, factual.
   - B2: Vocabulary = mid-level topic vocab, some idioms; Sentence = compound clauses, passive; Concept = abstract, cause-effect.
   - C1: Vocabulary = low-frequency, academic, metaphor, collocations; Sentence = complex, inversion, long sentences; Concept = editorial, philosophical, irony.
   If the response fits different levels on different dimensions, take the HIGHER level.
   At the very end, add exactly one line on its own line: CEFR_LEVEL: B1 or CEFR_LEVEL: B2 or CEFR_LEVEL: C1

Output format:
TRANSCRIPT:
[your transcript here]

发音问题：
[content]

语法问题：
[content]

可优化的说法：
[content]

CEFR_LEVEL: [B1 or B2 or C1]
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, audio_part],
    )
    text = (response.text or "").strip()
    transcript = ""
    feedback = text
    cefr_level = ""
    if "TRANSCRIPT:" in text:
        after_trans = text.split("TRANSCRIPT:", 1)[-1]
        for sep in ("发音问题：", "发音问题:", "发音问题 "):
            if sep in after_trans:
                parts = after_trans.split(sep, 1)
                transcript = (parts[0] or "").strip()
                feedback = sep + (parts[1] or "").strip() if len(parts) > 1 else text
                break
        else:
            transcript = after_trans.strip()
    for level in ("C1", "B2", "B1"):
        if f"CEFR_LEVEL: {level}" in text or f"CEFR_LEVEL:{level}" in text:
            cefr_level = level
            break
    if not cefr_level and re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I):
        cefr_level = re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I).group(1).upper()
    return transcript, feedback, cefr_level or "B2"


def _imitation_get_feedback(audio_path: str, reference_text: str) -> tuple:
    """断句模仿：对比参考文本与用户录音，分析断句/重音/连读偏差。返回 (transcript, feedback)。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("需要 GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    suffix = (audio_path or "").lower()
    mime = "audio/wav" if suffix.endswith(".wav") else "audio/mp3" if suffix.endswith(".mp3") else "audio/webm"
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)
    ref = (reference_text or "").strip()[:3000]
    prompt = f"""You are an English pronunciation and prosody coach. The learner has imitated the following reference text. Compare their recording to the reference and give detailed feedback in Chinese.

REFERENCE TEXT (what they should have imitated):
"{ref}"

Listen to the attached audio. Then:

1) TRANSCRIPT: Write exactly what the learner said (transcribe accurately).

2) FEEDBACK in Chinese, in three sections:
   - 断句与停顿：Did they pause at natural phrase boundaries? Where did they break inappropriately or miss a pause? (断句、意群、停顿是否合理；哪里多停/少停)
   - 重音：Were the correct words or syllables stressed? Any misplaced or missing stress? (重音是否准确；哪些词重音有偏差)
   - 连读与省音：Linking, weak forms, contractions—did they sound natural or too careful/choppy? (连读、弱读、省音是否自然；哪里可以更连贯)

Be specific: quote the word or phrase where the deviation occurs and suggest how to improve.

Output format:
TRANSCRIPT:
[your transcript]

断句与停顿：
[content]

重音：
[content]

连读与省音：
[content]
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, audio_part],
    )
    text = (response.text or "").strip()
    transcript = ""
    feedback = text
    if "TRANSCRIPT:" in text:
        after_trans = text.split("TRANSCRIPT:", 1)[-1]
        for sep in ("断句与停顿：", "断句与停顿:"):
            if sep in after_trans:
                parts = after_trans.split(sep, 1)
                transcript = (parts[0] or "").strip()
                feedback = sep + (parts[1] or "").strip() if len(parts) > 1 else text
                break
        else:
            transcript = after_trans.strip()
    return transcript, feedback


def _read_to_speak_get_topics(article: str, difficulty: str = "C1"):
    """根据文章内容生成 3 个具有争议性的讨论话题，难度与全站 CEFR 一致（B1/B2/C1）。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("需要 GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    text = (_strip_html(article) if article else "")[:6000]
    diff = difficulty.upper() if difficulty and difficulty.upper() in ("B1", "B2", "C1") else "C1"
    prompt = f"""Based on the following English article, generate exactly 3 controversial or debatable discussion topics for a {diff}-level speaking practice (CEFR: same criteria as reading — vocabulary, sentence structure, concept & logic).
Each topic should be one clear sentence that invites opinion or argument, related to the article's theme, and answerable at {diff} level.

Article (excerpt):
{text}

Output ONLY a JSON object with one key "topics", value is an array of exactly 3 strings (the 3 topic sentences in English). No other text.
Example: {{"topics": ["Topic one sentence.", "Topic two sentence.", "Topic three sentence."]}}"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = (response.text or "").strip()
    try:
        data = _parse_model_json(raw)
        topics = data.get("topics") or data.get("topic") or []
        if isinstance(topics, str):
            topics = [topics]
        return [str(t).strip() for t in topics[:3] if t]
    except Exception:
        traceback.print_exc()
    return [
        "What is the main controversy or dilemma raised by this article?",
        "Do you agree with the author's perspective? Why or why not?",
        "How might this issue develop in the future?",
    ]


def _read_to_speak_get_feedback(audio_path: str, topic: str, article: str, difficulty: str = "C1") -> tuple:
    """Read-to-Speak 反馈：转写 + 发音/语法/可优化说法 + Upgrade my vocabulary + CEFR。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("需要 GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    suffix = (audio_path or "").lower()
    mime = "audio/wav" if suffix.endswith(".wav") else "audio/mp3" if suffix.endswith(".mp3") else "audio/webm"
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)
    article_excerpt = (_strip_html(article) if article else "")[:3000]
    diff = difficulty.upper() if difficulty and difficulty.upper() in ("B1", "B2", "C1") else "C1"
    prompt = f"""You are an English speaking examiner. The candidate has just read an article and is answering a discussion topic based on it.
Article excerpt (for context): {article_excerpt[:2000]}
Topic: "{topic}"
Practice level: {diff}

Listen to the attached audio of their spoken response. Then:

1) TRANSCRIPT: Write exactly what the candidate said in English (transcribe accurately).

2) FEEDBACK in Chinese, in FOUR sections:
   - 发音问题：(pronunciation issues, or 无 if none notable)
   - 语法问题：(grammar issues, or 无 if none notable)
   - 可优化的说法：(suggestions for better wording or expressions)
   - Upgrade my vocabulary（词汇升级）: List words or phrases the candidate used that are simpler or more common. For each, suggest C1-level alternatives. Example format: "你刚才用了 important，在 C1 水平下可以尝试使用 pivotal 或 crucial." If they already used advanced vocabulary, say "无或较少，可保持."

3) CEFR assessment: Use the same CEFR criteria as elsewhere (Vocabulary, Sentence structure, Concept & logic). If dimensions disagree, take the higher level. At the very end, add exactly one line: CEFR_LEVEL: B1 or CEFR_LEVEL: B2 or CEFR_LEVEL: C1

Output format:
TRANSCRIPT:
[your transcript]

发音问题：
[content]

语法问题：
[content]

可优化的说法：
[content]

Upgrade my vocabulary（词汇升级）：
[content - give specific replacements like "你用了 X，可尝试 Y 或 Z"]

CEFR_LEVEL: [B1 or B2 or C1]
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, audio_part],
    )
    text = (response.text or "").strip()
    transcript = ""
    feedback = text
    cefr_level = ""
    if "TRANSCRIPT:" in text:
        after_trans = text.split("TRANSCRIPT:", 1)[-1]
        for sep in ("发音问题：", "发音问题:", "发音问题 "):
            if sep in after_trans:
                parts = after_trans.split(sep, 1)
                transcript = (parts[0] or "").strip()
                feedback = sep + (parts[1] or "").strip() if len(parts) > 1 else text
                break
        else:
            transcript = after_trans.strip()
    for level in ("C1", "B2", "B1"):
        if f"CEFR_LEVEL: {level}" in text or f"CEFR_LEVEL:{level}" in text:
            cefr_level = level
            break
    if not cefr_level and re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I):
        cefr_level = re.search(r"CEFR_LEVEL\s*:\s*(B1|B2|C1)", text, re.I).group(1).upper()
    return transcript, feedback, cefr_level or "B2"


@app.route("/speaking", methods=["GET"])
def speaking_page():
    return render_template_string(SPEAKING_TEMPLATE, provider=session.get("provider", "gemini"), difficulty=session.get("difficulty", "C1"))


@app.route("/speaking/topic", methods=["POST"])
def speaking_topic():
    try:
        difficulty = _current_difficulty()
        try:
            topic = _speaking_get_topic(difficulty)
        except Exception:
            # fallback to DeepSeek/Gemini text generation
            prompt = f"Give exactly ONE discussion topic sentence suitable for CEFR {difficulty.upper()} level speaking practice. Output ONLY one English sentence."
            topic = _llm_text_generate(prompt, provider=_current_provider())
        return jsonify({"topic": topic})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e) or "获取话题失败，请稍后再试。"}), 500


@app.route("/speaking/feedback", methods=["POST"])
def speaking_feedback():
    topic = (request.form.get("topic") or "").strip()
    difficulty = _current_difficulty()
    audio_file = request.files.get("audio")
    if not topic:
        return jsonify({"error": "缺少话题。"}), 400
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "请上传录音。"}), 400
    suffix = ".webm"
    if audio_file.filename.lower().endswith(".wav"):
        suffix = ".wav"
    elif audio_file.filename.lower().endswith(".mp3"):
        suffix = ".mp3"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(audio_file.read())
            path = tf.name
        try:
            # 1) Try Gemini multimodal first when available.
            if _current_provider() == "gemini" and GEMINI_API_KEY:
                transcript, feedback, cefr_level = _speaking_get_feedback(path, topic, difficulty)
                return jsonify({"transcript": transcript, "feedback": feedback, "cefr_level": cefr_level})

            # 2) Fallback: Ming-UniAudio ASR -> text LLM feedback
            if not _ming_uniaudio_enabled():
                return jsonify({"error": "当前无法分析录音：Gemini 不可用，且未配置 MING_UNIAUDIO_URL。"}), 400
            transcript = _ming_uniaudio_asr(path)
            # feedback text model: prefer current provider; if it's gemini but gemini not available, fallback to deepseek when configured
            text_provider = _current_provider()
            if text_provider == "gemini" and not GEMINI_API_KEY and DEEPSEEK_API_KEY:
                text_provider = "deepseek"
            transcript2, feedback2, cefr_level2 = _speaking_feedback_from_transcript(transcript, topic, difficulty, provider=text_provider)
            return jsonify({"transcript": transcript2, "feedback": feedback2, "cefr_level": cefr_level2, "note": "Fallback: Ming-UniAudio ASR"})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "分析语音时出错，请稍后再试。"}), 500


# ---------- 断句模仿 ----------

@app.route("/imitation/feedback", methods=["POST"])
def imitation_feedback():
    """断句模仿：根据参考文本与用户录音，返回转写与断句/重音/连读偏差分析。"""
    reference_text = (request.form.get("text") or request.form.get("reference_text") or "").strip()
    audio_file = request.files.get("audio")
    if not reference_text:
        return jsonify({"error": "请先选中要模仿的段落文字。"}), 400
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "请上传模仿录音。"}), 400
    suffix = ".webm"
    if audio_file.filename.lower().endswith(".wav"):
        suffix = ".wav"
    elif audio_file.filename.lower().endswith(".mp3"):
        suffix = ".mp3"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(audio_file.read())
            path = tf.name
        try:
            # 1) Try Gemini multimodal first when available.
            if _current_provider() == "gemini" and GEMINI_API_KEY:
                transcript, feedback = _imitation_get_feedback(path, reference_text)
                return jsonify({"transcript": transcript, "feedback": feedback})

            # 2) Fallback: Ming-UniAudio ASR -> text LLM feedback
            if not _ming_uniaudio_enabled():
                return jsonify({"error": "当前无法分析录音：Gemini 不可用，且未配置 MING_UNIAUDIO_URL。"}), 400
            transcript = _ming_uniaudio_asr(path)
            text_provider = _current_provider()
            if text_provider == "gemini" and not GEMINI_API_KEY and DEEPSEEK_API_KEY:
                text_provider = "deepseek"
            transcript2, feedback2 = _imitation_feedback_from_transcript(transcript, reference_text, provider=text_provider)
            return jsonify({"transcript": transcript2, "feedback": feedback2, "note": "Fallback: Ming-UniAudio ASR"})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except ClientError as e:
        if getattr(e, "code", None) == 429:
            return jsonify({
                "error": "Gemini 免费额度已用尽或请求过于频繁，请约 10 秒后再试，或查看 API 用量与计费：https://ai.google.dev/gemini-api/docs/rate-limits"
            }), 429
        traceback.print_exc()
        return jsonify({"error": "分析模仿录音时出错，请稍后再试。"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "分析模仿录音时出错，请稍后再试。"}), 500


# ---------- 每日练习：开放性问题 ----------

@app.route("/daily-practice/questions", methods=["POST"])
def daily_practice_questions():
    """根据文章与难度生成 3 个开放性问题（供每日练习使用）。"""
    payload = request.get_json(silent=True) or {}
    article = (payload.get("article") or "").strip()
    difficulty = (payload.get("difficulty") or "").strip().upper() or _current_difficulty()
    if difficulty not in ("B1", "B2", "C1"):
        difficulty = "C1"
    if not article or len(article) < 50:
        return jsonify({"error": "请先获取今日练习文章后再生成问题。"}), 400
    try:
        # Prefer Gemini implementation; fallback to text LLM if Gemini unavailable.
        try:
            topics = _read_to_speak_get_topics(article, difficulty)
        except Exception:
            diff = difficulty.upper() if difficulty in ("B1", "B2", "C1") else "C1"
            text = (_strip_html(article) if article else "")[:2000]
            prompt = f"""Based on the following English article, generate exactly 3 discussion questions for a {diff}-level speaking practice.\nEach question should be one sentence in English.\n\nArticle excerpt:\n{text}\n\nOutput ONLY JSON: {{\"topics\": [\"Q1\", \"Q2\", \"Q3\"]}}"""
            raw = _llm_text_generate(prompt, provider=_current_provider())
            data = _parse_model_json(raw)
            topics = data.get("topics") or []
            if isinstance(topics, str):
                topics = [topics]
            topics = [str(t).strip() for t in topics if str(t).strip()][:3]
        return jsonify({"topics": topics or []})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e) or "生成问题失败，请稍后再试。"}), 500


# ---------- Read-to-Speak 联动 ----------

@app.route("/read-to-speak/topics", methods=["POST"])
def read_to_speak_topics():
    """根据用户刚读的文章生成 3 个争议性讨论话题。"""
    payload = request.get_json(silent=True) or {}
    article = (payload.get("article") or "").strip()
    difficulty = (payload.get("difficulty") or "").strip().upper() or _current_difficulty()
    if difficulty not in ("B1", "B2", "C1"):
        difficulty = "C1"
    if not article:
        return jsonify({"error": "缺少文章内容。"}), 400
    try:
        try:
            topics = _read_to_speak_get_topics(article, difficulty)
        except Exception:
            diff = difficulty.upper() if difficulty in ("B1", "B2", "C1") else "C1"
            text = (_strip_html(article) if article else "")[:2500]
            prompt = f"""Based on the following English article, generate exactly 3 controversial or debatable discussion topics for a {diff}-level speaking practice.\nEach topic must be one English sentence.\n\nArticle excerpt:\n{text}\n\nOutput ONLY JSON: {{\"topics\": [\"Topic 1.\", \"Topic 2.\", \"Topic 3.\"]}}"""
            raw = _llm_text_generate(prompt, provider=_current_provider())
            data = _parse_model_json(raw)
            topics = data.get("topics") or []
            if isinstance(topics, str):
                topics = [topics]
            topics = [str(t).strip() for t in topics if str(t).strip()][:3]
        return jsonify({"topics": topics})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e) or "生成话题失败。"}), 500


@app.route("/read-to-speak/feedback", methods=["POST"])
def read_to_speak_feedback():
    """Read-to-Speak 录音反馈（含 Upgrade my vocabulary）。"""
    topic = (request.form.get("topic") or "").strip()
    article = (request.form.get("article") or "").strip()
    difficulty = _current_difficulty()
    audio_file = request.files.get("audio")
    if not topic:
        return jsonify({"error": "缺少话题。"}), 400
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "请上传录音。"}), 400
    suffix = ".webm"
    if audio_file.filename.lower().endswith(".wav"):
        suffix = ".wav"
    elif audio_file.filename.lower().endswith(".mp3"):
        suffix = ".mp3"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(audio_file.read())
            path = tf.name
        try:
            # 1) Try Gemini multimodal first when available.
            if _current_provider() == "gemini" and GEMINI_API_KEY:
                transcript, feedback, cefr_level = _read_to_speak_get_feedback(path, topic, article, difficulty)
                return jsonify({"transcript": transcript, "feedback": feedback, "cefr_level": cefr_level})

            # 2) Fallback: Ming-UniAudio ASR -> text LLM feedback
            if not _ming_uniaudio_enabled():
                return jsonify({"error": "当前无法分析录音：Gemini 不可用，且未配置 MING_UNIAUDIO_URL。"}), 400
            transcript = _ming_uniaudio_asr(path)
            text_provider = _current_provider()
            if text_provider == "gemini" and not GEMINI_API_KEY and DEEPSEEK_API_KEY:
                text_provider = "deepseek"
            transcript2, feedback2, cefr2 = _read_to_speak_feedback_from_transcript(transcript, topic, article, difficulty, provider=text_provider)
            return jsonify({"transcript": transcript2, "feedback": feedback2, "cefr_level": cefr2, "note": "Fallback: Ming-UniAudio ASR"})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "分析语音时出错，请稍后再试。"}), 500


READ_TO_SPEAK_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>Read-to-Speak 联动 - B1/B2/C1 英语教练</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e5e7eb; min-height: 100vh; padding: 16px; }
      .card { background: rgba(15, 23, 42, 0.9); border-radius: 16px; padding: 28px 24px; max-width: 720px; width: 100%; margin: 0 auto; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.3); }
      h1 { margin: 0 0 10px; font-size: 22px; }
      a { color: #5eead4; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .topic-box { margin: 16px 0; padding: 14px; border-radius: 12px; background: rgba(15,23,42,0.75); border: 1px solid rgba(148,163,184,0.35); font-size: 15px; line-height: 1.5; }
      .topic-btn { display: block; width: 100%; margin: 8px 0; padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.5); background: rgba(15,23,42,0.8); color: #e5e7eb; font-size: 14px; text-align: left; cursor: pointer; }
      .topic-btn:hover { background: rgba(94, 234, 212, 0.15); border-color: rgba(94, 234, 212, 0.5); }
      .topic-btn.selected { border-color: #22c55e; background: rgba(34, 197, 94, 0.15); }
      .btn { padding: 10px 18px; border-radius: 999px; font-weight: 600; font-size: 14px; border: none; cursor: pointer; }
      .btn-primary { background: linear-gradient(135deg, #22c55e, #22d3ee); color: #020617; }
      .btn-primary:hover:not(:disabled) { filter: brightness(1.05); }
      .btn-primary:disabled { opacity: 0.7; cursor: not-allowed; }
      .btn-danger { background: #dc2626; color: #fff; }
      .section { margin-top: 18px; }
      .section h3 { font-size: 15px; margin: 0 0 6px; color: #e5e7eb; }
      .section .text { font-size: 13px; line-height: 1.6; color: #9ca3af; white-space: pre-wrap; word-break: break-word; background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; }
      .small { font-size: 12px; color: #6b7280; margin-top: 8px; }
      .cefr-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: rgba(94, 234, 212, 0.25); color: #5eead4; }
      .no-article { padding: 20px; text-align: center; color: #9ca3af; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>Read-to-Speak 联动</h1>
      <p><a href="/">← 返回首页</a> &nbsp;·&nbsp; <a href="/speaking">发音练习</a></p>
      <div class="provider-bar" style="margin-top: 8px; padding: 8px 12px; border-radius: 10px; background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.35); font-size: 13px;">
        <span style="color: #94a3b8;">当前模型：</span>
        <button type="button" onclick="setProvider('gemini')" style="margin-left: 8px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'gemini' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">Gemini</button>
        <button type="button" onclick="setProvider('deepseek')" style="margin-left: 4px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if provider == 'deepseek' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">DeepSeek</button>
        <span style="color: #94a3b8; margin-left: 12px;">当前难度：</span>
        <button type="button" onclick="setDifficulty('B1')" style="margin-left: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B1' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B1</button>
        <button type="button" onclick="setDifficulty('B2')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'B2' %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">B2</button>
        <button type="button" onclick="setDifficulty('C1')" style="margin-left: 4px; padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.6); background: {% if difficulty == 'C1' or not difficulty %}linear-gradient(135deg, #22c55e, #22d3ee); color: #020617{% else %}transparent; color: #e5e7eb{% endif %}; cursor: pointer; font-weight: 600;">C1</button>
        <span class="small" style="margin-left: 8px; color: #6b7280;">（本页需 Gemini）</span>
      </div>
      <script>function setProvider(p){ fetch("/set-provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p }) }).then(function(r){ if(r.ok) window.location.reload(); }); } function setDifficulty(d){ fetch("/set-difficulty", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ difficulty: d }) }).then(function(r){ if(r.ok) window.location.reload(); }); }</script>
      <p class="small">根据你刚读的文章与<strong>当前难度（B1/B2/C1）</strong>，AI 生成 3 个讨论话题；录音作答后可获得发音、语法、表达优化及 CEFR 水平评估（与全站同一套词汇/句式/概念标准）。</p>

      <div id="no-article" class="no-article section" style="display: none;">
        请先阅读一篇文章，在文章页点击「基于此文练口语」后再进入本页。
      </div>

      <div id="has-article" style="display: none;">
        <div id="topics-loading" class="section" style="display: none;">正在根据文章生成 3 个讨论话题…</div>
        <div id="topics-section" class="section" style="display: none;">
          <h3>选择话题（选一个后录音作答）</h3>
          <div id="topics-list"></div>
          <p class="small" style="margin-top: 10px;">选择上方一个话题后，点击「开始录音」作答。</p>
        </div>
        <div id="record-section" class="section" style="display: none;">
          <h3>本话题</h3>
          <div id="chosen-topic" class="topic-box"></div>
          <button id="record-btn" type="button" class="btn btn-primary" onclick="toggleRecord()">开始录音</button>
          <button id="submit-btn" type="button" class="btn btn-primary" style="display: none; margin-left: 8px;" onclick="submitAudio()" disabled>停止并提交</button>
          <div id="record-status" class="small" style="margin-top: 6px;"></div>
        </div>
        <div id="result-section" class="section" style="display: none;">
          <div id="cefr-box" style="margin-bottom: 14px; padding: 10px 12px; border-radius: 10px; background: rgba(15,23,42,0.8); border: 1px solid rgba(94, 234, 212, 0.4); display: none;">
            <span style="font-size: 13px; color: #9ca3af;">你的口语水平评估：</span>
            <span id="cefr-badge" class="cefr-badge"></span>
          </div>
          <h3>识别文字</h3>
          <div id="transcript" class="text"></div>
          <h3 style="margin-top: 14px;">AI 反馈（含词汇升级）</h3>
          <div id="feedback" class="text"></div>
          <p class="small" style="margin-top: 10px;"><button type="button" class="btn btn-primary" onclick="startOver()">再练一题</button></p>
        </div>
      </div>
    </main>
    <script>
      var articleText = '';
      var currentDifficulty = "{{ difficulty|default('C1') }}";
      var topicsList = [];
      var chosenTopic = '';
      var mediaRecorder = null;
      var audioChunks = [];

      function startOver() {
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('record-section').style.display = 'none';
        document.getElementById('topics-section').style.display = 'block';
        document.getElementById('chosen-topic').textContent = '';
        chosenTopic = '';
      }

      (function init() {
        try {
          articleText = sessionStorage.getItem('readToSpeakArticle') || '';
        } catch (e) {}
        if (!articleText || articleText.length < 50) {
          document.getElementById('no-article').style.display = 'block';
          return;
        }
        document.getElementById('has-article').style.display = 'block';
        document.getElementById('topics-loading').style.display = 'block';
        fetch('/read-to-speak/topics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ article: articleText, difficulty: currentDifficulty }) })
          .then(function(r) { return r.json(); })
          .then(function(d) {
            document.getElementById('topics-loading').style.display = 'none';
            if (d.error) { document.getElementById('topics-loading').textContent = d.error; return; }
            topicsList = d.topics || [];
            var listEl = document.getElementById('topics-list');
            listEl.innerHTML = '';
            topicsList.forEach(function(t, i) {
              var btn = document.createElement('button');
              btn.className = 'topic-btn';
              btn.textContent = (i + 1) + '. ' + t;
              btn.onclick = function() {
                document.querySelectorAll('.topic-btn').forEach(function(b) { b.classList.remove('selected'); });
                btn.classList.add('selected');
                chosenTopic = t;
                document.getElementById('record-section').style.display = 'block';
                document.getElementById('chosen-topic').textContent = t;
              };
              listEl.appendChild(btn);
            });
            document.getElementById('topics-section').style.display = 'block';
          })
          .catch(function() {
            document.getElementById('topics-loading').style.display = 'none';
            document.getElementById('topics-loading').textContent = '获取话题失败，请重试。';
          });
      })();

      function toggleRecord() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          return;
        }
        audioChunks = [];
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = function(e) { if (e.data.size) audioChunks.push(e.data); };
          mediaRecorder.onstop = function() {
            stream.getTracks().forEach(function(t) { t.stop(); });
            document.getElementById('record-btn').textContent = '开始录音';
            document.getElementById('submit-btn').style.display = 'inline-block';
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('record-status').textContent = '已停止，可点击「停止并提交」上传。';
          };
          mediaRecorder.start();
          document.getElementById('record-btn').textContent = '停止录音';
          document.getElementById('record-btn').classList.add('btn-danger');
          document.getElementById('record-status').textContent = '正在录音…';
        }).catch(function() {
          document.getElementById('record-status').textContent = '无法使用麦克风，请允许权限。';
        });
      }

      function submitAudio() {
        if (!audioChunks.length || !chosenTopic) return;
        var btn = document.getElementById('submit-btn');
        btn.disabled = true;
        document.getElementById('record-status').textContent = '上传并分析中…';
        var blob = new Blob(audioChunks, { type: 'audio/webm' });
        var form = new FormData();
        form.append('topic', chosenTopic);
        form.append('article', articleText);
        form.append('difficulty', currentDifficulty);
        form.append('audio', blob, 'speech.webm');
        fetch('/read-to-speak/feedback', { method: 'POST', body: form })
          .then(function(r) { return r.json(); })
          .then(function(d) {
            document.getElementById('record-status').textContent = '';
            if (d.error) {
              document.getElementById('record-status').textContent = d.error;
              btn.disabled = false;
              return;
            }
            document.getElementById('transcript').textContent = d.transcript || '（无识别文字）';
            document.getElementById('feedback').textContent = d.feedback || '';
            if (d.cefr_level) {
              document.getElementById('cefr-badge').textContent = d.cefr_level;
              document.getElementById('cefr-box').style.display = 'block';
            }
            document.getElementById('result-section').style.display = 'block';
          })
          .catch(function() {
            document.getElementById('record-status').textContent = '上传失败，请重试。';
            btn.disabled = false;
          });
      }
    </script>
  </body>
</html>
"""


@app.route("/read-to-speak", methods=["GET"])
def read_to_speak_page():
    return render_template_string(READ_TO_SPEAK_TEMPLATE, provider=session.get("provider", "gemini"), difficulty=session.get("difficulty", "C1"))


def _run_midnight_crawl():
    """每日 0 点（北京时间）爬取 3 篇并缓存。"""
    try:
        provider = "gemini" if GEMINI_API_KEY else ("deepseek" if DEEPSEEK_API_KEY else "gemini")
        _crawl_daily_pool(provider)
        print(f"[{datetime.now(timezone.utc).isoformat()}] 每日爬取完成，已缓存 3 篇")
    except Exception as e:
        traceback.print_exc()
        print(f"[{datetime.now(timezone.utc).isoformat()}] 每日爬取失败: {e}")


def _start_scheduler():
    """启动定时任务：每日 0 点（北京时间 UTC+8）执行爬取。启动时若今日池为空则立即爬一次。"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        tz = TZ_BEIJING
        sched = BackgroundScheduler(timezone=tz)
        sched.add_job(_run_midnight_crawl, CronTrigger(hour=0, minute=0))
        sched.start()
        print("[Scheduler] 已设置每日 0:00 自动爬取")
        # 启动时若爬取库各难度不足(每难度≤1篇)或今日池为空，则后台爬一次
        rec = _load_recommend_data()
        today = _today_date()
        today_pool = rec.get("daily", {}).get(today, [])
        need_crawl = not _pool_has_enough_per_difficulty(rec) or len(today_pool) < 3
        if need_crawl:
            import threading
            threading.Thread(target=_run_midnight_crawl, daemon=True).start()
    except ImportError:
        print("[Scheduler] 未安装 APScheduler，跳过定时爬取")
    except Exception as e:
        traceback.print_exc()
        print(f"[Scheduler] 启动失败: {e}")


if __name__ == "__main__":
    _start_scheduler()
    app.run(debug=True)

