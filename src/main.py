import os
import re
import json
import html
import asyncio
import requests
import feedparser
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from xml.sax.saxutils import escape
from openai import OpenAI
import edge_tts
from pydub import AudioSegment

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DOCS_DIR = BASE_DIR / "docs"
EPISODES_DIR = DOCS_DIR / "episodes"
TEMP_DIR = BASE_DIR / "temp"
EPISODES_JSON = DOCS_DIR / "episodes.json"
RSS_PATH = DOCS_DIR / "rss.xml"
CST = timezone(timedelta(hours=8))


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()


def get_base_url():
    cfg_url = CONFIG.get("podcast", {}).get("base_url", "auto")
    if cfg_url and cfg_url != "auto":
        return cfg_url.rstrip("/")

    repo = os.getenv("GITHUB_REPOSITORY", "")
    if repo and "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner.lower()}.github.io/{name}"

    return "http://localhost"


def clean_text(text, max_len=500):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def fetch_news():
    items = []
    seen = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    max_items = CONFIG.get("news", {}).get("max_items", 12)
    per_source = CONFIG.get("news", {}).get("per_source", 3)

    for source in CONFIG.get("news", {}).get("sources", []):
        try:
            resp = requests.get(source["url"], headers=headers, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            count = 0
            for entry in feed.entries:
                if count >= per_source:
                    break

                title = clean_text(entry.get("title", ""), 200)
                link = entry.get("link", "")
                summary = clean_text(entry.get("summary", entry.get("description", "")), 300)

                if not title or not link or link in seen:
                    continue

                seen.add(link)
                items.append({
                    "source": source.get("name", "RSS"),
                    "title": title,
                    "summary": summary,
                    "link": link
                })
                count += 1

        except Exception as e:
            print(f"Failed to fetch {source.get('name', 'unknown')}: {e}")

    return items[:max_items]


def get_llm_client():
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url)


def ask_llm(client, messages, temperature=0.7, max_tokens=4000):
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM request failed: {e}")

        if max_tokens > 2000:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            return resp.choices[0].message.content or ""

        raise


def build_script(client, news):
    target_minutes = CONFIG.get("audio", {}).get("target_minutes", 10)
    target_chars = int(target_minutes * 300)
    news_json = json.dumps(news, ensure_ascii=False, indent=2)

    system_prompt = """你是一档中文每日新闻播客的双人编剧。你需要把新闻素材改写成自然、有趣、适合播出的双人对谈剧本。
主播A：逗哏，负责提问、好奇、吐槽、替听众追问。
主播B：捧哏，负责专业分析、解释背景、补充事实。
要求：
1. 只输出对话内容，不要标题，不要说明，不要Markdown，不要代码块。
2. 每一行必须以“A：”或“B：”开头。
3. A和B交替说话，每段不要过长，适合口语播出。
4. 内容必须基于新闻素材，不得编造事实。
5. 涉及政治内容时保持客观、中立、谨慎，不做煽动性评论。
6. 开头要有节目开场，结尾要有总结和告别。
7. 语言口语化，不要书面化。
8. 不要出现“根据素材”“作为AI”“以下是剧本”等词语。
"""

    user_prompt = f"""请根据下面新闻素材，生成一期《每日新闻速递》双人对谈播客剧本。
目标时长约{target_minutes}分钟，总字数尽量接近{target_chars}个中文字。
请选择最重要、最有可听性的新闻，覆盖科技、财经、国际政治或全球热点。
每条新闻之间要有自然过渡。

新闻素材：
{news_json}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    max_tokens = CONFIG.get("llm", {}).get("max_script_tokens", 8000)
    temperature = CONFIG.get("llm", {}).get("temperature", 0.7)

    return ask_llm(client, messages, temperature=temperature, max_tokens=max_tokens)


def parse_dialogue(text):
    if not text:
        return []

    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    lines = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"\*\*", "", line)

        m = re.match(r"^(?:主播)?\s*([AB])\s*[：:]\s*(.+)$", line)
        if m:
            speaker = m.group(1).upper()
            content = m.group(2).strip()
            if content:
                lines.append((speaker, content))

    if len(lines) < 6:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        lines = []
        for i, p in enumerate(paragraphs):
            speaker = "A" if i % 2 == 0 else "B"
            lines.append((speaker, p))

    return lines


def split_text(text, max_len=240):
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_len:
        return [text]

    parts = []

    while len(text) > max_len:
        cut = max_len

        for sep in ["。", "！", "？", "；", "，", "、", " "]:
            idx = text.rfind(sep, 0, max_len)
            if idx > max_len // 2:
                cut = idx + 1
                break

        parts.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        parts.append(text)

    return parts


async def synth_line(text, voice, path, retries=3):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(path))
            return True
        except Exception as e:
            print(f"TTS failed attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 + attempt * 2)

    return False


async def build_speech(lines):
    speech = AudioSegment.empty()
    gap_ms = CONFIG.get("audio", {}).get("gap_ms", 500)
    gap = AudioSegment.silent(duration=gap_ms)
    voices = CONFIG.get("voices", {})

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (speaker, content) in enumerate(lines):
        voice = voices.get(speaker, voices.get("B", "zh-CN-YunyangNeural"))
        chunks = split_text(content, 240)

        for chunk_idx, chunk in enumerate(chunks):
            path = TEMP_DIR / f"line_{idx:04d}_{chunk_idx:02d}.mp3"

            ok = await synth_line(chunk, voice, path)
            if not ok:
                print(f"Skip line {idx} chunk {chunk_idx}: TTS failed")
                continue

            try:
                seg = AudioSegment.from_file(str(path))
                speech += seg + gap
            except Exception as e:
                print(f"Failed to load audio {path}: {e}")

            await asyncio.sleep(0.12)

    return speech


def mix_audio(speech):
    final = speech

    assets_dir = BASE_DIR / "assets"
    intro_path = assets_dir / "intro.mp3"
    outro_path = assets_dir / "outro.mp3"
    bgm_path = assets_dir / "bgm.mp3"

    if intro_path.exists():
        intro = AudioSegment.from_file(str(intro_path))
        final = intro + final

    if outro_path.exists():
        outro = AudioSegment.from_file(str(outro_path))
        final = final + outro

    if bgm_path.exists() and len(final) > 1000:
        bgm = AudioSegment.from_file(str(bgm_path))

        if len(bgm) < len(final):
            repeat_times = (len(final) // len(bgm)) + 1
            bgm = bgm * repeat_times

        bgm = bgm[:len(final)]

        bgm_volume = CONFIG.get("audio", {}).get("bgm_volume", -24)
        bgm = bgm + bgm_volume

        final = final.overlay(bgm)

    return final


def ms_to_duration(ms):
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_metadata(client, script_text, date_str):
    system_prompt = "你是播客运营，负责生成标题和简介。只输出合法JSON，不要Markdown。"

    user_prompt = """请根据下面的播客剧本，生成JSON。
要求：
{
  "title": "不超过60字，适合播客平台，包含日期",
  "description": "不超过200字，概括本期内容，不要使用Markdown"
}

日期：""" + date_str + """
剧本：
""" + script_text[:8000]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        content = ask_llm(client, messages, temperature=0.3, max_tokens=500).strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)

        title = str(data.get("title", f"{date_str} 每日新闻速递")).strip()
        description = str(data.get("description", "每日新闻速递：全球科技财经政治双人对谈。")).strip()

        return title, description

    except Exception as e:
        print(f"Metadata generation failed: {e}")
        return f"{date_str} 每日新闻速递", "每日新闻速递：全球科技财经政治双人对谈，覆盖今日重要新闻。"


def load_episodes():
    if EPISODES_JSON.exists():
        try:
            return json.loads(EPISODES_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []

    return []


def save_episodes(episodes):
    EPISODES_JSON.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")


def escape_attr(value):
    return escape(str(value), {'"': "&quot;"})


def build_rss(episodes, base_url):
    p = CONFIG.get("podcast", {})
    items = []

    for ep in episodes:
        title = escape(ep.get("title", ""))
        description = escape(ep.get("description", ""))
        url = escape_attr(ep.get("mp3_url", ""))
        guid = escape_attr(ep.get("guid", ""))

        items.append(f"""    <item>
      <title>{title}</title>
      <description>{description}</description>
      <pubDate>{ep.get('pub_date', '')}</pubDate>
      <enclosure url="{url}" type="audio/mpeg" length="{ep.get('size', 0)}"/>
      <guid isPermaLink="false">{guid}</guid>
      <itunes:duration>{ep.get('duration', '')}</itunes:duration>
    </item>""")

    items_xml = "\n".join(items)

    cover = p.get("cover", "")
    cover_xml = ""
    if cover:
        cover_xml = f'    <itunes:image href="{escape_attr(cover)}"/>\n'

    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:atom="http://www.w3.org/2005/Atom" version="2.0">
  <channel>
    <title>{escape(p.get('title', '每日新闻速递'))}</title>
    <link>{escape_attr(base_url)}</link>
    <atom:link href="{escape_attr(base_url + '/rss.xml')}" rel="self" type="application/rss+xml"/>
    <language>{escape(p.get('language', 'zh-cn'))}</language>
    <description>{escape(p.get('description', '每日新闻速递'))}</description>
    <itunes:author>{escape(p.get('author', '每日新闻速递'))}</itunes:author>
    <itunes:summary>{escape(p.get('description', '每日新闻速递'))}</itunes:summary>
{cover_xml}    <itunes:category text="News"/>
    <itunes:category text="Technology"/>
    <itunes:category text="Business"/>
    <itunes:explicit>false</itunes:explicit>
    <itunes:owner>
      <itunes:name>{escape(p.get('author', '每日新闻速递'))}</itunes:name>
      <itunes:email>{escape(p.get('email', 'you@example.com'))}</itunes:email>
    </itunes:owner>
{items_xml}
  </channel>
</rss>
"""

    RSS_PATH.write_text(rss, encoding="utf-8")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    now_cst = datetime.now(CST)
    date_str = now_cst.strftime("%Y-%m-%d")

    print(f"Generating episode for {date_str}")

    news = fetch_news()
    if not news:
        raise RuntimeError("No news fetched. Check RSS sources.")

    client = get_llm_client()

    script = build_script(client, news)
    (DOCS_DIR / f"{date_str}_script.txt").write_text(script, encoding="utf-8")

    lines = parse_dialogue(script)
    if not lines:
        raise RuntimeError("No dialogue lines parsed from LLM output.")

    speech = asyncio.run(build_speech(lines))
    if len(speech) < 10000:
        raise RuntimeError("Generated speech too short.")

    final_audio = mix_audio(speech)

    bitrate = str(CONFIG.get("audio", {}).get("bitrate", "96k"))
    mp3_path = EPISODES_DIR / f"{date_str}.mp3"

    final_audio.export(str(mp3_path), format="mp3", bitrate=bitrate)

    title, description = generate_metadata(client, script, date_str)

    base_url = get_base_url()
    mp3_url = f"{base_url}/episodes/{date_str}.mp3"
    size = os.path.getsize(mp3_path)
    duration = ms_to_duration(len(final_audio))

    episodes = load_episodes()
    episodes = [ep for ep in episodes if ep.get("guid") != date_str]

    episode = {
        "guid": date_str,
        "title": title,
        "description": description,
        "mp3_url": mp3_url,
        "pub_date": format_datetime(datetime.now(timezone.utc)),
        "duration": duration,
        "size": size
    }

    episodes.insert(0, episode)
    episodes = episodes[:100]

    save_episodes(episodes)
    build_rss(episodes, base_url)

    print("Done")
    print(f"MP3: {mp3_path}")
    print(f"RSS: {RSS_PATH}")
    print(f"Public RSS URL: {base_url}/rss.xml")


if __name__ == "__main__":
    main()
