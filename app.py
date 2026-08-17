import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64
import io
import json
import re
import shutil
import socket
import time
import traceback
import requests
from urllib.parse import urlsplit, urlunsplit
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from openai import OpenAI

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

def auto_detect_proxy() -> str:
    """智能自动探测本地代理端口与系统代理"""
    for env_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        if os.environ.get(env_key):
            return os.environ.get(env_key)
    for port in [7897, 7890, 10809, 10808, 20811]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.15)
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return ""

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.apifast.tech/v1",
    "dual_api": False,
    "text_api_key": "",
    "text_base_url": "",
    "image_api_key": "",
    "image_base_url": "",
    "proxy": "auto",
    "base_img_path": "pig_hero.png",
    "output_dir": "./output_pigs",
    "text_model": "gemini-2.5-flash",
    "image_model": "gemini-3.1-flash-image-preview",
}

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

config = load_config()
os.makedirs(os.path.join(BASE_DIR, config.get("output_dir", "./output_pigs")), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

app = FastAPI(title="AutoPig Studio v1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_openai_client(api_key: str, base_url: str, proxy_val: str = "auto"):
    active_proxy = auto_detect_proxy() if (proxy_val == "auto" or not proxy_val) else proxy_val
    if active_proxy:
        os.environ["HTTP_PROXY"] = active_proxy
        os.environ["HTTPS_PROXY"] = active_proxy
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url or "https://api.apifast.tech/v1",
    )

def get_text_client(cfg=None):
    c = cfg or config
    if c.get("dual_api") and c.get("text_api_key"):
        return get_openai_client(c.get("text_api_key"), c.get("text_base_url"), c.get("proxy", "auto"))
    return get_openai_client(c.get("api_key"), c.get("base_url"), c.get("proxy", "auto"))

def get_image_client(cfg=None):
    c = cfg or config
    if c.get("dual_api") and c.get("image_api_key"):
        return get_openai_client(c.get("image_api_key"), c.get("image_base_url"), c.get("proxy", "auto"))
    return get_openai_client(c.get("api_key"), c.get("base_url"), c.get("proxy", "auto"))

def decode_image_data(text: str) -> Image.Image:
    b64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)', text)
    if b64_match:
        raw = re.sub(r'[\s\)\_]+', '', b64_match.group(1))
        raw += '=' * ((4 - len(raw) % 4) % 4)
        return Image.open(io.BytesIO(base64.b64decode(raw)))

    url_match = re.search(r'!\[.*?\]\((https?://[^\s\)]+)\)', text)
    if url_match:
        r = requests.get(url_match.group(1), timeout=30)
        return Image.open(io.BytesIO(r.content))

    try:
        clean = re.sub(r'[\s\)\_]+', '', text)
        clean += '=' * ((4 - len(clean) % 4) % 4)
        return Image.open(io.BytesIO(base64.b64decode(clean)))
    except Exception:
        return None

# ================= 增量新增：DashScope / qwen-image 原生生图通道 =================
def _is_dashscope_compatible(client) -> bool:
    base = str(getattr(client, "base_url", "") or "").lower()
    return "dashscope.aliyuncs.com/compatible-mode" in base

def _dashscope_image_generate(client, model, prompt, images):
    parts = urlsplit(str(getattr(client, "base_url", "")))
    native_url = urlunsplit((parts.scheme, parts.netloc, "/api/v1/services/aigc/multimodal-generation/generation", "", ""))
    content = [{"image": im} for im in images] + [{"text": prompt}]
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {"prompt_extend": True, "watermark": False},
    }
    headers = {"Authorization": f"Bearer {getattr(client, 'api_key', '')}", "Content-Type": "application/json"}
    r = requests.post(native_url, headers=headers, json=body, timeout=300)
    r.raise_for_status()
    return r.json()

def _collect_image_sources(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("image", "image_url", "url"):
                if isinstance(v, str) and v:
                    out.append(v)
                elif isinstance(v, dict) and isinstance(v.get("url"), str) and v["url"]:
                    out.append(v["url"])
            else:
                _collect_image_sources(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_image_sources(item, out)

def _dashscope_image_sources(raw_obj) -> list:
    sources = []
    _collect_image_sources(raw_obj, sources)
    seen = set()
    unique = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

def _download_image(src) -> Image.Image:
    img = decode_image_data(src)
    if img:
        return img
    s = str(src or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        try:
            r = requests.get(s, timeout=30)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content))
        except Exception:
            return None
    return None

def _dashscope_render_image(client, model, prompt, images):
    try:
        raw_obj = _dashscope_image_generate(client, model, prompt, images)
    except Exception as e:
        return None, f"DashScope 原生接口调用失败: {e}"
    sources = _dashscope_image_sources(raw_obj)
    if not sources:
        return None, "DashScope 原生接口响应中未找到 image 字段"
    for src in sources:
        img = _download_image(src)
        if img:
            return img, None
    return None, f"DashScope 原生接口返回了图片地址但下载失败 (共 {len(sources)} 个候选源)"

# ================= 静态与媒体路由 =================
@app.get("/api/hero-image")
def get_hero_image_api():
    base_img = os.path.join(BASE_DIR, config.get("base_img_path", "pig_hero.png"))
    return FileResponse(base_img) if os.path.exists(base_img) else JSONResponse(status_code=404, content={"message": "Not found"})

@app.get("/api/scholar-image")
def get_scholar_image_api():
    scholar_img = os.path.join(BASE_DIR, "pig_scholar.png")
    if os.path.exists(scholar_img): return FileResponse(scholar_img)
    hero_img = os.path.join(BASE_DIR, config.get("base_img_path", "pig_hero.png"))
    return FileResponse(hero_img) if os.path.exists(hero_img) else JSONResponse(status_code=404, content={"message": "Not found"})

@app.get("/api/pig-gif")
def get_pig_gif_api():
    for name in ["pig1.gif", "pig.gif", "pig_walk.gif", "walk.gif", "anim.gif"]:
        gif_path = os.path.join(BASE_DIR, name)
        if os.path.exists(gif_path):
            return FileResponse(gif_path, media_type="image/gif", headers={"Cache-Control": "no-cache"})
    hero_img = os.path.join(BASE_DIR, config.get("base_img_path", "pig_hero.png"))
    return FileResponse(hero_img) if os.path.exists(hero_img) else JSONResponse(status_code=404, content={"message": "Not found"})

@app.get("/api/oink-sound")
def get_oink_sound_api():
    sound_path = os.path.join(BASE_DIR, "oink.wav")
    return FileResponse(sound_path, media_type="audio/wav") if os.path.exists(sound_path) else JSONResponse(status_code=404, content={"message": "Not found"})

@app.get("/videos.json")
def get_videos_json_api():
    for p in [os.path.join(BASE_DIR, "static", "videos.json"), os.path.join(BASE_DIR, "videos.json")]:
        if os.path.exists(p):
            return FileResponse(p, media_type="application/json")
    return JSONResponse(content={"current_version": None, "general_guides": []})

@app.get("/api/config")
def get_config_api():
    cfg = load_config()
    cfg["detected_proxy"] = auto_detect_proxy() or "直连 (未检测到本地代理)"
    return cfg

@app.post("/api/config")
def save_config_api(cfg: dict):
    global config
    config.update(cfg)
    save_config(config)
    return {"status": "ok", "config": config}

@app.post("/api/test-connectivity")
def test_connectivity_api(cfg: dict):
    try:
        client = get_openai_client(cfg.get("api_key"), cfg.get("base_url"), cfg.get("proxy", "auto"))
        start_t = time.time()
        models_data = client.models.list()
        latency = int((time.time() - start_t) * 1000)
        model_ids = [m.id for m in models_data.data]
        return {"status": "success", "latency": latency, "models_count": len(model_ids), "models": model_ids}
    except Exception as e:
        return {"status": "error", "message": str(e), "log": traceback.format_exc()}

@app.post("/api/test-models-usability")
def test_models_usability_api(cfg: dict):
    try:
        txt_client = get_text_client(cfg)
        test_text_model = cfg.get("text_model", "gemini-2.5-flash")
        test_image_model = cfg.get("image_model", "gemini-3.1-flash-image-preview")

        text_ok, text_msg = False, ""
        try:
            start_t = time.time()
            res = txt_client.chat.completions.create(model=test_text_model, messages=[{"role": "user", "content": "1+1="}], max_tokens=5)
            text_ok, text_msg = True, f"响应正常 ({int((time.time() - start_t) * 1000)}ms)"
        except Exception as e_text:
            text_msg = f"调用失败: {str(e_text)}"

        img_keywords = ["image", "dall-e", "flux", "imagen", "diffusion", "sd"]
        image_ok = any(k in test_image_model.lower() for k in img_keywords)
        image_msg = "生图通道匹配正常" if image_ok else "模型未命中常见生图关键词"

        return {
            "status": "success",
            "text_model": test_text_model, "text_model_ok": text_ok, "text_model_msg": text_msg,
            "image_model": test_image_model, "image_model_ok": image_ok, "image_model_msg": image_msg
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "log": traceback.format_exc()}

# ================= 模式 1：智能风格文字量产 =================
@app.post("/api/generate-plan")
def generate_plan_api(payload: dict):
    count = int(payload.get("count", 3))
    style_vibe = payload.get("style_vibe", "趣味职业与生活角色")
    
    out_dir = os.path.join(BASE_DIR, config.get("output_dir", "./output_pigs"))
    existing = []
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.startswith("pig_") and f.endswith(".png"):
                slug = re.sub(r'_\d+$', '', f[4:-4])
                existing.append(slug)
    
    exclude_text = f"请勿重复或相似以下已有主题：{', '.join(set(existing))}。" if existing else ""
    
    system_prompt = (
        "你是顶级游戏美术设计师。构思小猪主题，仅返回纯JSON数组格式，严禁返回任何Markdown标签或额外文字。\n"
        "每个对象包含：\n"
        "- theme: 中文主题名\n"
        "- features: 1-2个极其简洁明确的中文特征道具\n"
        "- slug: 适合作为文件名的英文简短单词 (如 pirate, chef)"
    )
    user_prompt = f"请设计 {count} 个视觉特征鲜明但配饰极简的小猪角色。风格偏好：{style_vibe}。{exclude_text}"

    client = get_text_client()
    configured_model = config.get("text_model", "gemini-2.5-flash")
    models_to_try = [configured_model]
    for backup_model in ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3-pro-image-preview"]:
        if backup_model not in models_to_try:
            models_to_try.append(backup_model)

    last_err = None
    for model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.85
            )
            content = res.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                content = "\n".join(lines).strip()

            return {"status": "ok", "tasks": json.loads(content)}
        except Exception as e:
            last_err = e
            continue

    return JSONResponse(status_code=500, content={"status": "error", "message": f"策划失败: {str(last_err)}"})

@app.post("/api/render-image")
def render_image_api(payload: dict):
    theme = payload.get("theme")
    features = payload.get("features")
    extra_feedback = payload.get("extra_feedback", "")
    
    base_img_path = os.path.join(BASE_DIR, config.get("base_img_path", "pig_hero.png"))
    if not os.path.exists(base_img_path):
        return JSONResponse(status_code=400, content={"status": "error", "message": f"基准参考图 '{base_img_path}' 不存在！"})

    with open(base_img_path, "rb") as f:
        hero_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        f"请为参考图中的这只小猪设计【{theme}】造型（带有简单的{features}）。\n"
        f"严格要求：\n"
        f"1. 必须完全基于原图小猪进行创作，保持小猪趴卧的四足动物形态与原图肉色/浅色调皮肤，严禁画成双腿站立的人物姿态。\n"
        f"2. 保持与参考图一致的二维手绘矢量画风。\n"
        f"3. 装饰配饰保持极简点缀，不要过于复杂繁琐。\n"
        f"4. 背景为纯白色，无任何背景场景。"
    )
    if extra_feedback:
        prompt += f"\n5. 额外微调修改要求：{extra_feedback}"

    client = get_image_client()
    configured_model = config.get("image_model", "gemini-3.1-flash-image-preview")

    # 增量：DashScope(qwen-image) 原生生图通道，触发条件：image base_url 为 dashscope compatible-mode
    if _is_dashscope_compatible(client):
        img, qwhy = _dashscope_render_image(client, configured_model, prompt, [f"data:image/png;base64,{hero_b64}"])
        if img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return {"status": "ok", "image_b64": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}
        return JSONResponse(status_code=500, content={"status": "error", "message": f"图像生成失败：{qwhy}"})

    # 原版 gemini 多模态生图
    models_to_try = [configured_model]
    for backup_model in ["gemini-3.1-flash-image-preview", "gemini-3.1-flash-image", "gemini-3-pro-image-preview"]:
        if backup_model not in models_to_try:
            models_to_try.append(backup_model)

    for m in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{hero_b64}"}}
                    ]
                }]
            )
            raw = resp.choices[0].message.content
            img = decode_image_data(raw)
            if img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return {"status": "ok", "image_b64": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}
        except Exception:
            continue

    return JSONResponse(status_code=500, content={"status": "error", "message": "图像生成失败，请检查提示词或模型通道"})

# ================= 模式 2：角色参考图转小猪（玩偶穿搭黄金方案） =================
@app.post("/api/render-character-pig")
def render_character_pig_api(payload: dict):
    character_name = payload.get("character_name", "").strip() or "特定角色"
    character_b64 = payload.get("character_image_b64", "")
    extra_feedback = payload.get("extra_feedback", "")
    
    if "," in character_b64:
        character_b64 = character_b64.split(",", 1)[1]

    base_img_path = os.path.join(BASE_DIR, config.get("base_img_path", "pig_hero.png"))
    if not os.path.exists(base_img_path):
        return JSONResponse(status_code=400, content={"status": "error", "message": f"基准参考图 '{base_img_path}' 不存在！"})

    with open(base_img_path, "rb") as f:
        hero_b64 = base64.b64encode(f.read()).decode("utf-8")

    txt_client = get_text_client()
    img_client = get_image_client()

    # 1. 结构化提炼三件套：发型发套 + 迷你披风领结 + 标志性点缀
    extracted_features = "角色标志性发色假发套、经典配色迷你小披肩与代表性小饰品"
    ana_prompt = (
        f"请仔细观察参考图中的角色（{character_name}），为一只Q版趴卧肉色小猪设计一套专属的【Cosplay 迷你换装三件套】。\n"
        "请精准提炼并直接输出以下 3 项具体内容（文字精炼，总字数在 40 字以内）：\n"
        "1. 头顶假发套：具体的发型发色与标志性头饰（如：灰白色齐肩假发配黑白女仆发箍）；\n"
        "2. 背部小斗篷：提取角色服装的经典配色与领口（如：黑白色调的小披风，系着红色小领结）；\n"
        "3. 标志性小点缀：1个核心代表物（如：微型十字发夹/专属胸针）。\n"
        "格式要求：直接输出一段流畅的穿戴描述，不要带序号或解释。"
    )
    
    txt_models = [config.get("text_model", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-3-flash-preview"]
    for tm in txt_models:
        try:
            ana_resp = txt_client.chat.completions.create(
                model=tm,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ana_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{character_b64}"}}
                    ]
                }]
            )
            extracted_features = ana_resp.choices[0].message.content.strip()
            break
        except Exception:
            continue

    # 2. 采用具象穿搭提示词，杜绝死板负面词打架
    prompt = (
        f"请参考图2中【{character_name}】的外貌设定，为图1中的趴卧肉色小猪换上一套【{character_name} 主题 Cosplay 专属装扮】。\n\n"
        f"装扮设计要求（{extracted_features}）：\n"
        f"1. 【头部装扮】：小猪头顶服帖地戴着【{character_name}】标志性的发色假发与头饰，但面部必须完整露出图1小猪原生可爱的黑色豆豆眼与粉色椭圆猪鼻子。\n"
        f"2. 【背部装扮】：小猪圆滚滚的背上披着【{character_name}】经典服饰配色的小斗篷/小披肩，领口点缀精致的小领结或代表性配饰。\n"
        f"3. 【画风与体态】：保持与图1完全一致的二维手绘矢量画风与四足趴卧姿势，既能一眼看出是图1的小猪本体，又能一眼认出是【{character_name}】的同人换装！\n"
        f"4. 【背景】：纯白色背景（#FFFFFF），无任何阴影或杂物。"
    )
    if extra_feedback:
        prompt += f"\n5. 额外微调修改要求：{extra_feedback}"

    # 3. 双图多模态对照渲染 (图1: 小猪模板, 图2: 角色立绘)
    configured_img_model = config.get("image_model", "gemini-3.1-flash-image-preview")

    # 增量：DashScope(qwen-image) 原生生图通道，触发条件：image base_url 为 dashscope compatible-mode
    if _is_dashscope_compatible(img_client):
        img, qwhy = _dashscope_render_image(
            img_client, configured_img_model, prompt,
            [f"data:image/png;base64,{hero_b64}", f"data:image/png;base64,{character_b64}"]
        )
        if img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return {"status": "ok", "image_b64": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}
        return JSONResponse(status_code=500, content={"status": "error", "message": f"未能生成有效图片：{qwhy}"})

    # 原版 gemini 双图多模态生图
    img_models_to_try = [configured_img_model]
    for backup_model in ["gemini-3.1-flash-image-preview", "gemini-3.1-flash-image", "gemini-3-pro-image-preview"]:
        if backup_model not in img_models_to_try:
            img_models_to_try.append(backup_model)

    for im in img_models_to_try:
        try:
            resp = img_client.chat.completions.create(
                model=im,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{hero_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{character_b64}"}}
                    ]
                }]
            )
            img = decode_image_data(resp.choices[0].message.content)
            if img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return {"status": "ok", "image_b64": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}
        except Exception:
            continue

    return JSONResponse(status_code=500, content={"status": "error", "message": "未能生成有效图片，请检查网络通道或提示词"})
    
# ================= 保存与画廊管理 =================
@app.post("/api/save-image")
def save_image_api(payload: dict):
    slug = payload.get("slug", "custom")
    img_b64 = payload.get("image_b64", "").split(",", 1)[-1]
    if not img_b64:
        return JSONResponse(status_code=400, content={"status": "error", "message": "无图片数据"})

    out_dir = os.path.join(BASE_DIR, config.get("output_dir", "./output_pigs"))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"pig_{slug}.png")
    c = 2
    while os.path.exists(save_path):
        save_path = os.path.join(out_dir, f"pig_{slug}_{c}.png")
        c += 1
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(img_b64))
    return {"status": "ok", "filename": os.path.basename(save_path)}

@app.post("/api/delete-image")
def delete_image_api(payload: dict):
    target = os.path.join(BASE_DIR, config.get("output_dir", "./output_pigs"), payload.get("filename", ""))
    if os.path.exists(target): 
        try:
            os.remove(target)
            return {"status": "ok"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    return JSONResponse(status_code=404, content={"status": "error", "message": "文件不存在"})

@app.get("/api/gallery")
def get_gallery_api():
    out_dir = os.path.join(BASE_DIR, config.get("output_dir", "./output_pigs"))
    if not os.path.exists(out_dir): return {"images": []}
    imgs = []
    for f in sorted(os.listdir(out_dir), reverse=True):
        if f.endswith((".png", ".jpg")):
            p = os.path.join(out_dir, f)
            with open(p, "rb") as im:
                imgs.append({
                    "filename": f, 
                    "data": f"data:image/png;base64,{base64.b64encode(im.read()).decode()}", 
                    "size": f"{os.path.getsize(p)/1024:.1f} KB"
                })
    return {"images": imgs}

@app.post("/api/upload-base-img")
async def upload_base_img_api(file: UploadFile = File(...)):
    dest = os.path.join(BASE_DIR, "pig_hero.png")
    with open(dest, "wb") as buf: shutil.copyfileobj(file.file, buf)
    config["base_img_path"] = "pig_hero.png"
    save_config(config)
    return {"status": "ok"}

# 静态资源挂载
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
