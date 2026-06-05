import html
import json
import os
import subprocess
import sys
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="Riyanjaly & Ansh Media Voice Studio",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="en_US")
MODE = 'local'

VOICES_DIR = os.path.join(current_dir, "Voices")

def get_voices_map():
    if not os.path.exists(VOICES_DIR):
        return {}
    voices = {}
    for f in os.listdir(VOICES_DIR):
        if f.endswith((".mp3", ".wav", ".flac", ".m4a")):
            name = f.replace(".mp3", "").replace(".wav", "").replace(".flac", "").replace(".m4a", "")
            if "Weirdoo (Leo)" in name: name = "Leo"
            elif "Weirdoo (Rose)" in name: name = "Rose"
            elif "Monster's Voice" in name: name = "Monster"
            elif "Villan's Voice" in name or "Villan" in name: name = "Villain"
            elif "Cartoon2's Voice" in name: name = "Cartoon Male 2"
            elif "CartoonMale1's Voice" in name: name = "Cartoon Male 1"
            elif "MaleTeacher's Voice" in name: name = "Male Teacher"
            elif "Female1" in name: name = "Female 1"
            else: name = name.replace("'s Voice", "").strip()
            voices[name] = os.path.join(VOICES_DIR, f)
    return voices

def get_voice_choices():
    choices = list(get_voices_map().keys())
    top_voices = []
    
    order = ["Leo", "Rose", "Villain", "Monster"]
    for voice in order:
        if voice in choices:
            top_voices.append(voice)
            choices.remove(voice)
            
    return top_voices + sorted(choices)

tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )

# Branding
LOGO_URL = "https://i.ibb.co/KxFs8mW4/riyanjalyandanshmedia.jpg"

# Head: favicon + meta + small JS bridge to open the hidden Gradio Settings from our header button
custom_head = f"""
<link rel="icon" href="{LOGO_URL}">
<meta property="og:title" content="Riyanjaly & Ansh Media Voice Studio">
<meta property="og:image" content="{LOGO_URL}">
<meta name="theme-color" content="#7c3aed">
<title>Riyanjaly & Ansh Media Voice Studio</title>
<script>
  function toggleTheme() {{
    document.body.classList.toggle('dark');
    document.documentElement.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('theme-preference', isDark ? 'dark' : 'light');
  }}

  // Auto-apply theme on load
  document.addEventListener('DOMContentLoaded', () => {{
    const saved = localStorage.getItem('theme-preference');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && systemDark)) {{
      document.body.classList.add('dark');
      document.documentElement.classList.add('dark');
    }} else {{
      document.body.classList.remove('dark');
      document.documentElement.classList.remove('dark');
    }}
  }});

  function openGradioSettings() {{
    try {{
      const footer = document.querySelector('footer');
      if (!footer) return;
      let btn = footer.querySelector('button[aria-label="Settings"], a[aria-label="Settings"]');
      if (!btn) {{
        const nodes = Array.from(footer.querySelectorAll('*'));
        btn = nodes.find(n => (n.textContent || '').trim().toLowerCase() === 'settings');
      }}
      if (btn) btn.click();
    }} catch (e) {{
      console.warn('Settings open failed:', e);
    }}
  }}
  document.addEventListener('keydown', function(event) {{
    if (event.altKey && event.code === 'KeyH') {{
      const btn = document.querySelector('#secret_btn');
      if (btn) btn.click();
    }}
  }});
</script>
<style>#secret_btn {{ display: none !important; }}</style>
"""

# Languages and choices
LANGUAGES = {"中文": "zh_CN", "English": "en_US"}
EMO_CHOICES_ALL = [
    i18n("与音色参考音频相同"),
    i18n("使用情感参考音频"),
    i18n("使用情感向量控制"),
    i18n("使用情感描述文本控制"),
]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # hide experimental label by default

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70

def gen_single(emo_control_method, voice_library_selection, prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
               *args, progress=gr.Progress()):
    if not prompt and voice_library_selection:
        prompt = get_voices_map().get(voice_library_selection)

    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None
    if emo_control_method == 1:  # reference audio
        pass
    if emo_control_method == 2:  # custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    return gr.update(interactive=True)

def create_warning_message(warning_text):
    # Safe f-string quoting for inline CSS HTML
    return gr.HTML(
        f'<div style="padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold">{html.escape(warning_text)}</div>'
    )

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

# Theming
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Fira Code'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    button_primary_background_fill="*primary_400",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="white",
    loader_color="*primary_500",
    block_shadow="*shadow_drop_md",
    block_border_width="1px",
    input_border_width="2px",
)

# CSS
APP_CSS = """
/* Layout */
.gradio-container { max-width: 1120px; margin: 0 auto; }

/* Branded top bar */
.app-header.brand {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 10px 0 8px;
}
.brand-left { display: flex; align-items: center; gap: 12px; }
.brand-logo { height: 44px; width: auto; border-radius: 8px; box-shadow: 0 6px 16px rgba(124,58,237,.35); }
.brand-title { line-height: 1.1; }
.brand-title h2 { margin: 0; font-weight: 800; letter-spacing: .2px; }
.brand-title .tagline { margin: 2px 0 0; opacity: .75; font-size: 14px; }

/* Header actions */
.header-actions { display: flex; align-items: center; gap: 8px; }
.ghost-btn {
  height: 36px; padding: 0 12px; border-radius: 8px; font-weight: 600;
  background: transparent; color: var(--body-text-color, #e5e7eb);
  border: 1px solid rgba(124,58,237,.35);
}
.ghost-btn:hover { background: rgba(124,58,237,.12); }

/* Primary generate button full width */
#primary_gen { width: 100%; height: 52px; font-size: 16px; }

/* Accordions emphasis */
.gr-accordion .label { font-weight: 600; }

/* Hide default footer but keep for Settings hook */
footer { position: fixed !important; bottom: -200vh !important; opacity: 0 !important;
         pointer-events: none !important; height: 0 !important; overflow: hidden !important; }
"""

with gr.Blocks(title="Riyanjaly & Ansh Media Voice Studio", theme=theme, css=APP_CSS, head=custom_head) as demo:
    mutex = threading.Lock()

    # Header
    gr.HTML(f'''
      <div class="app-header brand">
        <div class="brand-left">
          <img src="{LOGO_URL}" alt="Razebait Logo" class="brand-logo" />
          <div class="brand-title">
            <h2>Riyanjaly & Ansh Media Voice Studio</h2>
          </div>
        </div>
        <div class="header-actions">
          <button class="ghost-btn" onclick="toggleTheme()" title="Toggle Dark/Light Mode" style="font-size: 1.2rem; padding: 0 10px;">🌓</button>
        </div>
      </div>
    ''')

    with gr.Column():
        with gr.Row():
            os.makedirs("prompts", exist_ok=True)
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():
                    with gr.Tab("Voice Library"):
                        voice_library = gr.Dropdown(label="Select Default Voice", choices=get_voice_choices(), value="Leo", interactive=True)
                        library_preview = gr.Audio(label="Voice Preview", interactive=False, value=get_voices_map().get("Leo"))
                    with gr.Tab("➕ Custom Voice"):
                        prompt_audio = gr.Audio(label="Upload Audio File", key="prompt_audio",
                                                sources=["upload", "microphone"], type="filepath")
                
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio")

            with gr.Column(scale=2, min_width=0):
                input_text_single = gr.TextArea(label=i18n("文本"), key="input_text_single",
                                                placeholder=i18n("请输入目标文本"),
                                                info="Powered by Riyanjaly & Ansh Media Voice Studio Core", lines=20)
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",
                                       interactive=True, variant="primary", elem_id="primary_gen")

        secret_btn = gr.Button("Secret", elem_id="secret_btn")
        with gr.Column(visible=False) as secret_settings_group:
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)

            # Settings with default = Use emotion vectors
            with gr.Accordion(i18n("功能设置")):
                with gr.Row():
                    emo_control_method = gr.Radio(
                        choices=EMO_CHOICES_OFFICIAL,
                        type="index",
                        value=EMO_CHOICES_OFFICIAL[2],  # default to "使用情感向量控制"
                        label=i18n("情感控制方式"),
                    )

            # Panels: set initial visibility to match "vectors" mode
            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            with gr.Row(visible=True) as emotion_randomize_group:
                emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

            with gr.Group(visible=True) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=1.0, step=0.05)  # Surprised = 1.0
                        vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            with gr.Group(visible=False) as emo_text_group:
                create_experimental_warning_message()
                with gr.Row():
                    emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                          placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                          value="",
                                          info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

            with gr.Row(visible=True) as emo_weight_group:
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.5, step=0.01)

            with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"{i18n('GPT2 采样设置')}")
                        with gr.Row():
                            do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                            temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                        with gr.Row():
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                            num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                        with gr.Row():
                            repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                            length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                        max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    with gr.Column(scale=2):
                        gr.Markdown(f'{i18n("分句设置")}')
                        with gr.Row():
                            initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                            max_text_tokens_per_segment = gr.Slider(
                                label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                                info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                            )
                        with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                            segments_preview = gr.Dataframe(
                                headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                                key="segments_preview",
                                wrap=True,
                            )
                advanced_params = [
                    do_sample, top_p, top_k, temperature,
                    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                ]

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)
            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {segments_preview: gr.update(value=data, visible=True, type="array")}
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {segments_preview: gr.update(value=df)}

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True))
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True))
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True))
        else:  # same as speaker
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False))

    emo_control_method.change(
        on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0
        return gr.update(choices=new_choices, value=new_choices[new_index])

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])

    def update_library_preview(voice_name):
        return gr.update(value=get_voices_map().get(voice_name))

    voice_library.change(update_library_preview, inputs=[voice_library], outputs=[library_preview])

    gen_button.click(
        gen_single,
        inputs=[emo_control_method, voice_library, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                *advanced_params],
        outputs=[output_audio]
    )

    secret_state = gr.State(value=False)
    def toggle_secret(is_visible):
        new_state = not is_visible
        return new_state, gr.update(visible=new_state)
    
    secret_btn.click(toggle_secret, inputs=[secret_state], outputs=[secret_state, secret_settings_group])

def start_cloudflare_tunnel(token):
    import sys
    import platform
    import urllib.request
    if not os.path.exists("cloudflared"):
        print("Downloading cloudflared...")
        if sys.platform == "darwin":
            if platform.machine() == "arm64":
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
            else:
                url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
            urllib.request.urlretrieve(url, "cloudflared.tgz")
            subprocess.run(["tar", "-xzf", "cloudflared.tgz"])
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            urllib.request.urlretrieve(url, "cloudflared")
        subprocess.run(["chmod", "+x", "cloudflared"])
    subprocess.Popen(["./cloudflared", "tunnel", "--no-autoupdate", "run", "--token", token])

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True)
