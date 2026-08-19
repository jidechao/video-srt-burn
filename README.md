# video-srt-burn

确定性视频字幕流水线：一条命令跑完 **转录 → 校对 → 断句 → 章节 → 人工预览**，再一条命令**烧录成片**。

```
视频 → FFmpeg 提取音频 → 百炼 FunAudio ASR（词级时间戳）
     → hotwords/glossary 术语纠错
     → qwen3.7-flash 语义校对 + 疑点抽帧视觉核对
     → Qwen 字幕断句 + 长视频章节生成
     → 本地预览编辑器（localhost:8765）人工校对
     → 烧录：字幕/章节进度条/成片 MP4 + ASS + SRT
```

核心包名 `videotrans`，由 [oil-subtitle](../oil-subtitle) skill 独立重实现而来：不依赖 Agent 编排，四个阶段顺序执行、可断点续跑、行为完全确定。

## 功能特性

- **FunAudio ASR 转录**：百炼 `fun-asr` 模型，词级时间戳，原始 ASR 结果落盘留存
- **hotwords 热词**：识别前上传远程热词表（SHA-256 内容哈希缓存，只在词表变化时重建），把专有名词纠正在识别阶段
- **glossary 术语表**：转录后按"容忍空格、忽略大小写"的正则替换（`GPT55` 也能命中 `GPT 55`），烧录时同样应用，保证两端一致
- **两阶段自动校对**：qwen3.7-flash 语义检查（置信度 ≥0.97 才替换）→ 型号/版本号/命令/文件名等疑点抽取前后三帧，视觉模型核对（≥0.90）；无法确认的写入 `unresolved`，绝不猜测
- **Qwen 断句**：LLM 语义断句（逐字符回验，失败自动回退规则打分），24 视觉字宽 / 4.2 秒硬上限兜底
- **章节进度条**：视频严格超过 3 分钟时，底部半透明渐变承载宽粒度章节（2–6 章、间距 ≥75s）
- **本地预览编辑器**：视频+字幕对照，双击改字、勾选删除、查找替换、批量删除，保存即生效
- **错题本自学习**：保存时自动对比修改前后内容，高置信（≥0.97）、带安全上下文的错词写入个人 glossary；润色/删句/标点一律忽略
- **一次烧录**：自动发现校对后的字幕与章节，一条命令产出 `*_subtitled.mp4/.ass/.srt` 三件套

## 环境要求

- Python ≥ 3.10
- FFmpeg 与 ffprobe 在 PATH 上（或设 `FFMPEG` / `FFPROBE` 环境变量）
- 阿里云百炼（DashScope）API Key：在[百炼控制台](https://bailian.console.aliyun.com/)申请

## 安装

```bash
cd video-srt-burn
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

## 配置

把 `.env.example` 复制为 `.env`，填入你的 Key：

```ini
DASHSCOPE_API_KEY=sk-your-key-here
```

也可以用命令写入（会在项目根 `.env` 中新增/更新一行，不动其他内容）：

```bash
python -m videotrans --save-api-key sk-your-key-here
```

API Key **只认 `.env`**（或真实环境变量 `DASHSCOPE_API_KEY`，环境变量优先）。`.env` 已被 `.gitignore` 忽略，不会提交。

### 可选：hotwords / glossary

启用方式只有两种（都不配则不启用）：

1. `.env` / 环境变量：

   ```ini
   OIL_SUBTITLE_HOTWORDS=./hotwords.json
   OIL_SUBTITLE_GLOSSARY=./glossary.json
   ```

2. 命令行：`--hotwords path.json` / `--glossary path.json`

词表格式（两个文件相同）：

```json
[
  {"wrong": "克劳德", "correct": "Claude"},
  {"wrong": "百链", "correct": "百炼"}
]
```

glossary 是项目的错题本：预览编辑器保存时自动学习高置信错词写入其中，下次转录即自动纠正。`glossary.json` 默认在项目根，已被 `.gitignore` 忽略。

## 使用

### 第一步：跑流水线（止于预览保存）

```bash
python -m videotrans 视频文件.mp4
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--resume` | 跳过已有产物的前序阶段（断点续跑） |
| `--language zh` | ASR 语言（默认 zh） |
| `--no-hotwords` / `--no-glossary` | 本次禁用热词/术语表 |
| `--progress` / `--no-progress` | 强制开/关章节进度条（默认：>3 分钟开启） |
| `--port 8765` | 预览编辑器端口 |
| `--text-model` / `--vision-model` / `--split-model` | 覆盖各阶段 Qwen 模型 |

中途产物在工作目录 `<视频名>.subtitle-work/`：`transcript.json`（转录）→ `reviewed-transcript.json`（校对后）→ `subtitle-transcript.json`（断句+排版后，编辑器编辑的就是它）。

流水线最后自动打开预览编辑器 `http://localhost:8765`：双击修改文字、勾选删除、查找替换，确认后点「保存并关闭」，然后 Ctrl-C 退出。保存时会自动判断人工修改是否值得进错题本（结果在 `manual-edit-review.json`）。

### 第二步：烧录成片

```bash
python -m videotrans.burn 视频文件.mp4
```

自动读取 `<视频名>.subtitle-work/` 里校对后的字幕与章节，产出（与源视频同目录）：

- `<视频名>_subtitled.mp4` — 成片
- `<视频名>_subtitled.ass` — ASS 字幕
- `<视频名>_subtitled.srt` — SRT 字幕

常用选项：`--output` 指定输出、`--srt-input` 直烧已审阅的 SRT（文本/换行/时间码原样保留）、`--draft-only` 只出 SRT 草稿、`--square-output` / `--output-height` 方形裁剪/缩放、`--no-progress` 不渲染章节进度条、`--font` 自定义字幕字体、`--no-beauty` 关闭摄像头区域轻度美颜（仅 macOS 支持检测，其他平台自动跳过）。

字幕字体默认按平台选择真实存在的字体：Windows 用微软雅黑、macOS 用 PingFang SC、Linux 用 Noto Sans CJK SC，避免 libass 静默替换成不可控的兜底字体。

**低分辨率视频的清晰度技巧**：`--output-height` 传大于源尺寸的值会放大渲染（lanczos），例如 270p 源加 `--output-height 1080`，字幕会按 1080p 分辨率渲染，清晰度接近预览编辑器；画面本身仍受源分辨率限制。缩小路径行为与历史版本完全一致。

## 测试

93 个离线单元测试（不触网、不消耗百炼额度）：

```bash
# Windows（GBK 控制台需指定 UTF-8）
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -m unittest discover -s tests

# macOS / Linux
.venv/bin/python -m unittest discover -s tests
```

## 流水线各阶段与模型

| 阶段 | 模型 | 产物 |
|---|---|---|
| 1 转录 | `fun-asr`（+热词表） | `transcript.json` |
| 2 校对 | `qwen3.7-flash`（文本+视觉） | `reviewed-transcript.json`、`subtitle-review.json`、`review-frames/` |
| 3 准备 | `qwen-plus`（断句+章节） | `subtitle-transcript.json`、`subtitle-chapters.json`、`subtitle-manifest.json` |
| 4 预览 | — | 人工保存 + `manual-edit-review.json` |
| 烧录 | — | `*_subtitled.mp4/.ass/.srt` |

## 致谢

本项目的流水线逻辑移植自 [oil-subtitle](../oil-subtitle) skill，与其核心算法保持一致。
