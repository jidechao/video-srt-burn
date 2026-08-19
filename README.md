# video-srt-burn

一条命令跑完 **转录 → 校对 → 断句 → 章节 → 人工预览**，再一条命令**烧录成片**的确定性中文字幕流水线。

```
视频 → FFmpeg 提取音频 → 百炼 FunAudio ASR（词级时间戳）
     → hotwords/glossary 术语纠错
     → qwen3.7-flash 语义校对 + 疑点抽帧视觉核对
     → Qwen 字幕断句 + 长视频章节生成
     → 本地预览编辑器（localhost:8765）人工校对
     → 烧录：字幕/章节进度条/成片 MP4 + ASS + SRT
```

识别用阿里云百炼 FunAudio ASR，校对/断句/章节用 Qwen，烧录用 FFmpeg。四个阶段顺序执行、可断点续跑，同一输入得到同一产出，中间没有任何"看情况跳步"的模型决策。

## 功能特性

- **FunAudio ASR 转录**：词级时间戳，原始 ASR 结果落盘留存
- **hotwords 热词**：识别前上传远程热词表（SHA-256 内容哈希缓存，只在词表变化时更新远端），把专有名词纠正在识别阶段
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

两种文件格式不同：

**hotwords.json** —— 百炼定制热词格式（`text` 必填、`weight` 为 1–5 整数必填、`lang` 可选），上传后在**识别阶段**生效，专治专有名词（文件内容变化时自动更新远端词表，ID 缓存在 `~/.cache/videotrans/`）：

```json
[
  {"text": "周谷堆", "weight": 4, "lang": "zh"},
  {"text": "Claude Code", "weight": 4, "lang": "en"}
]
```

**glossary.json** —— 错词到正确词的映射，在**转录后和烧录时**做"容忍空格、忽略大小写"的替换（`GPT55` 也能命中 `GPT 55`）：

```json
[
  {"wrong": "周古堆", "correct": "周谷堆"},
  {"wrong": "克劳德", "correct": "Claude"},
  {"wrong": "gpt55", "correct": "GPT-5"}
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
| `--work-dir <dir>` | 自定义工作目录（默认 `<视频名>.subtitle-work`） |
| `--language zh` | ASR 语言（默认 zh） |
| `--no-hotwords` / `--no-glossary` | 本次禁用热词/术语表 |
| `--progress` / `--no-progress` | 强制开/关章节进度条（默认：>3 分钟开启） |
| `--port 8765` | 预览编辑器端口 |
| `--text-model` / `--vision-model` / `--split-model` / `--prepare-model` | 覆盖各阶段 Qwen 模型 |

中途产物都在工作目录 `<视频名>.subtitle-work/`（**删除该目录即可全量重跑**；`--resume` 会跳过产物已存在的阶段）：

| 文件 | 阶段 | 用途 |
|---|---|---|
| `bailian_asr.json` | 转录 | 原始 ASR 结果留存 |
| `transcript.json` | 转录 | 断句后的字幕段（含词级时间戳） |
| `reviewed-transcript.json` | 校对 | 语义 + 视觉校对后的字幕 |
| `subtitle-review.json` | 校对 | 校对报告（applied / unresolved 及抽帧路径） |
| `review-frames/` | 校对 | 疑点时间点抽取的帧图片 |
| `subtitle-transcript.json` | 准备 | 显示版字幕，**编辑器编辑的就是它**；烧录默认读取它 |
| `subtitle-transcript.json.orig.json` | 预览 | 保存前的自动备份（错题本学习的比对基准） |
| `subtitle-chapters.json` | 准备 | 章节定义（>3 分钟视频） |
| `cache/chapters-response.json` | 准备 | 章节 LLM 响应缓存（`--resume` 复用，签名匹配才生效） |
| `manual-edit-review.json` | 预览 | 人工修改的错题本学习判定结果 |
| `pipeline-report.json` | 全程 | 各阶段耗时与状态 |

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

## 常见问题

**编辑器里的字幕和烧录后的观感为什么不同？**
编辑器字幕是浏览器矢量文本，按显示器分辨率渲染，永远锐利；烧录字幕则画进视频像素里，受源分辨率和视频编码影响。低分辨率源加 `--output-height 1080` 放大渲染即可大幅接近编辑器效果（画面本身仍受源分辨率限制）。

**只想改几条字幕再重新烧录，要重跑整个流水线吗？**
不用。直接编辑 `<视频名>.subtitle-work/subtitle-transcript.json` 里对应条目的 `text`（或编辑已生成的 `.ass` 微调字号/位置），然后重跑 `python -m videotrans.burn 视频文件.mp4` 即可，前面的阶段不会重跑、不消耗 API。

**真实运行消耗百炼额度吗？**
消耗。一次完整流水线 = 1 次 ASR 转录 + 数次 Qwen 调用（校对按 80 条/块分批、断句、章节）。参考耗时（3.5 分钟 480×270 视频）：转录 28s、校对 15s、准备 2s、烧录 11s（270p）/ 约 1 分钟（放大到 1080p）。

## 测试

离线单元测试全量运行（不触网、不消耗百炼额度）：

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

## 许可证

[MIT](LICENSE)
