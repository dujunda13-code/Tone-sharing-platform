# 音色创作客户端（本地零样本 + 云端情绪模式）

这是一个不训练用户权重的音色创作客户端。参考音频的质检、转写、审核、建档和安全复检在本机完成；语音创作可选择本机 GPT-SoVITS V2ProPlus 正常合成，或在单独授权后调用 ModelVerse IndexTTS-2 完成情绪合成。

## 当前产品边界

客户端固定流程为：

```text
本地登录
→ 上传已授权的 WAV/FLAC/MP3 参考音频
→ 本机质检、切片、转写、自动情绪识别与人工确认
→ 建立 ready 零样本音色档案（不创建训练任务）
→ 选择正常合成（本机 GPT-SoVITS）或情绪合成（授权后调用 API）
→ CAM++ 相似度、水印复检、高频指纹记录
→ 下载已验证音频
```

客户端支持普通用户和本地 `admin` 管理员。账号、音色档案、任务、输出和审计记录保存在本机；不使用 Redis、Celery、第三方身份服务或通用云存储。只有用户主动选择“情绪合成”并勾选云端处理授权时，所选参考音频和文本才会发送到固定 API。

后续如果需要训练增强版基础模型，应在独立的离线服务器工程中训练完整的通用 S1、S2 和真实解耦模块，完成质量、情绪迁移和消融验证后，以新的 SHA-256 登记版本替换 `active_voice_base`。这不是客户端用户操作的一部分，客户端不读取离线训练配置，也不会在运行时联网下载模型。

## 硬件、软件和磁盘预算

- Windows 10/11，NVIDIA `cuda:0`，设备总显存至少 8000 MiB（当前 8188 MiB 级设备可接受）。
- Python 3.10、PyTorch 2.5.1、CUDA 12.4、FunASR 1.4.12、AudioSeal 0.2.0。
- 基础权重首次部署约占 14.95 GiB；正常启动不会复制一份基础模型。
- 零样本流程不会产生 S1/S2 训练检查点、训练特征目录或用户专属权重；只会在 `data/` 下保存参考音频、转写/情绪分析结果、任务日志和带水印输出。
- `data/temp/` 下的合成中间 WAV 在任务完成或失败后会清理。需要长期复用的只有本地参考片段、音色档案元数据、审计记录和已验证输出。

## 安装和模型校验

1. 安装 Miniconda、Node.js（Node 20.19+ 或 22.12+）和 FFmpeg。
2. 在仓库根目录执行部署脚本：

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
   ```

3. 按 `scripts/download_models.ps1` 的白名单从固定上游下载并放置基础模型，然后执行：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/download_models.ps1
   ```

   该脚本只在部署阶段工作，并校验 `models/checksums.sha256`。模型缺失、版本不符或哈希不符时，应用保持 fail-closed。

4. 项目命令统一直接使用固定解释器 `D:\miniconda3\envs\timbre-share\python.exe`，不需要通过 Conda 包装器启动项目命令。

## 启动

正常合成无需云端密钥。需要启用情绪合成时，在启动脚本所在的同一 PowerShell 会话中设置有效的模型 API Key；不要把值写入仓库文件、命令历史、日志或前端环境变量：

```powershell
$env:COMPSHARE_API_KEY = '<从服务商控制台复制的模型 API Key>'
```

生产默认固定调用 `https://api.modelverse.cn/v1/audio/infer`。仅测试/私有网关可通过 `MODELVERSE_API_BASE_URL` 覆盖基础地址。情绪模式会发送所选授权参考和不超过 600 字的文本；参考必须为 5–30 秒 WAV/MP3、不超过 20 MiB 且采样率至少 16 kHz。云端支持标签、情绪描述、八维向量、情绪强度、22.05/44.1/48 kHz 输出采样率、0.25–4 倍语速、音量增益、随机采样和句间静音。提供方未声明的发音规则不会伪装成可用配置。

确保 `frontend/node_modules` 已安装后，在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-dev.ps1
```

脚本直接启动固定 Python 的 FastAPI、真实本地 GPU Worker 和 `node.exe` 的 Vite 入口，不经过额外 shell 包装。地址为：

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`
- 进程记录：`data/logs/dev-pids.json`
- 日志：`data/logs/*.out.log` 与 `data/logs/*.err.log`

按 `Ctrl+C` 退出脚本会清理子进程；如果终端被直接关闭，下一次启动会根据 PID 和端口执行安全的遗留进程清理。不会结束与本项目无关的端口占用进程。

## 本地账号和 SQLite

- SQLite 数据库：`data/app.db`。
- 普通用户使用“用户名 + 密码”注册，密码至少 8 位；不需要短信、邮箱验证码或 Redis。
- 管理员通过本机初始化脚本创建，密码不会作为命令行参数传入：

  ```powershell
  & 'D:\miniconda3\envs\timbre-share\python.exe' scripts/bootstrap_admin.py
  ```

- 角色只有 `user` 和 `admin`。普通用户只能访问自己的数据，管理员只能执行受控的本机账号、任务和审计维护，不能读取密码明文或关闭安全门禁。

## 参考音频要求

- 单文件支持 WAV、FLAC、MP3，最大 200 MB；客户端零样本参考的有效人声时长必须为 3–10 秒。
- 音频应为单人、清晰、少混响、无背景音乐；削波比例不超过 1% 是阻断门禁。
- 20 dB 是建议告警值而非阻断门禁；低于该值仍可继续，但界面会提示可能降低音色相似度和清晰度。
- 用户不需要自己切片；系统会在本机规范化并生成参考片段。
- 上传前必须确认拥有声音主体和用途授权。未授权、文本为空、审核未完成或削波等阻断式质量门禁不通过时不能建档。

## 情绪与安全

自动情绪置信度低于 0.55 时映射为 `other`，用户可以在审核阶段确认或修改标签。正常模式可自动选择或按已审核参考情绪选择；情绪模式支持自动、标签、文字描述和八维向量控制，并要求独立的云端处理授权。任何云端错误都会 fail-closed，不会静默输出无情绪结果。

每次合成都必须经过授权确认、敏感词拦截、真实 GPU Worker、CAM++ 相似度、水印嵌入与复检和高频指纹记录。发布目标为 CAM++ `>=0.90`、AudioSeal 概率 `>=0.80`；任一门禁失败都不会暴露可下载文件。严重噪声或破坏性编辑后只能报告“来源无法可靠确认”，不能伪造肯定结论。

## 分层验证

日常 CPU 检查（不能作为最终验收）：

```powershell
& 'D:\ProjectPractice\jianxi\scripts\verify.ps1' -SkipGpu
```

最终验收需要用户明确授权、原始时长和有效人声均为 3–10 秒的本地参考音频，并具备真实 GPU、本地模型和 AudioSeal/CAM++ 运行条件：

```powershell
& 'D:\ProjectPractice\jianxi\scripts\verify.ps1' -AuthorizedReference 'D:\数据集\authorized-reference.wav'
```

`-SkipGpu` 只会写入 `FINAL_RESULT=CPU_ONLY`。没有短参考、模型缺失、GPU 不可用、相似度/水印/指纹失败或存在跳过项时，`reports/final-evaluation.json` 必须保持 `final_result: INCOMPLETE`，不得伪造 `PASS`。

验证顺序为：模型哈希、后端 lint、CPU 单元/集成测试、匿名授权守卫、前端测试、mock E2E、CUDA 探针、GPT-SoVITS 零样本 smoke、真实零样本端到端、AudioSeal 攻击矩阵。所有用户数据和测试中间产物均留在本机目录。

## 目录职责

```text
backend/       FastAPI、SQLite、审核门禁、任务队列、本地 Worker 和安全链路
frontend/      React/Vite 登录、工作台、参考音频审核、音色档案、合成和管理界面
models/        部署期下载并校验的基础模型，只读用于客户端零样本合成
data/          本地账号、上传、参考片段、情绪结果、临时文件、输出和日志
scripts/       部署、模型校验、启动和验收脚本
docs/          实时交接日志、设计规格和实施计划
reports/       机器可读的验收和评测报告
```

## 清理本机数据

停止服务后，用户可以在应用中删除不再需要的参考音频和输出，也可以在确认影响后清理 `data/temp/`、分析特征和日志。删除参考音频会使依赖它的零样本档案不能继续合成；`models/` 中的基础权重只有在明确不再使用客户端时才应删除。
