# Codex 本地会话管理工具

`scripts/codex_session.py` 是一个零依赖的本地 CLI，用来管理当前仓库里的 Codex 工作会话。它会把会话记录保存为 JSON 文件，并支持导出 Markdown 交接文档，方便终端版 Codex 和 VS Code 插件版 Codex 之间交接上下文。

## 设计目标

- 本地保存会话摘要、任务、笔记和事件。
- 统一管理终端 Codex、VS Code 插件 Codex、混合协作会话。
- 支持扫描 `~/.codex/sessions/YYYY/MM/DD`，把 VS Code 插件生成的原生 Codex 会话映射成本地可管理记录。
- 避免直接读取 VS Code 插件私有存储，降低格式变更和隐私风险。
- 支持删除会话时同步清理 session 文件、索引和当前会话指针。

## 存储位置

工具优先使用：

```text
.codex/sessions/
```

当前工作区里 `.codex` 已经是一个普通文件，不是目录，所以工具会自动回退到：

```text
.codex-local/sessions/
```

生成文件包括：

```text
.codex-local/index.json
.codex-local/current_session
.codex-local/sessions/<session-id>.json
```

## 常用命令

初始化本地存储：

```bash
python3 scripts/codex_session.py init
```

创建新会话：

```bash
python3 scripts/codex_session.py start "分析 Android 活体检测 SDK" --tag android --tag sdk
```

记录笔记：

```bash
python3 scripts/codex_session.py note "定位到 LivenessDetectActivity 是系统相机活体入口。"
```

记录事件：

```bash
python3 scripts/codex_session.py event "新增本地视频喂帧测试 Activity。"
```

添加任务：

```bash
python3 scripts/codex_session.py task add "在 Android 设备上运行 SDK 测试" --status todo
```

查看当前会话：

```bash
python3 scripts/codex_session.py show -v
```

列出某天的原生 Codex/VS Code 插件会话：

```bash
python3 scripts/codex_session.py list-native --date 2026-04-23
```

把原生 Codex/VS Code 插件会话导入为本地一一映射：

```bash
python3 scripts/codex_session.py import-native --date 2026-04-23 --set-current
```

导出 Markdown 交接文档：

```bash
python3 scripts/codex_session.py export -o handoff.md
```

关闭当前会话：

```bash
python3 scripts/codex_session.py close --status done --summary "分析和测试入口开发完成。"
```

删除当前会话：

```bash
python3 scripts/codex_session.py delete --yes
```

删除指定会话：

```bash
python3 scripts/codex_session.py delete <session-id> --yes
```

清理已经缺失 session 文件的残留索引或 current 指针：

```bash
python3 scripts/codex_session.py delete <session-id> --missing-ok --yes
```

## 命令列表

```text
init                         创建本地存储目录。
start <title>                创建并切换到一个新会话。
list                         列出最近会话。
list-native                  列出 ~/.codex/sessions 下的原生 Codex JSONL 会话。
import-native                把原生 Codex JSONL 会话导入成本地映射。
use <session-id>             切换当前会话。
show [-v] [session-id]       查看当前或指定会话。
note <text>                  追加一条笔记。
event <text>                 追加一条带时间戳的事件。
task add <text>              添加任务。
task set <id> <status>       更新任务状态。
summary <text>               替换会话摘要。
attach-vscode [session-id]   给会话关联 VS Code 插件 Codex 元数据。
close [session-id]           标记会话为完成、暂停或归档。
delete [session-id]          删除会话文件，并同步清理索引和当前会话指针。
export [session-id]          导出 Markdown 交接文档。
```

任务状态：

```text
todo
doing
done
blocked
```

会话状态：

```text
active
done
paused
archived
```

## 管理 VS Code 插件版 Codex

VS Code 插件版 Codex 的原生会话通常保存在：

```text
~/.codex/sessions/YYYY/MM/DD/*.jsonl
```

工具可以把这些 JSONL 原生会话映射成本地管理器里的稳定 ID，规则是：

```text
native-<原生 Codex session id>
```

例如当前路径 `~/.codex/sessions/2026/04/23` 里发现的插件会话会映射为：

```text
本地管理 ID: native-019db94d-b155-74c2-b69f-8da35732f2a2
原生会话 ID: 019db94d-b155-74c2-b69f-8da35732f2a2
原生文件: /home/scc/.codex/sessions/2026/04/23/rollout-2026-04-23T15-46-09-019db94d-b155-74c2-b69f-8da35732f2a2.jsonl
来源: vscode / codex_vscode
```

导入后可以直接查看映射：

```bash
python3 scripts/codex_session.py show native-019db94d-b155-74c2-b69f-8da35732f2a2 -v
```

也可以导出交接文档：

```bash
python3 scripts/codex_session.py export native-019db94d-b155-74c2-b69f-8da35732f2a2 -o handoff.md
```

创建一个 VS Code 插件 Codex 会话：

```bash
python3 scripts/codex_session.py start "审查活体检测 Android SDK" \
  --client vscode \
  --vscode-workspace /supercloud/llm-code/scc/scc/Liveness_Detection \
  --external-ref "VS Code Codex chat: liveness review"
```

把当前终端会话关联到 VS Code 插件 Codex：

```bash
python3 scripts/codex_session.py attach-vscode \
  --keep-cli \
  --workspace /supercloud/llm-code/scc/scc/Liveness_Detection \
  --external-ref "VS Code Codex chat title or copied reference" \
  --note "同一任务从 VS Code 插件继续。"
```

说明：

- `--client vscode` 表示会话主要来自 VS Code 插件。
- `--client cli` 表示会话主要来自终端。
- `--client mixed` 或 `attach-vscode --keep-cli` 表示终端和 VS Code 插件共用同一任务上下文。
- `--external-ref` 可以记录 VS Code 插件聊天标题、复制出来的链接、人工编号或其他可追踪引用。

## 删除命令说明

`delete` 会同时处理三件事：

- 删除 `.codex-local/sessions/<session-id>.json`。
- 从 `.codex-local/index.json` 移除对应索引。
- 如果 `.codex-local/current_session` 指向该会话，则删除 current 指针。

默认情况下，`delete` 会要求输入 `yes` 二次确认。自动化脚本中可以加 `--yes` 跳过确认。

注意：这里删除的是本工具记录的本地会话，不会关闭 VS Code 插件或终端里正在运行的真实聊天窗口。

如果这个会话是通过 `import-native` 从 `~/.codex/sessions` 导入的映射，并且你确定要连原生 Codex JSONL 文件一起删除，可以显式加 `--native`：

```bash
python3 scripts/codex_session.py delete native-019db94d-b155-74c2-b69f-8da35732f2a2 --native --yes
```

`--native` 会额外执行两件事：

- 删除映射记录里保存的 `~/.codex/sessions/.../*.jsonl` 原生会话文件。
- 从 `~/.codex/session_index.jsonl` 清理对应原生会话索引。

建议只在确认 VS Code 插件/终端里已经不再需要该会话时使用 `--native`。如果只是整理交接文档或本地管理索引，使用不带 `--native` 的 `delete` 更安全。
