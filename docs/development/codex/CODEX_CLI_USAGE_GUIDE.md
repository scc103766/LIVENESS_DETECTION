# Codex CLI 使用教程

本文基于本机安装的 `codex-cli 0.125.0` 整理，更新时间：2026-05-08。Codex CLI 版本迭代较快，遇到差异时优先以本机命令输出为准：

```bash
codex --help
codex <command> --help
codex --version
```

## 1. Codex CLI 是什么

Codex CLI 是 OpenAI 提供的本地命令行编码代理。它可以在你的项目目录中读取代码、执行命令、修改文件、做代码审查、管理会话，并通过 MCP 或插件扩展能力。

常见使用方式有两类：

| 使用方式 | 命令 | 适合场景 |
| --- | --- | --- |
| 交互式 | `codex` | 日常开发、边聊边改、需要持续确认的任务 |
| 非交互式 | `codex exec` | 脚本化任务、CI 辅助、一次性生成或检查 |

## 2. 安装与验证

### 2.1 安装

常见安装方式：

```bash
npm install -g @openai/codex
```

macOS 也可以使用 Homebrew：

```bash
brew install --cask codex
```

也可以从 OpenAI Codex 的 GitHub Release 下载对应平台的二进制文件。安装后，确认 `codex` 已经在 `PATH` 中。

### 2.2 验证安装

```bash
command -v codex
codex --version
codex --help
```

如果 `command -v codex` 没有输出，说明 shell 找不到可执行文件，需要检查 npm 全局安装目录或 Homebrew 安装路径是否加入了 `PATH`。

## 3. 登录与认证

### 3.1 交互式登录

```bash
codex login
```

执行后按终端提示完成登录。常见选择包括使用 ChatGPT 账号登录，或使用 API key。

### 3.2 查看登录状态

```bash
codex login status
```

### 3.3 使用 API key 登录

`codex login --with-api-key` 会从标准输入读取 API key：

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
```

注意不要把 API key 直接写进 shell 历史、README、脚本或提交到 Git。

### 3.4 退出登录

```bash
codex logout
```

该命令会移除本地保存的认证凭据。

## 4. 交互式使用：`codex`

### 4.1 进入交互界面

```bash
codex
```

不带子命令时，`codex` 会进入交互式 TUI。你可以像和工程助手对话一样描述任务，例如：

```text
请阅读这个项目，找出训练入口脚本，并说明如何启动。
```

### 4.2 带初始提示启动

```bash
codex "阅读 README，告诉我这个项目如何运行"
```

### 4.3 指定工作目录

```bash
codex -C /path/to/project
```

`-C, --cd <DIR>` 用于告诉 Codex 使用哪个目录作为工作根目录。日常建议在 Git 仓库根目录运行，方便 Codex 理解项目结构并查看 diff。

### 4.4 附加图片

```bash
codex -i screenshot.png "根据截图修复前端布局问题"
```

`-i, --image <FILE>` 可以把图片附加到初始提示中，适合 UI、视觉检查、报错截图分析等任务。

### 4.5 指定模型

```bash
codex -m <MODEL> "帮我重构这个模块"
```

`-m, --model <MODEL>` 用于覆盖当前配置中的模型。具体可用模型取决于你的账号、配置和当前 Codex 版本。

### 4.6 启用联网搜索

```bash
codex --search "查阅最新官方文档后升级这段 API 调用"
```

`--search` 会启用 live web search。只在确实需要外部最新资料时使用；普通代码修改通常不需要联网。

### 4.7 保留终端滚动历史

```bash
codex --no-alt-screen
```

`--no-alt-screen` 会关闭 alternate screen 模式，让输出保留在终端滚动历史中。它适合在部分终端复用器里使用。

### 4.8 交互界面 Slash 命令速查

进入 `codex` 交互界面后，在输入框输入 `/` 会打开 TUI 内置命令菜单。以下表格基于本机 `codex-cli 0.125.0` 的菜单整理；如果升级后命令有变化，以实际界面为准。

| Slash 命令 | 功能 | 常见用途或注意事项 |
| --- | --- | --- |
| `/model` | 选择模型和 reasoning effort | 临时切换模型、推理强度，影响质量、速度和用量 |
| `/fast` | 切换 Fast mode | 使用更快推理，通常会增加 plan usage |
| `/permissions` | 配置 Codex 被允许执行的操作 | 调整沙箱、审批、自动执行相关权限 |
| `/experimental` | 切换实验性功能 | 只建议在明确需要试用新功能时开启 |
| `/memories` | 配置记忆使用和生成 | 管理 Codex 是否使用、生成长期记忆 |
| `/skills` | 使用 skills 改善特定任务表现 | 查看或触发可用技能，例如 OpenAI 文档、图片生成等 |
| `/review` | 审查当前改动并查找问题 | 在当前会话里做代码审查，重点找 bug、风险和测试缺口 |
| `/rename` | 重命名当前 thread | 方便后续从会话列表中识别当前工作 |
| `/new` | 在当前会话中开启新聊天 | 清理当前聊天上下文，开始新的任务线 |
| `/resume` | 恢复已保存的聊天 | 从历史会话继续工作 |
| `/fork` | fork 当前聊天 | 基于当前上下文派生另一条实现或分析路径 |
| `/init` | 创建 `AGENTS.md` 指令文件 | 初始化项目级 Codex 指令；已有文件时会避免覆盖 |
| `/compact` | 压缩并总结对话上下文 | 上下文过长时保留关键信息，降低撞到上下文上限的概率 |
| `/plan` | 切换到 Plan mode | 需要先讨论方案、暂不直接改代码时使用 |
| `/collab` | 切换协作模式 | 实验性功能，用于改变交互/协作方式 |
| `/agent` | 切换当前活跃 agent thread | 使用多 agent 或子线程时切换焦点 |
| `/side` | 在临时 fork 中开启 side conversation | 做旁路讨论或验证，不污染主会话 |
| `/copy` | 复制上一条回复为 Markdown | 便于把结果放入文档、Issue 或 PR 描述 |
| `/diff` | 显示 git diff，包括 untracked files | 快速查看当前工作区改动 |
| `/mention` | 提及一个文件 | 把指定文件加入当前输入上下文 |
| `/status` | 显示当前 session 配置和 token usage | 排查模型、目录、权限、上下文用量等状态 |
| `/title` | 配置终端标题显示项 | 调整终端窗口标题里展示的信息 |
| `/statusline` | 配置状态栏显示项 | 调整 TUI 底部状态栏展示的信息 |
| `/theme` | 选择语法高亮主题 | 改变代码块和 diff 的显示风格 |
| `/mcp` | 列出已配置 MCP tools | 可用 `/mcp verbose` 查看更详细信息 |
| `/plugins` | 浏览插件 | 查看和管理 Codex 插件能力 |
| `/logout` | 退出 Codex 登录 | 清除当前 Codex 登录状态 |
| `/exit` | 退出 Codex | 结束当前 TUI 会话 |
| `/feedback` | 向维护者发送日志 | 反馈问题时使用，注意日志可能包含本地运行信息 |
| `/ps` | 列出后台终端 | 查看 Codex 启动或保留的 background terminals |
| `/stop` | 停止所有后台终端 | 终止仍在后台运行的命令 |
| `/clear` | 清空终端并开始新聊天 | 清理屏幕，同时开启新的聊天上下文 |
| `/personality` | 选择 Codex 的沟通风格 | 调整回复风格；部分模型可能不支持 personalities |

## 5. 非交互式执行：`codex exec`

`codex exec` 用于一次性运行任务，适合自动化和脚本化场景。

### 5.1 基本用法

```bash
codex exec "阅读项目并总结主要入口文件"
```

### 5.2 指定工作目录执行

```bash
codex exec -C /path/to/project "运行测试并修复失败项"
```

### 5.3 从标准输入读取任务

```bash
codex exec -
```

如果标准输入被管道连接，并且同时提供了 prompt，stdin 内容会被追加为 `<stdin>` 块。

示例：

```bash
git diff | codex exec "审查这次改动，指出风险"
```

### 5.4 输出 JSONL 事件

```bash
codex exec --json "检查项目结构"
```

`--json` 会把事件以 JSONL 形式输出，适合脚本解析。

### 5.5 保存最后一条回复

```bash
codex exec -o result.md "生成项目启动说明"
```

`-o, --output-last-message <FILE>` 会把代理最后一条消息写入指定文件。

### 5.6 使用 JSON Schema 约束最终输出

```bash
codex exec --output-schema schema.json "分析日志并输出结构化结果"
```

`--output-schema <FILE>` 用于指定最终响应的 JSON Schema，适合需要机器读取结果的场景。

### 5.7 临时会话

```bash
codex exec --ephemeral "解释这个函数的作用"
```

`--ephemeral` 表示不把本次会话持久化到磁盘，适合不需要后续恢复的临时查询。

### 5.8 跳过 Git 仓库检查

```bash
codex exec --skip-git-repo-check "整理这个目录里的文件"
```

默认建议在 Git 仓库中使用 Codex。只有在处理非 Git 目录时才使用 `--skip-git-repo-check`。

## 6. 代码审查：`codex review`

`codex review` 用于非交互式代码审查，重点输出问题、风险和测试缺口。

### 6.1 审查当前未提交改动

```bash
codex review --uncommitted
```

会审查 staged、unstaged 和 untracked changes。

### 6.2 和指定分支比较

```bash
codex review --base main
```

适合在 feature branch 上审查相对 `main` 的所有改动。

### 6.3 审查某个 commit

```bash
codex review --commit <SHA>
```

### 6.4 添加自定义审查要求

```bash
codex review --uncommitted "重点检查数据泄漏、异常处理和测试覆盖"
```

### 6.5 指定审查标题

```bash
codex review --base main --title "flash liveness v3 review"
```

## 7. 应用 Codex 生成的 diff：`codex apply`

```bash
codex apply <TASK_ID>
```

`codex apply` 会把指定 Codex 任务产生的最新 diff 作为 `git apply` 应用到本地工作树。应用前建议先确认当前工作区状态：

```bash
git status
git diff
```

如果本地有未提交改动，先确认这些改动不会和即将应用的 patch 冲突。

## 8. 会话管理：`resume` 与 `fork`

Codex 会记录会话。你可以恢复旧会话，也可以从旧会话派生一个新分支。

### 8.1 恢复最近的交互式会话

```bash
codex resume --last
```

### 8.2 通过会话 ID 恢复

```bash
codex resume <SESSION_ID>
```

`SESSION_ID` 可以是 UUID，也可以是线程名称。UUID 优先。

### 8.3 恢复会话并追加新任务

```bash
codex resume <SESSION_ID> "继续修复刚才没有完成的测试"
```

### 8.4 显示所有会话

```bash
codex resume --all
```

`--all` 会禁用当前目录过滤，并显示 CWD 列。

### 8.5 包含非交互式会话

```bash
codex resume --include-non-interactive
```

该选项会把 `codex exec` 产生的非交互式会话也纳入选择范围。

### 8.6 Fork 最近会话

```bash
codex fork --last
```

`fork` 会基于历史会话创建一个新会话，适合尝试另一种实现路径，同时保留原会话上下文。

### 8.7 非交互式恢复会话

```bash
codex exec resume --last "继续运行验证并总结结果"
```

`codex exec resume` 可以在脚本或自动化流程里继续旧会话。

## 9. 沙箱与审批策略

Codex 在执行命令或编辑文件时会受到 sandbox 和 approval policy 控制。正确设置这些选项可以降低误操作风险。

### 9.1 沙箱模式：`--sandbox`

```bash
codex -s read-only
codex -s workspace-write
codex -s danger-full-access
```

| 模式 | 功能 | 建议 |
| --- | --- | --- |
| `read-only` | 只读沙箱，适合代码阅读和解释 | 最安全 |
| `workspace-write` | 可写当前工作区和允许的目录 | 日常开发推荐 |
| `danger-full-access` | 更高权限 | 谨慎使用 |

### 9.2 审批策略：`--ask-for-approval`

```bash
codex -a on-request
codex -a untrusted
codex -a never
```

| 策略 | 功能 |
| --- | --- |
| `untrusted` | 只有可信命令可直接运行，其他命令需要审批 |
| `on-request` | 由 Codex 判断何时请求审批 |
| `never` | 不请求审批，失败会直接返回给模型 |
| `on-failure` | 已弃用，不建议新流程使用 |

### 9.3 低摩擦自动执行

```bash
codex --full-auto
codex exec --full-auto "修复 lint 错误"
```

`--full-auto` 是低摩擦自动执行模式。它适合边界明确、可通过测试验证的任务。

### 9.4 高危选项

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

该选项会跳过确认提示并绕过沙箱，非常危险。只有在外部环境已经提供可靠隔离时才考虑使用。

## 10. 配置与特性开关

### 10.1 临时覆盖配置

```bash
codex -c key=value
codex -c nested.key=value
```

`-c, --config <key=value>` 会覆盖 `~/.codex/config.toml` 中的配置。key 支持点号路径，value 会按 TOML 解析。

示例：

```bash
codex -c 'model="<MODEL>"'
codex -c shell_environment_policy.inherit=all
```

### 10.2 使用配置 profile

```bash
codex -p <PROFILE>
codex exec -p <PROFILE> "运行项目检查"
```

`-p, --profile <CONFIG_PROFILE>` 用于选择 `config.toml` 中的配置 profile。

### 10.3 临时启用或禁用 feature

```bash
codex --enable <FEATURE>
codex --disable <FEATURE>
```

等价于覆盖配置中的：

```toml
[features]
name = true
```

### 10.4 管理 feature flags

```bash
codex features list
codex features enable <FEATURE>
codex features disable <FEATURE>
```

`codex features list` 可以查看已知 feature 的阶段和当前生效状态。

## 11. MCP 管理：`codex mcp`

MCP（Model Context Protocol）用于给 Codex 接入外部工具、资源和服务。

### 11.1 查看 MCP 服务器

```bash
codex mcp list
codex mcp list --json
```

### 11.2 查看某个 MCP 配置

```bash
codex mcp get <NAME>
codex mcp get <NAME> --json
```

### 11.3 添加 HTTP MCP 服务器

```bash
codex mcp add <NAME> --url <URL>
```

示例：

```bash
codex mcp add openaiDeveloperDocs --url https://developers.openai.com/mcp
```

如果服务需要 bearer token，可以指定环境变量名：

```bash
codex mcp add <NAME> --url <URL> --bearer-token-env-var <ENV_VAR>
```

### 11.4 添加 stdio MCP 服务器

```bash
codex mcp add <NAME> -- <COMMAND> ...
```

需要设置环境变量时：

```bash
codex mcp add <NAME> --env KEY=VALUE -- <COMMAND> ...
```

### 11.5 MCP OAuth 登录与退出

```bash
codex mcp login <NAME>
codex mcp login <NAME> --scopes scope1,scope2
codex mcp logout <NAME>
```

### 11.6 移除 MCP 服务器

```bash
codex mcp remove <NAME>
```

## 12. 插件管理：`codex plugin`

Codex 插件通过 marketplace 分发和管理。

### 12.1 查看插件命令

```bash
codex plugin --help
codex plugin marketplace --help
```

### 12.2 添加 marketplace

```bash
codex plugin marketplace add <SOURCE>
```

`<SOURCE>` 支持：

| 来源类型 | 示例 |
| --- | --- |
| GitHub owner/repo | `owner/repo` |
| 指定 ref | `owner/repo@ref` |
| HTTP(S) Git URL | `https://github.com/owner/repo.git` |
| SSH Git URL | `git@github.com:owner/repo.git` |
| 本地 marketplace 根目录 | `/path/to/marketplace` |

可选参数：

```bash
codex plugin marketplace add <SOURCE> --ref <REF>
codex plugin marketplace add <SOURCE> --sparse <PATH>
```

### 12.3 升级 marketplace

```bash
codex plugin marketplace upgrade
codex plugin marketplace upgrade <MARKETPLACE_NAME>
```

### 12.4 移除 marketplace

```bash
codex plugin marketplace remove <MARKETPLACE_NAME>
```

## 13. Shell 补全：`codex completion`

生成 shell completion 脚本：

```bash
codex completion bash
codex completion zsh
codex completion fish
codex completion powershell
codex completion elvish
```

默认 shell 是 `bash`：

```bash
codex completion
```

把生成结果放到对应 shell 的补全目录后，重启 shell 即可生效。不同系统的补全目录不同，建议先查看当前 shell 的官方补全文档。

## 14. 独立沙箱命令：`codex sandbox`

`codex sandbox` 可以在 Codex 提供的沙箱里运行命令。

```bash
codex sandbox linux <COMMAND>
codex sandbox macos <COMMAND>
codex sandbox windows <COMMAND>
```

| 子命令 | 平台 | 说明 |
| --- | --- | --- |
| `linux` | Linux | 默认使用 bubblewrap，也有 `landlock` alias |
| `macos` | macOS | 使用 Seatbelt，也有 `seatbelt` alias |
| `windows` | Windows | 使用 Windows restricted token |

日常开发中一般通过 `codex --sandbox ...` 控制代理执行环境；只有需要直接测试沙箱行为时才使用 `codex sandbox`。

## 15. Codex Cloud 与服务模式

以下命令偏高级或实验性质，使用前建议先查看 help：

```bash
codex cloud --help
codex mcp-server --help
codex app-server --help
codex exec-server --help
```

### 15.1 Codex Cloud

```bash
codex cloud list
codex cloud status <TASK_ID>
codex cloud diff <TASK_ID>
codex cloud apply <TASK_ID>
codex cloud exec "提交一个云端任务"
```

`codex cloud` 用于浏览 Codex Cloud 任务，并把云端任务 diff 应用到本地。

### 15.2 MCP Server 模式

```bash
codex mcp-server
```

该命令会把 Codex 作为 stdio MCP server 启动，供其他 MCP client 调用。

### 15.3 App Server 与 Exec Server

```bash
codex app-server
codex exec-server
```

这两个命令主要用于 IDE、桌面应用、远程控制或内部集成场景。普通命令行开发通常不需要直接使用。

## 16. 常用全局选项速查

| 选项 | 功能 | 示例 |
| --- | --- | --- |
| `-C, --cd <DIR>` | 指定工作目录 | `codex -C .` |
| `-m, --model <MODEL>` | 指定模型 | `codex -m <MODEL>` |
| `-i, --image <FILE>` | 附加图片 | `codex -i ui.png "检查布局"` |
| `-p, --profile <PROFILE>` | 使用配置 profile | `codex -p work` |
| `-s, --sandbox <MODE>` | 设置沙箱模式 | `codex -s workspace-write` |
| `-a, --ask-for-approval <POLICY>` | 设置审批策略 | `codex -a on-request` |
| `-c, --config <key=value>` | 临时覆盖配置 | `codex -c 'model="<MODEL>"'` |
| `--search` | 启用联网搜索 | `codex --search "查最新文档"` |
| `--full-auto` | 低摩擦自动执行 | `codex --full-auto` |
| `--no-alt-screen` | 保留终端滚动历史 | `codex --no-alt-screen` |
| `--enable <FEATURE>` | 临时启用 feature | `codex --enable <FEATURE>` |
| `--disable <FEATURE>` | 临时禁用 feature | `codex --disable <FEATURE>` |
| `-h, --help` | 查看帮助 | `codex exec --help` |
| `-V, --version` | 查看版本 | `codex --version` |

## 17. 常用命令速查

| 命令 | 功能 | 常见用法 |
| --- | --- | --- |
| `codex` | 启动交互式 CLI | `codex` |
| `codex "<prompt>"` | 带初始任务启动交互式 CLI | `codex "解释项目结构"` |
| `codex exec` | 非交互式执行任务 | `codex exec "生成 README"` |
| `codex exec resume` | 非交互式恢复会话 | `codex exec resume --last "继续验证"` |
| `codex review` | 非交互式代码审查 | `codex review --uncommitted` |
| `codex login` | 登录 | `codex login` |
| `codex login status` | 查看登录状态 | `codex login status` |
| `codex logout` | 退出登录 | `codex logout` |
| `codex resume` | 恢复历史交互会话 | `codex resume --last` |
| `codex fork` | 从历史会话派生新会话 | `codex fork --last` |
| `codex apply` | 应用 Codex 任务 diff | `codex apply <TASK_ID>` |
| `codex mcp list` | 查看 MCP server | `codex mcp list --json` |
| `codex mcp add` | 添加 MCP server | `codex mcp add docs --url <URL>` |
| `codex mcp remove` | 移除 MCP server | `codex mcp remove docs` |
| `codex plugin marketplace add` | 添加插件 marketplace | `codex plugin marketplace add owner/repo` |
| `codex plugin marketplace upgrade` | 升级插件 marketplace | `codex plugin marketplace upgrade` |
| `codex completion` | 生成 shell 补全 | `codex completion zsh` |
| `codex sandbox` | 直接运行沙箱命令 | `codex sandbox linux <COMMAND>` |
| `codex features list` | 查看 feature flags | `codex features list` |
| `codex cloud` | 管理 Codex Cloud 任务 | `codex cloud list` |
| `codex mcp-server` | 以 MCP server 启动 | `codex mcp-server` |
| `codex app-server` | 启动 app server | `codex app-server` |
| `codex exec-server` | 启动 exec server | `codex exec-server` |

## 18. 推荐工作流

### 18.1 日常开发

```bash
cd /path/to/project
git status
codex
```

在交互中描述任务：

```text
请先阅读项目结构，然后修复 xxx 问题。只修改必要文件，完成后运行相关测试。
```

完成后本地确认：

```bash
git diff
git status
```

### 18.2 一次性自动化任务

```bash
codex exec -C /path/to/project "检查 README 是否和当前代码一致，并给出需要修改的地方"
```

### 18.3 生成文档

```bash
codex exec -o ARCHITECTURE.md "阅读代码并生成架构说明，要求条理清晰"
```

### 18.4 审查 PR 或分支改动

```bash
codex review --base main "重点关注兼容性、数据安全和测试覆盖"
```

### 18.5 恢复之前的开发上下文

```bash
codex resume --last
```

如果想从旧思路分叉出新方案：

```bash
codex fork --last
```

## 19. Prompt 编写建议

好的 Codex prompt 应该明确目标、范围、约束和验证方式。

### 19.1 好的示例

```text
请修复 scripts/evaluate.py 里评估结果不稳定的问题。
要求：
1. 先定位随机性来源。
2. 只修改评估相关代码，不重构训练流程。
3. 修复后运行最小可行测试，并说明验证结果。
```

```text
请为当前项目生成部署说明，写入 DEPLOYMENT.md。
要求：
1. 包含环境依赖、启动命令、配置项和常见故障。
2. 不要虚构不存在的服务。
3. 先从 README、scripts 和配置文件中提取事实。
```

### 19.2 不推荐的示例

```text
优化一下项目
```

问题是范围太大，Codex 需要猜测目标，容易产生无关改动。

```text
把所有代码都重构一下
```

问题是风险高、验证成本高，也容易和现有工作冲突。

## 20. 安全与协作建议

1. 在 Git 仓库中使用 Codex，修改前后都看 `git status` 和 `git diff`。
2. 日常开发优先使用 `workspace-write`，不要默认使用 `danger-full-access`。
3. 不要把 API key、token、私有路径写进 prompt、文档或提交记录。
4. 让 Codex 先读代码再修改，避免直接要求它凭空生成大规模改动。
5. 对生产代码要求 Codex 运行相关测试；如果测试跑不了，让它说明原因。
6. 对大任务分阶段执行：先分析，再实现，再验证，再总结。
7. 使用 `codex review` 审查改动，但不要把审查结果当作唯一质量门禁。
8. 使用 `--search` 时优先要求引用官方文档或权威来源。
9. 使用 `codex apply` 前确认当前工作区没有冲突风险。
10. 对实验性命令，例如 `cloud`、`app-server`、`exec-server`，先查看 `--help`。

## 21. 排错

### 21.1 `codex: command not found`

检查是否安装成功：

```bash
npm list -g --depth=0
```

检查 npm 全局安装前缀和模块目录：

```bash
npm prefix -g
npm root -g
```

通常需要把 `npm prefix -g` 输出目录下的 `bin` 子目录加入 `PATH`。

如果使用 Homebrew，检查：

```bash
brew list --cask codex
```

### 21.2 登录后仍然无法使用

检查登录状态：

```bash
codex login status
```

如果认证异常，可以退出后重新登录：

```bash
codex logout
codex login
```

### 21.3 Codex 不修改文件

检查是否使用了只读沙箱：

```bash
codex -s workspace-write
```

如果在非 Git 目录执行，可能需要：

```bash
codex exec --skip-git-repo-check "你的任务"
```

### 21.4 命令执行需要审批

查看当前 approval policy。如果你希望由 Codex 判断何时请求审批：

```bash
codex -a on-request
```

如果是自动化场景，并且外部环境已经有隔离，可以考虑更低摩擦的配置，但要先评估风险。

### 21.5 输出太适合人读，不适合脚本解析

使用 JSONL：

```bash
codex exec --json "你的任务"
```

或者要求最后输出符合 schema：

```bash
codex exec --output-schema schema.json "你的任务"
```

## 22. 查看帮助的习惯

Codex CLI 命令层级较多，最可靠的学习方式是逐层查看 help：

```bash
codex --help
codex exec --help
codex review --help
codex mcp --help
codex mcp add --help
codex plugin marketplace --help
codex resume --help
codex fork --help
```

当本文和你的本地输出不一致时，以你的本地 `--help` 为准。

## 23. 官方资料入口

后续如果需要确认最新安装方式、认证方式或 IDE 集成，可以从这些入口开始：

| 资料 | 链接 |
| --- | --- |
| Codex 文档 | <https://developers.openai.com/codex> |
| Codex GitHub 仓库 | <https://github.com/openai/codex> |
| Codex Release | <https://github.com/openai/codex/releases/latest> |
| Codex IDE 文档 | <https://developers.openai.com/codex/ide> |
