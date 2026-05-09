# Codex Skill 构建教程

本文说明如何构建自己的 Codex Skill。Skill 可以把某一类任务的专用流程、领域知识、脚本和模板沉淀下来，让 Codex 在遇到相关任务时自动加载并按你的规则工作。

## 1. Skill 是什么

Codex Skill 是一个自包含目录，至少包含一个 `SKILL.md` 文件。

它主要解决这类问题：

1. 某类任务经常重复做，例如评估模型、整理报告、调用固定 API。
2. 任务有固定流程，直接靠临场 prompt 容易遗漏步骤。
3. 任务依赖公司内部规范、项目结构、数据字段或模板。
4. 有些操作适合脚本化，避免每次重新生成相同代码。

你可以把 Skill 理解为“给 Codex 看的专业操作手册”。

## 2. 推荐存放位置

推荐把自定义 Skill 放到：

```bash
${CODEX_HOME:-$HOME/.codex}/skills
```

如果没有设置 `CODEX_HOME`，实际路径通常是：

```bash
~/.codex/skills
```

例如：

```text
~/.codex/skills/my-skill/
```

放在这个目录下，Codex 更容易自动发现并加载。

## 3. Skill 目录结构

最小结构：

```text
my-skill/
└── SKILL.md
```

推荐结构：

```text
my-skill/
├── SKILL.md              # 必需，Skill 的触发描述和核心流程
├── agents/
│   └── openai.yaml       # 推荐，UI 展示元信息
├── scripts/              # 可选，可执行脚本
├── references/           # 可选，详细文档、规范、表结构、API 说明
└── assets/               # 可选，模板、图片、字体、样例文件
```

各目录用途：

| 路径 | 是否必需 | 用途 |
| --- | --- | --- |
| `SKILL.md` | 必需 | 定义 Skill 名称、触发条件和工作流程 |
| `agents/openai.yaml` | 推荐 | 给 Codex UI 展示用的名称、描述和默认提示 |
| `scripts/` | 可选 | 放稳定、可重复执行的脚本 |
| `references/` | 可选 | 放较长的参考资料，需要时再读取 |
| `assets/` | 可选 | 放输出要用的模板、素材或样例 |

## 4. 命名规则

Skill 名称建议：

1. 只使用小写字母、数字和短横线。
2. 不使用空格、下划线和中文。
3. 尽量短，最好能表达动作或领域。
4. 目录名和 `SKILL.md` frontmatter 里的 `name` 保持一致。

好例子：

```text
pdf-editor
frontend-builder
liveness-eval-helper
company-api-docs
gh-address-comments
```

不推荐：

```text
My Skill
skill_for_pdf
一个很强的技能
helper
```

## 5. 创建 Skill 的标准流程

推荐按下面顺序做：

1. 明确 Skill 要解决的问题。
2. 收集 3 到 5 个真实使用场景。
3. 设计需要的 `scripts/`、`references/`、`assets/`。
4. 初始化 Skill 目录。
5. 编写 `SKILL.md`。
6. 添加脚本、参考资料或模板。
7. 运行校验脚本。
8. 用真实任务试用并迭代。

## 6. 初始化 Skill

Codex 自带了 Skill 初始化脚本。推荐用它创建目录和模板。

### 6.1 创建最小 Skill

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills"
```

执行后会生成：

```text
~/.codex/skills/my-skill/
├── SKILL.md
└── agents/
    └── openai.yaml
```

### 6.2 创建带资源目录的 Skill

如果你需要脚本和参考资料：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references
```

如果还需要素材模板：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references,assets
```

### 6.3 带示例文件初始化

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references --examples
```

注意：如果使用 `--examples`，后续要删除或替换占位示例文件，不要把无用示例留在最终 Skill 里。

## 7. `SKILL.md` 写法

`SKILL.md` 是 Skill 的核心文件，由两部分组成：

1. YAML frontmatter
2. Markdown 正文

### 7.1 最小模板

```markdown
---
name: my-skill
description: Use when Codex needs to handle xxx tasks, including aaa, bbb, and ccc. Trigger when the user asks for xxx, yyy, or zzz.
---

# My Skill

## Workflow

1. Inspect the relevant files or inputs.
2. Identify the task type.
3. Follow the matching procedure.
4. Validate the result.

## Procedures

### For aaa tasks

- Do this.
- Then do that.
- Use `scripts/example.py` when deterministic processing is needed.

### For bbb tasks

- Read `references/bbb.md` only when the user asks about bbb.
- Keep edits scoped to the requested files.

## Validation

- Run the relevant command or script.
- Report skipped validation clearly.
```

### 7.2 Frontmatter 规则

`SKILL.md` 开头必须有 YAML frontmatter：

```yaml
---
name: my-skill
description: Use when Codex needs to ...
---
```

只需要写两个字段：

| 字段 | 是否必需 | 说明 |
| --- | --- | --- |
| `name` | 必需 | Skill 名称，和目录名一致 |
| `description` | 必需 | 触发条件和能力描述 |

不要在 frontmatter 里随便增加额外字段。

### 7.3 `description` 是最重要的字段

Codex 是否会加载这个 Skill，主要取决于 `description`。

好的 `description` 要同时说明：

1. 这个 Skill 做什么。
2. 用户在什么场景下应该触发它。
3. 覆盖哪些具体任务。

好例子：

```yaml
description: Use when Codex needs to evaluate face liveness detection models, parse evaluation logs, compare checkpoints, generate metrics tables, or debug false accept and false reject behavior in anti-spoofing projects.
```

不好的例子：

```yaml
description: A useful helper skill.
```

问题是太泛，Codex 无法判断什么时候应该使用它。

### 7.4 正文应该写什么

正文只写 Codex 执行任务时真正需要知道的内容：

1. 标准工作流程。
2. 必须遵守的约束。
3. 什么时候读取 `references/` 中的文件。
4. 什么时候运行 `scripts/` 中的脚本。
5. 如何验证结果。
6. 常见失败情况怎么处理。

不要写给人看的冗长背景介绍。

## 8. `scripts/` 的使用

适合放进 `scripts/` 的内容：

1. 重复执行的处理逻辑。
2. 容易写错的转换流程。
3. 需要稳定输出的校验或生成工具。
4. 文件格式处理脚本。

例子：

```text
pdf-editor/
├── SKILL.md
└── scripts/
    └── rotate_pdf.py
```

在 `SKILL.md` 中写清楚什么时候使用脚本：

```markdown
## PDF Rotation

- Use `scripts/rotate_pdf.py` for PDF page rotation.
- Run the script instead of rewriting PDF rotation logic.
- Verify the output file exists and has the expected page count.
```

脚本要求：

1. 参数清晰。
2. 有错误提示。
3. 尽量可重复执行。
4. 添加后实际运行一次，确认没有语法错误和路径问题。

## 9. `references/` 的使用

适合放进 `references/` 的内容：

1. 数据库表结构。
2. API 文档。
3. 公司内部规范。
4. 业务规则。
5. 长示例。
6. 不适合直接塞进 `SKILL.md` 的详细说明。

示例结构：

```text
company-api-helper/
├── SKILL.md
└── references/
    ├── auth.md
    ├── users-api.md
    └── billing-api.md
```

在 `SKILL.md` 中只写导航：

```markdown
## Reference Selection

- For authentication tasks, read `references/auth.md`.
- For user API tasks, read `references/users-api.md`.
- For billing API tasks, read `references/billing-api.md`.
```

这样 Codex 只有在需要时才读取对应文件，避免一次性加载太多上下文。

## 10. `assets/` 的使用

适合放进 `assets/` 的内容：

1. 报告模板。
2. 前端项目模板。
3. 图片、图标、字体。
4. 配置样例。
5. 输出文件需要复用的静态资源。

示例：

```text
report-builder/
├── SKILL.md
└── assets/
    ├── report-template.md
    └── company-logo.png
```

`assets/` 里的文件通常不是给 Codex 逐字阅读的，而是用于复制、修改或嵌入最终产物。

## 11. `agents/openai.yaml`

`agents/openai.yaml` 是推荐文件，用于 UI 展示 Skill 名称、短描述和默认提示。

通常不需要手写，可以用初始化脚本或生成脚本创建。

如果需要重新生成：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py ~/.codex/skills/my-skill --interface display_name="My Skill" --interface short_description="Handle xxx workflows" --interface default_prompt="Use this skill to handle xxx tasks."
```

一般只需要包含：

1. `display_name`
2. `short_description`
3. `default_prompt`

除非确实需要，不要额外添加图标、颜色等字段。

## 12. 校验 Skill

创建或修改完成后，运行：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/quick_validate.py ~/.codex/skills/my-skill
```

校验内容包括：

1. `SKILL.md` 是否存在。
2. YAML frontmatter 是否合法。
3. `name` 和 `description` 是否存在。
4. Skill 名称是否符合规则。

如果校验失败，按错误提示修改后重新运行。

## 13. 完整示例：构建一个模型评估 Skill

假设你经常让 Codex 评估活体检测模型、解析日志、对比 checkpoint，可以创建：

```text
liveness-eval-helper/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── scripts/
│   └── summarize_eval_metrics.py
└── references/
    └── metrics.md
```

### 13.1 初始化

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py liveness-eval-helper --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references
```

### 13.2 `SKILL.md` 示例

```markdown
---
name: liveness-eval-helper
description: Use when Codex needs to evaluate face liveness detection models, parse evaluation logs, compare checkpoints, summarize APCER/BPCER/ACER metrics, or debug false accept and false reject behavior in anti-spoofing projects.
---

# Liveness Eval Helper

## Workflow

1. Identify the evaluation target: checkpoint, log file, dataset split, or script.
2. Inspect existing project commands before inventing new ones.
3. Use existing evaluation scripts when available.
4. Summarize metrics with APCER, BPCER, ACER, threshold, and dataset split.
5. Compare checkpoints only when they were evaluated on the same protocol.
6. Report missing logs, missing checkpoints, or inconsistent protocols clearly.

## References

- Read `references/metrics.md` when explaining APCER, BPCER, ACER, FAR, FRR, or threshold selection.

## Scripts

- Use `scripts/summarize_eval_metrics.py` when multiple evaluation logs need to be summarized.

## Validation

- Verify that every reported metric comes from an actual log, script output, or dataset file.
- Do not mix results from different protocols without explicitly labeling them.
- If evaluation cannot be run, explain the blocker and provide the exact command that should be run next.
```

### 13.3 `references/metrics.md` 示例

```markdown
# Liveness Metrics

## Common Metrics

- APCER: Attack Presentation Classification Error Rate.
- BPCER: Bona Fide Presentation Classification Error Rate.
- ACER: Average of APCER and BPCER.

## Reporting Rules

- Always include the dataset split.
- Always include the threshold when available.
- Do not compare metrics from different protocols as if they were equivalent.
```

### 13.4 脚本示例

```text
scripts/summarize_eval_metrics.py
```

脚本可以做这些事：

1. 扫描日志目录。
2. 提取 checkpoint 名称。
3. 提取 APCER/BPCER/ACER。
4. 输出 Markdown 表格或 JSON。

在 `SKILL.md` 中只需要告诉 Codex 什么时候运行它，不需要把脚本内容全部复制进去。

## 14. 好 Skill 和差 Skill 的区别

### 14.1 好 Skill

```markdown
---
name: gh-address-comments
description: Use when Codex needs to address GitHub PR review comments, inspect changed files, map each review comment to a concrete code change, update tests, and summarize which comments were resolved.
---

# GH Address Comments

## Workflow

1. Fetch or read the review comments.
2. Group comments by file and issue type.
3. Inspect the current diff before editing.
4. Make the smallest code changes that resolve the comments.
5. Run targeted tests.
6. Summarize each resolved comment and any remaining blocker.
```

优点：

1. 触发条件明确。
2. 工作流程具体。
3. 没有无关背景。
4. 容易被 Codex 正确使用。

### 14.2 差 Skill

```markdown
---
name: helper
description: Helps with coding.
---

# Helper

This skill helps with many things. Try to write good code and be careful.
```

问题：

1. 名称太泛。
2. 描述太泛。
3. 没有明确触发条件。
4. 没有可执行流程。

## 15. 常见错误

| 错误 | 后果 | 修正方式 |
| --- | --- | --- |
| `description` 太短 | Codex 不知道何时触发 | 写清任务类型和触发场景 |
| `SKILL.md` 太长 | 占用上下文，降低效率 | 把细节拆到 `references/` |
| 所有资料都塞进正文 | 每次触发都加载太多内容 | 用渐进式加载 |
| 脚本没有测试 | Codex 调用时失败 | 添加后实际运行脚本 |
| 名称不规范 | 校验失败或难以识别 | 使用小写和短横线 |
| 目录里放 README/CHANGELOG | 增加噪声 | 只保留执行任务需要的文件 |
| 触发范围过大 | 不该用时也会用 | 缩小 description 范围 |
| 没有验证步骤 | 输出不可靠 | 在正文写清如何验证 |

## 16. 编写原则

1. 假设 Codex 已经很聪明，只补充它不知道的流程和上下文。
2. `SKILL.md` 保持精简，最好不要超过 500 行。
3. 详细资料放到 `references/`。
4. 稳定重复的操作放到 `scripts/`。
5. 模板和素材放到 `assets/`。
6. `description` 要具体，这是触发 Skill 的关键。
7. 每个 Skill 只解决一类清晰问题。
8. 创建后用真实任务测试，不要只看格式正确。

## 17. 速查命令

| 任务 | 命令 |
| --- | --- |
| 创建最小 Skill | `/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills"` |
| 创建带脚本和参考资料的 Skill | `/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references` |
| 创建带全部资源目录的 Skill | `/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references,assets` |
| 校验 Skill | `/home/scc/.codex/skills/.system/skill-creator/scripts/quick_validate.py ~/.codex/skills/my-skill` |
| 重新生成 UI 元信息 | `/home/scc/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py ~/.codex/skills/my-skill --interface display_name="My Skill" --interface short_description="Handle xxx workflows" --interface default_prompt="Use this skill to handle xxx tasks."` |

## 18. 推荐实践模板

新建 Skill 前，可以先填写这个清单：

```markdown
## Skill 设计清单

- Skill 名称：
- 要解决的问题：
- 用户可能怎么提问：
- 必须遵守的流程：
- 需要哪些脚本：
- 需要哪些参考资料：
- 需要哪些模板或素材：
- 如何验证输出：
- 哪些情况不应该使用这个 Skill：
```

根据清单写出的 Skill 通常更稳定，也更容易维护。

## 19. 最小可用示例

如果你只是想快速起步，可以创建一个最小 Skill：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/init_skill.py project-doc-writer --path "${CODEX_HOME:-$HOME/.codex}/skills"
```

然后把 `~/.codex/skills/project-doc-writer/SKILL.md` 改成：

```markdown
---
name: project-doc-writer
description: Use when Codex needs to write or update project documentation, including README files, setup guides, usage tutorials, architecture notes, and command references based on the actual repository contents.
---

# Project Doc Writer

## Workflow

1. Inspect existing documentation before writing new content.
2. Read relevant source files, scripts, and configuration files.
3. Prefer facts from the repository over assumptions.
4. Write concise Markdown with clear headings and command examples.
5. Do not invent commands, services, paths, or dependencies.
6. Validate command examples against files that exist in the repository.

## Validation

- Check that referenced files and scripts exist.
- Report any commands that were inferred but not executed.
- Keep the final response focused on files changed and validation performed.
```

校验：

```bash
/home/scc/.codex/skills/.system/skill-creator/scripts/quick_validate.py ~/.codex/skills/project-doc-writer
```

这个 Skill 就可以用于“根据项目生成文档”这类任务。
