# Solvation Labs — YES!Delft 路演：ML Engineer 准备手册

> 日期：2026-10-15  
> 基于 `Pitch_SolvationLabs_YESDelft.pptx` 第 4、6、7、8、9 页的公司叙事整理，并结合工程建议扩展。**既有科研成果、当前产品能力与建议中的未来架构须严格区分。**

## 1. 核心定位

**I turn scientific research workflows into a scalable, reproducible AI-driven engineering platform.**

你的价值不是“让 ChatGPT 生成配方”，而是把材料科学家的知识与脚本，组织为可运行、可追踪、可扩展的工程流程。

### 建议架构（目标状态，不代表已完全建成）

```text
Customer Brief (voltage / anode / safety constraints)
       |
Requirements parser / task orchestration
       |
       +--> Literature Agent --> evidence extraction
       +--> Simulation Jobs --> DFT-based labeling
       +--> ML Screening --> property estimation / ranking
       +--> Evaluation --> provenance / uncertainty / quality checks
       |
Ranked Shortlist + Evidence + Uncertainty
       |
Partner Lab Experimental Validation
       |
Data feedback for future model assessment and improvement
```

**重点表达：** Agent 管编排，physics-based simulation 提供科学依据，ML 辅助筛选，实验决定实际是否有效。

## 2. 英文现场话术

### Q1 — 30 秒自我介绍

> Hi, I'm Alex, the ML Engineer at Solvation Labs.
>
> My background is in distributed systems, backend engineering and machine learning. I have a Master's degree in Computational Science from the University of Amsterdam.
>
> My role is to turn our scientific expertise into a scalable AI-driven platform. Specifically, I focus on orchestrating AI agents, integrating simulation tools and ML models, and building a reliable pipeline that can automate repetitive research tasks.
>
> Our goal is to help scientists explore more electrolyte candidates with less manual work.

### Q2 — What exactly does AI do in your product?

> We don't use AI to randomly generate chemical recipes.
>
> We use AI to automate and accelerate the scientific workflow.
>
> For example, an agent can search scientific literature and extract useful information. Another component can prepare simulation tasks, while our ML models help predict properties and rank promising candidates.
>
> The final decisions are supported by physics-based simulations and experimental validation.
>
> So AI doesn't replace the scientist. It helps scientists explore a much larger design space more efficiently.

注：若功能尚在开发，使用 *can / is designed to / we plan to*，不要误称已经上线。

### Q3 — Why do you need an ML engineer if you already have a materials scientist?

> A materials scientist knows what to investigate and how to evaluate the chemistry.
>
> My responsibility is to make that process automated, reproducible and scalable.
>
> For example, a scientist can manually run a simulation, but running thousands of simulations requires task orchestration, scheduling, failure recovery, data management and model evaluation.
>
> That's where my engineering background becomes valuable.
>
> I focus on building the infrastructure that allows our scientific knowledge to scale beyond individual researchers.

### Q4 — Why can't customers just use ChatGPT?

> ChatGPT can help with scientific literature and reasoning, but it cannot reliably validate electrolyte performance on its own.
>
> Our value comes from combining domain-specific scientific knowledge, simulation tools, experimental data and a structured AI workflow.
>
> The language model is just one component of the system. The real value is in the scientific models, validated data and reproducible results.

### Q5 — How can this become a SaaS product?

> Initially, we plan to work directly with customers through paid pilot projects.
>
> Each project gives us more experimental data and helps us improve our screening pipeline.
>
> Over time, we want to standardize the process so customers can submit their requirements, run screening workflows and review ranked candidates through a self-service platform.
>
> We start with scientific services, but our long-term goal is a scalable software product.

### Q6 — What would you personally deliver during the ten-week program?

> First, a reproducible workflow connecting existing simulation tools and data processing. Second, a lightweight screening dashboard showing customer requirements, progress and candidate rankings. Third, structured records that can link model outputs to future experimental feedback.

## 3. 最值得背的一段：45 秒接棒稿

适用：创始人介绍完材料科学，由你介绍 AI 部分。

> Sure. From the engineering perspective, our goal is to turn electrolyte research into an automated and scalable workflow.
>
> Traditionally, scientists need to search papers, prepare simulations, analyze results and decide what to test next. Many of these tasks are repetitive and time-consuming.
>
> What we want to build is an AI-driven pipeline that connects literature research, physics-based simulations, machine learning and experimental feedback.
>
> My role is to orchestrate these components and make the entire process reliable, reproducible and scalable.
>
> And importantly, we're not trying to replace scientific expertise with AI. We're trying to make that expertise much more productive.
>
> In the long run, we want to turn this workflow into a self-service platform for battery R&D teams.

## 4. 演示建议与配套 Python demo

随附 `solvation_ai_demo/` 是可以本地运行的 Streamlit 原型，旨在**展示产品交互而非科学结果**。

演示环节：

1. 在左侧提交 Customer Brief：`4.6 V / Silicon / Prioritize non-flammable`。
2. 点击 **Run illustrative screening**，讲解需求解析、文献 Agent、DFT 模拟、ML 筛选、排序和实验验证的流程。
3. 指出所有科研步骤目前在 demo 中是 simulated，**没有执行模拟、训练模型或查阅论文**。
4. 展示 synthetic shortlist 及 score；解释真实版本需要提供具体 simulation evidence、model version、uncertainty 和 paper/measurement IDs。
5. 展示 JSON/CSV 下载：强调可追溯的结果记录，对后续 pilot 的数据闭环重要。

**Demo 英文旁白：**

> Imagine a battery manufacturer needs an electrolyte for a high-voltage cathode and a silicon anode. Instead of manually testing hundreds of combinations, they can submit the requirements here. Our pipeline is designed to coordinate scientific tools, screen possible candidates and return a ranked shortlist with supporting evidence. The most promising candidates will then go through experimental validation.

**最佳展示优先级：** 真实 pipeline 运行录屏 > 真实研究数据驱动的可视化 > 明确标注 mock 的交互原型 > 单纯 LLM 聊天机器人。

## 5. 常见技术追问及回答框架

| Question | Key response |
|---|---|
| Are you training a foundation model? | Not necessarily. LLMs may orchestrate tasks; specialized models evaluate properties. |
| How do you ensure scientific accuracy? | Physics-based methods, reproducible evidence and experimental confirmation. |
| How do you handle failed simulations? | Job state, retry policy, timeouts, logs and recovery. |
| What about limited training data? | Use physics-based data where appropriate; incorporate validated feedback; track uncertainty. |
| What is the defensible advantage? | Domain expertise, provenance-rich datasets, scientifically grounded workflows and real validation feedback. |
| What is already implemented? | Separate completed science, current prototype, and roadmap with explicit evidence. |

### Active Learning：如果问“下一批该实验哪些配方？”

> One possible direction is active learning. Instead of testing every candidate, we can use our predictive models to identify promising formulations while also considering prediction uncertainty.
>
> We then prioritize a small number of candidates for expensive simulations or experiments. Once new results come back, they can be used to improve the model.
>
> This creates a feedback loop between AI predictions and real scientific experiments.

**措辞注意：** 没有实现时，只能说 *one possible direction*。

## 6. 路演风险与边界

### PPT 第 4 页：AI Agent pipeline

PPT 描述：literature mining、DFT labelling、ML models training、simulation across thousands of formulations。应准备回答：

- 当前实际上运行了多少 DFT jobs？
- 训练标签有多少、多少来自实测？
- 是否有 baseline、holdout 和误差指标？
- 相比既有科研流程到底快了多少？
- 哪些是已经建成，哪些是计划？

**不能用 mock demo 的 candidate score 冒充真实性能、证据或加速效果。**

### PPT 第 6 页：630 Wh/kg 结果

PPT 将其介绍为内部尚未发表的 lead formulation，并在页面明确提出寻求 independent validation。不要将其表述为已由第三方独立验证的性能，也不要把不同电池体系/测试条件的公司指标看作直接的严格横向对照。

## 7. 十周建议交付计划（建议而非原始 PPT 的既有承诺）

| Milestone | Concrete engineering deliverable | How to demonstrate |
|---|---|---|
| 1. Reproducible workflow | Wrap existing simulations and evaluation into reliable tasks | Logs, task graph, versions, repeatable run |
| 2. Screening dashboard | Customer brief → job status → ranked shortlist | Interactive UI / stored pipeline records |
| 3. Pilot feedback loop | Map predictions and simulations to lab-measured outcomes | Record schema, evidence links and comparison reports |

## 8. 现场最后检查

- 熟悉 30 秒自我介绍和 45 秒技术接棒稿。
- 一句话解释 **LLM orchestration ≠ scientific validation**。
- 明确当前已实现的代码/数据、正在开发的模块和 roadmap。
- 能说清楚 partner-lab validation 的必要性，以及为何需要数据闭环。
- Demo 提前运行和录屏，现场网络不稳定时可用离线演示。
- 准备好被问到：*Why not ChatGPT?*、*Why ML Engineer?*、*What's built today?*、*What will you deliver in ten weeks?*

---

**一句话总结：** 一个可验证的小型科研工作流，比一个虚构多 Agent 的华丽界面更能提高早期 Validation Program 的可信度。
