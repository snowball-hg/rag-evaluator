# RAG Evaluator — Codex 技能

[![Codex Skill](https://img.shields.io/badge/Codex-Skill-4F46E5)](https://codex.ai)

**RAG Evaluator** 是一个 Codex 技能（Skill），用于对检索增强生成（RAG）系统进行系统化的定量评估。它提供了完整的评估流水线指南、指标参考、框架对比和可复用的 Python 评估脚本。

简体中文 | [English](SKILL.md)

---

## 功能特性

- **完整的评估流水线** — 从定义评估目标到生成分析报告，覆盖全流程
- **8 类核心指标** — Faithfulness、Answer Relevance、Context Precision/Recall、Hallucination Rate、MRR/NDCG/Recall@k、BERTScore、LLM-as-Judge
- **5 种评估框架对比** — RAGAS、DeepEval、TruLens、LangSmith、自定义 LLM-as-Judge
- **可复用的 Python 脚本** — 加载测试数据集、调用评估框架计算指标、输出结构化 JSON 报告
- **结果解读指南** — 分数分档策略和按元数据切片分析的方法

## 快速开始

### 安装为 Codex 技能

```bash
# 将技能目录复制到 Codex 的技能目录
cp -r rag-evaluator ~/.codex/skills/

# Codex 将在下次对话时自动发现该技能
```

### 准备测试数据集

以 JSONL 格式准备测试数据，每行格式如下：

```json
{"question": "风机发电机绝缘等级是多少？", "ground_truth": "H级，防护等级IP54", "reference_contexts": ["..."], "metadata": {"domain": "spec", "difficulty": "easy"}}
```

### 运行评估

```bash
# 安装依赖
pip install ragas deepeval datasets pandas openai

# 执行评估
python scripts/evaluate_rag.py \
  --dataset queries.jsonl \
  --output report.json \
  --metrics faithfulness,answer_relevancy,context_precision
```

## 评估流水线

```
定义评估目标 → 构建测试集 → 选择指标 → 选择框架 → 运行评估 → 分析报告
```

每个步骤的详细说明见 [SKILL.md](SKILL.md)。

## 指标速查

| 评估目标 | 指标 | 说明 |
|---------|------|------|
| 答案是否忠于上下文 | Faithfulness | 将答案拆分为陈述，逐一核对上下文 |
| 答案是否回答了问题 | Answer Relevance | 从答案反向生成问题，计算相似度 |
| 检索块是否相关 | Context Precision | 评估检索结果中相关块的比例和排序 |
| 所有必要信息是否召回 | Context Recall | 检查真实答案所需信息是否在上下文中 |
| 答案是否编造事实 | Hallucination Rate | 统计无法归因于上下文的陈述比例 |
| 检索排序质量 | MRR / NDCG@k | 需要人工标注的相关性判断 |

详细定义和公式见 [references/metrics.md](references/metrics.md)。

## 评估框架对比

| 框架 | 适用场景 |
|------|---------|
| **RAGAS** | 快速标准评估，最少代码 |
| **DeepEval** | CI/CD 集成，自定义评分规则，对话式 RAG |
| **TruLens** | 调试 trace，迭代优化 |
| **LangSmith** | LangChain/LangGraph 生态，托管实验追踪 |
| **自定义 LLM-as-Judge** | 领域特定评分规则 |

详细对比见 [references/frameworks.md](references/frameworks.md)。

## 目录结构

```
rag-evaluator/
├── README.md                        # 项目说明
├── SKILL.md                         # 技能主文件（评估指南）
├── .gitignore
├── agents/
│   └── openai.yaml                  # Codex UI 元数据
├── references/
│   ├── metrics.md                   # 指标参考（定义、公式、适用场景）
│   └── frameworks.md                # 框架对比（特点、代码示例、选型建议）
└── scripts/
    └── evaluate_rag.py              # Python 评估运行器（加载数据 → 计算指标 → 输出报告）
```

## 许可

MIT
