---
name: rag-evaluator
description: 设计、执行和解读 RAG（检索增强生成）系统的定量评估。当用户需要评估、基准测试、审计或衡量 RAG 流水线的质量时使用——包括检索准确性、生成答案忠实度、幻觉检测、端到端答案质量或不同配置间的性能对比。涵盖指标选择、测试数据集构建、评估框架集成（RAGAS、DeepEval、TruLens、LangSmith）、LLM-as-Judge 设置和结果分析。
---

# RAG 评估器

## 评估流水线

按以下流程端到端执行：

1. **定义评估目标** — 你的 RAG 系统哪些方面最重要？（检索精度、幻觉率、端到端答案质量、延迟等）
2. **构建或加载测试数据集** — 包含查询及其真实答案和/或相关文档段落的测试集
3. **选择评估指标** — 根据评估目标选择指标，详见 [references/metrics.md](references/metrics.md)
4. **选择评估框架** — 详见 [references/frameworks.md](references/frameworks.md) 的框架对比
5. **运行评估** — 对每条查询收集检索到的上下文和生成的答案
6. **分析和报告** — 汇总得分，按查询类型/分块策略/模型进行切片分析，发现回归问题

对于典型场景，可使用辅助脚本 `scripts/evaluate_rag.py`——解析测试数据集、运行 RAG 检索和生成、计算指标并输出结构化报告。

## 指标选择速查表

| 评估目标 | 首选指标 | 框架函数 |
|---------|---------|---------|
| 答案不超出检索上下文 | Faithfulness / Factuality | `ragas.faithfulness`, `deepeval.GEval` |
| 答案回答了问题 | Answer Relevance | `ragas.answer_relevancy` |
| 召回的块与问题相关 | Context Precision | `ragas.context_precision` |
| 所有必要信息都召回了 | Context Recall | `ragas.context_recall` |
| 答案编造事实 | Hallucination Rate | `deepeval.HallucinationMetric` |
| 检索排序质量 | MRR / NDCG@k / Recall@k | 自定义（scikit-learn, pytrec_eval） |
| 与参考答案的语义相似度 | BERTScore | `bert_score` |
| 端到端输出质量 | LLM-as-Judge 评分规则 | 通过 `deepeval.LLMTestCase` 自定义 |

详见 [references/metrics.md](references/metrics.md) 了解定义、计算方法和选择依据。

## 测试数据集格式

以 JSONL 或 CSV 格式存储测试数据。每行至少包含：

- `question` — 用户查询
- `ground_truth` — 理想答案或预期答案
- `reference_contexts`（可选）— 理想答案所依据的文档段落列表
- `metadata`（可选）— 用于对结果进行切片分析的标签（如 `domain`、`difficulty`、`query_type`）

## 选择评估框架

详见 [references/frameworks.md](references/frameworks.md) 的详细对比。快速选型规则：

- **RAGAS**：适合轻量级、以指标为中心的评估，样板代码最少
- **DeepEval**：适合 CI/CD 集成、自定义指标和 LLM-as-Judge 场景
- **TruLens**：适合调试单个 trace 和迭代优化
- **LangSmith**：适合已使用 LangChain/LangGraph 且需要内置评估的团队
- **自定义 LLM-as-Judge**：适合需要领域特定评分规则或非标准指标的场景

## 运行评估脚本

```bash
# 安装依赖（一次性）
pip install ragas deepeval datasets pandas openai

# 运行评估
python scripts/evaluate_rag.py \
  --dataset path/to/queries.jsonl \
  --output path/to/report.json \
  --metrics faithfulness,answer_relevancy,context_precision
```

## 结果解读

LLM 评判的指标常用分数范围（0–1 分制）：

- **0.8–1.0**：优秀 — 可投入生产
- **0.6–0.8**：良好 — 检索或生成存在小缺陷
- **0.4–0.6**：一般 — 明显问题，需要调查
- **< 0.4**：差 — 数据或流水线存在根本性问题

务必按元数据字段（查询类型、文档来源、分块策略）对结果进行切片分析，以发现系统性弱点。

## 性能基准测试

除质量指标外，还需衡量：

- **端到端延迟**（p50、p95、p99）
- **检索延迟**
- **生成延迟** / tokens-per-second
- **上下文窗口利用率**（已检索 token 数 vs 可用 token 数）

进行回归测试时，在每次流水线变更后运行相同的评估，并与基线报告进行对比。
