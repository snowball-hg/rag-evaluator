# RAG 评估框架对比

## RAGAS

| 方面 | 详情 |
|------|------|
| **官网** | https://docs.ragas.io |
| **安装** | `pip install ragas` |
| **核心指标** | faithfulness, answer_relevancy, context_precision, context_recall, aspect_critique |
| **优势** | API 简洁；内置指标函数；通过可调用函数适配任意 RAG 管线；开源社区活跃 |
| **劣势** | 自定义评分规则灵活性较低；无参考数据时指标得分不够直观 |
| **适用场景** | 用最少代码快速进行标准 RAG 质量基准测试 |

**使用示例**：
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
results.to_pandas()
```

## DeepEval

| 方面 | 详情 |
|------|------|
| **官网** | https://docs.confident-ai.com |
| **安装** | `pip install deepeval` |
| **核心指标** | FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, HallucinationMetric, GEval, SummarizationMetric |
| **优势** | 丰富的指标目录；通过 GEval 可组合自定义 LLM 评判规则；异步评估；pytest 集成支持 CI/CD；支持对话式 RAG |
| **劣势** | 依赖较重；自定义规则不当时代理解读性下降 |
| **适用场景** | 生产管线、CI/CD 门禁、自定义评分规则的评估、对话式 RAG |

**使用示例**：
```python
from deepeval import evaluate as deep_evaluate
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG stands for Retrieval-Augmented Generation..."
)
metric = FaithfulnessMetric(threshold=0.7)
deep_evaluate([test_case], [metric])
```

## TruLens

| 方面 | 详情 |
|------|------|
| **官网** | https://trulens.org |
| **安装** | `pip install trulens-eval` |
| **核心指标** | Answer Relevance, Context Relevance, Groundedness |
| **优势** | 交互式仪表盘可进行 trace 级别调试；反馈函数可单独调用；适合迭代开发 |
| **劣势** | 对反馈函数的组织方式有一定约束；内置标准指标较少 |
| **适用场景** | 调试单个 RAG trace、迭代优化提示词/分块策略 |

## LangSmith

| 方面 | 详情 |
|------|------|
| **官网** | https://smith.langchain.com |
| **安装** | `pip install langsmith` |
| **核心功能** | 运行追踪、数据集管理、评估运行、对比视图、自动化回归测试 |
| **优势** | 深度集成 LangChain/LangGraph；托管数据集版本管理；支持实验运行并排对比 |
| **劣势** | 需要 LangSmith API 密钥；仅云端（无完全自托管方案）；偏向 LangChain 生态 |
| **适用场景** | 已使用 LangChain/LangGraph 的团队，需要托管评估和实验追踪 |

## 自定义 LLM-as-Judge

**适用场景**：当标准指标无法捕捉特定领域的质量要求时。

**模式示例**：
```python
prompt = f"""You are evaluating a RAG answer.
Rubric: {rubric}
Question: {question}
Context: {context}
Answer: {answer}
Score (0-10) and one-sentence reason:"""
response = OpenAI().chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
)
```
