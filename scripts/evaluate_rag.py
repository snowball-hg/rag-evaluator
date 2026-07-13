#!/usr/bin/env python3
"""RAG evaluation runner.

Usage:
  python evaluate_rag.py --dataset queries.jsonl --output report.json
  python evaluate_rag.py --dataset queries.jsonl --metrics faithfulness,answer_relevancy --output report.json

Dataset format (JSONL, each line):
  {"question": "...", "ground_truth": "...", "reference_contexts": ["..."], "metadata": {"domain": "finance"}}

Dependencies: pip install ragas datasets pandas openai tiktoken
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Optional


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset from file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {i}: {e}", file=sys.stderr)
    print(f"Loaded {len(data)} queries from {path}")
    return data


def save_report(results: list[dict], scores: dict, output_path: str):
    """Save evaluation results and scores to a JSON file."""
    report = {
        "scores": scores,
        "num_queries": len(results),
        "config": vars(ARGS) if "ARGS" in dir() else {},
        "results": results,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {output_path}")


def parse_metrics(metric_str: str) -> list[str]:
    """Parse comma-separated metric names."""
    if not metric_str:
        return ["faithfulness", "answer_relevancy", "context_precision"]
    return [m.strip() for m in metric_str.split(",") if m.strip()]


def compute_with_ragas(questions, answers, contexts, ground_truths, metric_names):
    """Compute metrics using the RAGAS framework.

    Args:
        questions: list of query strings
        answers: list of generated answer strings
        contexts: list of list of context strings
        ground_truths: list of ground-truth answer strings
        metric_names: list of metric names

    Returns: dict of {metric_name: score} with average scores.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    METRIC_MAP = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    selected = [METRIC_MAP[m] for m in metric_names if m in METRIC_MAP]
    if not selected:
        print("Warning: no recognisable RAGAS metrics requested. Using defaults.", file=sys.stderr)
        selected = [faithfulness, answer_relevancy]

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = ragas_evaluate(dataset, metrics=selected)
    scores = result.to_pandas().to_dict(orient="list")
    # compute mean for each metric
    avg_scores = {}
    for metric_name in metric_names:
        if metric_name in METRIC_MAP:
            values = [v for v in scores.get(metric_name, []) if v is not None]
            avg_scores[metric_name] = sum(values) / len(values) if values else None
    return avg_scores


def compute_with_deepeval(questions, answers, contexts, ground_truths, metric_names):
    """Compute metrics using the DeepEval framework.

    Returns: dict of {metric_name: score}.
    """
    from deepeval import evaluate as deepeval_evaluate
    from depeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from depeval.test_case import LLMTestCase

    metric_map = {
        "faithfulness": FaithfulnessMetric(threshold=0.0),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.0),
    }

    test_cases = [
        LLMTestCase(
            input=q,
            actual_output=a,
            retrieval_context=c,
            expected_output=g,
        )
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
    ]

    selected = [metric_map[m] for m in metric_names if m in metric_map]
    deepeval_evaluate(test_cases, selected, run_async=False)

    avg_scores = {}
    for metric, cases in zip(selected, test_cases):
        name = metric.__class__.__name__.replace("Metric", "").lower()
        scores = [getattr(c, f"{name}_score", None) for c in test_cases]
        scores = [s for s in scores if s is not None]
        avg_scores[name] = sum(scores) / len(scores) if scores else None
    return avg_scores


def run_custom_judge(questions, answers, contexts, rubric: str, model: str = "gpt-4o") -> list[float]:
    """Score answers using an LLM judge with a custom rubric.

    Args:
        rubric: Scoring rubric description.
        model: Model ID for the judge.

    Returns:
        List of scores (one per query).
    """
    from openai import OpenAI

    client = OpenAI()
    scores = []

    for q, a, ctx in zip(questions, answers, contexts):
        prompt = f"""You are evaluating a RAG answer.
Scoring rubric: {rubric}

Question: {q}

Retrieved context:
{chr(10).join(ctx[:5])}

Generated answer: {a}

Score the answer 0-10 based on the rubric.
Respond with only the numeric score."""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        try:
            score = float(response.choices[0].message.content.strip())
        except (ValueError, AttributeError):
            score = 0.0
        scores.append(score)

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG system")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", "-o", default="rag_eval_report.json", help="Output path")
    parser.add_argument("--metrics", default="", help="Comma-separated metric names")
    parser.add_argument("--framework", choices=["ragas", "deepeval", "custom"], default="ragas",
                        help="Evaluation framework to use")
    parser.add_argument("--rubric", default="",
                        help="Custom rubric for LLM-as-judge (only with --framework custom)")
    parser.add_argument("--judge-model", default="gpt-4o", help="Model ID for LLM judge")
    parser.add_argument("--retrieval", help="Path to a Python file with get_retriever() and get_generator() functions")
    args = parser.parse_args()

    # Store args so save_report can reference them
    global ARGS
    ARGS = args

    # Load dataset
    dataset = load_dataset(args.dataset)
    questions = [d["question"] for d in dataset]
    ground_truths = [d.get("ground_truth", "") for d in dataset]
    reference_contexts = [d.get("reference_contexts", []) for d in dataset]
    metadata_list = [d.get("metadata", {}) for d in dataset]

    # For now, require answers and contexts to be pre-populated in the dataset
    # (or provided from a previous pipeline run)
    has_precomputed = all(("answer" in d and "contexts" in d) for d in dataset)
    if not has_precomputed and not args.retrieval:
        print("Dataset must contain answer and contexts fields, or provide --retrieval script.",
              file=sys.stderr)
        sys.exit(1)

    answers = [d.get("answer", "") for d in dataset]
    contexts = [d.get("contexts", []) for d in dataset]

    metric_names = parse_metrics(args.metrics)

    # Compute scores
    if args.framework == "ragas":
        scores = compute_with_ragas(questions, answers, contexts, ground_truths, metric_names)
    elif args.framework == "deepeval":
        scores = compute_with_deepeval(questions, answers, contexts, ground_truths, metric_names)
    elif args.framework == "custom" and args.rubric:
        judge_scores = run_custom_judge(questions, answers, contexts, args.rubric, args.judge_model)
        scores = {"custom_judge": sum(judge_scores) / len(judge_scores) if judge_scores else 0.0}
    else:
        scores = {"error": "Unknown framework configuration"}

    # Print results
    print("Evaluation results:")
    for metric, score in sorted(scores.items()):
        if score is not None:
            print(f"  {metric}: {score:.4f}")
        else:
            print(f"  {metric}: N/A")

    # Build per-query results
    results = [
        {
            "question": q,
            "ground_truth": g,
            "answer": a,
            "contexts": c,
            "metadata": m,
        }
        for q, g, a, c, m in zip(questions, ground_truths, answers, contexts, metadata_list)
    ]

    save_report(results, scores, args.output)


if __name__ == "__main__":
    main()
