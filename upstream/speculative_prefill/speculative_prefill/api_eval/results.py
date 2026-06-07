from collections import defaultdict
import re
import string
from typing import Iterable

try:
    from eval.long_bench.metrics import (classification_score, code_sim_score,
                                         count_score, qa_f1_score,
                                         qa_f1_zh_score, retrieval_score,
                                         retrieval_zh_score, rouge_score,
                                         rouge_zh_score)
except ImportError:
    from collections import Counter

    def _normalize_answer(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(text.split())

    def _f1(prediction: list[str], ground_truth: list[str]) -> float:
        common = Counter(prediction) & Counter(ground_truth)
        same = sum(common.values())
        if same == 0:
            return 0.0
        precision = same / len(prediction)
        recall = same / len(ground_truth)
        return 2 * precision * recall / (precision + recall)

    def qa_f1_score(prediction, ground_truth, **kwargs):
        return _f1(
            _normalize_answer(prediction).split(),
            _normalize_answer(ground_truth).split(),
        )

    def retrieval_score(prediction, ground_truth, **kwargs):
        matches = re.findall(r"Paragraph (\d+)", ground_truth)
        if not matches:
            return 0.0
        nums = re.findall(r"\d+", prediction)
        return float(any(num == matches[0] for num in nums))

    def count_score(prediction, ground_truth, **kwargs):
        nums = re.findall(r"\d+", prediction)
        return float(any(num == str(ground_truth) for num in nums))

    def classification_score(prediction, ground_truth, **kwargs):
        return float(str(ground_truth) in prediction)

    def rouge_score(prediction, ground_truth, **kwargs):
        return qa_f1_score(prediction, ground_truth)

    qa_f1_zh_score = qa_f1_score
    rouge_zh_score = rouge_score
    retrieval_zh_score = retrieval_score
    code_sim_score = qa_f1_score


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def score_longbench_predictions(rows: Iterable[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    scores = {}
    for dataset, dataset_rows in grouped.items():
        total = 0.0
        for row in dataset_rows:
            pred = row["pred"]
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip("\n").split("\n")[0]
            best = 0.0
            for answer in row["answers"]:
                best = max(
                    best,
                    dataset2metric[dataset](
                        pred,
                        answer,
                        all_classes=row.get("all_classes", []),
                    ),
                )
            total += best
        scores[dataset] = {
            "score": round(100 * total / len(dataset_rows), 2),
            "n": len(dataset_rows),
        }
    return scores


def group_scores(rows: Iterable[dict], keys: tuple[str, ...]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        label = " | ".join(str(row[key]) for key in keys)
        grouped[label].append(row)
    return {
        label: score_longbench_predictions(group_rows)
        for label, group_rows in sorted(grouped.items())
    }
