# Most of this code is taken from lm-evaluation-harness library:
# https://github.com/EleutherAI/lm-evaluation-harness

import re
import string
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring
import evaluate
import collections


ROUGE_SCORER = None
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def normalize_answer(
    s: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_articles: bool = True,
    fix_whitespace: bool = True,
):
    text = s

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    if fix_whitespace:
        text = " ".join(text.split())

    return text

def exact_match(
    predictions,
    references,
):
    assert len(predictions) == len(references), \
        "predictions and references must have the same length"

    # Convert predictions to numpy array
    predictions = np.asarray(predictions)


    scores = []

    # Loop per example (alias matching happens here)
    for pred, refs in zip(predictions, references):
        # Normalize all aliases for this example
        refs = np.asarray(refs)

        # Exact match against ANY alias
        scores.append(np.any(pred == refs))

    return float(np.mean(scores))

def contains_match(
    predictions,
    references,
):
    """
    predictions: List[str]
    references:  List[List[str]]

    Returns:
        float: mean contains score over dataset
    """

    assert len(predictions) == len(references), \
        "predictions and references must have the same length"

    predictions = np.asarray(predictions)

    scores = []

    for pred, refs in zip(predictions, references):
        refs = np.asarray(refs)

        # Check if prediction contains ANY reference
        scores.append(
            any(ref in pred for ref in refs)
        )

    return float(np.mean(scores))


def corpus_bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge_score(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """

    rouge_types = ["rouge1", "rouge2", "rougeLsum"]

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        # init RougeScorer once (https://github.com/EleutherAI/lm-evaluation-harness/issues/1692)--rouge_types are constant
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

def f1(predictions, references):
    """
    predictions: List[str]
    references:  List[List[str]]
    """
    
    def compute_f1(gold_toks, pred_toks):
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    assert len(predictions) == len(references), \
        "predictions and references must have the same length"

    scores = []

    for pred, refs in zip(predictions, references):
        scores.append(
            max(compute_f1(pred, ref) for ref in refs)
        )

    return float(np.mean(scores))

def get_task_metrics(
    preds,
    refs,
):
    """
    preds: List[str]
    refs:  List[List[str]]
    """

    assert len(preds) == len(refs), \
        "predictions and references must have the same length"
        
    # Normalize metrics
    predictions = [normalize_answer(p) for p in preds]
    references = [[normalize_answer(r) for r in r_list] for r_list in refs]


    bleu_max_list = []
    rouge1_max_list = []
    rouge2_max_list = []
    rougeL_max_list = []

    # ---------- Alias-max metrics (per example) ----------
    for pred, refs in zip(predictions, references):
        # BLEU
        bleu_scores = [
            corpus_bleu([[ref]], [pred]) for ref in refs
        ]
        bleu_max_list.append(np.nanmax(bleu_scores))

        # ROUGE
        rouge_scores = [
            rouge_score([ref], [pred]) for ref in refs
        ]

        rouge1_max_list.append(
            np.nanmax([s["rouge1"] for s in rouge_scores])
        )
        rouge2_max_list.append(
            np.nanmax([s["rouge2"] for s in rouge_scores])
        )
        rougeL_max_list.append(
            np.nanmax([s["rougeLsum"] for s in rouge_scores])
        )
        
    # ---------- Dataset-level BLEU / ROUGE ----------
    bleu_results = bleu.compute(
        predictions=predictions,
        references=references,
    )

    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return {
        # HF corpus metrics
        "bleu": bleu_results["bleu"],
        "rouge1": float(rouge_results["rouge1"]) * 100,
        "rouge2": float(rouge_results["rouge2"]) * 100,
        "rougeL": float(rouge_results["rougeL"]) * 100,

        # Alias-max metrics (mean over examples)
        "bleu_max": float(np.mean(bleu_max_list)),
        "rouge1_max": float(np.mean(rouge1_max_list)),
        "rouge2_max": float(np.mean(rouge2_max_list)),
        "rougeL_max": float(np.mean(rougeL_max_list)),

        # Exact match
        "exact_match": exact_match(predictions, references),
        
        # F1
        "f1": f1(predictions, references),
        
        # Contains
        "contains": contains_match(predictions, references)
    }
    
    
    
# ******* ANN Metrics ********

def get_ann_metrics(results, true_neighbors):
    """
    results: list of dicts with keys:
        - neighbors: List[int]
        - retrievalLatencyNs: float or int

    true_neighbors: List[List[int]]
    """
    
    assert len(results) == len(true_neighbors)
    
    num_queries = len(results)
    K = len(results[0]["neighbors"])

    # Accumulators
    recall_at_k = {k: 0.0 for k in range(1, K + 1)}
    precision_at_k = {k: 0.0 for k in range(1, K + 1)}
    processing_times = []

    for res, gt in zip(results, true_neighbors):
        processing_times.append(res["retrievalLatencyNs"])
        retrieved = res["neighbors"]
        true_nn = gt[0]
        gt_set = set(gt)
        hits = 0
        for k in range(1, K + 1):
            if retrieved[k - 1] in gt_set:
                hits += 1
                
            precision_at_k[k] += hits / k
                
            if true_nn in retrieved[:k]:
                recall_at_k[k] += 1

    # Average over all queries
    for k in range(1, K + 1):
        precision_at_k[k] /= num_queries
        recall_at_k[k] /= num_queries
        
    metrics = {
        "recall@k": recall_at_k,
        "precision@k": precision_at_k,
        "latency": {
            "mean_ns": float(np.mean(processing_times)),
            "median_ns": float(np.median(processing_times)),
            "p95_ns": float(np.percentile(processing_times, 95)),
            "p99_ns": float(np.percentile(processing_times, 99)),
        },
        "qps": 1_000_000_000.0 / float(np.mean(processing_times)) if np.mean(processing_times) > 0 else None,
    }

    return metrics