#!/usr/bin/env python3
"""Scoring shared by agent.py and compare_pipelines.py: hotpot_v1 normalize /
EM / F1, REASON/ANSWER contract extraction, and retrieval_hit."""
import re
import string
from collections import Counter


def norm(s):
    """Official hotpot_evaluate_v1 normalize_answer: lower, strip every
    string.punctuation char (not a subset), drop articles, collapse space."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def first_line(s):
    """The device config stops on newline (generate_until); our path has no stop
    sequences, so truncate here instead of adding one to core Kotlin."""
    return s.strip().split("\n")[0].strip()


def answer_of(text):
    """The ANSWER: line of the REASON/ANSWER contract; falls back to the first
    line so pre-contract traces score exactly as before."""
    for line in text.splitlines():
        m = re.match(r"(?i)^answer\s*:\s*(.*)$", line.strip())
        if m:
            return m.group(1).strip()
    return first_line(text)


def em(pred, golds):
    p = norm(pred)
    return any(p == norm(g) for g in golds)


def f1(pred, golds):
    """hotpot_evaluate_v1 f1_score: yes/no/noanswer must match exactly (else 0),
    otherwise token-overlap F1; best over reference answers."""
    p = norm(pred)
    best = 0.0
    for g in golds:
        gt = norm(g)
        if p in ("yes", "no", "noanswer") or gt in ("yes", "no", "noanswer"):
            best = max(best, float(p == gt))
            continue
        pt, gt_t = p.split(), gt.split()
        common = Counter(pt) & Counter(gt_t)
        num_same = sum(common.values())
        if not num_same:
            continue
        pr, rc = num_same / len(pt), num_same / len(gt_t)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def retrieval_hit(golds, events):
    """Binary retrieval-quality metric: 1 if any gold answer string appears in
    the retrieved chunks (all Retriever context events, normalized both sides,
    whole words only, yes/no/noanswer golds skipped), else 0. Isolates
    retrieval from generation (EM/F1)."""
    ctx = norm("\n".join(e["output"] for e in events
                        if e["agent"] == "Retriever" and e["step"] == "context"))
    # whole-word match: bare substring lets "no" hit on "not", "1" on "1962";
    # yes/no/noanswer golds would match almost any corpus (same set f1 handles)
    gs = [norm(g) for g in golds]
    gs = [g for g in gs if g and g not in ("yes", "no", "noanswer")]
    return int(any(re.search(r"\b" + re.escape(g) + r"\b", ctx) for g in gs))


if __name__ == "__main__":
    # Self-test: python3 score.py  (moved here from tests/test_score.py)
    import unittest

    class NormalizeTest(unittest.TestCase):
        def test_articles_punct_case(self):
            self.assertEqual(norm("The  Paris, France!"), "paris france")

        def test_em_exact_match(self):
            self.assertTrue(em("The Paris", ["paris"]))
            self.assertFalse(em("London", ["paris"]))

        def test_f1_partial_overlap(self):
            # "paris france" vs "paris": p=0.5, r=1 -> F1 = 2/3
            self.assertAlmostEqual(f1("paris france", ["Paris"]), 2 / 3)

        def test_f1_yes_no_must_match_exactly(self):
            self.assertEqual(f1("yes", ["yes"]), 1.0)
            self.assertEqual(f1("yes", ["no"]), 0.0)

    class ContractTest(unittest.TestCase):
        def test_answer_line_wins(self):
            text = "REASON: passage 2 says so.\nANSWER: Arlo Vance"
            self.assertEqual(answer_of(text), "Arlo Vance")

        def test_falls_back_to_first_line(self):
            self.assertEqual(answer_of("flat answer\nmore"), "flat answer")

    class RetrievalHitTest(unittest.TestCase):
        def _ctx(self, output):
            return {"agent": "Retriever", "step": "context", "output": output}

        def test_gold_in_context_hits(self):
            events = [self._ctx("Blair High School is in Pasadena.")]
            self.assertEqual(retrieval_hit(["Pasadena"], events), 1)

        def test_gold_missing_is_zero(self):
            events = [self._ctx("unrelated passage")]
            self.assertEqual(retrieval_hit(["Pasadena"], events), 0)

        def test_yes_no_gold_ignored(self):
            # yes/no skipped: they'd match (or never match) any corpus
            events = [self._ctx("some passage")]
            self.assertEqual(retrieval_hit(["yes"], events), 0)
            events = [self._ctx("the north pole, I know it, 1962")]
            self.assertEqual(retrieval_hit(["no"], events), 0)

        def test_substring_is_not_a_hit(self):
            events = [self._ctx("I know the north of 1962")]
            self.assertEqual(retrieval_hit(["no", "1"], events), 0)
            self.assertEqual(retrieval_hit(["1962"], events), 1)

    unittest.main()
