"""Reduce full-vocabulary vLLM prompt logprobs before returning teacher results."""

import math


def compact_full_prompt_logprobs(logprobs_dict, scored_id: int, topk: int):
    """Return top-k IDs/logprobs, scored-token logprob, and exact full entropy."""
    if topk < 1 or len(logprobs_dict) < topk:
        raise ValueError("vLLM did not return enough full-vocabulary prompt logprobs.")
    sampled = logprobs_dict.get(scored_id, logprobs_dict.get(str(scored_id)))
    if sampled is None:
        raise ValueError(f"Prompt logprobs missing scored token {scored_id}.")

    ids = [None] * topk
    logprobs = [None] * topk
    entropy = 0.0
    for token_id, entry in logprobs_dict.items():
        lp = float(entry.logprob)
        if math.isfinite(lp):
            entropy -= math.exp(lp) * lp
        rank = entry.rank
        if rank is not None and 1 <= rank <= topk:
            ids[rank - 1] = int(token_id)
            logprobs[rank - 1] = lp
    if any(value is None for value in ids) or any(value is None for value in logprobs):
        raise ValueError("vLLM full-vocabulary prompt logprobs are missing ranked top-k tokens.")
    if not math.isfinite(entropy):
        raise ValueError("vLLM full-vocabulary prompt entropy is not finite.")
    return ids, logprobs, float(sampled.logprob), entropy
