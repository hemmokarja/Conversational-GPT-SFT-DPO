from torch.nn import functional as F


def _compute_logprobs(logits, y):
    """
    Compute log probabilities for all tokens in batched sequences.

    Args:
        logits: [B, S, vocab_size]
        y: [B, S] - target token ids

    Returns:
        logprobs: [B, S] - log probabilities for each position
    """
    logprobs = F.log_softmax(logits, dim=-1)  # [B, S, vocab_size]
    token_logprobs = logprobs.gather(dim=-1, index=y.unsqueeze(-1))
    token_logprobs = token_logprobs.squeeze(-1)  # [B, S]
    return token_logprobs


def _compute_completion_logprobs(logits, y, completion_mask):
    logprobs = _compute_logprobs(logits, y)  # [B, S]
    
    # sum logprobs only for completion tokens
    completion_logprobs = (logprobs * completion_mask).sum(dim=-1)  # [B]
    return completion_logprobs


def dpo_loss(
    logits_accepted,
    logits_rejected,
    logits_accepted_reference,
    logits_rejected_reference,
    y_accepted,
    y_rejected,
    cmask_accepted,
    cmask_rejected,
    beta=0.1,
):
    logprobs_accepted = _compute_completion_logprobs(
        logits_accepted, y_accepted, cmask_accepted
    )
    logprobs_rejected = _compute_completion_logprobs(
        logits_rejected, y_rejected, cmask_rejected
    )
    logprobs_accepted_ref = _compute_completion_logprobs(
        logits_accepted_reference, y_accepted, cmask_accepted
    )
    logprobs_rejected_ref = _compute_completion_logprobs(
        logits_rejected_reference, y_rejected, cmask_rejected
    )
    policy_logratios = logprobs_accepted - logprobs_rejected
    reference_logratios = logprobs_accepted_ref - logprobs_rejected_ref
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()
    return loss
