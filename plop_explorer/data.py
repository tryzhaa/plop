"""Task prompt sets, matching the sources used by the PLoP reference code.

Unlike the reference, sampling is seeded so a (task, n, seed) triple always
yields the same prompts, which is what makes caching by task possible.
"""

import random

from datasets import load_dataset

TASKS = {
    "math": "GSM8K (grade-school math word problems)",
    "code": "HumanEval (Python function stubs)",
    "history": "MMLU high-school European history",
    "logic": "MMLU logical fallacies",
}


def _mmlu(subject):
    rows = load_dataset("cais/mmlu", subject, split="test")
    return [
        r["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(r["choices"]))
        for r in rows
    ]


def load_prompts(task, n=100, seed=0, pool_size=None):
    """`pool_size` overrides the math pool (the reference uses the first 2n questions, so it grows with n)."""
    if task == "math":
        # The reference samples from the first 2n training questions.
        pool = load_dataset("openai/gsm8k", "main", split="train")["question"][: pool_size or 2 * n]
    elif task == "code":
        pool = [f"# Write a Python function\n{p}" for p in load_dataset("openai/openai_humaneval", split="test")["prompt"]]
    elif task == "history":
        pool = _mmlu("high_school_european_history")
    elif task == "logic":
        pool = _mmlu("logical_fallacies")
    else:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    return random.Random(seed).sample(pool, min(n, len(pool)))


CONTROLS = {
    "random_tokens": "Uniformly random vocabulary tokens, same lengths as the math prompts",
    "shuffled_words": "The math prompts' words in random order (same words, no meaning)",
    "prose": "Plain English sentences unrelated to any task (WikiText-2)",
}


def load_control(kind, n=100, seed=0, tokenizer=None):
    """Control prompt sets for X4. `random_tokens` needs the tokenizer (vocabulary and lengths)."""
    rng = random.Random(seed)
    if kind == "shuffled_words":
        out = []
        for prompt in load_prompts("math", n, seed):
            words = prompt.split()
            rng.shuffle(words)
            out.append(" ".join(words))
        return out
    if kind == "random_tokens":
        special = set(tokenizer.all_special_ids)
        ids = [i for i in range(len(tokenizer)) if i not in special]
        # Decoding and re-tokenizing changes some tokens, but the text stays noise.
        return [
            tokenizer.decode([rng.choice(ids) for _ in range(len(tokenizer(p)["input_ids"]))])
            for p in load_prompts("math", n, seed)
        ]
    if kind == "prose":
        lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
        pool = [t.strip() for t in lines if not t.lstrip().startswith("=") and 20 <= len(t.split()) <= 80]
        return rng.sample(pool, min(n, len(pool)))
    raise ValueError(f"Unknown control {kind!r}; expected one of {sorted(CONTROLS)}")


def make_batches(prompts, tokenizer, batch_size=8, seq_len=256, padding_side=None):
    """padding_side="right" with an EOS pad token mimics the reference setup."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return [
        tokenizer(
            prompts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
            **({"padding_side": padding_side} if padding_side else {}),
        )
        for i in range(0, len(prompts), batch_size)
    ]
