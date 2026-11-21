"""Character edit distance and CER, shared by stt/cli.py and finetune/."""


def edits(a: str, b: str) -> int:
    """Levenshtein distance between two strings, by character."""
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate, ignoring punctuation and spaces."""
    r = "".join(c for c in ref if c.isalnum())
    h = "".join(c for c in hyp if c.isalnum())
    if not r:
        return 0.0 if not h else 1.0
    return edits(r, h) / len(r)
