"""Where things live. Everything private (data/, models/, secrets/) is gitignored."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
SECRETS = os.path.join(ROOT, "secrets")
LABELS = os.path.join(DATA, "labels")          # labeling page: labels, drafts, 说法表, exports
GLOSSARY = os.path.join(LABELS, "glossary.json")  # 说法表
WU_TOKENS = os.path.join(MODELS, "wu-u2pp-int8", "tokens.txt")  # the Wu models' 5946-char vocabulary
