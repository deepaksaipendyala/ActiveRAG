import re, json, string
from tqdm import tqdm
import numpy as np
import collections
from sklearn.metrics.pairwise import cosine_similarity


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_recall(a_gold, a_pred):
    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold[0])
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(gold_toks)

    return recall

def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False

def get_embedding(texts, embed_model):
    if isinstance(texts, str):
        texts = [texts]
    return np.array([embed_model.get_text_embedding(t) for t in texts])

def ras_score(x, y, embed_model):
    x_q, x_r = x['query'], x['references']
    y_q, y_r = y['query'], y['references']

    x_p = x_q + ' ' + x_r
    y_p = y_q + ' ' + y_r

    emb_q = get_embedding([x_q, y_q], embed_model)
    emb_r = get_embedding([x_r, y_r], embed_model)
    emb_p = get_embedding([x_p, y_p], embed_model)

    sim_q = cosine_similarity([emb_q[0]], [emb_q[1]])[0][0]
    sim_r = cosine_similarity([emb_r[0]], [emb_r[1]])[0][0]
    sim_p = cosine_similarity([emb_p[0]], [emb_p[1]])[0][0]

    return min(sim_p, 0.5 * (sim_q + sim_r))
