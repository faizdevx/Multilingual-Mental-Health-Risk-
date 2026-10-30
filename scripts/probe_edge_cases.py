"""Behavioural probe: 50 hand-written edge cases (sarcasm, hidden sadness, contradictions, Hinglish, neutral text, ...)
sent to a running API for every loaded model; writes reports/results/edge_case_probe.json.

The model has 6 emotion labels (anger, disgust, fear, joy, sadness, surprise), so each expected label is mapped to the
nearest one by a fixed rule (RULES below). This is a heuristic check, not a human judgement; 8 cases with no
defensible single answer are recorded but not scored.

  uvicorn src.app.main:app_factory --factory --port 8000      # in another shell
  python scripts/probe_edge_cases.py [--url http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CKPT_DIR  # noqa: E402

CASES = [
    ("sarcasm", "E01", "Wow, another amazing day of being ignored. Living the dream!", "Sadness / Sarcasm"),
    ("sarcasm", "E02", "I'm so happy that nobody remembers I exist.", "Sadness / Sarcasm"),
    ("sarcasm", "E03", "Kya baat hai, ek aur din barbaad. Life toh set hai meri.", "Frustration"),
    ("sarcasm", "E04", "Best feeling ever, watching everyone move ahead while I remain stuck.", "Envy / Sadness"),
    ("sarcasm", "E05", "Haha, it's funny how nobody gives a damn.", "Sadness / Sarcasm"),
    ("hidden", "E06", "I stopped checking my phone for messages. It doesn't make a difference anymore.", "Loneliness"),
    ("hidden", "E07", "Ab kisi ke aane ya jaane se farak nahi padta.", "Emotional numbness"),
    ("hidden", "E08", "I used to tell everyone everything. Now I just say nothing happened.", "Suppressed sadness"),
    ("hidden", "E09", "My birthday was yesterday. It was just another Tuesday.", "Loneliness / Disappointment"),
    ("hidden", "E10", "Room mein sab kuch same hai, bas ab pehle jaisi feeling nahi aati.", "Nostalgia / Sadness"),
    ("contradiction", "E11", "I am happy for her, but I wish it was me.", "Happiness + Envy"),
    ("contradiction", "E12", "Finally got what I wanted, but somehow it feels empty.", "Happiness + Emptiness"),
    ("contradiction", "E13", "Usse milkar achha laga, but ghar aake phir wahi loneliness.", "Happiness + Sadness"),
    ("contradiction", "E14", "I miss those days, but I never want to go back.", "Nostalgia + Regret"),
    ("contradiction", "E15", "I hate him, but I still wait for his messages.", "Anger + Attachment"),
    ("hinglish", "E16", "Scene toh sorted hai, bas andar se thoda messed up hoon.", "Hidden sadness"),
    ("hinglish", "E17", "Bhai, sab changa si... bas main hi theek nahi hoon.", "Sadness"),
    ("hinglish", "E18", "Mujhe farak nahi padta, seriously, bilkul bhi nahi.", "Possible denial / Emotional numbness"),
    ("hinglish", "E19", "Usne seen karke chhod diya. Koi na, aadat hai.", "Disappointment"),
    ("hinglish", "E20", "Life mast chal rahi hai, bas mann nahi lagta kisi cheez mein.", "Emotional numbness"),
    ("negwords", "E21", "I hate waking up early, but I love my job.", "Mixed / Happiness"),
    ("negwords", "E22", "I'm crying because I finally achieved my dream.", "Happiness"),
    ("negwords", "E23", "This is painfully beautiful.", "Awe / Appreciation"),
    ("negwords", "E24", "I am terrified of how much I love this place.", "Love + Fear"),
    ("negwords", "E25", "Worst day ever, but my friends made it unforgettable.", "Mixed / Gratitude"),
    ("neutral", "E26", "I went to college, attended lectures, and returned home.", "Neutral"),
    ("neutral", "E27", "He didn't reply to my message for three hours.", "Neutral / Ambiguous"),
    ("neutral", "E28", "I am sitting alone in my room.", "Neutral"),
    ("neutral", "E29", "She left the classroom before I arrived.", "Neutral"),
    ("neutral", "E30", "Nobody called me yesterday.", "Ambiguous"),
    ("ambiguity", "E31", "It's okay.", "Context-dependent"),
    ("ambiguity", "E32", "Whatever happens, happens.", "Acceptance / Resignation"),
    ("ambiguity", "E33", "I guess I'm used to it now.", "Resignation / Sadness"),
    ("ambiguity", "E34", "Koi baat nahi, hota hai.", "Acceptance / Hidden disappointment"),
    ("ambiguity", "E35", "I'm fine. Don't worry about me.", "Ambiguous / Possible suppressed sadness"),
    ("slang", "E36", "lol I'm cooked", "Frustration / Humour"),
    ("slang", "E37", "kuch ni yrr", "Ambiguous"),
    ("slang", "E38", "bruh, seriously?", "Surprise / Frustration"),
    ("slang", "E39", "mann nhi krra ab", "Low motivation"),
    ("slang", "E40", "sab temporary hai :) ", "Context-dependent"),
    ("slang", "E41", "Lmao, I'm literally falling apart.", "Distress / Humour"),
    ("slang", "E42", "thik hu bhai, chill", "Ambiguous"),
    ("adversarial", "E43", "I am not sad. I am just tired of pretending to be happy.", "Sadness"),
    ("adversarial", "E44", "I don't hate my life; I just don't enjoy it.", "Emotional numbness"),
    ("adversarial", "E45", "She didn't reject me. She simply never chose me.", "Disappointment"),
    ("adversarial", "E46", "Everyone says I'm lucky. I wish I could feel that way.", "Sadness"),
    ("adversarial", "E47", "I'm smiling because explaining my sadness is exhausting.", "Suppressed sadness"),
    ("adversarial", "E48", "Khush rehna seekh liya hai, khush hoon ya nahi pata nahi.", "Emotional numbness"),
    ("adversarial", "E49", "Nothing bad happened today, and somehow that makes me feel worse.", "Ambiguous sadness"),
    ("adversarial", "E50", "I have everything I once prayed for, yet I feel disconnected from everything.", "Emptiness"),
]
CATEGORIES = {"sarcasm": "Sarcasm and fake positivity", "hidden": "Hidden sadness, no emotion words",
              "contradiction": "Contradictory emotions", "hinglish": "Hinglish code-switching",
              "negwords": "Negative words, positive emotion", "neutral": "Neutral statements",
              "ambiguity": "Ambiguity and context", "slang": "Short text and slang", "adversarial": "Adversarial phrasing"}

# (rule, emotion set or groups, allowed sentiments or None); None = recorded but not scored
#   any: one of the emotions detected | sad_or_none: sadness or no emotion | none: no emotion | groups: one from each group
NEG, NEG_NEU = {"negative"}, {"negative", "neutral"}
SAD, SAD_ANGER = {"sadness"}, {"sadness", "anger"}
RULES = {
    "E01": ("any", SAD_ANGER, NEG), "E02": ("any", SAD_ANGER, NEG), "E03": ("any", {"anger"}, NEG),
    "E04": ("any", SAD_ANGER, NEG), "E05": ("any", SAD_ANGER, NEG),
    "E06": ("any", SAD, NEG), "E07": ("sad_or_none", SAD, NEG_NEU), "E08": ("any", SAD, NEG),
    "E09": ("any", SAD, NEG_NEU), "E10": ("any", SAD, NEG_NEU),
    "E11": ("groups", [{"joy"}, SAD_ANGER], None), "E12": ("groups", [{"joy"}, SAD], None),
    "E13": ("groups", [{"joy"}, SAD], None), "E14": ("any", SAD, None),
    "E15": ("groups", [{"anger"}, {"sadness", "joy", "fear"}], None),
    "E16": ("any", SAD, None), "E17": ("any", SAD, None), "E18": ("sad_or_none", SAD, None),
    "E19": ("any", SAD, None), "E20": ("sad_or_none", SAD, None),
    "E21": ("any", {"joy"}, None), "E22": ("any", {"joy"}, None), "E23": ("any", {"joy", "surprise"}, None),
    "E24": ("groups", [{"joy"}, {"fear"}], None), "E25": ("any", {"joy"}, None),
    "E26": ("none", None, {"neutral"}), "E27": None, "E28": ("none", None, {"neutral"}),
    "E29": ("none", None, {"neutral"}), "E30": None,
    "E31": None, "E32": ("sad_or_none", SAD, None), "E33": ("any", SAD, None), "E34": None, "E35": None,
    "E36": ("any", {"anger", "joy", "surprise"}, None), "E37": None, "E38": ("any", {"surprise", "anger"}, None),
    "E39": ("sad_or_none", SAD, None), "E40": None, "E41": ("any", {"sadness", "fear", "joy"}, None), "E42": None,
    "E43": ("any", SAD, NEG), "E44": ("sad_or_none", SAD, None), "E45": ("any", SAD, NEG),
    "E46": ("any", SAD, NEG), "E47": ("any", SAD, NEG), "E48": ("sad_or_none", SAD, None),
    "E49": ("any", SAD, None), "E50": ("sad_or_none", SAD, None),
}
NEUTRAL_SCORED = ("E26", "E28", "E29")


def check(cid: str, sentiment: str, emotions: set) -> bool | None:
    rule = RULES[cid]
    if rule is None:
        return None
    kind, emo, sent = rule
    ok = {"any": lambda: bool(emotions & emo), "sad_or_none": lambda: bool(emotions & emo) or not emotions,
          "none": lambda: not emotions, "groups": lambda: all(emotions & g for g in emo)}[kind]()
    return ok and (sent is None or sentiment in sent)


def predict(url: str, text: str, model: str) -> dict:
    body = json.dumps({"text": text, "model": model, "language": "auto", "explain": False}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(ROOT / "reports/results/edge_case_probe.json"))
    a = ap.parse_args()
    health = json.load(urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=30))
    out = {"n_cases": len(CASES), "n_scored": sum(r is not None for r in RULES.values()),
           "categories": CATEGORIES, "models": {}}
    for model in health["models_loaded"]:
        rows, cats = [], {k: {"scored": 0, "passed": 0, "n": 0} for k in CATEGORIES}
        for cat, cid, text, expected in CASES:
            r = predict(a.url, text, model)
            emotions = {e["label"] if isinstance(e, dict) else e for e in r.get("emotions_detected") or []}
            ok = check(cid, r["sentiment"], emotions)
            c = cats[cat]
            c["n"] += 1
            if ok is not None:
                c["scored"] += 1
                c["passed"] += int(ok)
            rows.append({"id": cid, "category": cat, "expected": expected, "language_tag": r.get("language_tag"),
                         "sentiment": r["sentiment"], "emotions": sorted(emotions), "risk": r.get("risk"),
                         "risk_probability": r.get("risk_probability"), "check": ok})
        meta_p = CKPT_DIR / model / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        emo_counts = Counter(e for r in rows for e in r["emotions"])
        out["models"][model] = {
            "checkpoint": {k: meta.get(k) for k in ("best_epoch", "best_val_mean_macro_f1")} | {
                "epochs_planned": (meta.get("train_config") or {}).get("epochs"),
                "device": (meta.get("train_config") or {}).get("device")},
            "by_category": cats,
            "scored": sum(c["scored"] for c in cats.values()), "passed": sum(c["passed"] for c in cats.values()),
            "n_stress": sum(r["risk"] == "stress" for r in rows),
            "n_two_plus_emotions": sum(len(r["emotions"]) >= 2 for r in rows),
            "n_no_emotion": sum(not r["emotions"] for r in rows),
            "neutral_flagged_stress": sum(r["risk"] == "stress" for r in rows if r["id"] in NEUTRAL_SCORED),
            "emotion_counts": dict(emo_counts), "sentiment_counts": dict(Counter(r["sentiment"] for r in rows)),
            "cases": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for m, v in out["models"].items():
        print(f"{m}: {v['passed']}/{v['scored']} passed, stress {v['n_stress']}/50")


if __name__ == "__main__":
    main()
