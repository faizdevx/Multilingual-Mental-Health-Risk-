"""Build the IEEE-format research paper (HTML -> PDF) from paper/paper.html.j2.

All numbers in tables and most numbers in prose come from the generated result files:
  reports/results/summary.csv, reports/results/comparisons.json, reports/metrics/comparison.json,
  reports/results/selective_prediction.json, reports/results/faithfulness.json, reports/results/error_breakdown.json,
  reports/results/tweets_hindi_langid_stratified.json, reports/results/hingcorpus_overlap.json
References come from paper/refs.json (author lists fetched from the ACL Anthology, arXiv and Crossref) and the venue
strings below.

  python paper/build_paper.py            # writes paper/research_paper.html and paper/research_paper.pdf
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "reports"

summary = pd.read_csv(R / "results/summary.csv")
comps = json.loads((R / "results/comparisons.json").read_text())
orig = {m["name"]: m for m in json.loads((R / "metrics/comparison.json").read_text())["models"]}
boot = json.loads((R / "metrics/bootstrap_ci.json").read_text())
sel = json.loads((R / "results/selective_prediction.json").read_text())
faith = json.loads((R / "results/faithfulness.json").read_text())["groups"]
errb = json.loads((R / "results/error_breakdown.json").read_text())
lid = json.loads((R / "results/tweets_hindi_langid_stratified.json").read_text())
overlap = json.loads((R / "results/hingcorpus_overlap.json").read_text())
probe = json.loads((R / "results/edge_case_probe.json").read_text(encoding="utf-8"))
refs = json.loads((HERE / "refs.json").read_text())
refs["chancellor"]["authors"] = "S. Chancellor and M. De Choudhury"

VENUE = {
    "turcan": "in Proc. 10th Int. Workshop Health Text Mining Inf. Anal. (LOUHI), 2019, pp. 97-107.",
    "barbieri": "in Proc. 13th Lang. Resour. Eval. Conf. (LREC), 2022, pp. 258-266.",
    "conneau": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 8440-8451.",
    "khanuja_cs": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 3575-3585.",
    "liu": "in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2019, pp. 4487-4496.",
    "pruk": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 5231-5247.",
    "deyoung": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4443-4458.",
    "jacovi": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4198-4205.",
    "desai": "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2020, pp. 295-302.",
    "bergk": "in Proc. Joint Conf. Empirical Methods Natural Lang. Process. Comput. Natural Lang. Learn. (EMNLP-CoNLL), 2012, pp. 995-1005.",
    "dror": "in Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2018, pp. 1383-1392.",
    "harrigian": "in Findings Assoc. Comput. Linguistics: EMNLP 2020, pp. 3774-3788.",
    "benton": "in Proc. 1st ACL Workshop Ethics Natural Lang. Process., 2017, pp. 94-102.",
    "patwa": "in Proc. 14th Workshop Semantic Eval. (SemEval), 2020, pp. 774-790.",
    "muril": "arXiv:2103.10730, 2021.",
    "hingbert": "arXiv:2204.08398, 2022.",
    "brighter": "in Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL), 2025, arXiv:2502.11926.",
    "semeval25": "arXiv:2503.07269, 2025.",
    "yu": "in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, arXiv:2001.06782.",
    "dodge": "arXiv:2002.06305, 2020.",
    "mosbach": "in Int. Conf. Learn. Represent. (ICLR), 2021, arXiv:2006.04884.",
    "sun": "arXiv:1905.05583, 2019.",
    "guo": "in Proc. 34th Int. Conf. Mach. Learn. (ICML), 2017, arXiv:1706.04599.",
    "geifman": "in Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, arXiv:1705.08500.",
    "hendrycks": "in Int. Conf. Learn. Represent. (ICLR), 2017, arXiv:1610.02136.",
    "sundararajan": "in Proc. 34th Int. Conf. Mach. Learn. (ICML), 2017, arXiv:1703.01365.",
    "loshchilov": "in Int. Conf. Learn. Represent. (ICLR), 2019, arXiv:1711.05101.",
    "wolf": "in Proc. EMNLP: Syst. Demonstrations, 2020, arXiv:1910.03771.",
    "paszke": "in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019, arXiv:1912.01703.",
    "roark": "in Proc. 12th Lang. Resour. Eval. Conf. (LREC), 2020, arXiv:2007.01176.",
    "madhani": "arXiv:2205.03018, 2022.",
}
ACL_TITLES = {
    "turcan": "Dreaddit: A Reddit dataset for stress analysis in social media",
    "barbieri": "XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond",
    "conneau": "Unsupervised cross-lingual representation learning at scale",
    "khanuja_cs": "GLUECoS: An evaluation benchmark for code-switched NLP",
    "liu": "Multi-task deep neural networks for natural language understanding",
    "pruk": "Intermediate-task transfer learning with pretrained language models: When and why does it work?",
    "deyoung": "ERASER: A benchmark to evaluate rationalized NLP models",
    "jacovi": "Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?",
    "desai": "Calibration of pre-trained transformers",
    "bergk": "An empirical investigation of statistical significance in NLP",
    "dror": "The Hitchhiker's guide to testing statistical significance in natural language processing",
    "harrigian": "Do models of mental health based on social media data generalize?",
    "benton": "Ethical research protocols for social media health research",
    "patwa": "SemEval-2020 Task 9: Overview of sentiment analysis of code-mixed tweets",
}
EXTRA = {  # references that are not in refs.json
    "pedregosa": 'F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011.',
    "ytdata": 'shae2977, "Hinglish YouTube comments sentiment dataset," Hugging Face Datasets, revision 84abda5, accessed Oct. 7, 2026. [Online]. Available: https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset',
    "umsab": 'cardiffnlp, "Tweet sentiment multilingual (UMSAB)," Hugging Face Datasets, revision 14b13ed, accessed Oct. 7, 2026. [Online]. Available: https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual',
}


def title_case_ieee(t: str) -> str:
    return t


def fmt_ref(k: str) -> str:
    if k in EXTRA:
        return EXTRA[k]
    r = refs[k]
    if k in ("chancellor", "schuster", "hochreiter"):
        j = r["journal"]
        jn = {"npj Digital Medicine": "npj Digit. Med.", "IEEE Transactions on Signal Processing": "IEEE Trans. Signal Process.", "Neural Computation": "Neural Comput."}.get(j, j)
        extra = f", vol. {r['vol']}" if r.get("vol") else ""
        extra += f", no. {r['issue']}" if r.get("issue") else ""
        pg = r["page"]
        extra += (f", Art. no. {pg}" if k == "chancellor" else f", pp. {pg}")
        return f'{r["authors"]}, "{r["title"][0].upper() + r["title"][1:]}," <i>{jn}</i>{extra}, {r["year"]}.'
    title = ACL_TITLES.get(k) or r["title"]
    return f'{r["authors"]}, "{title}," {VENUE[k]}'


# --------------------------------------------------------------------------- data helpers
def m(exp, task, metric, scope="all", split="test", d=3):
    r = summary[(summary.experiment == exp) & (summary.split == split) & (summary.task == task) & (summary.scope == scope) & (summary.metric == metric)]
    if r.empty:
        raise KeyError((exp, task, metric, scope, split))
    return f"{float(r['mean'].iloc[0]):.{d}f}"


def ms(exp, task, metric, scope="all", split="test", d=3):
    r = summary[(summary.experiment == exp) & (summary.split == split) & (summary.task == task) & (summary.scope == scope) & (summary.metric == metric)].iloc[0]
    sd = 0.0 if pd.isna(r["sd"]) else float(r["sd"])
    return f"{float(r['mean']):.{d}f} &plusmn; {sd:.{d}f}"


def _c(a, b, split, task, metric, scope="all"):
    for c in comps:
        if (c["a"], c["b"], c["split"], c["task"], c["metric"], c["scope"]) == (a, b, split, task, metric, scope) and c["status"] == "ok":
            return c
    raise KeyError((a, b, split, task, metric, scope))


def dlt(a, b, split, task, metric, scope="all"):
    c = _c(a, b, split, task, metric, scope)
    return f"{c['diff_b_minus_a']:+.3f}"


def ci(a, b, split, task, metric, scope="all"):
    c = _c(a, b, split, task, metric, scope)
    return f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"


def dci(*args, **kw):
    return f"{dlt(*args, **kw)} {ci(*args, **kw)}"


def dnum(a, b, split, task, metric, scope="all"):
    return _c(a, b, split, task, metric, scope)["diff_b_minus_a"]


def sig(a, b, split, task, metric, scope="all"):
    return _c(a, b, split, task, metric, scope)["ci_excludes_zero"]


def n_of(c):
    return _c(*c)["n_examples"]


def covf(exp, task, cov):
    return f"{sel[exp][task]['curve_mean'][cov]['macro_f1']:.3f}"


def fth(group, key):
    return f"{faith[group][key]:.3f}"


def f2(x, d=3):
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------- numbering (two passes)
class Reg:
    def __init__(self, prev=None):
        self.cites, self.tabs, self.figs = [], [], []
        self.prev = prev

    def cite(self, *keys):
        out = []
        for k in keys:
            if k not in self.cites:
                self.cites.append(k)
            out.append(f"[{self.cites.index(k) + 1}]")
        return ", ".join(out)

    def tab_def(self, key):
        if key not in self.tabs:
            self.tabs.append(key)
        return roman(self.tabs.index(key) + 1)

    def fig_def(self, key):
        if key not in self.figs:
            self.figs.append(key)
        return str(self.figs.index(key) + 1)

    def tref(self, key):
        order = (self.prev or self).tabs
        return "Table " + (roman(order.index(key) + 1) if key in order else "?")

    def fref(self, key):
        order = (self.prev or self).figs
        return "Fig. " + (str(order.index(key) + 1) if key in order else "?")


def roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in vals:
        while n >= v:
            s += r; n -= v
    return s


def render(prev):
    reg = Reg(prev)
    env = Environment(loader=FileSystemLoader(str(HERE)), autoescape=False)
    env.globals.update(dnum=dnum, m=m, ms=ms, dlt=dlt, ci=ci, dci=dci, sig=sig, covf=covf, fth=fth, f2=f2, orig=orig, boot=boot, errb=errb, lid=lid,
                       overlap=overlap, probe=probe, cite=reg.cite, tab_def=reg.tab_def, fig_def=reg.fig_def, tref=reg.tref, fref=reg.fref, sel=sel, faith=faith, summary=summary)
    html = env.get_template("paper.html.j2").render(refs_html=lambda: "".join(f'<li>{fmt_ref(k)}</li>' for k in reg.cites))
    return html, reg


if __name__ == "__main__":
    _, r1 = render(None)
    html, r2 = render(r1)
    # citations must have the same order in both passes
    assert r1.cites == r2.cites and r1.tabs == r2.tabs and r1.figs == r2.figs
    # humanizer check: no em/en dashes in the paper text
    body = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    bad = [c for c in ("—", "–") if c in text]
    assert not bad, f"dash characters found: {bad}"
    (HERE / "research_paper.html").write_text(html, encoding="utf-8")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        linux_chrome = Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        if linux_chrome.exists():
            b = p.chromium.launch(executable_path=str(linux_chrome), args=["--no-sandbox"])
        else:  # e.g. Windows: use the installed Edge, no browser download needed
            b = p.chromium.launch(channel="msedge")
        pg = b.new_page()
        pg.goto((HERE / "research_paper.html").as_uri())
        pg.wait_for_timeout(500)
        pg.pdf(path=str(HERE / "research_paper.pdf"), format="Letter", print_background=True, prefer_css_page_size=True)
        b.close()
    print("cited", len(r2.cites), "references; tables", len(r2.tabs), "; figures", len(r2.figs))
