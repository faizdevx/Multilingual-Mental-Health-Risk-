/* All dynamic content is built with textContent (never innerHTML) so user text can't inject markup. */
(function () {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const cap = (x) => (x ? x.charAt(0).toUpperCase() + x.slice(1) : x);
  const fmt = (x) => (x === null || x === undefined ? "Not available" : typeof x === "number" ? x.toFixed(3) : String(x));
  const showErr = (msg) => { const b = $("error"); if (!b) return; b.textContent = msg; b.hidden = !msg; };
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Request failed (" + r.status + ")");
    return j;
  }
  function card(label, value, sub, probs) {
    const c = el("div", "card res"); c.append(el("div", "k", label), el("div", "v", value));
    if (sub) c.append(el("div", "p", sub));
    if (probs) { const m = el("div", "mini-bars"); Object.entries(probs).forEach(([k, v]) => { const row = el("div", "mini"); const t = el("span", "t"); const f = el("span", "f"); f.style.width = Math.round(v * 100) + "%"; t.append(f); row.append(el("span", "l", k), t, el("span", "", v.toFixed(2))); m.append(row); }); c.append(m); }
    return c;
  }
  function renderCards(r, container) {
    container.replaceChildren(
      card("Language", r.language_name + " (" + r.language + ")", "confidence " + fmt(r.language_confidence) + " · " + r.language_method),
      card("Script", r.script, "tag " + r.language_tag),
      card("Sentiment", cap(r.sentiment), null, r.sentiment_probabilities),
      card("Emotion", r.emotion === "none" ? "None above threshold" : cap(r.emotion), r.emotions_detected.length > 1 ? "also: " + r.emotions_detected.slice(1).map((e) => e.label).join(", ") : null, r.emotion_probabilities),
      card("Risk signal", r.risk_signal, "Probability (stress class): " + r.risk_probability.toFixed(2) + " · model label: " + r.risk),
      card("Inference latency", r.inference_latency_ms + " ms", r.model_display_name + " · " + r.device));
  }
  function renderNotes(r, safety, warnings) {
    const s = r.safety; safety.replaceChildren(); safety.hidden = !s.notice;
    if (s.notice) {
      safety.append(el("b", "", s.notice));
      if (s.show_resources) { const ul = el("ul"); s.resources.forEach((x) => { const li = el("li"); const a = el("a", "", x.name); a.href = x.url; a.rel = "noopener"; li.append(a, document.createTextNode(" - " + x.description)); ul.append(li); }); safety.append(el("div", "", "If this reflects an immediate safety concern, please seek help from a qualified professional or local emergency/crisis service. Official directories (verified " + s.resources_verified_on + "):"), ul); }
    }
    if (warnings) {
      warnings.replaceChildren();
      const v = r.validated_for_language, notes = [];
      if (!r.validated_for_language.risk) notes.push(s.validation_note);
      if (!v.sentiment) notes.push("Sentiment was not evaluated for " + r.language_tag + "; treat as unvalidated.");
      if (!v.emotion) notes.push("Emotion was not evaluated for " + r.language_tag + "; treat as unvalidated.");
      if (!r.probabilities_calibrated) notes.push("No calibration file found for this model: probabilities are raw model outputs.");
      notes.push(s.probability_note);
      const box = el("div", "alert warn"); notes.forEach((n) => box.append(el("div", "", "• " + n))); warnings.append(box);
    }
  }
  function renderExplain(box, ex) {
    box.replaceChildren(); if (!ex) { box.hidden = true; return; }
    box.hidden = false; box.append(el("h3", "", "Model attribution"), el("p", "muted", ex.risk.caveat));
    [["risk", "Stress-class score"], ["sentiment", "Predicted sentiment score"]].forEach(([k, title]) => {
      const e = ex[k]; box.append(el("h4", "", title + " (target: " + e.target_label + ")")); const w = el("div");
      e.tokens.forEach((t) => { const s = el("span", "tok", t.token); const a = t.attribution; s.style.background = a >= 0 ? "rgba(59,91,219," + Math.min(0.85, Math.abs(a)) + ")" : "rgba(214,90,49," + Math.min(0.85, Math.abs(a)) + ")"; s.style.color = Math.abs(a) > 0.5 ? "#fff" : "inherit"; s.title = a.toFixed(3); w.append(s); });
      box.append(w);
    });
    box.append(el("p", "muted", "Blue pushes toward the target label, orange away. Not clinical reasoning."));
  }

  const af = $("analyze-form");
  if (af) af.addEventListener("submit", async (ev) => {
    ev.preventDefault(); showErr(""); const btn = af.querySelector("button"); btn.disabled = true;
    try {
      const r = await post("/predict", { text: $("text").value, model: $("model").value, language: $("language").value, explain: $("explain").checked });
      $("result").hidden = false; renderCards(r, $("cards")); renderNotes(r, $("safety"), $("warnings")); renderExplain($("explain-box"), r.explanation);
    } catch (e) { showErr(e.message); $("result").hidden = true; } finally { btn.disabled = false; }
  });

  const pf = $("play-form");
  if (pf) pf.addEventListener("submit", async (ev) => {
    ev.preventDefault(); showErr(""); const btn = pf.querySelector("button"); btn.disabled = true; const out = $("play-out"); out.replaceChildren();
    try {
      const j = await post("/api/compare", { text: $("text").value, models: ["bilstm", "transformer"] });
      Object.entries(j.results).forEach(([name, r]) => {
        const col = el("section", "card"); col.append(el("h3", "", r.model_display_name));
        const g = el("div", "grid"); renderCards(r, g); col.append(g);
        const s = el("div", "alert warn"); const w = el("div"); renderNotes(r, s, w); if (!s.hidden) col.append(s); col.append(w); out.append(col);
      });
    } catch (e) { showErr(e.message); } finally { btn.disabled = false; }
  });

  const bf = $("batch-form"); let lastRows = null;
  if (bf) {
    bf.addEventListener("submit", async (ev) => {
      ev.preventDefault(); showErr(""); const btn = bf.querySelector("button[type=submit]"); btn.disabled = true;
      try {
        const fd = new FormData(); fd.append("file", $("file").files[0]); fd.append("model", $("model").value);
        const r = await fetch("/api/batch", { method: "POST", body: fd }); const j = await r.json();
        if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Request failed");
        lastRows = j.rows; const out = $("batch-out"); out.hidden = false;
        const cols = ["id", "language", "sentiment", "emotion", "risk", "risk_probability", "note"]; const t = el("table", "small"); const h = el("tr"); cols.forEach((c) => h.append(el("th", "", c))); t.append(h);
        j.rows.slice(0, 200).forEach((row) => { const tr = el("tr"); cols.forEach((c) => tr.append(el("td", "", row[c] === undefined ? "" : String(row[c])))); t.append(tr); });
        out.replaceChildren(el("p", "muted", j.n_rows + " rows processed" + (j.n_rows > 200 ? " (first 200 shown)" : "") + ". " + j.probability_note), t); $("download").hidden = false;
      } catch (e) { showErr(e.message); } finally { btn.disabled = false; }
    });
    $("download").addEventListener("click", () => {
      if (!lastRows) return; const cols = ["id", "language", "sentiment", "emotion", "risk", "risk_probability"];
      const esc = (v) => '"' + String(v === undefined ? "" : v).replace(/"/g, '""') + '"';
      const csv = [cols.join(",")].concat(lastRows.map((r) => cols.map((c) => esc(r[c])).join(","))).join("\n");
      const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "results.csv"; a.click(); URL.revokeObjectURL(a.href);
    });
  }
})();
