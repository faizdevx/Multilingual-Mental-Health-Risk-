"""FastAPI application: JSON API + Jinja2 research-dashboard UI. No user text is logged or stored."""
from __future__ import annotations

import csv
import io
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.app.reports import SUPPORTED_LANGUAGES, Reports
from src.config import DISCLAIMER, EVALUATED_LANGUAGES, MODEL_VERSION, PROBABILITY_NOTE, TASK_LANGUAGES
from src.data.datasets import processed_available
from src.evaluation.adapters import ARCH, DISPLAY
from src.inference import MAX_CHARS, ModelRegistry, Predictor
from src.privacy import log_event, new_request_id

HERE = Path(__file__).parent
BATCH_MAX_ROWS = int(os.environ.get("MHRISK_BATCH_MAX_ROWS", "500"))
BATCH_MAX_BYTES = int(os.environ.get("MHRISK_BATCH_MAX_BYTES", str(1_000_000)))
NAV = [("/", "Dashboard"), ("/analyze", "Analyze"), ("/playground", "Playground"), ("/models", "Model Comparison"),
       ("/languages", "Language Analysis"), ("/errors", "Error Analysis"), ("/batch", "Batch"), ("/health-ui", "Health"), ("/about", "About")]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    model: str = "transformer"
    language: str | None = None  # None / "auto" => detect
    explain: bool = False


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    models: list[str] = ["bilstm", "transformer"]
    language: str | None = None


def _csv_safe(v) -> str:
    """Neutralise spreadsheet formula injection in echoed cells."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def create_app(registry: ModelRegistry | None = None, reports: Reports | None = None) -> FastAPI:
    app = FastAPI(title="Multilingual Mental Health Risk & Sentiment Analyzer", version=MODEL_VERSION,
                  description="Research / screening NLP system. " + DISCLAIMER)
    app.state.registry = registry if registry is not None else ModelRegistry()
    app.state.predictor = Predictor(app.state.registry)
    app.state.reports = reports or Reports()
    app.state.started = time.time()
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["fmt"] = lambda v, d=3: "Not available" if v is None or v == "" else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    templates.env.filters["pct"] = lambda v: 0 if v is None else max(0, min(100, round(float(v) * 100, 1)))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def page(request: Request, name: str, **ctx):
        base = dict(nav=NAV, path=request.url.path, disclaimer=DISCLAIMER, version=MODEL_VERSION, prob_note=PROBABILITY_NOTE,
                    models=app.state.registry.names(), display=DISPLAY)
        return templates.TemplateResponse(request, name, {**base, **ctx})

    # ---------------------------------------------------------------- middleware / errors (privacy-safe)
    @app.middleware("http")
    async def privacy_logging(request: Request, call_next):
        rid, t0 = new_request_id(), time.perf_counter()
        status, err = 500, None
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:  # never include message/args (could contain text)
            err = type(e).__name__
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"
        route = request.scope.get("route")
        log_event(request_id=rid, method=request.method, route=getattr(route, "path", request.url.path), status=status,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 1), model_version=MODEL_VERSION, error_type=err)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # default handler echoes the submitted input back; strip it so text never appears in error payloads
        errs = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse({"detail": errs}, status_code=422)

    # ---------------------------------------------------------------- API
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse({"name": app.title, "version": MODEL_VERSION, "disclaimer": DISCLAIMER,
                                 "endpoints": {"health": "GET /health", "predict": "POST /predict", "docs": "GET /docs"},
                                 "models_loaded": app.state.registry.names(), "evaluated_languages": EVALUATED_LANGUAGES})
        return page(request, "dashboard.html", **dashboard_context())

    @app.get("/health")
    async def health():
        r = app.state.registry
        return {"status": "healthy", "models_loaded": r.names(), "models_failed": list(r.errors), "device": r.device,
                "version": MODEL_VERSION, "dataset_available": processed_available(),
                "uptime_s": round(time.time() - app.state.started, 1)}

    def _need_model(name: str):
        if name not in app.state.registry.adapters:
            return JSONResponse({"detail": f"model '{name}' is not loaded", "available": app.state.registry.names()}, status_code=503 if not app.state.registry.names() else 400)
        return None

    @app.post("/predict")
    async def predict(req: PredictRequest):
        if (bad := _need_model(req.model)) is not None:
            return bad
        r = app.state.predictor.analyze(req.text, req.model, req.language, req.explain)
        log_event(event="predict", model=req.model, language_tag=r["language_tag"], risk_label=r["risk"])
        return r

    @app.post("/api/compare")
    async def compare(req: CompareRequest):
        out = {}
        for m in req.models:
            if (bad := _need_model(m)) is not None:
                return bad
            out[m] = app.state.predictor.analyze(req.text, m, req.language)
        return {"results": out}

    @app.post("/api/batch")
    async def batch(file: UploadFile = File(...), model: str = Form("transformer"), format: str = Form("json")):
        if (bad := _need_model(model)) is not None:
            return bad
        raw = await file.read(BATCH_MAX_BYTES + 1)  # kept in memory only; nothing is written to disk
        if len(raw) > BATCH_MAX_BYTES:
            return JSONResponse({"detail": f"file too large (max {BATCH_MAX_BYTES} bytes)"}, status_code=413)
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        except (UnicodeDecodeError, csv.Error):
            return JSONResponse({"detail": "could not parse file as UTF-8 CSV"}, status_code=400)
        finally:
            del raw
        if not rows or "text" not in rows[0]:
            return JSONResponse({"detail": "CSV must have a 'text' column (optional: id, language)"}, status_code=400)
        if len(rows) > BATCH_MAX_ROWS:
            return JSONResponse({"detail": f"too many rows (max {BATCH_MAX_ROWS})"}, status_code=413)
        texts = [(r.get("text") or "").strip() for r in rows]
        valid = [i for i, t in enumerate(texts) if t]
        res = app.state.predictor.analyze_many([texts[i] for i in valid], model, [(rows[i].get("language") or None) for i in valid]) if valid else []
        by_idx = dict(zip(valid, res))
        out = []
        for i, r in enumerate(rows):
            a = by_idx.get(i)
            out.append({"id": _csv_safe(r.get("id") or i + 1), "language": a["language_tag"] if a else "", "sentiment": a["sentiment"] if a else "",
                        "emotion": a["emotion"] if a else "", "risk": a["risk"] if a else "", "risk_probability": a["risk_probability"] if a else "",
                        "note": "" if a else "empty text"})
        log_event(event="batch", model=model, n_items=len(rows))
        if format == "csv":
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
            return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=results.csv"})
        return {"model": model, "n_rows": len(out), "rows": out, "disclaimer": DISCLAIMER, "probability_note": PROBABILITY_NOTE}

    # ---- JSON views of the evaluation artefacts
    def dashboard_context() -> dict:
        rp, reg = app.state.reports, app.state.registry
        best = rp.best_model_entry()
        stats = rp.dataset_stats()
        return dict(best=best, stats=stats, supported=SUPPORTED_LANGUAGES, loaded=reg.names(), device=reg.device if reg.names() else "Not available",
                    calibration=rp.calibration_status(best["name"]) if best else "Not available", comparison=rp.comparison(),
                    task_langs=TASK_LANGUAGES, langid=rp.langid())

    @app.get("/api/dashboard")
    async def api_dashboard():
        c = dashboard_context()
        return {k: c[k] for k in ("best", "stats", "supported", "loaded", "device", "calibration")}

    @app.get("/api/models")
    async def api_models():
        return app.state.reports.comparison() or {"models": []}

    @app.get("/api/languages")
    async def api_languages():
        return {"evaluated_languages": SUPPORTED_LANGUAGES, "metrics": app.state.reports.language_table()}

    def filtered_errors(language="", task="", model="", prediction="", limit=200):
        df = app.state.reports.error_table()
        if df is None:
            return None, 0
        for col, val in (("language", language), ("task", task), ("model", model), ("predicted_label", prediction)):
            if val:
                if col == "predicted_label":
                    df = df[(df.predicted_sentiment == val) | (df.predicted_emotion == val) | (df.predicted_risk == val)]
                else:
                    df = df[df[col] == val]
        return df.head(limit), len(df)

    @app.get("/api/errors")
    async def api_errors(language: str = "", task: str = "", model: str = "", prediction: str = ""):
        df, total = filtered_errors(language, task, model, prediction)
        return {"total": total, "rows": [] if df is None else df.to_dict("records"), "summary": app.state.reports.error_summary()}

    # ---------------------------------------------------------------- pages
    @app.get("/analyze", response_class=HTMLResponse)
    async def analyze_page(request: Request):
        return page(request, "analyze.html")

    @app.get("/playground", response_class=HTMLResponse)
    async def playground_page(request: Request):
        return page(request, "playground.html")

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        rp = app.state.reports
        comp = rp.comparison()
        loaded = set(app.state.registry.names())
        rows = []
        for m in (comp or {}).get("models", []):
            rows.append({**m, "checkpoint_status": "loaded" if m["name"] in loaded else "evaluated, not loaded"})
        for n in ARCH:
            if n not in {r["name"] for r in rows}:
                rows.append({"name": n, "display_name": DISPLAY[n], "architecture": ARCH[n], "checkpoint_status": "not available"})
        hist = {n: rp.history(n) for n in ("bilstm", "transformer")}
        return page(request, "models.html", rows=rows, comp=comp, history=hist)

    @app.get("/languages", response_class=HTMLResponse)
    async def languages_page(request: Request):
        rp = app.state.reports
        return page(request, "languages.html", table=rp.language_table(), supported=SUPPORTED_LANGUAGES, task_langs=TASK_LANGUAGES,
                    by_source=(rp.by_language() or {}).get("by_source"), langid=rp.langid())

    @app.get("/errors", response_class=HTMLResponse)
    async def errors_page(request: Request, language: str = "", task: str = "", model: str = "", prediction: str = ""):
        rp = app.state.reports
        df, total = filtered_errors(language, task, model, prediction)
        opts = {}
        full = rp.error_table()
        if full is not None:
            opts = {"language": sorted(full.language.unique()), "task": sorted(full.task.unique()), "model": sorted(full.model.unique()),
                    "prediction": sorted(set(full.predicted_sentiment) | set(full.predicted_risk) | set(x for v in full.predicted_emotion for x in v.split("|")) - {""})}
        return page(request, "errors.html", rows=None if df is None else df.to_dict("records"), total=total, summary=rp.error_summary(),
                    opts=opts, sel=dict(language=language, task=task, model=model, prediction=prediction))

    @app.get("/batch", response_class=HTMLResponse)
    async def batch_page(request: Request):
        return page(request, "batch.html", max_rows=BATCH_MAX_ROWS)

    @app.get("/health-ui", response_class=HTMLResponse)
    async def health_ui(request: Request):
        rp, reg = app.state.reports, app.state.registry
        ckpts = {n: {"loaded": n in reg.adapters, "failed": n in reg.errors, "present": (rp.ckpt_dir / n).exists()} for n in ("baseline", "bilstm", "transformer")}
        return page(request, "health.html", ckpts=ckpts, artifacts=rp.artifact_status(), device=reg.device, torch_v=torch.__version__,
                    tf_v=transformers.__version__, py=platform.python_version(), dataset=processed_available(), loaded=reg.names(),
                    uptime=round(time.time() - app.state.started, 1))

    @app.get("/about", response_class=HTMLResponse)
    async def about_page(request: Request):
        return page(request, "about.html", supported=SUPPORTED_LANGUAGES, task_langs=TASK_LANGUAGES)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory src.app.main:app_factory
    return create_app()


app = None  # populated lazily: `uvicorn src.app.main:app_factory --factory`
