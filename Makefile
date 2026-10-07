.PHONY: install data baseline lstm transformer langid evaluate serve test smoke all
install:      ; pip install -r requirements.txt
data:         ; python scripts/download_data.py && python scripts/prepare_data.py
baseline:     ; python scripts/train_baseline.py
langid:       ; python scripts/train_langid.py
lstm:         ; python scripts/train_lstm.py
transformer:  ; python scripts/train_transformer.py   # ~2-3 h on a 4-core CPU; much faster on GPU
evaluate:     ; python scripts/evaluate.py
serve:        ; uvicorn src.app.main:app_factory --factory --host 127.0.0.1 --port 8000
test:         ; python -m pytest -q
smoke:        ; python scripts/train_lstm.py --smoke && python scripts/train_transformer.py --smoke
all: data baseline langid lstm transformer evaluate test
