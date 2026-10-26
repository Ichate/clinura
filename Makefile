PY ?= python3

.PHONY: install run model test lint fmt preview record replay benchmark ui build-ui

install:
	$(PY) -m pip install -r requirements-dev.txt
	cd app && npm install

run:
	$(PY) main.py

model:
	$(PY) scripts/download_model.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check engine scripts main.py

fmt:
	$(PY) -m ruff format engine scripts main.py

preview:
	PYTHONPATH=engine $(PY) -m clinura.cli preview

record:
	PYTHONPATH=engine $(PY) -m clinura.cli record --seconds 30

replay:
	PYTHONPATH=engine $(PY) -m clinura.cli replay $(SESSION)

benchmark:
	PYTHONPATH=engine $(PY) scripts/benchmark_synthetic.py

build-ui:
	cd app && npm run build

ui:
	cd app && npm run dev
