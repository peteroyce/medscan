.PHONY: setup train evaluate serve clean

setup:
	pip install -r requirements.txt

train:
	python -m src.train --config configs/train_config.yaml

evaluate:
	python -m src.evaluate --config configs/train_config.yaml

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf outputs/ checkpoints/ __pycache__ .ipynb_checkpoints
	find . -name "*.pyc" -delete
