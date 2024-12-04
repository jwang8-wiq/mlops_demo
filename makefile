.PHONY: lint test build deploy run-data-preparation run-model-training run-evaluate run-retrain run-fastapi

# Linting
lint:
	flake8 application/ scripts/ tests/

# Testing
test:
	pytest tests/

# Building Docker Images
build-fastapi:
	docker build -t fastapi-service -f docker/Dockerfile.fastapi .

build-retrain:
	docker build -t retrain-job -f docker/Dockerfile.retrain .

build-mlserver:
	docker build -t mlserver -f docker/Dockerfile.mlserver .

# Deploying to Kubernetes
deploy-fastapi:
	kubectl apply -f manifests/fastapi/

deploy-retrain:
	kubectl apply -f manifests/retraining/

deploy-mlserver:
	kubectl apply -f manifests/mlserver/

# Data Preparation
run-data-preparation:
	python scripts/run_data_preparation.py \
		--data-path data/raw/telco_churn.csv \
		--config-path config/process.yaml \
		--output-dir data/processed

# Model Training
run-model-training:
	python scripts/model_training.py

# Model Evaluation
run-evaluate:
	python scripts/evaluate_model.py

# Model Retraining
run-retrain:
	python scripts/retrain_model.py

# Run FastAPI Locally
run-fastapi:
	uvicorn application.src.create_app:create_app --factory --host 0.0.0.0 --port 8000 --reload
