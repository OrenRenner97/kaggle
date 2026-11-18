# make download C=titanic
download:
	@test -n "$(C)" || (echo "укажи C=<slug соревнования>"; exit 1)
	mkdir -p data/kaggle/$(C)
	kaggle competitions download -c $(C) -p data/kaggle/$(C)
	cd data/kaggle/$(C) && unzip -o -q *.zip && rm -f *.zip

# make submit C=titanic F=competitions/01_titanic/submissions/x.csv M="lgbm v1"
submit:
	kaggle competitions submit -c $(C) -f $(F) -m "$(M)"

infra-up:
	docker compose -f infra/docker-compose.yml up -d

infra-down:
	docker compose -f infra/docker-compose.yml down

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: download submit infra-up infra-down mlflow
