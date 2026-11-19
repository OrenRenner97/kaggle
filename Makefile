# make download C=titanic
download:
	@test -n "$(C)" || (echo "укажи C=<slug соревнования>"; exit 1)
	mkdir -p data/kaggle/$(C)
	kaggle competitions download -c $(C) -p data/kaggle/$(C)
	cd data/kaggle/$(C) && unzip -o -q *.zip && rm -f *.zip

# make submit C=titanic F=competitions/01_titanic/submissions/catboost.csv M="catboost v1"
submit:
	kaggle competitions submit -c $(C) -f $(F) -m "$(M)"

# make nb N=competitions/01_titanic/02_models.ipynb — выполнить ноутбук целиком и сохранить выводы
nb:
	cd $(dir $(N)) && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $(notdir $(N))

.PHONY: download submit nb
