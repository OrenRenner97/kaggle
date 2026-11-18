# data/ (в git не попадает)

```
data/
├── kaggle/<slug>/          # как скачалось с Kaggle (make download C=<slug>)
└── lake/<slug>/
    ├── sources/            # имитация сырых систем: CSV-выгрузки, JSON-логи, дампы
    ├── raw/                # то, что забрал ingestion, без изменений
    ├── staging/            # очищено, типизировано, дедуплицировано
    ├── marts/              # витрины под задачу
    └── features/           # offline-часть feature store
```

Всё здесь воспроизводится кодом из `competitions/NN_slug/`, руками ничего не правим.
