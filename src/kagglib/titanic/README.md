# kagglib.titanic — прод

| Файл | Что делает |
|------|------------|
| `features.py` | сырые колонки → 21 признак (float32). `fit` → `FeatureState`, `transform` с явным `Context` (кого знаем) |
| `cv.py` | честная CV: признаки, включая групповые, пересчитываются внутри фолда |
| `model.py` | обучение CatBoost (основная) и LogReg (резервная), экспорт пакета артефактов |
| `predict.py` | `Predictor(bundle, backend="onnx" \| "native" \| "logreg")` — инференс по пакету |

## Пакет артефактов (`models/titanic/`)

| Файл | Для чего | Зависимости у потребителя |
|------|----------|---------------------------|
| `catboost.onnx` | основная модель, вход `features` float32[N,21] → `probabilities` float32[N,2] | любой ONNX Runtime: C/C++, C#, Java, JS, Go, Python, мобильные |
| `libtitanic.so` | основная модель как нативная библиотека, C ABI | **никаких**, кроме libstdc++ |
| `catboost_model.cpp` + `titanic_native.cpp` | исходники `.so` — собрать под любую платформу: `g++ -O2 -shared -fPIC *.cpp -o libtitanic.so` | C++17 компилятор |
| `catboost.cbm` | родной формат CatBoost | catboost |
| `logreg.onnx`, `logreg.json` | резервная модель; JSON — формула: стандартизация → веса → сигмоида | ничего (формулу можно написать руками) |
| `feature_state.json` | статистики заполнения пропусков | — |
| `reference_outcomes.csv` | архив исходов (PassengerId, Ticket, Survived) для групповых признаков | — |
| `golden.csv` | эталон: признаки + ожидаемые вероятности для 50 пассажиров | — |
| `manifest.json` | версия, порядок признаков, порог, sha256 всех файлов | — |

`Predictor` проверяет sha256 всех файлов при загрузке: подменённый или битый артефакт не загрузится.

### C ABI

```c
int    titanic_n_features(void);   // 21
int    titanic_predict_proba(const float* X, size_t rows, size_t cols, double* out);  // 0 = ok, 1 = неверное cols
```

### Перенос на другую платформу

Модель переносится бит-в-бит (ONNX и `.so` совпадают с CatBoost до 1e-7). **Признаки** нужно реализовать на стороне
потребителя по `features.py` — это ~60 строк: титул из имени, медианы из `feature_state.json`, размер группы по билету,
судьба попутчиков по `reference_outcomes.csv`. Проверка реализации — `golden.csv`: признаки и вероятности должны совпасть.

Важно: `TicketGroupSize` зависит от **состава партии** (кто ещё едет с этим билетом), поэтому эталон посчитан для
конкретной партии — первые 50 строк `test.csv` + архив.

## Тесты

`tests/titanic/`: нет утечки собственного исхода, валидационные исходы не попадают в признаки, невиданные титулы и пропуски,
совпадение ONNX / native / CatBoost, воспроизведение `golden.csv`, формула из `logreg.json` = ONNX, отказ при подмене файла.
