# Рекомендации банковских продуктов

Учебный ML-проект, в котором по ежемесячным клиентским снапшотам банка строится система рекомендаций новых продуктов на следующий месяц.

## Бизнес-задача

Банку нужно предложить клиенту релевантный **дополнительный** продукт, а не повторить уже имеющийся портфель. Это влияет на конверсию оффера, кросс-сейл и пользовательский опыт: плохая рекомендация тратит слот коммуникации и снижает вероятность отклика.

## ML-задача

Задача сформулирована как **multilabel recommendation / ranking**:

- на входе есть клиентский профиль в месяце `t`;
- таргет — продукты, которые клиент **добавит в месяце `t+1`**;
- при ранжировании из кандидатов исключаются продукты, которые у клиента уже есть в `t`.

Это принципиально отличается от задачи предсказания текущего портфеля: модель должна искать именно будущие новые подключения.

## Данные

Источник данных: `train_ver2.csv` со снапшотами клиентов с января 2015 по май 2016.

Основные сущности:

- `fecha_dato` — месяц наблюдения;
- `ncodpers` — идентификатор клиента;
- `ind_*_ult1` — 24 продуктовых признака;
- остальное — демография, канал, активность, сегмент, провинция, доход и служебные атрибуты клиента.

Особенности данных:

- временная структура, где критично не допустить утечку будущего;
- грязные строковые значения с пробелами;
- шум в типах;
- пропуски в доходе, дате подключения, части категориальных полей;
- сильный дисбаланс: новые продукты появляются редко.

## Формулировка офлайн-эксперимента

Используется time-based split:

- train: срезы `2015-02` ... `2015-11`;
- validation: `2015-12`, `2016-01`, `2016-02`;
- test: `2016-03`, `2016-04`.

Для каждого месяца `t` таргет формируется как `max(products_(t+1) - products_t, 0)`.

## Метрики

Основные метрики:

- `MAP@3` — основная ranking-метрика, потому что учитывает и попадание, и порядок в top-3;
- `Precision@3` — насколько качественны три показанных оффера;
- `Recall@3` — какую долю реальных новых продуктов удаётся покрыть;
- `NDCG@3` — насколько правильно упорядочены рекомендации.

Почему `K = 3`:

- в банковском сценарии разумно не показывать слишком длинный список офферов;
- три продукта достаточно, чтобы не перегружать клиента и сохранить ценность ranking-модели;
- `top-3` хорошо интерпретируется и для CRM-коммуникаций, и для оператора, и для фронта в цифровом канале.

`Accuracy` и случайный `train_test_split` здесь не используются, потому что они не отражают бизнес-смысл задачи и дают утечку.

## Что реализовано

### EDA

Ноутбук [01_eda.ipynb](/home/what/praktika/practicum-sem4-praktika1/notebooks/01_eda.ipynb) строится поверх кода из `src/` и анализирует:

- структуру данных и временную динамику;
- пропуски и грязные значения;
- проникновение продуктов по месяцам;
- динамику новых подключений;
- клиентские сегменты;
- выводы, которые подводят к ranking-постановке.

### Моделирование

Эксперименты выстроены как поэтапная лестница улучшений:

- `stage_00_global_popularity`: самый простой baseline по глобальной частоте новых подключений;
- `stage_01_segment_popularity`: baseline после улучшения сегментацией по `segmento`;
- `stage_02_catboost_basic`: первая CatBoost-модель на сырых клиентских признаках и текущем продуктном портфеле;
- `stage_03_catboost_feature_engineering`: CatBoost после добавления engineered temporal / portfolio features;
- `stage_04_catboost_tuned`: финальный CatBoost после двухступенчатого, ресурсно-ограниченного подбора гиперпараметров.

Итоговая основная модель проекта — **CatBoost**. Для продового артефакта `best_model.joblib` выбирается лучшая именно среди CatBoost-стадий, а baseline'ы остаются как честная точка сравнения.

Пайплайн включает:

- chunked-разрезание исходного CSV на очищенные помесячные снапшоты;
- построение только нужных modeling columns на месяце `t` и раннее отбрасывание неиспользуемых полей;
- формирование multilabel-таргета на `t+1`;
- time-based split;
- downsampling негативных примеров на train без затрагивания validation/test;
- ограничение полного train sample через cap `TRAIN_MAX_ROWS`, чтобы CatBoost не раздувался по памяти на редких продуктах;
- облегчённые dtypes в modeling parquet и column-pruned чтение под конкретную стадию;
- memory-safe one-vs-rest CatBoost: продуктовые модели обучаются строго последовательно, сразу сохраняются в stage-директорию и выгружаются из памяти;
- явный `del` и `gc.collect()` после каждой продуктовой модели и после каждой крупной stage-структуры;
- CatBoost с нативной работой по категориальным признакам без ручного one-hot и с memory-aware ограничениями по CTR/threads;
- фиксированный `eval_set` и `early_stopping` для всех CatBoost-стадий;
- двухступенчатый tuning вместо тяжелого глобального поиска:
  - Stage A: быстрый screening компактного ручного набора memory-safe конфигураций на последних train-месяцах и sampled validation;
  - Stage B: подтверждение только top-N конфигураций на полном train sample;
- feature importance только для осмысленных CatBoost-стадий, а не для каждого промежуточного run;
- расчёт ranking-метрик и анализ ошибок.

### MLflow

Логируются отдельные runs для каждой стадии улучшения:

- параметры эксперимента;
- ranking-метрики на validation и test;
- одна главная scalar-метрика `val_map_at_3` для удобного сравнения runs;
- bundle-артефакты и финальные stage summary;
- артефакты анализа ошибок и важности признаков только для ключевых CatBoost-стадий;
- отдельные MLflow runs для всех tuning-кандидатов из Stage A и Stage B;
- leaderboard'ы `stage_04_stage_a_leaderboard.csv` и `stage_04_stage_b_leaderboard.csv` для tuned-стадии.

Лучший CatBoost-run дополнительно регистрируется в **MLflow Model Registry** под именем `bank-product-recommendations-catboost` и получает alias `champion`.

Хранилище по умолчанию локальное:

- backend store: `sqlite:///mlruns/mlflow.db`
- artifact store: `./mlruns/artifacts`

### API

FastAPI-сервис находится в [app.py](/home/what/praktika/practicum-sem4-praktika1/src/service/app.py).

Endpoint'ы:

- `GET /health`
- `POST /predict`
- `GET /metrics`

`POST /predict` принимает текущий профиль клиента и список уже имеющихся продуктов, а в ответ возвращает `top_k` новых рекомендаций со score. Скоринг строится на выходах CatBoost-модели, а уже имеющиеся продукты принудительно исключаются из top-k.

## Результаты экспериментов

Отложенная оценка считалась на sampled holdout-месяцах по `120,000` клиентов на месяц с фиксированным `random_state=42`.

Текущая идея сравнения не в “зоопарке моделей”, а в понятной траектории улучшений:

- что даёт простой baseline;
- что даёт более сильный baseline;
- что даёт переход к CatBoost как основной модели;
- что даёт feature engineering;
- что даёт более устойчивый двухступенчатый tuning.

Главная метрика для сравнения runs в MLflow: `val_map_at_3`.

Актуальные метрики сохраняются в `models/model_metadata.json` и в MLflow по каждому stage-run отдельно.

### Мониторинг

Из API в Prometheus отдаются:

- `bank_reco_request_total`
- `bank_reco_error_total`
- `bank_reco_latency_seconds`
- `bank_reco_score_distribution`
- `bank_reco_empty_total`
- `bank_reco_suspicious_total`

Подробности — в [MONITORING.md](/home/what/praktika/practicum-sem4-praktika1/monitoring/MONITORING.md).

## Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_experiments.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── service/
│   └── utils/
├── scripts/
├── monitoring/
├── models/
├── Dockerfile
└── docker-compose.yml
```

## Установка

```bash
python3.10 -m venv .venv_rec_prod
source .venv_rec_prod/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
## Как запустить MLflow

```bash
source .venv_rec_prod/bin/activate
bash scripts/run_mlflow.sh
```
## Как запустить обучение

```bash
source .venv_rec_prod/bin/activate
bash scripts/train_model.sh
```

Ключевые memory-safe параметры:

```bash
CATBOOST_PROFILE=memory_safe
CATBOOST_RAM_LIMIT=12gb
CATBOOST_THREAD_COUNT=2
TRAIN_MAX_ROWS=350000
TUNING_STAGE_A_CANDIDATE_COUNT=4
TUNING_STAGE_B_TOP_N=2
```


## Как запустить EDA

Сначала подготовить артефакты и модельные месяцы:

```bash
source .venv_rec_prod/bin/activate
python -m src.models.train
```

После этого открыть ноутбук 01_eda.ipynb


Если нагрузку на машину, можно запускать так:

```bash
CATBOOST_PROFILE=memory_safe CATBOOST_RAM_LIMIT=12gb CATBOOST_THREAD_COUNT=2 CATBOOST_TUNING_ENABLED=true TRAIN_MAX_ROWS=300000 TUNING_STAGE_A_MAX_ROWS=120000 TUNING_STAGE_A_CANDIDATE_COUNT=4 bash scripts/train_model.sh
```

Если нужен только быстрый smoke-прогон без hyperparameter search:

```bash
CATBOOST_PROFILE=memory_safe CATBOOST_RAM_LIMIT=12gb CATBOOST_THREAD_COUNT=1 CATBOOST_TUNING_ENABLED=false TRAIN_MAX_ROWS=150000 EVAL_MONTH_SAMPLE_SIZE=40000 CATBOOST_FIT_EVAL_SIZE=20000 bash scripts/train_model.sh
```




После старта UI доступен на `http://localhost:5000`.

В UI нужно смотреть:

- runs внутри эксперимента `bank-product-recommendations`;
- scalar metric `val_map_at_3` для сравнения стадий и tuning-кандидатов;
- раздел **Models**, где зарегистрирована `bank-product-recommendations-catboost`;
- alias `champion`, который указывает на текущую основную версию CatBoost-модели.

## Как запустить API локально

```bash
source .venv_rec_prod/bin/activate
bash scripts/run_api.sh
```

Swagger UI будет доступен на `http://localhost:8000/docs`.

Пример запроса:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 123456,
    "snapshot_date": "2016-04-28",
    "top_k": 3,
    "current_products": ["ind_cco_fin_ult1", "ind_recibo_ult1"],
    "profile": {
      "sexo": "H",
      "age": 42,
      "antiguedad": 64,
      "ind_actividad_cliente": 1,
      "segmento": "02 - PARTICULARES",
      "renta": 120000
    }
  }'
```

## Как запустить в Docker

Сначала обучить и сохранить модель локально, затем:

```bash
docker compose up --build
```

Сервисы:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`

## Артефакты модели

- `best_model.joblib` — основной bundle с лучшей CatBoost-стадией по validation `MAP@3`;
- `stage_00_*.joblib ... stage_04_*.joblib` — отдельные артефакты каждой стадии;
- `stage_*_models/` — каталоги с последовательными per-product CatBoost-моделями, которые сохраняются сразу после fit;
- `product_mapping.json` — человекочитаемые названия продуктов;
- `reference_stats.json` — база для простого drift-check в API;
- `experiment_leaderboard.csv` — единая таблица сравнения baseline / CatBoost / tuned CatBoost;
- `split_summary.csv` — помесячная сводка train / validation / test для ноутбука и ревью;
- `feature_list.json` / `categorical_features.json` — зафиксированный список признаков для продового CatBoost bundle;
- `stage_03_*_feature_importance.csv`, `stage_04_*_feature_importance.csv` — топ важностей признаков только для ключевых CatBoost-стадий;
- `stage_*_validation_errors.csv` / `stage_*_test_errors.csv` — примеры промахов для CatBoost-стадий;
- `stage_04_stage_a_leaderboard.csv` — результаты быстрого screening-а конфигураций;
- `stage_04_stage_b_leaderboard.csv` — результаты финального подтверждения top-N конфигураций;
- `stage_04_tuning_summary.json` — краткое описание новой стратегии подбора.

## Retraining-подход

Практичный цикл дообучения:

1. Появился новый месячный снапшот.
2. Пайплайн пересобирает `t -> t+1` выборки.
3. Переобучается модель со скользящим историческим окном в последовательном memory-safe режиме.
4. Обновляются MLflow-эксперименты, leaderboard'ы тюнинга, `reference_stats.json` и alias `champion` в Model Registry.
5. После smoke-check выкатывается новый `best_model.joblib`.

Для автоматического запуска подготовлен [retrain.py](/home/what/praktika/practicum-sem4-praktika1/scripts/retrain.py).

## Ограничения текущего решения

- На текущих данных сильный popularity-сигнал всё ещё может быть конкурентным по сравнению с CatBoost.
- В офлайне пока нет отдельной калибровки score.
- Для ускорения обучения негативные примеры на train семплируются.
- В drift-контроле реализован разумный минимум, а не полноценный отдельный сервис мониторинга данных.

## Что можно улучшить

- product-specific модели или бустинг для редких продуктов;
- больше лаговых признаков по истории изменений портфеля;
- калибровка вероятностей;
- отдельный online feedback loop и бизнес-метрики конверсии;
- полноценный drift-job с ежедневным batch-отчётом.
