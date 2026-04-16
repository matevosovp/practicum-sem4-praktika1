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

Реализованы два подхода:

- baseline 1: глобальная популярность продукта по частоте следующего подключения;
- baseline 2: сегментная популярность (`segmento`) со сглаживанием;
- supervised experiment: `OneVsRest(SGDClassifier loss="log_loss")` с числовыми, категориальными и продуктовыми признаками.

По sampled backtest на holdout-периодах лучшей офлайн-моделью оказался **segment popularity baseline**, поэтому именно он сохраняется в `models/best_model.joblib` и используется в API. Supervised-модель сохранена отдельно в `models/supervised_model.joblib`.

Пайплайн включает:

- chunked-разрезание исходного CSV на очищенные помесячные снапшоты;
- построение фичей на месяце `t`;
- формирование multilabel-таргета на `t+1`;
- time-based split;
- downsampling негативных примеров на train без затрагивания validation/test;
- расчёт ranking-метрик и анализ ошибок.

### MLflow

Логируются:

- параметры эксперимента;
- ranking-метрики на validation и test;
- модель;
- артефакты анализа ошибок и важности признаков;
- входной пример и сигнатура модели.

Хранилище по умолчанию локальное:

- backend store: `sqlite:///mlruns/mlflow.db`
- artifact store: `./mlruns/artifacts`

### API

FastAPI-сервис находится в [app.py](/home/what/praktika/practicum-sem4-praktika1/src/service/app.py).

Endpoint'ы:

- `GET /health`
- `POST /predict`
- `GET /metrics`

`POST /predict` принимает текущий профиль клиента и список уже имеющихся продуктов, а в ответ возвращает `top_k` новых рекомендаций со score.

## Результаты экспериментов

Отложенная оценка считалась на sampled holdout-месяцах по `120,000` клиентов на месяц с фиксированным `random_state=42`.

Validation:

- `global popularity`: `MAP@3 = 0.5849`, `NDCG@3 = 0.6502`, `Precision@3 = 0.0789`
- `segment popularity`: `MAP@3 = 0.6215`, `NDCG@3 = 0.6592`, `Precision@3 = 0.0796`
- `supervised SGD`: `MAP@3 = 0.1840`, `NDCG@3 = 0.2264`, `Precision@3 = 0.0330`

Test:

- `global popularity`: `MAP@3 = 0.5753`, `NDCG@3 = 0.6415`, `Precision@3 = 0.0704`
- `segment popularity`: `MAP@3 = 0.6180`, `NDCG@3 = 0.6546`, `Precision@3 = 0.0701`
- `supervised SGD`: `MAP@3 = 0.1851`, `NDCG@3 = 0.2277`, `Precision@3 = 0.0306`

Практический вывод: в этих данных сильный popularity-сигнал, а сегментное сглаживание даёт лучший ranking, чем и глобальная популярность, и текущая линейная supervised-модель. Поэтому в продовый артефакт вынесен именно segment-based baseline, а supervised-пайплайн оставлен как воспроизводимый эксперимент с артефактами ошибок и коэффициентов.

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

## Как запустить EDA

Сначала подготовить артефакты и модельные месяцы:

```bash
source .venv_rec_prod/bin/activate
python -m src.models.train
```

После этого открыть ноутбук:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Как запустить обучение

```bash
source .venv_rec_prod/bin/activate
bash scripts/train_model.sh
```

Полезные выходы после обучения:

- `models/best_model.joblib`
- `models/supervised_model.joblib`
- `models/model_metadata.json`
- `models/feature_importance.csv`
- `models/validation_errors.csv`
- `models/test_errors.csv`
- `models/reference_stats.json`

## Как запустить MLflow

```bash
source .venv_rec_prod/bin/activate
bash scripts/run_mlflow.sh
```

После старта UI доступен на `http://localhost:5000`.

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

- `best_model.joblib` — основной bundle с лучшей офлайн-моделью `segment_popularity`;
- `supervised_model.joblib` — отдельный bundle с supervised `SGD`-экспериментом;
- `product_mapping.json` — человекочитаемые названия продуктов;
- `reference_stats.json` — база для простого drift-check в API;
- `feature_importance.csv` — топ коэффициентов именно для supervised-модели;
- `validation_errors.csv` / `test_errors.csv` — примеры промахов модели.

## Retraining-подход

Практичный цикл дообучения:

1. Появился новый месячный снапшот.
2. Пайплайн пересобирает `t -> t+1` выборки.
3. Переобучается модель со скользящим историческим окном.
4. Обновляются MLflow-эксперименты и `reference_stats.json`.
5. После smoke-check выкатывается новый `best_model.joblib`.

Для автоматического запуска подготовлен [retrain.py](/home/what/praktika/practicum-sem4-praktika1/scripts/retrain.py).

## Ограничения текущего решения

- В текущем наборе признаков supervised-линейная модель уступила сегментному popularity baseline.
- В офлайне пока нет отдельной калибровки score.
- Для ускорения обучения негативные примеры на train семплируются.
- В drift-контроле реализован разумный минимум, а не полноценный отдельный сервис мониторинга данных.

## Что можно улучшить

- product-specific модели или бустинг для редких продуктов;
- больше лаговых признаков по истории изменений портфеля;
- калибровка вероятностей;
- отдельный online feedback loop и бизнес-метрики конверсии;
- полноценный drift-job с ежедневным batch-отчётом.
