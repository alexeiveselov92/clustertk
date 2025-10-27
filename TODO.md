# ClusterTK - План разработки библиотеки

## Видение проекта

ClusterTK - это библиотека для полного пайплайна кластерного анализа, разработанная для аналитиков. Цель: предоставить удобный API, где пользователь может передать датафрейм и настраивать каждый шаг через параметры, без написания кода.

## Архитектура

### Принципы дизайна

1. **Pipeline-based подход** (аналогично sklearn)
   - Один главный класс `ClusterAnalysisPipeline`
   - Настройка через параметры при инициализации
   - Методы для каждого шага можно вызывать отдельно или через `.fit()`

2. **Модульная структура**
   - Каждый модуль независим и может использоваться отдельно
   - Чёткое разделение ответственности

3. **Визуализации как optional dependency**
   - Устанавливаются через `pip install clustertk[viz]`
   - Основная функциональность работает без matplotlib/seaborn
   - Методы визуализации доступны только если установлен viz extra

4. **Гибкость и расширяемость**
   - Возможность передать собственные функции обработки
   - Поддержка различных алгоритмов кластеризации
   - Простое добавление новых методов

### Структура проекта

```
clustertk/
├── __init__.py                     # Главный API
├── pipeline.py                     # ClusterAnalysisPipeline класс
│
├── preprocessing/                  # Модуль предобработки
│   ├── __init__.py
│   ├── missing.py                  # Обработка пропущенных значений
│   ├── outliers.py                 # Детекция и обработка выбросов
│   ├── scaling.py                  # Нормализация (StandardScaler, RobustScaler)
│   └── transforms.py               # Трансформации (log, box-cox, etc.)
│
├── feature_selection/              # Модуль отбора признаков
│   ├── __init__.py
│   ├── correlation.py              # Корреляционный анализ и фильтрация
│   └── variance.py                 # Фильтрация по дисперсии
│
├── dimensionality/                 # Модуль снижения размерности
│   ├── __init__.py
│   ├── pca.py                      # PCA с автоподбором компонент
│   └── manifold.py                 # t-SNE, UMAP для визуализации
│
├── clustering/                     # Модуль кластеризации
│   ├── __init__.py
│   ├── kmeans.py                   # K-Means обёртка
│   ├── gmm.py                      # Gaussian Mixture Model
│   ├── hierarchical.py             # Иерархическая кластеризация
│   └── dbscan.py                   # DBSCAN
│
├── evaluation/                     # Модуль оценки качества
│   ├── __init__.py
│   ├── metrics.py                  # Silhouette, Calinski-Harabasz, Davies-Bouldin
│   └── optimal_k.py                # Подбор оптимального числа кластеров
│
├── interpretation/                 # Модуль интерпретации
│   ├── __init__.py
│   ├── profiles.py                 # Профилирование кластеров
│   └── naming.py                   # Автоматическое именование кластеров
│
└── visualization/                  # Модуль визуализации (ОПЦИОНАЛЬНЫЙ)
    ├── __init__.py
    ├── correlation.py              # Корреляционные матрицы
    ├── distributions.py            # Распределения, boxplots
    ├── dimensionality.py           # Scree plots, PCA biplots
    ├── clusters.py                 # t-SNE, scatter plots кластеров
    └── profiles.py                 # Heatmaps, radar charts профилей
```

### Пример использования (целевой API)

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Загружаем данные
df = pd.read_csv('data.csv')

# Создаём pipeline с настройками
pipeline = ClusterAnalysisPipeline(
    # Предобработка
    handle_missing='median',              # или 'mean', 'drop', callable
    handle_outliers='robust',             # 'robust', 'clip', 'remove', None
    scaling='robust',                     # 'standard', 'robust', 'minmax'

    # Отбор признаков
    correlation_threshold=0.85,           # Удалить признаки с корреляцией > 0.85
    variance_threshold=0.01,              # Удалить признаки с низкой дисперсией

    # PCA
    pca_variance=0.9,                     # Оставить компоненты, объясняющие 90% дисперсии
    pca_min_components=2,                 # Минимум компонент

    # Кластеризация
    clustering_algorithm='kmeans',        # 'kmeans', 'gmm', 'hierarchical', 'dbscan'
    n_clusters=None,                      # None = автоподбор, или список [3,4,5]
    n_clusters_range=(2, 10),            # Диапазон для автоподбора

    # Другое
    random_state=42,
    verbose=True                          # Вывод прогресса
)

# Запускаем полный пайплайн
pipeline.fit(df, feature_columns=['col1', 'col2', 'col3'])

# Или запускаем пошагово
pipeline.preprocess(df)
pipeline.select_features()
pipeline.reduce_dimensions()
pipeline.find_optimal_clusters()
pipeline.cluster(n_clusters=5)
pipeline.create_profiles()

# Получаем результаты
labels = pipeline.labels_                  # Метки кластеров
profiles = pipeline.cluster_profiles_      # Профили кластеров
metrics = pipeline.metrics_                # Метрики качества

# Визуализации (если установлен viz extra)
pipeline.plot_correlation_matrix()
pipeline.plot_pca_variance()
pipeline.plot_optimal_clusters()
pipeline.plot_clusters_2d()               # t-SNE/UMAP
pipeline.plot_cluster_profiles()          # Heatmap + radar charts

# Экспорт результатов
pipeline.export_results('results.csv')    # CSV с метками
pipeline.export_report('report.html')     # HTML отчёт со всеми графиками
```

## План разработки

### Фаза 1: Базовая инфраструктура

- [ ] Создать структуру проекта
  - [ ] Создать все директории и `__init__.py`
  - [ ] Настроить `setup.py` / `pyproject.toml`
  - [ ] Настроить `extras_require` для визуализаций
  - [ ] Создать `requirements.txt` и `requirements-viz.txt`

- [ ] Настроить окружение разработки
  - [ ] Настроить pre-commit hooks (black, flake8, mypy)
  - [ ] Настроить pytest
  - [ ] Создать `.gitignore`
  - [ ] Создать базовый `README.md`

- [ ] Создать базовый `ClusterAnalysisPipeline` класс
  - [ ] Определить интерфейс и основные методы
  - [ ] Реализовать `__init__` с параметрами
  - [ ] Создать заглушки для всех методов
  - [ ] Добавить docstrings

### Фаза 2: Модули предобработки

- [ ] `preprocessing/missing.py`
  - [ ] Функции для обработки пропусков (median, mean, drop)
  - [ ] Класс `MissingValueHandler`
  - [ ] Тесты

- [ ] `preprocessing/outliers.py`
  - [ ] IQR метод детекции выбросов
  - [ ] Z-score метод
  - [ ] Класс `OutlierHandler`
  - [ ] Тесты

- [ ] `preprocessing/scaling.py`
  - [ ] Обёртки для StandardScaler, RobustScaler, MinMaxScaler
  - [ ] Автоматический выбор scaler на основе данных
  - [ ] Тесты

- [ ] `preprocessing/transforms.py`
  - [ ] Log трансформации (с обработкой отрицательных/нулевых значений)
  - [ ] Box-Cox трансформация
  - [ ] Детекция скошенности
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.preprocess()`
  - [ ] Сохранение промежуточных результатов
  - [ ] Интеграционные тесты

### Фаза 3: Отбор признаков

- [ ] `feature_selection/correlation.py`
  - [ ] Вычисление корреляционной матрицы
  - [ ] Фильтрация сильно коррелирующих признаков
  - [ ] Класс `CorrelationFilter`
  - [ ] Тесты

- [ ] `feature_selection/variance.py`
  - [ ] Фильтрация признаков с низкой дисперсией
  - [ ] Класс `VarianceFilter`
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.select_features()`
  - [ ] Тесты

### Фаза 4: Снижение размерности

- [ ] `dimensionality/pca.py`
  - [ ] PCA с автоподбором компонент по variance threshold
  - [ ] Анализ loadings
  - [ ] Класс `PCAReducer`
  - [ ] Тесты

- [ ] `dimensionality/manifold.py`
  - [ ] t-SNE обёртка для 2D визуализации
  - [ ] UMAP обёртка (опционально)
  - [ ] Класс `ManifoldReducer`
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.reduce_dimensions()`
  - [ ] Тесты

### Фаза 5: Кластеризация

- [ ] `clustering/kmeans.py`
  - [ ] Обёртка для sklearn KMeans
  - [ ] Класс `KMeansClustering`
  - [ ] Тесты

- [ ] `clustering/gmm.py`
  - [ ] Обёртка для GaussianMixture
  - [ ] Класс `GMMClustering`
  - [ ] Тесты

- [ ] `clustering/hierarchical.py`
  - [ ] Обёртка для AgglomerativeClustering
  - [ ] Поддержка разных linkage методов
  - [ ] Класс `HierarchicalClustering`
  - [ ] Тесты

- [ ] `clustering/dbscan.py`
  - [ ] Обёртка для DBSCAN
  - [ ] Автоподбор eps и min_samples
  - [ ] Класс `DBSCANClustering`
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.cluster()`
  - [ ] Поддержка разных алгоритмов
  - [ ] Тесты

### Фаза 6: Оценка качества

- [ ] `evaluation/metrics.py`
  - [ ] Silhouette score
  - [ ] Calinski-Harabasz score
  - [ ] Davies-Bouldin score
  - [ ] Функция `compute_all_metrics()`
  - [ ] Тесты

- [ ] `evaluation/optimal_k.py`
  - [ ] Elbow method
  - [ ] Голосование по метрикам
  - [ ] Класс `OptimalKFinder`
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.find_optimal_clusters()`
  - [ ] Метод `pipeline.evaluate()`
  - [ ] Тесты

### Фаза 7: Интерпретация

- [ ] `interpretation/profiles.py`
  - [ ] Создание профилей кластеров (средние, медианы)
  - [ ] Топ-признаки для каждого кластера
  - [ ] Анализ по категориям признаков
  - [ ] Класс `ClusterProfiler`
  - [ ] Тесты

- [ ] `interpretation/naming.py`
  - [ ] Автоматическое предложение имён для кластеров
  - [ ] Эвристики на основе профилей
  - [ ] Класс `ClusterNamer`
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.create_profiles()`
  - [ ] Метод `pipeline.suggest_names()`
  - [ ] Тесты

### Фаза 8: Визуализация - ОПЦИОНАЛЬНО

- [ ] `visualization/correlation.py`
  - [ ] Heatmap корреляционной матрицы
  - [ ] Тесты с mock данными

- [ ] `visualization/distributions.py`
  - [ ] Boxplots до/после нормализации
  - [ ] Гистограммы распределений
  - [ ] Тесты

- [ ] `visualization/dimensionality.py`
  - [ ] Scree plot для PCA
  - [ ] Biplot для loadings
  - [ ] Тесты

- [ ] `visualization/clusters.py`
  - [ ] 2D scatter plot (t-SNE/UMAP)
  - [ ] Визуализация центроидов
  - [ ] Тесты

- [ ] `visualization/profiles.py`
  - [ ] Heatmap профилей кластеров
  - [ ] Radar charts
  - [ ] Bar charts по категориям
  - [ ] Тесты

- [ ] Интеграция в Pipeline
  - [ ] Методы `pipeline.plot_*()`
  - [ ] Проверка наличия viz dependencies
  - [ ] Тесты

### Фаза 9: Экспорт и отчёты

- [ ] Экспорт результатов
  - [ ] CSV с метками кластеров
  - [ ] JSON с профилями
  - [ ] Pickle для сохранения pipeline

- [ ] HTML отчёт
  - [ ] Шаблон для отчёта
  - [ ] Встраивание всех графиков
  - [ ] Таблицы с метриками
  - [ ] Экспорт профилей

- [ ] Интеграция в Pipeline
  - [ ] Метод `pipeline.export_results()`
  - [ ] Метод `pipeline.export_report()`
  - [ ] Тесты

### Фаза 10: Документация и примеры

- [ ] Документация
  - [ ] Настроить Sphinx
  - [ ] API reference для всех классов
  - [ ] User guide с примерами
  - [ ] Tutorial notebooks

- [ ] Примеры
  - [ ] Базовый пример использования
  - [ ] Продвинутый пример с кастомизацией
  - [ ] Пример с реальными данными
  - [ ] Пример без визуализаций

- [ ] README.md
  - [ ] Описание проекта
  - [ ] Инструкция по установке
  - [ ] Quick start
  - [ ] Ссылки на документацию

### Фаза 11: Полировка и релиз

- [ ] Покрытие тестами > 80%
- [ ] Проверка type hints
- [ ] Проверка docstrings
- [ ] Бенчмарки производительности
- [ ] Оптимизация узких мест
- [ ] Финальный review кода
- [ ] Подготовка к релизу (CHANGELOG, версия)
- [ ] Публикация на PyPI

## Дополнительные идеи для будущих версий

- [ ] Поддержка больших данных (Dask, chunking)
- [ ] GPU ускорение (cuML)
- [ ] Автоматический feature engineering
- [ ] Интеграция с MLflow для tracking
- [ ] Web UI для интерактивного анализа
- [ ] Поддержка временных рядов в кластеризации
- [ ] Ensemble clustering методы
- [ ] Incremental clustering для streaming данных

## Технические детали

### Зависимости

**Обязательные:**
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

**Опциональные (viz extra):**
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

**Опциональные (extras):**
- umap-learn >= 0.5.0 (для UMAP)
- hdbscan >= 0.8.0 (для HDBSCAN)

**Разработка:**
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950

### Стандарты кода

- Python 3.8+
- Type hints везде
- Docstrings в формате NumPy/Google
- Black для форматирования
- flake8 для линтинга
- mypy для type checking
- pytest для тестов (coverage > 80%)

### CI/CD

- GitHub Actions для CI
- Автоматический запуск тестов на PR
- Проверка code style
- Coverage report
- Автоматический deploy на PyPI при тегах

## Приоритеты

**MUST HAVE для v1.0:**
- Фазы 1-7 (базовая функциональность без визуализаций)
- Основная документация

**SHOULD HAVE для v1.0:**
- Фаза 8 (визуализации)
- Полная документация

**NICE TO HAVE для v1.1+:**
- HTML отчёты
- Дополнительные алгоритмы
- Оптимизации производительности
