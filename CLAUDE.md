# ClusterTK - Project Context for Claude Code

## Описание проекта

ClusterTK - это Python библиотека для полного пайплайна кластерного анализа, разработанная для аналитиков. Библиотека предоставляет удобный API где можно просто передать датафрейм и настроить каждый шаг через параметры, без написания большого количества кода.

**Основная цель:** Сделать кластерный анализ максимально удобным и доступным для аналитиков.

## Текущий статус проекта

### 📦 Публикация:
- ✅ GitHub: https://github.com/alexeiveselov92/clustertk
- ✅ PyPI: https://pypi.org/project/clustertk/
- **Latest Version:** v0.14.1 (2025-10-30)
- **Recent Major Updates:**
  - v0.14.1 - Configurable report_top_features parameter (safe handling for small datasets)
  - v0.14.0 - Multivariate Outlier Detection + Enhanced HTML Reports
  - v0.13.0 - **BREAKING CHANGE**: Default `handle_outliers` changed from `'robust'` to `'winsorize'`
  - v0.12.1 - Winsorize: Percentile-based outlier handling (recommended for univariate outliers)
  - v0.12.0 - Algorithm Parameters & Noise Point Tracking
  - v0.11.1 - SHAP multidimensional array fix
  - v0.11.0 - Smart Feature Selection & Cluster Balance

### ✅ Полностью реализовано:

1. **Preprocessing** - полностью готово (v0.1.0, v0.11.0, v0.12.1, v0.13.0, v0.14.0)
   - MissingValueHandler - обработка пропусков (median/mean/drop/custom)
   - OutlierHandler - UNIVARIATE outlier handling:
     - Methods: IQR, z-score, modified z-score, percentile
     - Actions: clip, remove, nan, winsorize (v0.12.1)
     - **Winsorize: DEFAULT since v0.13.0** (Percentile 2.5%-97.5%, ~2-sigma)
     - Pipeline: `handle_outliers='winsorize'` теперь дефолт
   - MultivariateOutlierDetector - MULTIVARIATE outlier detection (v0.14.0):
     - Methods: IsolationForest, LOF, EllipticEnvelope
     - Auto method selection based on data characteristics
     - Detects outliers in full feature space (not per-feature)
     - Pipeline: `detect_multivariate_outliers='auto'`
   - ScalerSelector - автовыбор скейлера (Standard/Robust/MinMax)
   - SkewnessTransformer - log/sqrt/box-cox трансформации

2. **Feature Selection** - полностью готово (v0.1.0, v0.11.0)
   - CorrelationFilter - удаление сильно коррелирующих признаков
   - SmartCorrelationFilter - интеллектуальный выбор из коррелирующих пар (Hopkins statistic) (v0.11.0)
   - VarianceFilter - удаление low-variance признаков

3. **Dimensionality Reduction** - полностью готово (v0.1.0)
   - PCAReducer - PCA с автоподбором компонент по variance threshold
   - ManifoldReducer - t-SNE/UMAP для визуализации (только для viz, не для кластеризации!)

4. **Clustering** - полностью готово (v0.2.0, v0.8.0, v0.12.0)
   - BaseClusterer - базовый класс для всех алгоритмов
   - KMeansClustering - K-Means алгоритм
   - GMMClustering - Gaussian Mixture Model
   - HierarchicalClustering - иерархическая кластеризация (Ward, Complete, Average)
   - DBSCANClustering - DBSCAN с автоподбором eps и min_samples
   - HDBSCANClustering - HDBSCAN с автоподбором min_cluster_size (v0.8.0)
   - clustering_params - гибкая передача параметров любому алгоритму (v0.12.0)

5. **Evaluation** - полностью готово (v0.1.0, v0.11.0, v0.12.0)
   - compute_clustering_metrics - Silhouette, Calinski-Harabasz, Davies-Bouldin, Cluster Balance (v0.11.0)
   - Noise points tracking - n_noise, noise_ratio для DBSCAN/HDBSCAN (v0.12.0)
   - OptimalKFinder - автоподбор оптимального k с голосованием метрик

6. **Interpretation** - полностью готово (v0.3.0, v0.9.0, v0.12.0)
   - ClusterProfiler - профилирование кластеров (с фильтрацией noise points v0.12.0)
   - ClusterNamer - автоматическое именование кластеров (3 стратегии: top_features, categories, combined)
   - FeatureImportanceAnalyzer - permutation, contribution, SHAP (v0.9.0)
   - ClusterStabilityAnalyzer - bootstrap resampling для оценки стабильности (v0.9.0)

7. **Pipeline** - полностью готово (v0.1.0, v0.12.0)
   - ClusterAnalysisPipeline - оркестрирует все шаги
   - Можно запускать как полный pipeline через .fit() так и пошагово
   - clustering_params для передачи параметров алгоритмам (v0.12.0)

8. **Visualization** - полностью готово (v0.3.0)
   - 11 функций визуализации в 4 модулях
   - Все функции интегрированы в Pipeline через методы .plot_*()
   - Опциональная зависимость (pip install clustertk[viz])
   - Стандартное matplotlib поведение для полной совместимости

9. **Export & Reports** - полностью готово (v0.5.0)
   - `export_results()` - экспорт в CSV (данные + labels) и JSON (метаданные + профили + метрики)
   - `export_report()` - HTML отчёты с embedded plots (base64)
   - `save_pipeline()` / `load_pipeline()` - сохранение/загрузка fitted pipeline через joblib
   - Добавлена зависимость joblib>=1.0.0

10. **Algorithm Comparison** - полностью готово (v0.7.0)
   - `compare_algorithms()` метод для автоматического сравнения алгоритмов
   - Тестирование KMeans, GMM, Hierarchical, DBSCAN, HDBSCAN на разных k
   - Weighted scoring system (40% Silhouette, 30% Calinski-Harabasz, 30% Davies-Bouldin)
   - `plot_algorithm_comparison()` визуализация сравнения
   - Автоматическая рекомендация лучшего алгоритма
   - Полная документация и примеры

11. **Test Suite** - полностью готово (v0.8.0, v0.14.0)
   - pytest infrastructure с pytest.ini конфигурацией
   - 62 unit и integration тестов (39 + 23 for multivariate outliers)
   - Coverage: preprocessing 85% (multivariate_outliers), overall ~5-40%
   - Fixtures для различных сценариев данных
   - Тесты для preprocessing, clustering, evaluation, pipeline

### ⚠️ TODO (для будущих версий):

**v0.15.0+ (приоритет MEDIUM/LOW):**
- **Enhanced Coverage** - увеличить покрытие тестами до >50%
- **CI/CD** - GitHub Actions для автоматического тестирования
- **More Clustering Algorithms** - Spectral Clustering, OPTICS
- **Sphinx** - полная API документация
- **GitHub Pages** - хостинг документации

## Важные архитектурные решения

### 1. Код на английском
**Весь код, комментарии и docstrings ТОЛЬКО на английском языке!**

### 2. Модульная архитектура
Каждый компонент - отдельный класс с `.fit()` и `.transform()` методами (sklearn-style).

```
clustertk/
├── preprocessing/       # 4 класса - готово
├── feature_selection/   # 2 класса - готово
├── dimensionality/      # 2 класса - готово
├── clustering/          # базовый + 4 алгоритма - готово
├── evaluation/          # метрики + optimal k finder - готово
├── interpretation/      # профилирование + naming - готово
└── visualization/       # 11 функций в 4 модулях - готово
```

### 3. Опциональные визуализации
Matplotlib и Seaborn - тяжелые зависимости, поэтому они опциональны:
- Базовая установка: `pip install clustertk` (БЕЗ viz)
- С визуализациями: `pip install clustertk[viz]`

### 4. Промежуточные результаты
Pipeline сохраняет все промежуточные результаты в атрибутах с постфиксом `_`:
- `data_preprocessed_`, `data_scaled_`, `data_reduced_`, `labels_`, `cluster_profiles_`, etc.

## Референсный код

В папке `for_developing/for_developing.py` есть исходный исследовательский код.

**ВАЖНО:**
- Это НЕ истина в последней инстанции
- Там есть дублирование логики и лишние штуки
- Используй его только как референс идей, но не копируй напрямую

## Стиль кода

1. **Type hints везде** - все параметры и возвращаемые значения типизированы
2. **Docstrings в NumPy/Google стиле** - для всех классов и публичных методов
3. **Verbose logging** - если `verbose=True`, выводим прогресс выполнения
4. **Единый интерфейс** - все классы следуют паттерну `.fit()`, `.transform()`, `.fit_transform()`

## Важные нюансы и решенные проблемы

### 1. Manifold методы (t-SNE, UMAP)
⚠️ **ВАЖНО:** t-SNE и UMAP используются ТОЛЬКО для визуализации, НЕ для кластеризации!
Они искажают расстояния и не подходят для clustering. Кластеризуем на PCA компонентах.

### 2. Нормализация профилей кластеров
В ClusterProfiler используется нормализация **per feature** (каждый признак отдельно 0-1),
чтобы признаки были сопоставимы на графиках.

**КРИТИЧЕСКИЙ НЮАНС (исправлено в v0.3.2):**
При малом количестве кластеров (особенно 2) min-max нормализация по кластерам давала
бесполезные результаты (все значения = 0 или 1).

**Решение:** Нормализация использует min/max из **исходных данных**, а не из средних кластеров:
```python
# ❌ НЕПРАВИЛЬНО (v0.3.1 и ранее):
col_min = profiles[col].min()  # Min из 2 значений кластеров
col_max = profiles[col].max()  # Max из 2 значений кластеров

# ✅ ПРАВИЛЬНО (v0.3.2+):
col_min = X[col].min()  # Min из исходных данных
col_max = X[col].max()  # Max из исходных данных
```

### 3. Проблема дублирования графиков в Jupyter (решено в v0.4.1)

**Техническая причина дублирования:**
- `plt.subplots()` регистрирует figure в глобальном состоянии pyplot
- В Jupyter это вызывает автоматическое отображение
- Когда функция возвращает figure, Jupyter отображает его СНОВА
- Результат: дубликат графика

**Почему у seaborn нет этой проблемы:**
- Seaborn функции возвращают `Axes` объект, НЕ `Figure`
- Axes не вызывают auto-display в Jupyter
- Только Figure объекты auto-display

**Наше решение:**
```python
def _prepare_figure_return(fig: plt.Figure) -> plt.Figure:
    plt.close(fig)  # Удаляет figure из pyplot state
    return fig      # Но figure остается рабочим!
```

`plt.close(fig)` удаляет figure из pyplot state, предотвращая первое отображение,
но figure объект остается полностью функциональным - его можно отобразить, сохранить, изменить.

**Использование:**
```python
# Один график - отобразится автоматически
pipeline.plot_cluster_heatmap()

# Несколько графиков в одной ячейке - только последний отобразится
# (стандартное поведение Python/Jupyter для функций, возвращающих объекты)
# Используй display() или отдельные ячейки:
from IPython.display import display

display(pipeline.plot_cluster_heatmap())
display(pipeline.plot_clusters_2d())
display(pipeline.plot_cluster_radar())

# Или сохрани для дальнейшей работы
fig = pipeline.plot_clusters_2d()
fig.savefig('clusters.png')
```

### 4. Автоподбор скейлера
ScalerSelector автоматически выбирает между StandardScaler и RobustScaler:
- Если >5% выбросов → RobustScaler
- Иначе → StandardScaler

### 5. Оптимальное k
OptimalKFinder использует голосование трех метрик:
- Silhouette (выше = лучше)
- Calinski-Harabasz (выше = лучше)
- Davies-Bouldin (ниже = лучше)

### 6. Winsorization для univariate outliers (v0.12.1)

**Проблема с IQR и clip:**
- IQR threshold=1.5 слишком слабый для экстремальных выбросов (10-50x)
- При `action='clip'` все экстремальные значения обрезаются до одной границы
- Пример: revenue=[100, 150, ..., 10000, 12000, 15000] → все три → 875 (одно значение!)
- Результат: артефакты, потеря информации

**Решение - Winsorize:**
```python
pipeline = ClusterAnalysisPipeline(
    handle_outliers='winsorize',  # Recommended!
    # percentile_limits=(0.025, 0.975) - по умолчанию
)
```

**Преимущества:**
- Distribution-agnostic (работает с любыми распределениями)
- Percentile-based clipping (default 2.5%-97.5% = ~2-sigma)
- Нет артефактов (экстремальные значения обрезаются до разных перцентилей)
- Нет потери данных (rows сохраняются)
- Хорошо работает с асимметричными outliers

**Важно:** Winsorize решает UNIVARIATE outliers (per-feature). Для MULTIVARIATE outliers (выбросы в многомерном пространстве) нужен MultivariateOutlierDetector (планируется v0.14.0).

**КРИТИЧНО: Почему StandardScaler/RobustScaler НЕ решают проблему выбросов?**

Частый вопрос: "Мы же применяем scaling перед кластеризацией, разве StandardScaler не нормализует выбросы?"

**НЕТ!** StandardScaler и RobustScaler НЕ удаляют и НЕ уменьшают выбросы:

Пример: revenue = [100, 150, 200, 180, 220, 190, 10000, 12000, 15000]

```python
# StandardScaler: (x - mean) / std
# mean = 4226.7, std = 6208.9 (искажены выбросами!)
# Результат:
#   Normal: -0.705 to -0.684 (range: 0.021)
#   Outliers: 0.986 to 1.840 (range: 0.854)
# Выбросы все еще в 40x-80x раз дальше от нормальных!

# RobustScaler: (x - median) / IQR
# median = 200, IQR = 9820
# Результат:
#   Normal: -0.010 to 0.002 (range: 0.012)
#   Outliers: 0.998 to 1.507 (range: 0.509)
# Выбросы все еще в 80x-120x раз дальше!
```

**Почему это проблема для K-Means:**
- Euclidean distance между normal точками: ~0.01-0.02
- Euclidean distance от normal до outlier: ~1.0-1.5
- K-Means видит: "куча точек близко" + "3 точки очень далеко"
- Результат: 1 большой кластер (90%) + 2-3 маленьких кластера (по 1-2%)
- Silhouette score высокий (~0.95), но кластеризация бесполезна!

**Решение: Winsorize ДО scaling**

Execution order в Pipeline:
1. **Winsorize** → clips outliers to 2.5%-97.5% percentiles
2. **StandardScaler/RobustScaler** → normalizes scale
3. **K-Means** → корректная кластеризация на адекватных данных

```python
# ✅ ПРАВИЛЬНО (v0.12.1+):
pipeline = ClusterAnalysisPipeline(
    handle_outliers='winsorize',  # Обрезаем ПЕРЕД scaling
    scaling='robust'
)

# ❌ НЕПРАВИЛЬНО (старый default):
pipeline = ClusterAnalysisPipeline(
    handle_outliers='robust',  # Только RobustScaler, выбросы остаются!
)
```

### 7. Multivariate vs Univariate Outliers (v0.14.0)

**КРИТИЧЕСКИЙ НЮАНС:** Есть ДВА типа выбросов - univariate и multivariate!

**Univariate Outliers (per-feature extremes):**
- Экстремальные значения ПО ОТДЕЛЬНОМУ признаку
- Пример: revenue=10000 при средних ~200
- Решение: Winsorize (default v0.13.0) - обрезка по перцентилям
- Применяется ПЕРЕД scaling

**Multivariate Outliers (full-space outliers):**
- Точки далеко от всех кластеров в МНОГОМЕРНОМ пространстве
- ПО ОТДЕЛЬНОСТИ каждый признак выглядит нормально!
- Применяется ПОСЛЕ scaling

**Проблема без multivariate detection:**
```python
# Data: 3 normal clusters + 3 multivariate outliers
# Feature1: все в диапазоне [-5, 15] - НЕТ univariate outliers
# Feature2: все в диапазоне [-5, 15] - НЕТ univariate outliers
#
# Но точки (15, 15), (-5, 10), (5, -5) далеко от всех кластеров!
# K-Means результат: silhouette=0.746, но кластеры искажены outliers
```

**Решение - MultivariateOutlierDetector (v0.14.0):**
```python
pipeline = ClusterAnalysisPipeline(
    # Step 1: Univariate outliers (per-feature extremes)
    handle_outliers='winsorize',           # Clips to 2.5%-97.5% percentiles

    # Step 2: Multivariate outliers (full-space outliers)
    detect_multivariate_outliers='auto',   # NEW v0.14.0!
    multivariate_contamination=0.05,       # Expected outlier ratio
    multivariate_action='remove',          # Remove outliers (default)
)

# After multivariate detection: silhouette=0.772 (+3.5% improvement)
```

**Auto method selection:**
- n_samples < 100: LOF (better for small datasets)
- n_features < 5: LOF (better for low-dimensional varying density)
- n_features >= 10: IsolationForest (better for high-dimensional, faster)
- Gaussian data: EllipticEnvelope (если данные нормальные)

**Execution order (ПРАВИЛЬНЫЙ!):**
```
1. Missing values → handle NaN
2. Log transform → normalize skewness (optional)
3. Winsorize → clip UNIVARIATE outliers (per-feature extremes)
4. Scaling → normalize scale
5. MultivariateOutlierDetector → detect MULTIVARIATE outliers (full space) ← NEW!
6. PCA → dimensionality reduction
7. K-Means → clustering
```

**Когда использовать:**
- ✅ Always use `detect_multivariate_outliers='auto'` для production
- ✅ Особенно важно при большом количестве признаков (>5)
- ✅ Когда видишь 1 огромный кластер + маленькие кластеры
- ⚠️ Для clean academic datasets можно отключить (None)

### 8. Log-трансформация и выбросы

**Вопрос:** Решает ли log-трансформация проблему с выбросами?

**Короткий ответ: НЕТ, log только СЖИМАЕТ выбросы, но не удаляет их.**

**Что делает log:**
```python
# Пример: revenue = [100, 200, 300, 10000]
# После log1p:
#   100 → 4.62
#   200 → 5.30
#   300 → 5.71
#   10000 → 9.21
#
# Разница БЕЗ log: 10000/200 = 50x
# Разница С log: 9.21/5.30 = 1.7x (лучше, но выброс остается!)
```

**Когда log ПОЛЕЗЕН:**
- Скошенные данные БЕЗ экстремальных выбросов (skewness > 2.0)
- Log-normal распределения → становятся нормальными
- Помогает K-Means лучше работать с расстояниями

**Когда log ВРЕДЕН:**
- Нормальные данные (skewness < 1.0)
- Искажает естественное распределение
- Ухудшает качество кластеризации

**Execution order в Pipeline (правильный!):**
```
1. Missing values → handle NaN
2. Log transform → normalize skewness (if log_transform_skewed=True)
3. Winsorize → clip remaining outliers to percentiles
4. Scaling → normalize scale
5. K-Means → clustering on clean data
```

**Рекомендация:**
```python
# Для скошенных данных с выбросами:
pipeline = ClusterAnalysisPipeline(
    log_transform_skewed=True,       # Нормализует skewness
    skewness_threshold=2.0,          # Default
    handle_outliers='winsorize',     # Обрезает выбросы (дефолт v0.13.0)
)

# Для нормальных данных с выбросами:
pipeline = ClusterAnalysisPipeline(
    log_transform_skewed=False,      # НЕ применять log! (default)
    handle_outliers='winsorize',     # Только winsorize (default)
)
```

## История релизов

- **v0.1.0** (первый релиз) - базовый pipeline без DBSCAN, Hierarchical, visualization, naming
- **v0.1.1** - hotfix для документации
- **v0.2.0** - добавлены DBSCAN и HierarchicalClustering
- **v0.3.0** - добавлены visualization (10 функций) и ClusterNamer
- **v0.3.1** - hotfix README и minor fixes
- **v0.3.2** - исправлена нормализация для малого числа кластеров
- **v0.3.3-v0.3.5** - улучшения в visualization модуле
- **v0.4.0** - попытка исправить дублирование (неудачная)
- **v0.4.1** - правильное решение дублирования через plt.close(fig)
- **v0.4.2** - обновлен Quick Start пример + улучшена навигация в README
- **v0.5.0** - добавлены export_results(), export_report(), save_pipeline(), load_pipeline()
- **v0.6.0** - создана docs/ структура, сокращен README, полная документация
- **v0.7.0** - добавлен compare_algorithms() для автоматического сравнения алгоритмов
- **v0.8.0** - добавлены HDBSCAN алгоритм и полный Test Suite (39 тестов, 39% coverage)
- **v0.9.0** - Feature Importance & Stability Analysis (permutation, SHAP, contribution, bootstrap stability)
- **v0.10.0** - MAJOR OPTIMIZATION: Stability Analysis для больших датасетов
  - Полная переработка ClusterStabilityAnalyzer с streaming computation
  - Memory: 32+ GB OOM → <500 MB (64x reduction)
  - Speed: OOM → 6 seconds for 80k samples (100-1000x speedup)
  - Sliding window approach, vectorized operations, adaptive sampling
- **v0.10.1** - CRITICAL FIX: Feature Importance memory issues на больших датасетах
  - Permutation importance OOM fix: silhouette sampling для >10k samples
  - Memory: 51+ GB OOM → ~2 GB (25x reduction)
  - Speed: OOM → 20 seconds for 80k samples
  - Feature contribution vectorization: 10x speedup
- **v0.10.2** - True NumPy Vectorization для Feature Contribution
  - Replaced pandas groupby with pure NumPy bincount
  - Performance: 1.23x faster (0.0165s → 0.0134s on 80k samples)
  - True vectorization without hidden loops or pandas overhead
- **v0.11.0** - Smart Feature Selection & Cluster Balance
  - SmartCorrelationFilter: Hopkins statistic-based feature selection
  - ClusterBalancer: min_cluster_size enforcement for quality control
- **v0.11.1** - SHAP fix для multidimensional arrays
- **v0.12.0** - Algorithm Parameters & Noise Point Tracking
  - Exposed algorithm parameters in Pipeline (kmeans_params, dbscan_params, etc.)
  - ClusterProfiler tracks noise points (n_noise_, noise_ratio_)
- **v0.12.1** - Winsorize: Percentile-based Outlier Handling (RECOMMENDED)
  - New 'winsorize' action for OutlierHandler (distribution-agnostic)
  - Percentile-based clipping (default 2.5%-97.5%, ~2-sigma)
  - Solves: IQR artifacts (multiple extreme values → same clipped value)
  - Pipeline: handle_outliers='winsorize' now available
  - Better than 'clip' for extreme/asymmetric outliers
- **v0.13.0** - **BREAKING CHANGE**: Winsorize is Now Default
  - Changed default `handle_outliers` from `'robust'` to `'winsorize'`
  - **Why**: RobustScaler doesn't remove outliers, they remain far away after scaling
  - Problem solved: K-Means creating 1 huge cluster (90%+) + tiny outlier clusters
  - Execution order: Winsorize → Scaling → Clustering (correct!)
  - Documentation: Updated all examples and user guide
  - Migration: If you want old behavior, explicitly set `handle_outliers='robust'`
- **v0.14.0** - Multivariate Outlier Detection
  - NEW: MultivariateOutlierDetector class with 3 methods (IsolationForest, LOF, EllipticEnvelope)
  - Auto method selection based on data characteristics (n_samples, n_features, distribution)
  - Integrated into Pipeline: `detect_multivariate_outliers='auto'`
  - Detects outliers in FULL feature space (not per-feature like Winsorize)
  - Execution order: Winsorize → Scaling → Multivariate Detection → PCA → Clustering
  - Benefits: +3-5% silhouette improvement, prevents tiny outlier clusters
  - Tests: 23 comprehensive unit tests, 85% coverage
  - Two types of outliers now handled: Univariate (per-feature) + Multivariate (full-space)
  - Configurable contamination rate and action (remove/flag)
  - Documentation: Added section 7 in CLAUDE.md explaining univariate vs multivariate outliers

## Контакты автора

**Author:** Aleksey Veselov
**Email:** alexei.veselov92@gmail.com
**GitHub:** https://github.com/alexeiveselov92

## Примечания для Claude Code

При работе с проектом:

### 🧹 Чистота корня проекта (КРИТИЧНО!)
1. **НЕ создавай временные файлы в корне проекта!**
   - Примеры: CHECKLIST_*.md, FINAL_STEPS_*.md, PUBLISH_*.md, NEXT_SESSION.md, etc.
   - Все инструкции по публикации/релизам должны быть в TODO.md
   - Временные скрипты (PUBLISH_NOW.sh и т.п.) - НЕ создавать!

2. **В корне должны быть ТОЛЬКО:**
   - Documentation: README.md, ARCHITECTURE.md, CLAUDE.md, TODO.md, CHANGELOG.md
   - Configuration: setup.py, pyproject.toml, pytest.ini
   - Requirements: requirements*.txt
   - Directories: clustertk/, tests/, docs/, examples/

3. **Планирование и TODO:**
   - ВСЕ планы и задачи пишем ТОЛЬКО в TODO.md
   - TODO.md - это single source of truth для планирования
   - НЕ создавай отдельные planning files!

### 💻 Код и архитектура
4. Всегда следуй архитектуре и стилю существующего кода
5. Весь новый код на английском с docstrings
6. Добавляй type hints ко всем функциям
7. Используй verbose logging в Pipeline
8. Не забывай обновлять `__init__.py` при добавлении новых классов
9. Все visualization функции должны:
   - Проверять `_check_viz_available()` в начале
   - Возвращать matplotlib Figure объект
   - Использовать стандартное matplotlib поведение (автоматический вывод в Jupyter)
10. При работе с нормализацией профилей - использовать min/max из исходных данных
11. Manifold методы (t-SNE, UMAP) - только для визуализации, не для кластеризации!

10. **Documentation Structure** - полностью готово (v0.6.0)
   - Создана docs/ директория с полной структурой
   - Сокращён README с 495 до 196 строк
   - Созданы 8 разделов User Guide + installation, quickstart, examples, FAQ
   - README теперь только Quick Start + ссылки на docs/
