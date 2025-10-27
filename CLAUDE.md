# ClusterTK - Project Context for Claude Code

## Описание проекта

ClusterTK - это Python библиотека для полного пайплайна кластерного анализа, разработанная для аналитиков. Библиотека предоставляет удобный API где можно просто передать датафрейм и настроить каждый шаг через параметры, без написания большого количества кода.

**Основная цель:** Сделать кластерный анализ максимально удобным и доступным для аналитиков.

## Текущий статус проекта

### ✅ Реализовано (v0.1.0):

1. **Preprocessing** - полностью готово
   - MissingValueHandler - обработка пропусков (median/mean/drop/custom)
   - OutlierHandler - детекция и обработка выбросов (IQR/z-score/modified z-score)
   - ScalerSelector - автовыбор скейлера (Standard/Robust/MinMax)
   - SkewnessTransformer - log/sqrt/box-cox трансформации

2. **Feature Selection** - полностью готово
   - CorrelationFilter - удаление сильно коррелирующих признаков
   - VarianceFilter - удаление low-variance признаков

3. **Dimensionality Reduction** - полностью готово
   - PCAReducer - PCA с автоподбором компонент по variance threshold
   - ManifoldReducer - t-SNE/UMAP для визуализации (только для viz, не для кластеризации!)

4. **Clustering** - базовая реализация готова
   - BaseClusterer - базовый класс для всех алгоритмов
   - KMeansClustering - K-Means алгоритм
   - GMMClustering - Gaussian Mixture Model
   - ⚠️ TODO: HierarchicalClustering, DBSCANClustering

5. **Evaluation** - полностью готово
   - compute_clustering_metrics - Silhouette, Calinski-Harabasz, Davies-Bouldin
   - OptimalKFinder - автоподбор оптимального k с голосованием метрик

6. **Interpretation** - базовая реализация готова
   - ClusterProfiler - профилирование кластеров, топ-признаки, анализ по категориям
   - ⚠️ TODO: ClusterNamer - автоматическое именование кластеров

7. **Pipeline** - полностью готово
   - ClusterAnalysisPipeline - оркестрирует все шаги
   - Можно запускать как полный pipeline через .fit() так и пошагово

8. **Visualization** - НЕ реализовано
   - Структура модуля создана, но функции не реализованы
   - Это опциональная зависимость (pip install clustertk[viz])

### 📦 Публикация:
- ✅ GitHub: https://github.com/alexeiveselov92/clustertk
- ✅ PyPI: https://pypi.org/project/clustertk/ (v0.1.0)

## Важные архитектурные решения

### 1. Код на английском
**Весь код, комментарии и docstrings ТОЛЬКО на английском языке!**

### 2. Модульная архитектура
Каждый компонент - отдельный класс с `.fit()` и `.transform()` методами (sklearn-style).

Пример структуры:
```
clustertk/
├── preprocessing/       # 4 класса
├── feature_selection/   # 2 класса
├── dimensionality/      # 2 класса
├── clustering/          # базовый + 2 алгоритма (TODO: +2)
├── evaluation/          # метрики + optimal k finder
├── interpretation/      # профилирование (TODO: naming)
└── visualization/       # TODO: полностью
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
- Там есть дублирование логики
- Есть лишние штуки (например seo_segments)
- Может не хватать гибкости
- Используй его только как референс идей, но не копируй напрямую

## Стиль кода

1. **Type hints везде** - все параметры и возвращаемые значения типизированы
2. **Docstrings в NumPy/Google стиле** - для всех классов и публичных методов
3. **Verbose logging** - если `verbose=True`, выводим прогресс выполнения
4. **Единый интерфейс** - все классы следуют паттерну `.fit()`, `.transform()`, `.fit_transform()`

## Что нужно сделать дальше

Смотри актуальный TODO.md для подробного списка задач.

### Приоритеты:
1. **High priority:**
   - Реализовать HierarchicalClustering и DBSCANClustering
   - Реализовать visualization модуль (все plot функции)
   - Добавить ClusterNamer для автоматического именования

2. **Medium priority:**
   - Написать тесты (pytest) - покрытие >80%
   - Создать больше примеров и notebooks
   - HTML export с отчетами

3. **Low priority:**
   - Документация Sphinx
   - CI/CD (GitHub Actions)
   - Дополнительные алгоритмы (Spectral, HDBSCAN)

## Важные нюансы

### 1. Manifold методы (t-SNE, UMAP)
⚠️ **ВАЖНО:** t-SNE и UMAP используются ТОЛЬКО для визуализации, НЕ для кластеризации!
Они искажают расстояния и не подходят для clustering. Кластеризуем на PCA компонентах.

### 2. Нормализация для профилей
В ClusterProfiler используем нормализацию **per feature** (каждый признак отдельно 0-1),
чтобы признаки были сопоставимы на графиках.

### 3. Автоподбор скейлера
ScalerSelector автоматически выбирает между StandardScaler и RobustScaler:
- Если >5% выбросов → RobustScaler
- Иначе → StandardScaler

### 4. Оптимальное k
OptimalKFinder использует голосование трех метрик:
- Silhouette (выше = лучше)
- Calinski-Harabasz (выше = лучше)
- Davies-Bouldin (ниже = лучше)

## Контакты автора

**Author:** Aleksey Veselov
**Email:** alexei.veselov92@gmail.com
**GitHub:** https://github.com/alexeiveselov92

## Примечания для Claude Code

При работе с проектом:
1. Всегда следуй архитектуре и стилю существующего кода
2. Весь новый код на английском с docstrings
3. Добавляй type hints ко всем функциям
4. Используй verbose logging в Pipeline
5. Не забывай обновлять `__init__.py` при добавлении новых классов
6. Все visualization функции должны проверять `check_viz_available()`
