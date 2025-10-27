# ClusterTK - Архитектура и дизайн-решения

## Обзор

ClusterTK построен на основе модульной архитектуры с единой точкой входа через класс `ClusterAnalysisPipeline`. Библиотека следует принципам:

- **Separation of Concerns**: каждый модуль отвечает за свою часть функциональности
- **Composition over Inheritance**: Pipeline композирует модули, а не наследует их
- **Optional Dependencies**: визуализации не являются обязательными
- **Scikit-learn compatibility**: API схож с sklearn для удобства пользователей

## Детальная архитектура

### 1. ClusterAnalysisPipeline - главный класс

Это единая точка входа для пользователей. Класс оркеструет работу всех модулей.

```python
class ClusterAnalysisPipeline:
    """
    Главный класс для полного пайплайна кластерного анализа.

    Attributes:
        data_ : pd.DataFrame
            Исходные данные
        data_preprocessed_ : pd.DataFrame
            Предобработанные данные
        data_scaled_ : pd.DataFrame
            Нормализованные данные
        data_reduced_ : pd.DataFrame
            Данные после PCA
        labels_ : np.ndarray
            Метки кластеров
        cluster_profiles_ : dict
            Профили кластеров
        metrics_ : dict
            Метрики качества
    """

    def __init__(
        self,
        # Preprocessing
        handle_missing='median',
        handle_outliers='robust',
        scaling='robust',
        log_transform_skewed=False,
        skewness_threshold=2.0,

        # Feature selection
        correlation_threshold=0.85,
        variance_threshold=0.01,

        # Dimensionality reduction
        pca_variance=0.9,
        pca_min_components=2,

        # Clustering
        clustering_algorithm='kmeans',
        n_clusters=None,
        n_clusters_range=(2, 10),

        # General
        random_state=42,
        verbose=True
    ):
        ...

    def fit(self, X, feature_columns=None, category_mapping=None):
        """Запускает весь пайплайн"""
        self.preprocess(X, feature_columns)
        self.select_features()
        self.reduce_dimensions()
        self.find_optimal_clusters()
        self.cluster()
        self.create_profiles(category_mapping)
        return self

    def preprocess(self, X, feature_columns=None):
        """Предобработка данных"""
        ...

    def select_features(self):
        """Отбор признаков"""
        ...

    def reduce_dimensions(self):
        """Снижение размерности"""
        ...

    def find_optimal_clusters(self):
        """Поиск оптимального числа кластеров"""
        ...

    def cluster(self, n_clusters=None, algorithm=None):
        """Кластеризация"""
        ...

    def create_profiles(self, category_mapping=None):
        """Создание профилей кластеров"""
        ...

    def export_results(self, path):
        """Экспорт результатов"""
        ...
```

### 2. Модуль preprocessing

Отвечает за предобработку данных. Каждый компонент - отдельный класс.

```python
# preprocessing/missing.py
class MissingValueHandler:
    """
    Обработка пропущенных значений.

    Parameters:
        strategy : str or callable
            'median', 'mean', 'drop', или функция
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy

    def fit(self, X):
        """Вычисляет параметры заполнения"""
        ...

    def transform(self, X):
        """Применяет заполнение"""
        ...

    def fit_transform(self, X):
        """Fit + transform"""
        ...


# preprocessing/outliers.py
class OutlierHandler:
    """
    Детекция и обработка выбросов.

    Parameters:
        method : str
            'iqr', 'zscore', 'isolation_forest'
        action : str
            'clip', 'remove', 'robust_scale'
        threshold : float
            Порог для детекции
    """
    def __init__(self, method='iqr', action='robust_scale', threshold=1.5):
        ...

    def detect_outliers(self, X):
        """Возвращает маску выбросов"""
        ...

    def handle_outliers(self, X):
        """Обрабатывает выбросы согласно action"""
        ...


# preprocessing/scaling.py
class ScalerSelector:
    """
    Автоматический выбор scaler на основе данных.

    Если много выбросов (>5% по IQR) -> RobustScaler
    Иначе -> StandardScaler
    """
    def __init__(self, auto=True, scaler_type='auto'):
        ...

    def fit(self, X):
        """Выбирает и обучает scaler"""
        ...

    def transform(self, X):
        """Применяет scaling"""
        ...


# preprocessing/transforms.py
class SkewnessTransformer:
    """
    Трансформация скошенных распределений.

    Автоматически детектирует скошенность и применяет
    log-трансформацию с обработкой отрицательных значений.
    """
    def __init__(self, threshold=2.0, method='log1p'):
        ...

    def detect_skewed_features(self, X):
        """Находит скошенные признаки"""
        ...

    def transform(self, X):
        """Применяет трансформацию"""
        ...
```

### 3. Модуль feature_selection

```python
# feature_selection/correlation.py
class CorrelationFilter:
    """
    Фильтрация сильно коррелирующих признаков.

    Удаляет один из пары признаков, если |корреляция| > threshold.
    """
    def __init__(self, threshold=0.85, method='pearson'):
        ...

    def fit(self, X):
        """Вычисляет корреляционную матрицу и определяет признаки для удаления"""
        ...

    def transform(self, X):
        """Удаляет сильно коррелирующие признаки"""
        ...

    def get_correlation_matrix(self):
        """Возвращает корреляционную матрицу"""
        ...

    def get_high_correlation_pairs(self):
        """Возвращает пары с высокой корреляцией"""
        ...


# feature_selection/variance.py
class VarianceFilter:
    """
    Фильтрация признаков с низкой дисперсией.
    """
    def __init__(self, threshold=0.01):
        ...

    def fit(self, X):
        """Вычисляет дисперсии"""
        ...

    def transform(self, X):
        """Удаляет признаки с низкой дисперсией"""
        ...
```

### 4. Модуль dimensionality

```python
# dimensionality/pca.py
class PCAReducer:
    """
    PCA с автоматическим подбором числа компонент.

    Parameters:
        variance_threshold : float
            Минимальная доля объяснённой дисперсии
        min_components : int
            Минимальное число компонент (для визуализации)
    """
    def __init__(self, variance_threshold=0.9, min_components=2, random_state=42):
        ...

    def fit(self, X):
        """Обучает PCA и определяет число компонент"""
        ...

    def transform(self, X):
        """Применяет PCA"""
        ...

    def get_n_components(self):
        """Возвращает выбранное число компонент"""
        ...

    def get_explained_variance(self):
        """Возвращает объяснённую дисперсию"""
        ...

    def get_loadings(self):
        """Возвращает loadings (веса признаков в компонентах)"""
        ...


# dimensionality/manifold.py
class ManifoldReducer:
    """
    t-SNE или UMAP для визуализации в 2D.

    Используется ТОЛЬКО для визуализации, не для кластеризации!
    """
    def __init__(self, method='tsne', n_components=2, random_state=42, **kwargs):
        ...

    def fit_transform(self, X):
        """Снижает размерность для визуализации"""
        ...
```

### 5. Модуль clustering

```python
# clustering/base.py
class BaseClusterer:
    """Базовый класс для всех кластеризаторов"""
    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X):
        return self.fit(X).predict(X)


# clustering/kmeans.py
class KMeansClustering(BaseClusterer):
    """K-Means кластеризация"""
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        ...


# clustering/gmm.py
class GMMClustering(BaseClusterer):
    """Gaussian Mixture Model кластеризация"""
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        ...


# clustering/hierarchical.py
class HierarchicalClustering(BaseClusterer):
    """Иерархическая кластеризация"""
    def __init__(self, n_clusters=3, linkage='ward', **kwargs):
        ...


# clustering/dbscan.py
class DBSCANClustering(BaseClusterer):
    """DBSCAN кластеризация"""
    def __init__(self, eps='auto', min_samples='auto', **kwargs):
        ...
```

### 6. Модуль evaluation

```python
# evaluation/metrics.py
def compute_clustering_metrics(X, labels):
    """
    Вычисляет все метрики качества кластеризации.

    Returns:
        dict: {
            'silhouette': float,
            'calinski_harabasz': float,
            'davies_bouldin': float
        }
    """
    ...


# evaluation/optimal_k.py
class OptimalKFinder:
    """
    Поиск оптимального числа кластеров.

    Тестирует разные значения k и применяет различные метрики.
    """
    def __init__(self, k_range=(2, 10), method='voting', random_state=42):
        ...

    def find_optimal_k(self, X, clusterer_class):
        """
        Находит оптимальное k.

        Returns:
            int: рекомендуемое число кластеров
            dict: метрики для всех k
        """
        ...
```

### 7. Модуль interpretation

```python
# interpretation/profiles.py
class ClusterProfiler:
    """
    Создание профилей кластеров.
    """
    def __init__(self, normalize_per_feature=True):
        ...

    def create_profiles(self, X, labels, feature_names=None):
        """
        Создаёт профили кластеров.

        Returns:
            pd.DataFrame: средние значения признаков по кластерам
        """
        ...

    def get_top_features(self, n=5):
        """
        Возвращает топ-N отличительных признаков для каждого кластера.
        """
        ...

    def analyze_by_categories(self, X, labels, category_mapping):
        """
        Анализ по категориям признаков (например, behavioral, social, etc.)

        Parameters:
            category_mapping : dict
                {'category_name': ['feature1', 'feature2', ...]}
        """
        ...


# interpretation/naming.py
class ClusterNamer:
    """
    Автоматическое предложение имён для кластеров.
    """
    def __init__(self, strategy='rule_based'):
        ...

    def suggest_names(self, profiles, category_analysis):
        """
        Предлагает названия на основе профилей.

        Returns:
            dict: {cluster_id: suggested_name}
        """
        ...
```

### 8. Модуль visualization (ОПЦИОНАЛЬНЫЙ)

```python
# visualization/base.py
def check_viz_available():
    """Проверяет, установлены ли viz dependencies"""
    try:
        import matplotlib
        import seaborn
        return True
    except ImportError:
        return False


# visualization/correlation.py
def plot_correlation_matrix(corr_matrix, save_path=None, **kwargs):
    """Heatmap корреляционной матрицы"""
    ...


# visualization/clusters.py
def plot_clusters_2d(X_2d, labels, method_name='', save_path=None, **kwargs):
    """2D scatter plot кластеров (t-SNE/UMAP)"""
    ...


# visualization/profiles.py
def plot_cluster_heatmap(profiles, save_path=None, **kwargs):
    """Heatmap профилей кластеров"""
    ...

def plot_cluster_radar_charts(profiles, save_path=None, **kwargs):
    """Radar charts для кластеров"""
    ...
```

## Дизайн-решения

### 1. Почему визуализации опциональны?

**Проблема**: matplotlib и seaborn тяжёлые зависимости, не нужные для основной функциональности.

**Решение**:
- Основная библиотека работает без них
- Визуализации устанавливаются через `pip install clustertk[viz]`
- При вызове методов визуализации проверяется наличие зависимостей

```python
def plot_clusters_2d(self):
    if not check_viz_available():
        raise ImportError(
            "Visualization dependencies not installed. "
            "Install with: pip install clustertk[viz]"
        )
    ...
```

### 2. Почему отдельные классы для каждого компонента?

**Проблема**: Монолитные функции сложно тестировать и расширять.

**Решение**:
- Каждый компонент - отдельный класс с `.fit()` и `.transform()`
- Единый интерфейс похожий на sklearn
- Легко добавлять новые методы
- Удобно тестировать изолированно

### 3. Как хранить промежуточные результаты?

**Проблема**: Пользователь может захотеть посмотреть результаты промежуточных шагов.

**Решение**:
- Все результаты сохраняются в атрибутах Pipeline с постфиксом `_`
- `data_preprocessed_`, `data_scaled_`, `data_reduced_`, `labels_`, etc.
- Пользователь может получить доступ после каждого шага

```python
pipeline.preprocess(df)
print(pipeline.data_preprocessed_)  # Доступ к промежуточному результату
```

### 4. Как обеспечить гибкость?

**Проблема**: Пользователи могут хотеть кастомные функции обработки.

**Решение**:
- Параметры могут быть строками или callable
- Пример: `handle_missing='median'` или `handle_missing=custom_function`

```python
def my_custom_imputer(df):
    # Своя логика
    return df.fillna(0)

pipeline = ClusterAnalysisPipeline(handle_missing=my_custom_imputer)
```

### 5. Как handle различные алгоритмы кластеризации?

**Проблема**: Разные алгоритмы имеют разные параметры.

**Решение**:
- Базовый класс `BaseClusterer` с единым интерфейсом
- Pipeline принимает как строку, так и экземпляр класса
- Дополнительные параметры через `**kwargs`

```python
# Простой способ
pipeline = ClusterAnalysisPipeline(clustering_algorithm='kmeans')

# Продвинутый способ
from clustertk.clustering import KMeansClustering
custom_kmeans = KMeansClustering(n_clusters=5, init='random')
pipeline = ClusterAnalysisPipeline(clustering_algorithm=custom_kmeans)
```

### 6. Обработка ошибок и валидация

**Принципы**:
- Ранняя валидация в `__init__` и в начале методов
- Понятные сообщения об ошибках
- Предупреждения (warnings) для некритичных ситуаций

```python
def preprocess(self, X, feature_columns=None):
    # Валидация входных данных
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if feature_columns is None:
        warnings.warn(
            "feature_columns not specified, using all numeric columns"
        )
        feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    # Проверка наличия колонок
    missing_cols = set(feature_columns) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in X: {missing_cols}")

    ...
```

### 7. Verbose режим

**Проблема**: Пользователь хочет видеть прогресс длительных операций.

**Решение**:
- Параметр `verbose` в Pipeline
- Использование logging или print для вывода прогресса
- Прогресс-бары для долгих операций (tqdm)

```python
if self.verbose:
    print("Step 1/6: Preprocessing...")
    print(f"  - Handling missing values: {self.handle_missing}")
    print(f"  - Handling outliers: {self.handle_outliers}")
```

### 8. Экспорт результатов

**Решение**:
- CSV с метками кластеров и исходными ID
- JSON с профилями кластеров
- HTML отчёт со всей информацией (если viz установлен)

```python
def export_results(self, path, format='csv', include_original=True):
    """
    Экспорт результатов.

    Parameters:
        path : str
            Путь для сохранения
        format : str
            'csv', 'json', 'html'
        include_original : bool
            Включить исходные данные в экспорт
    """
    ...
```

## Тестирование

### Стратегия тестирования

1. **Unit tests**: каждый класс тестируется изолированно
2. **Integration tests**: тестирование Pipeline целиком
3. **End-to-end tests**: примеры использования как тесты

### Структура тестов

```
tests/
├── test_preprocessing/
│   ├── test_missing.py
│   ├── test_outliers.py
│   ├── test_scaling.py
│   └── test_transforms.py
├── test_feature_selection/
│   └── test_correlation.py
├── test_dimensionality/
│   └── test_pca.py
├── test_clustering/
│   ├── test_kmeans.py
│   └── test_gmm.py
├── test_evaluation/
│   └── test_metrics.py
├── test_interpretation/
│   └── test_profiles.py
├── test_visualization/  (опционально)
│   └── test_plots.py
└── test_pipeline.py  (интеграционные тесты)
```

### Пример теста

```python
# tests/test_preprocessing/test_missing.py
import pytest
import pandas as pd
import numpy as np
from clustertk.preprocessing import MissingValueHandler


def test_missing_value_handler_median():
    # Arrange
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, np.nan, 7, 8]
    })
    handler = MissingValueHandler(strategy='median')

    # Act
    handler.fit(df)
    result = handler.transform(df)

    # Assert
    assert result.isna().sum().sum() == 0
    assert result['a'].iloc[2] == 2.0  # median of [1, 2, 4]
    assert result['b'].iloc[1] == 6.5  # median of [5, 7, 8]


def test_missing_value_handler_custom():
    # Arrange
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
    })

    def custom_imputer(series):
        return series.fillna(999)

    handler = MissingValueHandler(strategy=custom_imputer)

    # Act
    result = handler.fit_transform(df)

    # Assert
    assert result['a'].iloc[2] == 999
```

## Производительность

### Оптимизации

1. **Ленивые вычисления**: не считать метрики пока не нужно
2. **Кэширование**: сохранять промежуточные результаты
3. **Векторизация**: использовать numpy/pandas операции
4. **Параллелизм**: joblib для параллельных вычислений где возможно

### Бенчмарки

- Замерять время выполнения каждого шага
- Тестировать на датасетах разных размеров
- Профилировать узкие места

## Совместимость

- Python 3.8+
- pandas 1.3+
- numpy 1.20+
- scikit-learn 1.0+

## Roadmap для будущих версий

### v1.1
- HDBSCAN поддержка
- Ensemble clustering
- Больше метрик оценки

### v1.2
- Dask поддержка для больших данных
- Incremental clustering
- Time series clustering

### v2.0
- GPU ускорение (cuML)
- AutoML для автоподбора параметров
- Web UI
