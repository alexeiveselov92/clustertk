# ClusterTK - Project Context for Claude Code

## Описание проекта

ClusterTK - это Python библиотека для полного пайплайна кластерного анализа, разработанная для аналитиков. Библиотека предоставляет удобный API где можно просто передать датафрейм и настроить каждый шаг через параметры, без написания большого количества кода.

**Основная цель:** Сделать кластерный анализ максимально удобным и доступным для аналитиков.

## Текущий статус проекта

### 📦 Публикация:
- ✅ GitHub: https://github.com/alexeiveselov92/clustertk
- ✅ PyPI: https://pypi.org/project/clustertk/
- **Latest Version:** v0.6.0

### ✅ Полностью реализовано:

1. **Preprocessing** - полностью готово (v0.1.0)
   - MissingValueHandler - обработка пропусков (median/mean/drop/custom)
   - OutlierHandler - детекция и обработка выбросов (IQR/z-score/modified z-score)
   - ScalerSelector - автовыбор скейлера (Standard/Robust/MinMax)
   - SkewnessTransformer - log/sqrt/box-cox трансформации

2. **Feature Selection** - полностью готово (v0.1.0)
   - CorrelationFilter - удаление сильно коррелирующих признаков
   - VarianceFilter - удаление low-variance признаков

3. **Dimensionality Reduction** - полностью готово (v0.1.0)
   - PCAReducer - PCA с автоподбором компонент по variance threshold
   - ManifoldReducer - t-SNE/UMAP для визуализации (только для viz, не для кластеризации!)

4. **Clustering** - полностью готово (v0.2.0)
   - BaseClusterer - базовый класс для всех алгоритмов
   - KMeansClustering - K-Means алгоритм
   - GMMClustering - Gaussian Mixture Model
   - HierarchicalClustering - иерархическая кластеризация (Ward, Complete, Average)
   - DBSCANClustering - DBSCAN с автоподбором eps и min_samples

5. **Evaluation** - полностью готово (v0.1.0)
   - compute_clustering_metrics - Silhouette, Calinski-Harabasz, Davies-Bouldin
   - OptimalKFinder - автоподбор оптимального k с голосованием метрик

6. **Interpretation** - полностью готово (v0.3.0)
   - ClusterProfiler - профилирование кластеров, топ-признаки, анализ по категориям
   - ClusterNamer - автоматическое именование кластеров (3 стратегии: top_features, categories, combined)

7. **Pipeline** - полностью готово (v0.1.0)
   - ClusterAnalysisPipeline - оркестрирует все шаги
   - Можно запускать как полный pipeline через .fit() так и пошагово

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

### ⚠️ TODO (для будущих версий):

**v0.6.0 (приоритет HIGH):**
- **Tests** - базовые unit tests для критичных модулей (покрытие >50%)

**v0.7.0+ (приоритет MEDIUM/LOW):**
- **docs/ структура** - markdown документация + GitHub Pages
- **Короткий README** - сократить до ~100-150 строк
- **CI/CD** - GitHub Actions
- **Sphinx** - полная API документация

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
6. Все visualization функции должны:
   - Проверять `_check_viz_available()` в начале
   - Возвращать matplotlib Figure объект
   - Использовать стандартное matplotlib поведение (автоматический вывод в Jupyter)
7. При работе с нормализацией профилей - использовать min/max из исходных данных
8. Manifold методы (t-SNE, UMAP) - только для визуализации, не для кластеризации!

10. **Documentation Structure** - полностью готово (v0.6.0)
   - Создана docs/ директория с полной структурой
   - Сокращён README с 495 до 196 строк
   - Созданы 8 разделов User Guide + installation, quickstart, examples, FAQ
   - README теперь только Quick Start + ссылки на docs/
