# ClusterTK - TODO

## Текущий статус: v0.8.0 🚀

**Latest Release:** v0.8.0 (ready to publish on PyPI)
**PyPI:** https://pypi.org/project/clustertk/
**GitHub:** https://github.com/alexeiveselov92/clustertk

## ✅ Что завершено

### v0.1.0-v0.1.1 - Базовый функционал
- [x] Полный pipeline кластерного анализа
- [x] Preprocessing (4 модуля)
- [x] Feature selection (2 модуля)
- [x] Dimensionality reduction (PCA, t-SNE, UMAP)
- [x] Clustering (KMeans, GMM)
- [x] Evaluation (метрики + optimal k finder)
- [x] Interpretation (ClusterProfiler)
- [x] Публикация на PyPI

### v0.2.0 - Дополнительные алгоритмы
- [x] HierarchicalClustering (Ward, Complete, Average linkage)
- [x] DBSCANClustering с автоподбором параметров (eps, min_samples)

### v0.3.0 - Visualization & Naming
- [x] 11 visualization функций в 4 модулях
- [x] Интеграция с Pipeline (.plot_*() методы)
- [x] ClusterNamer с 3 стратегиями (top_features, categories, combined)
- [x] Условные импорты для опциональных зависимостей

### v0.3.1-v0.3.5 - Bugfixes & Improvements
- [x] v0.3.1: Hotfix README (plot_cluster_profiles → plot_cluster_heatmap)
- [x] v0.3.2: Исправлена нормализация для малого числа кластеров
- [x] v0.3.3-v0.3.5: Улучшения в visualization модуле

### v0.4.0-v0.4.2 - Visualization Fix & Documentation
- [x] v0.4.0: Попытка исправить дублирование (неудачная)
- [x] v0.4.1: Правильное решение через plt.close(fig)
- [x] v0.4.2: Обновлен Quick Start + навигация в README
- [x] Исследована причина дублирования (pyplot global state)
- [x] Изучен подход seaborn (возвращает Axes, не Figure)
- [x] Добавлены Table of Contents и Quick Links
- [x] Pipeline Components теперь в collapsed section

### v0.5.0 - Export & Reports ✅
- [x] Реализован `export_results()` для CSV формата
- [x] Реализован `export_results()` для JSON формата
- [x] Реализован `export_report()` - HTML отчёты с embedded plots
- [x] Реализованы `save_pipeline()` и `load_pipeline()` методы
- [x] Добавлена зависимость joblib
- [x] Обновлён README с примерами экспорта
- [x] Протестирован весь функционал экспорта

### v0.6.0 - Documentation Structure ✅
- [x] Создана docs/ директория с полной структурой
- [x] Сокращён README с 495 до 196 строк
- [x] Созданы все основные документы:
  - docs/installation.md - установка и требования
  - docs/quickstart.md - быстрый старт за 5 минут
  - docs/user_guide/ - 8 разделов (preprocessing, clustering, evaluation, etc.)
  - docs/examples.md - реальные примеры использования
  - docs/faq.md - часто задаваемые вопросы
  - docs/index.md - главная страница документации
- [x] README теперь содержит только Quick Start и ссылки на docs/
- [ ] TODO (optional): GitHub Pages или MkDocs setup

## ✅ v0.7.0 - Algorithm Comparison (Completed)
- [x] Метод `pipeline.compare_algorithms()` - автоматическое сравнение KMeans/GMM/Hierarchical/DBSCAN
- [x] Таблица с метриками для каждого алгоритма (DataFrame с результатами)
- [x] Рекомендация лучшего алгоритма на основе weighted scoring (40/30/30)
- [x] Визуализация сравнения алгоритмов (plot_algorithm_comparison)
- [x] Тесты функционала (test_comparison.py - все пройдены)
- [x] Документация с примерами (README, clustering.md, examples.md, FAQ)

## ✅ v0.8.0 - HDBSCAN & Test Suite (Completed)

### HDBSCAN Algorithm ✅
- [x] Реализация HDBSCANClustering класса
- [x] Автоподбор параметров (min_cluster_size, min_samples)
- [x] Интеграция в Pipeline
- [x] Добавление в compare_algorithms()
- [x] Документация и примеры

### Test Suite ✅
- [x] Unit tests для preprocessing модулей (missing, scaling)
- [x] Unit tests для clustering алгоритмов (kmeans)
- [x] Unit tests для evaluation (metrics)
- [x] Integration tests для Pipeline (fit, transform, full workflow)
- [x] Pytest infrastructure с pytest.ini
- [x] 39 тестов, 39% coverage (clustering 66-76%, preprocessing 61-69%)
- [x] Fixtures для различных сценариев данных

### Build & Release ✅
- [x] Build packages успешно (wheel + source distribution)
- [x] Packages прошли twine check
- [x] Git tag v0.8.0 создан и pushed
- [x] Version numbers обновлены (setup.py, pyproject.toml, __init__.py)
- [ ] **TODO: Publish to PyPI** (ручная публикация)
  ```bash
  # Get API token: https://pypi.org/manage/account/token/
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="pypi-YOUR_TOKEN"
  python3 -m twine upload dist/clustertk-0.8.0*
  ```
- [ ] **TODO: Create GitHub Release**
  - URL: https://github.com/alexeiveselov92/clustertk/releases/new
  - Tag: v0.8.0
  - Title: v0.8.0 - HDBSCAN Algorithm & Test Suite
  - Description: Copy from git tag message
  - Attach: dist/clustertk-0.8.0-py3-none-any.whl, dist/clustertk-0.8.0.tar.gz

## 🎯 Приоритеты для v0.9.0

### HIGH PRIORITY

1. **Enhanced Test Coverage** 🔥
   - [ ] Добавить тесты для outliers, transforms модулей
   - [ ] Тесты для GMM, Hierarchical, DBSCAN, HDBSCAN
   - [ ] Тесты для optimal_k
   - [ ] Тесты для export/report функционала
   - [ ] Тесты для visualization модуля
   - [ ] Цель: coverage >50%

2. **CI/CD Setup** 🔥
   - [ ] GitHub Actions для автоматических тестов
   - [ ] Автоматическая проверка code style (black, flake8)
   - [ ] Coverage reports (codecov integration)
   - [ ] Автоматическая публикация на PyPI при release tag

### MEDIUM PRIORITY

3. **More Clustering Algorithms**
   - [ ] Spectral Clustering
   - [ ] OPTICS (Ordering Points To Identify Clustering Structure)
   - [ ] Mini-Batch K-Means (для больших данных)

4. **Enhanced Feature Analysis**
   - [ ] SHAP values для интерпретации
   - [ ] Permutation importance
   - [ ] Feature contribution to cluster separation
   - [ ] Anomaly detection внутри кластеров (LOF, Isolation Forest)

### LOW PRIORITY (Backlog)

5. **Advanced Features**
   - [ ] Ensemble clustering (voting между алгоритмами)
   - [ ] Автоматический feature engineering (polynomial, interactions)
   - [ ] Stability analysis (bootstrap resampling)
   - [ ] Time series clustering support

## 📋 Backlog (для будущих версий)

- Полная Sphinx документация с API reference
- Оптимизация производительности для больших данных
- GPU support (cuML integration)
- Incremental/streaming clustering
- Time series clustering support
- Interactive web UI

## 🐛 Известные проблемы

Нет критических проблем в текущей версии.

## 💡 Идеи для улучшения

1. **User Experience:**
   - Прогресс-бар для долгих операций (tqdm)
   - Более информативные error messages
   - Warnings для потенциально проблемных настроек

2. **Performance:**
   - Кэширование промежуточных результатов
   - Параллелизация где возможно
   - Опциональная поддержка Dask для больших данных

3. **Features:**
   - Автоматический feature engineering
   - Anomaly detection внутри кластеров
   - Stability analysis (bootstrap resampling)
   - Cluster evolution tracking (для временных данных)

## 📝 Заметки по разработке

### Приоритизация
- **v0.4.0-v0.4.2** ✅ - исправлено дублирование графиков + улучшена навигация
- **v0.5.0** ✅ - export/reports (CSV, JSON, HTML, save/load)
- **v0.6.0** 🎯 - docs/ структура + algorithm comparison + enhanced analysis
- **v0.7.0** - HDBSCAN/Spectral + базовые тесты + CI/CD
- **v0.8.0+** - стабилизация API + advanced features
- **v1.0.0** - стабильный API, полная документация, production-ready

### Принципы
- Backward compatibility важна
- Опциональные зависимости остаются опциональными
- Код на английском, docstrings обязательны
- Type hints везде

### Для новых фич
Перед добавлением новой фичи проверь:
1. Соответствует ли общей архитектуре?
2. Нужны ли новые зависимости? (минимизировать!)
3. Как это повлияет на API?
4. Есть ли похожее в sklearn? (следовать их паттернам)
