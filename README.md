# Ma_contaPLANE

Проект для детекции самолетов на датасете HRPlanes (PyTorch, CPU-first).

## Что решается

Задача: обнаружение самолетов на спутниковых снимках высокого разрешения.
Разметка: YOLO (`*.txt` рядом с изображениями).

## Структура репозитория

- `src/` - код модели, датасета и утилит
- `scripts/train.py` - обучение
- `scripts/val.py` - валидация
- `scripts/test.py` - тестовая оценка
- `requirements.txt` - минимальный способ воспроизведения среды
- `pyproject.toml` + `uv.lock` - рекомендуемый способ воспроизведения среды

Основная логика запуска находится прямо в `scripts/*.py`, а общие функции
для сохранения артефактов и служебных операций лежат в `src/common.py`.

## 1) Установка зависимостей

Вариант A (минимум):

```bash
pip install -r requirements.txt
```

Вариант B (рекомендуется, воспроизводимо):

```bash
uv sync
```

## 2) Скачивание и подготовка датасета

Ссылка на датасет: HRPlanes (используйте источник, который вам выдали для проекта).

Если архив уже разбит на части и лежит в корне репозитория, распаковка:

```powershell
& "C:\Program Files\7-Zip\7z.exe" x .\HRPlanes.7z.001 -o.\src\data -y
```

Если первая часть называется `HRPlanes.7z`, сначала создайте копию с корректным именем:

```powershell
Copy-Item .\HRPlanes.7z .\HRPlanes.7z.001 -Force
& "C:\Program Files\7-Zip\7z.exe" x .\HRPlanes.7z.001 -o.\src\data -y
```

Проверка структуры:

```powershell
Get-ChildItem .\src\data
Get-ChildItem .\src\data\img -File | Measure-Object
```

Ожидаемая структура:

```text
src/data/
  img/
    *.jpg
    *.txt
  train.txt
  validation.txt
  test.txt
```

## 3) Как запустить обучение

Все гиперпараметры и пути запуска задаются в одном месте:

- `src/config.py`

Перед запуском при необходимости правь там `epochs`, `batch_size`, `lr`,
`weight_decay`, `num_workers`, `score_threshold`, пути к артефактам и флаги
визуализации.

```powershell
New-Item -ItemType Directory -Path .\logs -Force | Out-Null
python scripts/train.py | Tee-Object -FilePath .\logs\train.log
```

## 4) Как запустить валидацию

```powershell
python scripts/val.py | Tee-Object -FilePath .\logs\val.log
```

## 5) Как посмотреть метрики и финальный скор

Итоговый скор для сдачи:

- `F1` из `scripts/val.py` на `checkpoints/best.pt`.

Быстрые команды:

```powershell
Select-String -Path .\logs\train.log -Pattern "val_f1" | Select-Object -Last 1
Select-String -Path .\logs\val.log -Pattern "F1:" | Select-Object -Last 1
Get-Content .\outputs\metrics\final_summary.json
Get-Content .\outputs\metrics\val_metrics.json
Get-Content .\outputs\metrics\run_config.json
```

## Артефакты, которые сохраняются

После обучения `scripts/train.py` автоматически сохраняет:

- Веса модели:
  - `checkpoints/best.pt`
  - `checkpoints/last.pt`
- Метрики обучения:
  - `outputs/metrics/train_history.json`
  - `outputs/metrics/train_history.csv`
  - `outputs/metrics/final_summary.json`
- Графики:
  - `outputs/plots/loss_curve.png`
  - `outputs/plots/metrics_curve.png`

После валидации `scripts/val.py` сохраняет:

- `outputs/metrics/val_metrics.json`

После теста `scripts/test.py` сохраняет:

- `outputs/metrics/test_metrics.json`
- `outputs/reports/test_detailed_report.json`
- `outputs/reports/test_worst_cases.json`
- `outputs/reports/test_report.txt`

## Команды для полного воспроизведения

```powershell
uv sync
python scripts/train.py
python scripts/val.py
python scripts/test.py
```

## Pre-commit и линтеры

Установить dev-инструменты:

```powershell
uv sync --extra dev
```

Инициализировать pre-commit хуки:

```powershell
pre-commit install
pre-commit run --all-files
```

Используются `ruff` + `ruff-format` и базовые проверки `pre-commit-hooks`.

## Seed и конфиг запуска

- Фиксированный `random seed` задается в `src/config.py` (`seed = 42`).
- Полная конфигурация запуска сохраняется в `outputs/metrics/run_config.json`.
