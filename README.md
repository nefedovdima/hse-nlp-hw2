# HSE NLP HW2: генерация кода

Решение домашнего задания №2 «Генерация кода».
Модель получает текст математической задачи и генерирует Python-код для вычисления ответа.

**Модель:** encoder-decoder Transformer (8 слоёв энкодера, 4 слоя декодера, 100,8 млн параметров), обученный с нуля на [tinysms_qa_12m](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m).

**Результат:** public pass@1 — **74,27% (713/960)**. Генерация 1919 программ на NVIDIA Tesla T4 — **26,3 с** (ограничение — 180 с).

## Запуск в Google Colab

1. Откройте `notebooks/inference_colab_t4.ipynb` в [Google Colab](https://colab.research.google.com/) и выберите **T4 GPU**.
2. Скачайте `HW2_CODE_FINAL.zip` из [Releases](../../releases). В этом архиве есть обученные веса, токенизатор, код и тестовые входы.
3. Запустите ячейки ноутбука сверху вниз. Когда появится окно загрузки, выберите скачанный `HW2_CODE_FINAL.zip`.
4. На выходе получится `submission.jsonl` — 1919 строк формата `{"code": "..."}`.

В репозитории не хранятся обученные веса из-за их размера. Финальный Colab-ноутбук использует полный архив из Releases.

## Код

- `download_all.py`, `prepare_tokens.py` — загрузка и подготовка датасета.
- `v2/` — модель, обучение, валидация, экспорт и инференс.
- `v2/configs/v2_main_full_2ep.json` — конфигурация основного обучения.
- `e1/` — первый, меньший Transformer для сравнения.
- `notebooks/train_v2.ipynb`, `notebooks/train_e1.ipynb` — ноутбуки обучения.
- `notebooks/inference_colab_t4.ipynb` — итоговый ноутбук генерации на T4.
- `evidence/` — основные сохранённые метрики и логи экспериментов.

## Обучение

Для повторного обучения нужна CUDA GPU, исходные данные и свободное место для Arrow-кэша. Из корня проекта:

```bash
python -m pip install -r requirements.txt
python download_all.py
python prepare_tokens.py
python -m pytest -q v2/tests
python v2/hw2_data_v2.py --cache data/prepared/bpe8192 --output experiments/v2/validation4096.json
python v2/hw2_train_v2.py --root . --config v2/configs/v2_main_full_2ep.json
```

Основная модель обучалась две эпохи (111275 шагов). Для генерации использованы FP16, greedy decoding, batch size 32 и максимум 384 новых токена. Датасет, промежуточные веса и кэш обучения в Git не включены.
