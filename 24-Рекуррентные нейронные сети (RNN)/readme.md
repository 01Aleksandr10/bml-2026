# Рекуррентные нейронные сети (RNN)

Команда 24. Курс «Основы машинного обучения».

## Содержание

- `Рекуррентные нейронные сети.pdf` — презентация: устройство RNN, численный пример, обучение через время, LSTM и GRU, Python, приложения и альтернативы.
- `src/run_demo.py` — RNN на NumPy: прямой проход, BPTT, Adam и проверка градиентов.
- `src/numerical_example.py` — пошаговый скалярный пример и сравнение с обратным порядком.
- `src/demo.ipynb` — notebook с объяснениями и сохранённым выводом.
- `src/requirements.txt` — зависимости.
- `src/README.md` — устройство эксперимента и подробные результаты.

## Запуск

Нужен Python 3.11+. Команды выполняются из каталога этой темы:

```sh
python3 -m pip install -r src/requirements.txt
python3 src/numerical_example.py
python3 src/run_demo.py
```

Численный пример использует стандартную библиотеку, обучение — NumPy 2.3.5. Данные создаются локально; загружать датасеты не нужно. Скрипты создают `results/` с данными, метриками и весами. Результаты в notebook уже сохранены.

## Учебный эксперимент

RNN различает возрастающую последовательность и её обратную копию. В каждой паре одинаковые значения и среднее, но разное направление. Пары целиком разделены на train и test. Нормализация использует только train. Длина входа — 12, скрытое состояние — 8, параметров — 89.

В фиксированном запуске на 256 тестовых примерах RNN получила 100%, классификатор по среднему — 50%, а правило «последнее больше первого» — 100%. Это демонстрирует работу реализации и роль порядка, но не преимущество RNN. Эксперимент не проверяет устойчивость к шуму или качество на реальных данных.

Все 89 производных BPTT проверены конечными разностями. Численный пример: `[1, 0, -1]` даёт `h₃ ≈ -0,2137`, обратный порядок `[-1, 0, 1]` — `h₃ ≈ +0,2137`.

## Научные источники

1. [Zhang, Lipton, Li, Smola. Dive into Deep Learning: RNN](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) — скрытое состояние и общие веса.
2. [D2L: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) — обучение и цепочки производных.
3. [D2L: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html) — состояние памяти и вентили.
4. [D2L: GRU](https://d2l.ai/chapter_recurrent-modern/gru.html) — reset и update.
5. [Pascanu, Mikolov, Bengio. On the difficulty of training recurrent neural networks, 2013](https://arxiv.org/abs/1211.5063) — затухание, взрыв градиента и clipping.
6. [Hochreiter, Schmidhuber. Long Short-Term Memory, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) — исходная работа о LSTM. Показанная в презентации современная схема включает последующие усовершенствования, в частности forget gate.
7. [Cho et al. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, 2014](https://arxiv.org/abs/1406.1078) — GRU.
8. [Vaswani et al. Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762) — Transformer.
9. [Bai, Kolter, Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018](https://arxiv.org/abs/1803.01271) — TCN и сравнение с рекуррентными подходами.
10. [PyTorch nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) — дополнительная справка о библиотечной реализации. Наш пример реализован на NumPy.

## Где RNN применяются сейчас

Проверено по первичным источникам **31 октября 2026 года**. Это конкретные доступные функции и интеграции; примеры не являются рейтингом распространённости.

| Продукт или компонент | Применение | Подтверждение |
|---|---|---|
| OBS Studio → RNNoise | Подавление фонового шума в микрофонном сигнале; GRU обрабатывает последовательные аудиокадры | [Фильтр OBS](https://obsproject.com/kb/noise-suppression-filter), [код RNNoise](https://github.com/xiph/rnnoise/blob/main/src/rnn.c) |
| whisper.cpp → Silero VAD | Выделение речевых участков перед распознаванием; в декодере Silero используется LSTM | [Интеграция VAD](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad), [LSTMCell в Silero v6.2.3](https://github.com/snakers4/silero-vad/blob/v6.2.3/tuning/utils.py), [загрузка весов декодера](https://github.com/snakers4/silero-vad/blob/v6.2.3/tuning/tune.py) |
| Tesseract 5 | LSTM-движок распознаёт строки текста на изображениях; включается через `--oem 1` | [Руководство Tesseract](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html), [описание движка](https://github.com/tesseract-ocr/tesseract) |

Прикладной пример OCR — [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx/blob/dev/docs/index.md): система использует Tesseract, чтобы документы можно было искать и выделять в них текст. Конкретная конфигурация OCR зависит от установки. Сам Whisper использует Transformer; Silero VAD — отдельный подключаемый компонент.

### Схемы и лицензия

Слайды 4 и 9 используют изображения из Dive into Deep Learning авторов Aston Zhang, Zachary C. Lipton, Mu Li и Alexander J. Smola:

- [Развёртка RNN](https://d2l.ai/_images/unfolded-rnn.svg).
- [Современная ячейка LSTM](https://d2l.ai/_images/lstm-3.svg).

Лицензия изображений: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), [LICENSE учебника](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE). Содержание схем не изменено; выполнено только техническое преобразование SVG в PNG и пропорциональное масштабирование. Атрибуция есть на слайдах и в этом readme. На слайде 15 показано [тестовое изображение eurotext.png](https://tesseract-ocr.github.io/tessdoc/images/eurotext.png) из официальной документации Tesseract, а не результат нашего эксперимента. Собственные текст, расчёты, код и графики подготовлены для этого комплекта.
