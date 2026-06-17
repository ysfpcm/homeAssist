# Custom Wake Word Models

Place your custom trained `openWakeWord` ONNX models here.

For example, if you want to use the wake word **"gemma"**, place your trained `gemma.onnx` file in this directory. Then update your wake word configuration (e.g. `WAKE_WORD = "gemma"` in `main.py`).

## How to train a custom model:
You can use openWakeWord's official training utilities or their Google Colab notebook to generate a custom `.onnx` model using synthetic text-to-speech voice samples.
- **openWakeWord GitHub:** https://github.com/dscripka/openWakeWord
