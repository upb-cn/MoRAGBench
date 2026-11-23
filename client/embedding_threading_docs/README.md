# Android embedding threading investigation

Related issue: https://github.com/upb-cn/MoRAGBench/issues/24

## Scope

Investigate whether BGE-base and MiniLM embeddings differ between default
ONNX Runtime CPU threading and single-thread execution on the same Android
device, using Python ONNX Runtime as an independent reference.

## Environment

- Android device: OPPO CPH2791
- ADB serial used during testing: `3B15C6002GV00000`
- Android ONNX Runtime: 1.23.0
- Python reference: ONNX Runtime 1.30.0, NumPy 2.4.3
- Python execution provider: CPUExecutionProvider
- Python graph optimization: ORT_ENABLE_ALL
- BGE-base: FP32, 768 dimensions, CLS pooling
- MiniLM: INT8, 384 dimensions, attention-masked mean pooling

The Python reference runs on macOS, not on the Android device. Consequently,
the Android/Python comparison also crosses operating systems and CPU
architectures; it is not a controlled threading-only comparison.

## Test input

Sentence:

    The quick brown fox jumps over the lazy dog.

BGE-base prepends:

    "Represent this sentence for searching relevant passages: "

MiniLM uses the sentence without a prefix.

Each model was evaluated with default intra-op threading and with
`intraOpNumThreads = 1`, using three repetitions per configuration and two
separate instrumentation executions.

## Android diagnostic setup

The instrumentation tests require model files to be provisioned manually
inside the app-private files directory under `embedding_threading_models/`:

- `bge-base-model.onnx`
- `bge-base-tokenizer.json`
- `minilm-model.onnx`
- `minilm-tokenizer.json`

The model and tokenizer sources used locally are:

- `client/task_files/embedding/bge-base-en-v1.5_FLOAT32/`
- `client/task_files/embedding/all-minilm-l6-v2/`

From the `android/` directory, build both APKs:

    ./gradlew :cli:assembleDebug :cli:assembleDebugAndroidTest

Install both APKs on the intended Android device:

    adb -s 3B15C6002GV00000 install -r cli/build/outputs/apk/debug/cli-debug.apk
    adb -s 3B15C6002GV00000 install -r cli/build/outputs/apk/androidTest/debug/cli-debug-androidTest.apk

Then run the threading diagnostic on the same device:

    adb -s 3B15C6002GV00000 shell am instrument -w -e class com.example.cli.EmbeddingThreadingValidationTest com.example.cli.test/androidx.test.runner.AndroidJUnitRunner
It writes twelve JSON files per execution to the app's external files
directory under `embedding_threading_validation/`. Copy those files to
`client/embedding_threading_validation/` before repeating the test; otherwise
the second execution overwrites the first. Store the second set in
`client/embedding_threading_validation/repeat_2/`.

`EmbeddingTokenizerValidationTest` separately captures MiniLM's native
Android tokenizer output. Both tests are opt-in diagnostics requiring manual
file provisioning; they are not self-contained CI regression tests.

## Python comparison

Install Python dependencies `numpy`, `onnxruntime`, and `tokenizers` in a
virtual environment. From the repository root, run:

    python client/validate_embedding_threading.py

The script requires the local model files and all 24 Android JSON results.
It validates result metadata and compares normalized embeddings with Python
ONNX Runtime CPU outputs.

## Observations

### BGE-base FP32

- Three of six default-thread Android outputs differed noticeably from
  the stable output.
- Their cosine similarities to Python were approximately 0.962102,
  0.997951, and 0.998950.
- All six single-thread Android outputs were identical.
- The stable Android output matched the Python reference with maximum
  absolute difference approximately 2.36e-7.

### MiniLM INT8

- All twelve Android outputs were identical across default and single-thread
  configurations and across both instrumentation executions.
- Their cosine similarity to Python ONNX Runtime 1.30.0 was approximately
  0.997797, with maximum absolute difference approximately 0.0102673.
- Android and Python MiniLM token IDs, attention masks, and token type IDs
  matched exactly across 128 positions.
- Python ONNX Runtime 1.23.0 and 1.30.0 also produced different MiniLM
  embeddings in a separate comparison.

## Interpretation and limitations

BGE-base showed reproducible default-thread instability on the tested phone;
single-thread execution removed the observed instability. This supports the
BGE-base `intraOpNumThreads = 1` workaround.

MiniLM showed no threading-dependent variation in these runs. Its consistent
Android/Python difference remains unexplained and must not be attributed
to threading without further evidence.

A JVM configuration regression test, `SupportedEmbeddingModelsTest`, verifies
that BGE-base retains `intraOpNumThreads = 1`. Run it from `android/` with
`./gradlew :shared:testDebugUnitTest --tests com.example.shared.SupportedEmbeddingModelsTest`.
This test passed locally. It guards the production configuration but does not
verify numerical inference correctness.

The Python comparison script reports cosine similarity and maximum absolute
difference against Python without enforcing cross-runtime accuracy thresholds.
It does assert exact equality across the six saved BGE-base single-thread
Android embeddings. This checks the captured results, not future device runs.
The Android instrumentation tests are opt-in diagnostics rather than automated
CI coverage.

These findings are limited to one Android device, one input sentence, and
the model files used in this investigation. The Android diagnostic tests
currently verify output dimensions and export vectors; numerical comparison
is performed by the Python script.

## Provisioning Android model files

For a debuggable installation of `com.example.cli`, copy each model and tokenizer to `/data/local/tmp/` with `adb push`, then copy it into `files/embedding_threading_models/` using `adb shell run-as com.example.cli cp`. The four required destination filenames are listed in the Android diagnostic setup section above. Use `adb -s 3B15C6002GV00000` to target the tested OPPO phone. Verify file sizes or checksums after copying. Ordinary `adb push` cannot write directly into the app-private directory.

## Reproduction path details

The Python validator expects `model.onnx` and `tokenizer.json` inside each of `client/task_files/embedding/bge-base-en-v1.5_FLOAT32/` and `client/task_files/embedding/all-minilm-l6-v2/`. Override the parent directory with `--models-dir` if necessary. Saved Android results default to `client/embedding_threading_validation/`; override that directory with `--android-results`.

The Android instrumentation test requires the same artifacts under `files/embedding_threading_models/` in the `com.example.cli` app-private directory, renamed as follows:

| Source | Android filename |
| --- | --- |
| `bge-base-en-v1.5_FLOAT32/model.onnx` | `bge-base-model.onnx` |
| `bge-base-en-v1.5_FLOAT32/tokenizer.json` | `bge-base-tokenizer.json` |
| `all-minilm-l6-v2/model.onnx` | `minilm-model.onnx` |
| `all-minilm-l6-v2/tokenizer.json` | `minilm-tokenizer.json` |

The recorded ADB serial identifies the device used for this investigation; substitute your own device serial when reproducing the experiment. The model artifacts are local prerequisites and are not included with the captured JSON evidence.

## Collecting Android results

After each instrumentation execution, retrieve the twelve JSON files using `adb pull`. For example, from the repository root:

```sh
adb -s YOUR_DEVICE_SERIAL pull /sdcard/Android/data/com.example.cli/files/embedding_threading_validation/ client/embedding_threading_validation/
```

Perform this collection before running the instrumentation test again, because the next execution overwrites files with the same names. For the second execution, change the destination to `client/embedding_threading_validation/repeat_2/`. Replace `YOUR_DEVICE_SERIAL` with the serial reported by `adb devices`. Preserve both result sets for comparison.

## Provisioning command pattern

From the repository root, first create the app-private destination with `adb -s YOUR_DEVICE_SERIAL shell run-as com.example.cli mkdir -p files/embedding_threading_models`. For each of the four files in the mapping table above, transfer the source file using `adb -s YOUR_DEVICE_SERIAL push SOURCE_PATH /data/local/tmp/DESTINATION_FILENAME`, then install it using `adb -s YOUR_DEVICE_SERIAL shell run-as com.example.cli cp /data/local/tmp/DESTINATION_FILENAME files/embedding_threading_models/DESTINATION_FILENAME`. Replace the placeholders with the verified filenames in the mapping table. These commands overwrite existing destination files; use them only when provisioning the models intentionally. Remove temporary staging copies after verifying the transfers.
