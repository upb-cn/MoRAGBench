# Android embedding threading investigation

Related issue: https://github.com/upb-cn/MoRAGBench/issues/24

## Scope

Investigate whether BGE-small, BGE-base, and MiniLM embeddings differ between
default ONNX Runtime CPU threading and single-thread execution on the same
Android device, using Python ONNX Runtime as an independent reference.
BGE-large FP32 was investigated separately.

## Environment

- Android device: OPPO CPH2791
- ADB serial for reproduction: `YOUR_DEVICE_SERIAL` (replace with your device's serial)
- Android ONNX Runtime: 1.23.0
- Python reference: ONNX Runtime 1.30.0, NumPy 2.4.3
- Python execution provider: CPUExecutionProvider
- Python graph optimization: ORT_ENABLE_ALL
- BGE-small: FP32, 384 dimensions, CLS pooling
- BGE-base: FP32, 768 dimensions, CLS pooling
- MiniLM: INT8, 384 dimensions, attention-masked mean pooling

The Python reference runs on macOS, not on the Android device. Consequently,
the Android/Python comparison also crosses operating systems and CPU
architectures; it is not a controlled threading-only comparison.

## Test input

Sentence:

    The quick brown fox jumps over the lazy dog.

BGE-small and BGE-base prepend:

    "Represent this sentence for searching relevant passages: "

MiniLM uses the sentence without a prefix.

Each model was evaluated with default intra-op threading and with
`intraOpNumThreads = 1`, using three repetitions per configuration and two
separate instrumentation executions.

## Android diagnostic setup

The instrumentation tests require model files to be provisioned manually
inside the app-private files directory under `embedding_threading_models/`:

- `bge-small-model.onnx`
- `bge-small-tokenizer.json`
- `bge-base-model.onnx`
- `bge-base-tokenizer.json`
- `minilm-model.onnx`
- `minilm-tokenizer.json`

The model and tokenizer sources used locally are:

- `client/task_files/embedding/bge-small-en-v1.5_float32/`
- `client/task_files/embedding/bge-base-en-v1.5_FLOAT32/`
- `client/task_files/embedding/all-minilm-l6-v2/`

From the `android/` directory, build both APKs:

    ./gradlew :cli:assembleDebug :cli:assembleDebugAndroidTest

Install both APKs on the intended Android device:

    adb -s YOUR_DEVICE_SERIAL install -r cli/build/outputs/apk/debug/cli-debug.apk
    adb -s YOUR_DEVICE_SERIAL install -r cli/build/outputs/apk/androidTest/debug/cli-debug-androidTest.apk

Then run the threading diagnostic on the same device:

    adb -s YOUR_DEVICE_SERIAL shell am instrument -w -e class com.example.cli.EmbeddingThreadingValidationTest com.example.cli.test/androidx.test.runner.AndroidJUnitRunner
It writes eighteen JSON files per execution to the app's external files
directory under `embedding_threading_validation/`. Copy the first execution
to `client/embedding_threading_validation/bge_small_run_1/` before repeating
the test, and the second to `bge_small_run_2/`. Each execution overwrites
the previous results on the device.

`EmbeddingTokenizerValidationTest` separately captures MiniLM's native
Android tokenizer output. Both tests are opt-in diagnostics requiring manual
file provisioning; they are not self-contained CI regression tests.

## Python comparison

Install Python dependencies `numpy`, `onnxruntime`, and `tokenizers` in a
virtual environment. From the repository root, run:

    python client/validate_embedding_threading.py

By default, the script validates BGE-base and MiniLM using their original
24 Android JSON results. For the expanded experiment, specify the model
and both result directories explicitly. From `client/`, run:

    python validate_embedding_threading.py --models-dir task_files/embedding --model bge-small-en-v1.5 --run-dirs embedding_threading_validation/bge_small_run_1 embedding_threading_validation/bge_small_run_2

The validator checks result metadata and compares normalized embeddings with Python
ONNX Runtime CPU outputs.

## Observations

### BGE-small FP32

Two independent Android instrumentation executions produced six embeddings
per threading configuration, using the same sentence and model files.

- Default threading produced three distinct embedding vectors across six
  measurements. Four matched the Python reference closely, while the first
  inference in each execution differed noticeably.
- The two divergent default-thread outputs had cosine similarities to Python
  of approximately 0.984617 and 0.983590, with maximum absolute differences
  of approximately 0.025210 and 0.030781, respectively.
- All six single-thread Android embeddings were bit-for-bit identical.
- The single-thread output matched the Python ONNX Runtime CPU reference
  with cosine similarity approximately 1.0 and maximum absolute difference
  approximately 1.60e-7.

These results support setting `intraOpNumThreads = 1` for BGE-small on the
tested device, consistent with the existing BGE-base and BGE-large
workarounds. They do not establish that every FP32 embedding model requires
single-thread execution.

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

## Interpretation and limitations

BGE-small and BGE-base showed reproducible default-thread instability on
the tested phone; single-thread execution removed the observed instability.
This supports their `intraOpNumThreads = 1` workarounds. Together with the
separate BGE-large findings, the evidence supports single-thread execution
for the tested BGE FP32 models, not a universal FP32 threading policy.

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
It does assert exact equality across the six saved single-thread Android
embeddings for each selected BGE model. This checks the captured results, not future device runs.
The Android instrumentation tests are opt-in diagnostics rather than automated
CI coverage.

These findings are limited to one Android device, one input sentence, and
the model files used in this investigation. The Android diagnostic tests
currently verify output dimensions and export vectors; numerical comparison
is performed by the Python script.

## Provisioning Android model files

For a debuggable installation of `com.example.cli`, copy each model and tokenizer to `/data/local/tmp/` with `adb push`, then copy it into `files/embedding_threading_models/` using `adb shell run-as com.example.cli cp`. The six required destination filenames are listed in the Android diagnostic setup section above. Use `adb -s YOUR_DEVICE_SERIAL` to target your intended Android device. Verify file sizes or checksums after copying. Ordinary `adb push` cannot write directly into the app-private directory.

## Reproduction path details

The Python validator expects `model.onnx` and `tokenizer.json` inside the selected model directory under `client/task_files/embedding/`: `bge-small-en-v1.5_float32/`, `bge-base-en-v1.5_FLOAT32/`, or `all-minilm-l6-v2/`. Override the parent directory with `--models-dir` if necessary. Saved Android results default to `client/embedding_threading_validation/`; override that directory with `--android-results`.

The Android instrumentation test requires the same artifacts under `files/embedding_threading_models/` in the `com.example.cli` app-private directory, renamed as follows:

| Source | Android filename |
| --- | --- |
| `bge-small-en-v1.5_float32/model.onnx` | `bge-small-model.onnx` |
| `bge-small-en-v1.5_float32/tokenizer.json` | `bge-small-tokenizer.json` |
| `bge-base-en-v1.5_FLOAT32/model.onnx` | `bge-base-model.onnx` |
| `bge-base-en-v1.5_FLOAT32/tokenizer.json` | `bge-base-tokenizer.json` |
| `all-minilm-l6-v2/model.onnx` | `minilm-model.onnx` |
| `all-minilm-l6-v2/tokenizer.json` | `minilm-tokenizer.json` |

The recorded ADB serial identifies the device used for this investigation; substitute your own device serial when reproducing the experiment. The model artifacts are local prerequisites and are not included with the captured JSON evidence.

## Collecting Android results

After each expanded instrumentation execution, retrieve the eighteen JSON files using `adb pull`. For example, from the repository root:

```sh
adb -s YOUR_DEVICE_SERIAL pull /sdcard/Android/data/com.example.cli/files/embedding_threading_validation/. client/embedding_threading_validation/bge_small_run_1/
```

Perform this collection before running the instrumentation test again, because the next execution overwrites files with the same names. For the expanded experiment, use `client/embedding_threading_validation/bge_small_run_1/` for the first execution and `client/embedding_threading_validation/bge_small_run_2/` for the second. The original BGE-base/MiniLM experiment instead used the root results directory and `repeat_2/`. Replace `YOUR_DEVICE_SERIAL` with the serial reported by `adb devices`. Preserve both result sets for comparison.

## Provisioning command pattern

From the repository root, first create the app-private destination with `adb -s YOUR_DEVICE_SERIAL shell run-as com.example.cli mkdir -p files/embedding_threading_models`. For each of the six files in the mapping table above, transfer the source file using `adb -s YOUR_DEVICE_SERIAL push SOURCE_PATH /data/local/tmp/DESTINATION_FILENAME`, then install it using `adb -s YOUR_DEVICE_SERIAL shell run-as com.example.cli cp /data/local/tmp/DESTINATION_FILENAME files/embedding_threading_models/DESTINATION_FILENAME`. Replace the placeholders with the verified filenames in the mapping table. These commands overwrite existing destination files; use them only when provisioning the models intentionally. Remove temporary staging copies after verifying the transfers.
