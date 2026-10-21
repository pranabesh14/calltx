# calltx — call recording transcriber (C++)

Transcribes `.wav` / `.mp3` call recordings into timestamped transcripts using
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) (fully offline, no Python)
and [miniaudio](https://github.com/mackron/miniaudio) for decoding + resampling to 16 kHz.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. Dependencies are fetched automatically.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release          # add -DGGML_CUDA=ON for NVIDIA GPU
cmake --build build -j
```

#For Windows build as below 
```bash
cmake -B build -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build build --config Release
```

## Get a model

```bash
mkdir -p models
curl -L -o models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

| Model              | Size   | Use when                                  |
|--------------------|--------|-------------------------------------------|
| `large-v3`         | ~3 GB  | best accuracy (recommended for calls)     |
| `large-v3-turbo`   | ~1.6 GB| near large-v3 accuracy, much faster       |
| `medium` / `small` | smaller| CPU-only boxes, quick drafts              |

## Usage

```bash
# single file, txt + json
./build/calltx -m models/ggml-large-v3.bin call_001.wav

# whole folder, English, all formats, into ./out
./build/calltx -m models/ggml-large-v3.bin -l en -f txt,srt,json -o out recordings/

# stereo call (agent on L, customer on R) -> speaker-labelled transcript
./build/calltx -m models/ggml-large-v3.bin -s -w call_001.mp3

# domain vocabulary hint improves names/jargon
./build/calltx -m models/ggml-large-v3.bin -p "Refinitiv, Eikon, KYC, IBAN" call.wav
```

### Output

`call_001.txt`
```
[00:00:01.240 --> 00:00:04.880] Speaker A: Thank you for calling, how can I help you today?
[00:00:05.120 --> 00:00:09.600] Speaker B: Hi, I'm calling about a failed payment on my account.
```

`call_001.json` contains per-segment `start_ms`, `end_ms`, `speaker`, `confidence`,
`text`, and (with `-w`) per-word timestamps and probabilities.

## Accuracy notes

- `--split-channels` gives the most reliable speaker separation, but only for
  recordings where each party is on its own channel. Mono recordings get no
  speaker labels (that needs a separate diarization model).
- 8 kHz telephony audio is upsampled automatically; it works, but accuracy is
  lower than wideband audio — prefer `large-v3`.
- Use low `confidence` values in the JSON to flag segments for human review.
- Exit code is `2` if any file failed, so it's safe to use in batch pipelines.
