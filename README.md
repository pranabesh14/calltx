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

The two extra flags are only there so the .exe runs on other Windows machines without first installing the Visual C++ Redistributable:

	•	CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded puts the C++ runtime inside the .exe.
	•	GGML_OPENMP=OFF drops Microsoft's OpenMP DLL; whisper.cpp uses its own threading instead.

## Get a model

```bash
mkdir -p models
curl -L -o models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
  
#Windows  
  Invoke-WebRequest -Uri " https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin" -OutFile ".\models\ggml-large-v3.bin"

```

| Model              | Size   | Use when                                  |
|--------------------|--------|-------------------------------------------|
| `large-v3`         | ~3 GB  | best accuracy (recommended for calls)     |
| `large-v3-turbo`   | ~1.6 GB| near large-v3 accuracy, much faster       |
| `medium` / `small` | smaller| CPU-only boxes, quick drafts              |

Optional but recommended — the Silero VAD model (~1 MB):

```bash
curl -L -o models/ggml-silero-v6.2.0.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
  
#Windows  
  Invoke-WebRequest -Uri "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin" -OutFile ".\models\ggml-silero-v6.2.0.bin"
```

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

# recommended for calls: VAD on (skips silence/hold music, separates dead air from music/calls)
./build/calltx -m models/ggml-large-v3.bin --vad-model models/ggml-silero-v6.2.0.bin -s call.wav
```

### Output

Each recording gets its own transcript named after it. For a folder input the folder
structure is kept, so same-named recordings in different folders don't overwrite each other:

```
recordings/2026-10-01/C1001/call.wav  ->  out/2026-10-01/C1001/call.txt
recordings/2026-10-01/C1002/call.wav  ->  out/2026-10-01/C1002/call.txt
recordings/agent_smith.mp3            ->  out/agent_smith.txt
```

Any remaining name clash is resolved with a suffix and reported on stderr
(`call.wav` + `call.mp3` in one folder -> `call.txt` + `call_wav.txt`).

`call_001.txt`
```
[00:00:01.240 --> 00:00:04.880] Speaker A: Thank you for calling, how can I help you today?
[00:00:05.120 --> 00:00:09.600] Speaker B: Hi, I'm calling about a failed payment on my account.
```

`call_001.json` contains per-segment `start_ms`, `end_ms`, `speaker`, `confidence`,
`text`, and (with `-w`) per-word timestamps and probabilities.

## Dead air

Gaps of at least `--min-gap` seconds (default 3) are written into the transcript:

```
[00:00:05.190 --> 00:00:08.250] Speaker B: Hi, I'm calling about a failed payment on my account.
[00:00:08.250 --> 00:00:16.480] [DEAD AIR 8.2s]
[00:00:16.480 --> 00:00:28.550] [NON-SPEECH 12.1s]
[00:00:28.550 --> 00:00:31.420] Speaker A: Thanks for waiting.
```

- **DEAD AIR**: no speech and the line is quiet (below `--silence-db`, default -45 dBFS).
- **NON-SPEECH**: no speech but there is sound, e.g. hold music or noise.
- In stereo, a stretch only counts when **both** channels have no speech.
- With `--vad-model`, speech is found by Silero VAD and silence vs. music is told apart by
  loudness. Without it, dead air is found by loudness alone and hold music isn't detected.
- Tune `--silence-db` if your lines have loud background hiss (raise it, e.g. -40) or
  very quiet speakers are being cut (lower `--vad-threshold`, e.g. 0.35).
- Disable with `--no-dead-air`.

The JSON gets a summary block and typed entries:

```json
"dead_air": {"method": "vad", "min_gap_sec": 3, "silence_db": -45, "count": 2,
             "total_sec": 12.4, "longest_sec": 8.2, "non_speech_count": 1, "non_speech_total_sec": 12.1},
"segments": [
  {"type": "speech", "start_ms": 5190, "end_ms": 8250, "speaker": "Speaker B", "confidence": 0.91, "text": "..."},
  {"type": "dead_air", "start_ms": 8250, "end_ms": 16480, "duration_sec": 8.2, "level_db": -68.2},
  {"type": "non_speech", "start_ms": 16480, "end_ms": 28550, "duration_sec": 12.1, "level_db": -24.7}
]
```

Consumers that only want speech should filter on `"type": "speech"`.

## Tests

```bash
cmake -B build -DCALLTX_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Accuracy notes

- `--split-channels` gives the most reliable speaker separation, but only for
  recordings where each party is on its own channel. Mono recordings get no
  speaker labels (that needs a separate diarization model).
- 8 kHz telephony audio is upsampled automatically; it works, but accuracy is
  lower than wideband audio — prefer `large-v3`.
- Use low `confidence` values in the JSON to flag segments for human review.
- Exit code is `2` if any file failed, so it's safe to use in batch pipelines.
