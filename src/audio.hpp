#pragma once

#include <string>
#include <vector>

// Decoded audio, resampled to 16 kHz float PCM (what Whisper expects).
struct AudioData {
    static constexpr int kSampleRate = 16000;

    int source_channels = 0;
    std::vector<std::vector<float>> channels;  // one buffer per channel

    std::vector<float> mono() const;            // average of all channels
    double duration_sec() const;
};

// Decodes .wav / .mp3 (also .flac) and resamples to 16 kHz.
// Channels are kept separate so stereo call recordings (agent/customer on
// L/R) can be transcribed per speaker.
bool load_audio(const std::string& path, AudioData& out, std::string& err);
