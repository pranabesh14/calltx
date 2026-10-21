#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_GENERATION
#include "miniaudio.h"

#include "audio.hpp"

std::vector<float> AudioData::mono() const {
    if (channels.empty()) return {};
    if (channels.size() == 1) return channels[0];
    const size_t n = channels[0].size();
    std::vector<float> out(n, 0.0f);
    const float scale = 1.0f / static_cast<float>(channels.size());
    for (const auto& ch : channels)
        for (size_t i = 0; i < n; ++i) out[i] += ch[i] * scale;
    return out;
}

double AudioData::duration_sec() const {
    return channels.empty() ? 0.0
                            : static_cast<double>(channels[0].size()) / kSampleRate;
}

bool load_audio(const std::string& path, AudioData& out, std::string& err) {
    // channels = 0 -> keep native channel count; sampleRate = 16000 -> resample.
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, AudioData::kSampleRate);
    ma_decoder dec;
    ma_result r = ma_decoder_init_file(path.c_str(), &cfg, &dec);
    if (r != MA_SUCCESS) {
        err = std::string("cannot decode audio: ") + ma_result_description(r);
        return false;
    }

    const ma_uint32 nch = dec.outputChannels;
    if (nch == 0) {
        ma_decoder_uninit(&dec);
        err = "audio has zero channels";
        return false;
    }

    out.source_channels = static_cast<int>(nch);
    out.channels.assign(nch, {});

    constexpr ma_uint64 kChunk = 8192;
    std::vector<float> buf(kChunk * nch);
    for (;;) {
        ma_uint64 got = 0;
        r = ma_decoder_read_pcm_frames(&dec, buf.data(), kChunk, &got);
        for (ma_uint64 f = 0; f < got; ++f)
            for (ma_uint32 c = 0; c < nch; ++c)
                out.channels[c].push_back(buf[f * nch + c]);
        if (r != MA_SUCCESS || got < kChunk) break;  // MA_AT_END or error
    }
    ma_decoder_uninit(&dec);

    if (out.channels[0].empty()) {
        err = "decoded audio is empty";
        return false;
    }
    return true;
}
