#include "deadair.hpp"

#include <algorithm>
#include <cmath>

namespace deadair {

namespace {

constexpr float kFloorDb = -120.0f;
constexpr int kMaxBurstFrames = 5;  // loud blips up to 100 ms don't break a silence

int64_t frame_index(int64_t ms) { return ms / kFrameMs; }

// 90th-percentile loudness of the frames in [t0, t1): robust against a single click,
// but a gap that is mostly music still reads as loud.
float gap_level(const std::vector<float>& levels, int64_t t0_ms, int64_t t1_ms) {
    const int64_t a = std::clamp<int64_t>(frame_index(t0_ms), 0, (int64_t)levels.size());
    const int64_t b = std::clamp<int64_t>(frame_index(t1_ms), a, (int64_t)levels.size());
    if (a == b) return kFloorDb;
    std::vector<float> v(levels.begin() + a, levels.begin() + b);
    const size_t k = static_cast<size_t>(0.9 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

Gap make_gap(int64_t t0, int64_t t1, const std::vector<float>& levels, const Params& p) {
    Gap g;
    g.t0_ms = t0;
    g.t1_ms = t1;
    g.level_db = gap_level(levels, t0, t1);
    g.kind = g.level_db < p.silence_db ? Kind::DeadAir : Kind::NonSpeech;
    return g;
}

}  // namespace

std::vector<float> frame_levels(const std::vector<std::vector<float>>& channels, int sample_rate) {
    if (channels.empty() || channels[0].empty()) return {};
    const size_t frame = static_cast<size_t>(sample_rate) * kFrameMs / 1000;
    const size_t n = channels[0].size();
    const size_t nframes = (n + frame - 1) / frame;

    std::vector<float> out(nframes, kFloorDb);
    for (size_t f = 0; f < nframes; ++f) {
        const size_t a = f * frame, b = std::min(n, a + frame);
        double loudest = 0.0;
        for (const auto& ch : channels) {
            double sum = 0.0;
            for (size_t i = a; i < b; ++i) sum += static_cast<double>(ch[i]) * ch[i];
            loudest = std::max(loudest, sum / static_cast<double>(b - a));
        }
        out[f] = loudest > 1e-12 ? static_cast<float>(10.0 * std::log10(loudest)) : kFloorDb;
    }
    return out;
}

std::vector<Interval> merge(std::vector<Interval> v) {
    std::sort(v.begin(), v.end(), [](const Interval& a, const Interval& b) { return a.t0_ms < b.t0_ms; });
    std::vector<Interval> out;
    for (const auto& iv : v) {
        if (iv.t1_ms <= iv.t0_ms) continue;
        if (!out.empty() && iv.t0_ms <= out.back().t1_ms)
            out.back().t1_ms = std::max(out.back().t1_ms, iv.t1_ms);
        else
            out.push_back(iv);
    }
    return out;
}

namespace {

// Quiet stretches of at least min_gap_ms inside [t0_ms, t1_ms), as DeadAir gaps.
std::vector<Gap> quiet_runs(const std::vector<float>& levels, int64_t t0_ms, int64_t t1_ms,
                            const Params& p) {
    const size_t a = static_cast<size_t>(std::clamp<int64_t>(frame_index(t0_ms), 0, (int64_t)levels.size()));
    const size_t b = static_cast<size_t>(std::clamp<int64_t>((t1_ms + kFrameMs - 1) / kFrameMs, (int64_t)a,
                                                             (int64_t)levels.size()));
    std::vector<bool> quiet(b - a);
    for (size_t i = a; i < b; ++i) quiet[i - a] = levels[i] < p.silence_db;

    // Treat short loud blips (clicks, line pops) surrounded by quiet as quiet.
    const size_t n = quiet.size();
    for (size_t i = 0; i < n;) {
        if (quiet[i]) { ++i; continue; }
        size_t j = i;
        while (j < n && !quiet[j]) ++j;
        if (i > 0 && j < n && (j - i) <= static_cast<size_t>(kMaxBurstFrames))
            std::fill(quiet.begin() + i, quiet.begin() + j, true);
        i = j;
    }

    std::vector<Gap> out;
    for (size_t i = 0; i < n;) {
        if (!quiet[i]) { ++i; continue; }
        size_t j = i;
        while (j < n && quiet[j]) ++j;
        const int64_t r0 = std::max(t0_ms, static_cast<int64_t>(a + i) * kFrameMs);
        const int64_t r1 = std::min(t1_ms, static_cast<int64_t>(a + j) * kFrameMs);
        if (r1 - r0 >= p.min_gap_ms) {
            Gap g;
            g.t0_ms = r0;
            g.t1_ms = r1;
            g.kind = Kind::DeadAir;
            g.level_db = gap_level(levels, r0, r1);
            out.push_back(g);
        }
        i = j;
    }
    return out;
}

// Splits one no-speech region into dead air (quiet runs) and non-speech (the loud rest),
// so e.g. 8 s of silence followed by 12 s of hold music is reported as two entries.
void split_region(const std::vector<float>& levels, int64_t t0, int64_t t1, const Params& p,
                  std::vector<Gap>& out) {
    int64_t cursor = t0;
    auto loud_piece = [&](int64_t a, int64_t b) {
        if (b - a < p.min_gap_ms) return;
        Gap g = make_gap(a, b, levels, p);
        g.kind = Kind::NonSpeech;
        out.push_back(g);
    };
    for (const Gap& q : quiet_runs(levels, t0, t1, p)) {
        loud_piece(cursor, q.t0_ms);
        out.push_back(q);
        cursor = q.t1_ms;
    }
    loud_piece(cursor, t1);
}

}  // namespace

std::vector<Gap> gaps_from_speech(const std::vector<Interval>& speech, int64_t duration_ms,
                                  const std::vector<float>& levels, const Params& p) {
    std::vector<Gap> gaps;
    int64_t cursor = 0;
    for (const auto& iv : merge(speech)) {
        const int64_t start = std::min(iv.t0_ms, duration_ms);
        if (start - cursor >= p.min_gap_ms) split_region(levels, cursor, start, p, gaps);
        cursor = std::max(cursor, std::min(iv.t1_ms, duration_ms));
    }
    if (duration_ms - cursor >= p.min_gap_ms) split_region(levels, cursor, duration_ms, p, gaps);
    return gaps;
}

std::vector<Gap> gaps_from_energy(int64_t duration_ms, const std::vector<float>& levels,
                                  const Params& p) {
    return quiet_runs(levels, 0, duration_ms, p);
}

const char* label(Kind k) { return k == Kind::DeadAir ? "DEAD AIR" : "NON-SPEECH"; }
const char* json_type(Kind k) { return k == Kind::DeadAir ? "dead_air" : "non_speech"; }

}  // namespace deadair
