#pragma once

// Dead-air detection: finds the gaps between speech and classifies them.
//
//   DeadAir   - no speech and the line is quiet (true silence)
//   NonSpeech - no speech but there is sound (hold music, noise)
//
// Pure functions with no whisper.cpp dependency, so they can be unit tested.

#include <cstdint>
#include <vector>

namespace deadair {

constexpr int kFrameMs = 20;  // loudness is measured in 20 ms frames

struct Interval {
    int64_t t0_ms = 0, t1_ms = 0;
};

enum class Kind { DeadAir, NonSpeech };

struct Gap {
    int64_t t0_ms = 0, t1_ms = 0;
    Kind kind = Kind::DeadAir;
    float level_db = -120.0f;  // 90th-percentile frame loudness inside the gap (dBFS)
    int64_t duration_ms() const { return t1_ms - t0_ms; }
};

struct Params {
    int64_t min_gap_ms = 3000;   // shorter pauses are not reported
    float silence_db = -45.0f;   // gaps quieter than this are dead air
};

// Loudness per 20 ms frame in dBFS, taking the loudest channel in each frame
// (so a stereo frame counts as quiet only when both parties are quiet).
std::vector<float> frame_levels(const std::vector<std::vector<float>>& channels, int sample_rate);

// Sorts and merges overlapping/touching intervals.
std::vector<Interval> merge(std::vector<Interval> v);

// With VAD: gaps between speech regions of at least min_gap_ms,
// each classified as DeadAir or NonSpeech by its loudness.
std::vector<Gap> gaps_from_speech(const std::vector<Interval>& speech, int64_t duration_ms,
                                  const std::vector<float>& levels, const Params& p);

// Without VAD: quiet stretches of at least min_gap_ms (all DeadAir).
// Hold music cannot be detected this way because it is not quiet.
std::vector<Gap> gaps_from_energy(int64_t duration_ms, const std::vector<float>& levels,
                                  const Params& p);

const char* label(Kind k);      // "DEAD AIR" / "NON-SPEECH"
const char* json_type(Kind k);  // "dead_air" / "non_speech"

}  // namespace deadair
