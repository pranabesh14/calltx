// calltx - call recording transcriber (WAV/MP3 -> timestamped transcript)
// Engine: whisper.cpp (+ Silero VAD). Decoding/resampling: miniaudio.

#include "audio.hpp"
#include "deadair.hpp"
#include "whisper.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ----------------------------------------------------------------- types ---

struct Options {
    std::string model;
    std::vector<std::string> inputs;
    std::string out_dir = ".";
    std::set<std::string> formats{"txt", "json"};
    std::string language = "auto";
    std::string prompt;  // domain vocabulary hint (product names, jargon)
    int threads = static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    int beam_size = 5;
    bool split_channels = false;
    bool words = false;
    bool use_gpu = true;
    bool verbose = false;

    // VAD + dead air
    std::string vad_model;       // Silero VAD ggml model; empty = VAD off
    float vad_threshold = 0.5f;  // speech probability threshold
    bool dead_air = true;
    double min_gap_sec = 3.0;
    float silence_db = -45.0f;
};

struct Word {
    int64_t t0_ms = 0, t1_ms = 0;
    std::string text;
    float prob = 1.0f;
};

// One transcript entry: either speech or a gap (dead air / non-speech).
struct Segment {
    std::string type = "speech";  // "speech" | "dead_air" | "non_speech"
    int64_t t0_ms = 0, t1_ms = 0;
    std::string speaker;          // speech only, when --split-channels
    std::string text;
    float confidence = 0.0f;      // speech only: mean token probability
    float level_db = 0.0f;        // gaps only: loudness inside the gap
    std::vector<Word> words;

    bool is_speech() const { return type == "speech"; }
};

struct GapSummary {
    std::string method;  // "vad" | "energy" | "off"
    int dead_air_count = 0, non_speech_count = 0;
    int64_t dead_air_ms = 0, non_speech_ms = 0, longest_dead_air_ms = 0;
};

// --------------------------------------------------------------- helpers ---

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// HH:MM:SS.mmm (sep='.') or HH:MM:SS,mmm (sep=',' for SRT)
static std::string format_ts(int64_t ms, char sep = '.') {
    if (ms < 0) ms = 0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld%c%03lld",
                  (long long)(ms / 3600000), (long long)(ms / 60000 % 60),
                  (long long)(ms / 1000 % 60), sep, (long long)(ms % 1000));
    return buf;
}

static std::string fmt1(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}

static std::string secs(int64_t ms) { return fmt1(ms / 1000.0); }

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += static_cast<char>(c);  // UTF-8 passes through
                }
        }
    }
    return o;
}

// Display text for an entry: the transcript, or e.g. "[DEAD AIR 36.7s]".
static std::string display_text(const Segment& s) {
    if (s.is_speech()) return s.speaker.empty() ? s.text : s.speaker + ": " + s.text;
    const char* lbl = s.type == "dead_air" ? "DEAD AIR" : "NON-SPEECH";
    return std::string("[") + lbl + " " + secs(s.t1_ms - s.t0_ms) + "s]";
}

static void quiet_log(enum ggml_log_level, const char*, void*) {}

// ------------------------------------------------------------------- CLI ---

static void usage(const char* argv0) {
    std::cerr <<
        "Usage: " << argv0 << " -m MODEL [options] INPUT [INPUT ...]\n"
        "\n"
        "INPUT may be a .wav/.mp3 file or a directory (scanned for .wav/.mp3).\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH        whisper ggml model (e.g. ggml-large-v3.bin)  [required]\n"
        "  -o, --out-dir DIR       output directory (default: .)\n"
        "  -f, --format LIST       comma list of txt,srt,json (default: txt,json)\n"
        "  -l, --lang CODE         language code, e.g. en, hi, de, or auto (default: auto)\n"
        "  -t, --threads N         CPU threads (default: min(8, cores))\n"
        "  -b, --beam N            beam size, 1 = greedy (default: 5)\n"
        "  -p, --prompt TEXT       vocabulary hint, e.g. \"Refinitiv, KYC, IBAN\"\n"
        "  -s, --split-channels    stereo calls: transcribe L/R separately as speakers\n"
        "  -w, --words             add word-level timestamps (json)\n"
        "\n"
        "Voice activity detection and dead air:\n"
        "      --vad-model PATH    Silero VAD model (e.g. ggml-silero-v6.2.0.bin): skips\n"
        "                          non-speech during transcription and separates dead air\n"
        "                          from hold music\n"
        "      --vad-threshold P   speech probability threshold 0-1 (default: 0.5)\n"
        "      --min-gap SEC       report gaps at least this long (default: 3)\n"
        "      --silence-db DB     gaps quieter than this are dead air (default: -45)\n"
        "      --no-dead-air       don't report gaps\n"
        "\n"
        "      --no-gpu            force CPU\n"
        "  -v, --verbose           show whisper.cpp logs\n"
        "  -h, --help\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else if (a == "-m" || a == "--model")   o.model = need(i);
        else if (a == "-o" || a == "--out-dir") o.out_dir = need(i);
        else if (a == "-l" || a == "--lang")    o.language = need(i);
        else if (a == "-p" || a == "--prompt")  o.prompt = need(i);
        else if (a == "-t" || a == "--threads") o.threads = std::max(1, std::stoi(need(i)));
        else if (a == "-b" || a == "--beam")    o.beam_size = std::max(1, std::stoi(need(i)));
        else if (a == "-s" || a == "--split-channels") o.split_channels = true;
        else if (a == "-w" || a == "--words")   o.words = true;
        else if (a == "--vad-model")            o.vad_model = need(i);
        else if (a == "--vad-threshold")        o.vad_threshold = std::clamp(std::stof(need(i)), 0.0f, 1.0f);
        else if (a == "--min-gap")              o.min_gap_sec = std::max(0.1, std::stod(need(i)));
        else if (a == "--silence-db")           o.silence_db = std::stof(need(i));
        else if (a == "--no-dead-air")          o.dead_air = false;
        else if (a == "--no-gpu")               o.use_gpu = false;
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (a == "-f" || a == "--format") {
            o.formats.clear();
            std::stringstream ss(need(i));
            std::string f;
            while (std::getline(ss, f, ',')) {
                f = lower(trim(f));
                if (f != "txt" && f != "srt" && f != "json")
                    throw std::runtime_error("unknown format: " + f);
                o.formats.insert(f);
            }
        }
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option: " + a);
        else o.inputs.push_back(a);
    }
    if (o.model.empty() || o.inputs.empty()) { usage(argv[0]); return false; }
    return true;
}

static bool is_audio(const fs::path& p) {
    const std::string ext = lower(p.extension().string());
    return ext == ".wav" || ext == ".mp3";
}

// An input file and the output name (without extension) its transcripts get.
struct InputFile {
    fs::path path;
    fs::path out_base;  // relative to --out-dir
};

static std::vector<InputFile> collect_inputs(const std::vector<std::string>& args) {
    std::vector<InputFile> files;
    for (const auto& a : args) {
        fs::path p(a);
        if (fs::is_directory(p)) {
            // Keep the folder structure: recordings/2026-10/a/call.wav -> out/2026-10/a/call.txt,
            // so recordings with the same file name in different folders don't overwrite each other.
            std::vector<fs::path> found;
            for (const auto& e : fs::recursive_directory_iterator(p))
                if (e.is_regular_file() && is_audio(e.path())) found.push_back(e.path());
            std::sort(found.begin(), found.end());
            for (const auto& f : found)
                files.push_back({f, fs::relative(f, p).replace_extension()});
        } else if (fs::is_regular_file(p)) {
            if (is_audio(p)) files.push_back({p, p.stem()});
            else std::cerr << "skip (not .wav/.mp3): " << p << "\n";
        } else {
            std::cerr << "skip (not found): " << p << "\n";
        }
    }

    // The same recording listed twice (e.g. a file and its folder) is transcribed once.
    std::set<fs::path> seen;
    files.erase(std::remove_if(files.begin(), files.end(), [&](const InputFile& f) {
                    std::error_code ec;
                    fs::path c = fs::weakly_canonical(f.path, ec);
                    return !seen.insert(ec ? f.path : c).second;
                }),
                files.end());

    // Remaining clashes get a distinguishing name so every input keeps its own transcript:
    //   same folder, different format (call.wav + call.mp3)   -> call_wav
    //   different folders passed as files (a/call.wav, b/call.wav) -> b_call
    std::map<std::string, fs::path> used;  // output name -> source folder of its first user
    auto key_of = [](const fs::path& b) { return lower(b.generic_string()); };
    for (auto& f : files) {
        const fs::path original = f.out_base;
        auto it = used.find(key_of(f.out_base));
        if (it != used.end()) {
            const fs::path src_dir = f.path.parent_path();
            const std::string folder = src_dir.filename().string();
            if (fs::absolute(src_dir) != fs::absolute(it->second) && !folder.empty() && folder != "." && folder != "..")
                f.out_base = original.parent_path() / (folder + "_" + original.filename().string());
            else
                f.out_base = fs::path(original.string() + "_" + lower(f.path.extension().string()).substr(1));
            const fs::path candidate = f.out_base;
            for (int n = 2; used.count(key_of(f.out_base)); ++n)
                f.out_base = fs::path(candidate.string() + "_" + std::to_string(n));
            std::cerr << "note: " << f.path.generic_string() << " -> " << f.out_base.generic_string()
                      << " (name " << original.generic_string() << " already used)\n";
        }
        used.emplace(key_of(f.out_base), f.path.parent_path());
    }
    return files;
}

// ------------------------------------------------------------------- VAD ---

static whisper_vad_params vad_params(const Options& o) {
    whisper_vad_params vp = whisper_vad_default_params();
    vp.threshold = o.vad_threshold;
    return vp;
}

// Speech regions of one channel, on the original timeline.
static std::vector<deadair::Interval> detect_speech(whisper_vad_context* vctx,
                                                    const std::vector<float>& pcm,
                                                    const Options& o) {
    whisper_vad_segments* segs =
        whisper_vad_segments_from_samples(vctx, vad_params(o), pcm.data(), static_cast<int>(pcm.size()));
    if (!segs) throw std::runtime_error("VAD failed");
    std::vector<deadair::Interval> out;
    const int n = whisper_vad_segments_n_segments(segs);
    for (int i = 0; i < n; ++i) {
        // VAD segment times are in centiseconds
        out.push_back({static_cast<int64_t>(whisper_vad_segments_get_segment_t0(segs, i) * 10.0f),
                       static_cast<int64_t>(whisper_vad_segments_get_segment_t1(segs, i) * 10.0f)});
    }
    whisper_vad_free_segments(segs);
    return out;
}

// ------------------------------------------------------------ transcribe ---

static std::vector<Segment> transcribe(whisper_context* ctx, const std::vector<float>& pcm,
                                       const Options& o, const std::string& speaker,
                                       std::string& detected_lang) {
    whisper_full_params wp = whisper_full_default_params(
        o.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wp.n_threads        = o.threads;
    wp.language         = o.language.c_str();
    wp.translate        = false;          // keep original language
    wp.no_timestamps    = false;
    wp.token_timestamps = o.words;
    wp.print_progress   = false;
    wp.print_realtime   = false;
    wp.print_special    = false;
    wp.print_timestamps = false;
    wp.suppress_blank   = true;
    wp.temperature      = 0.0f;           // deterministic first pass (fallback still kicks in)
    wp.no_speech_thold  = 0.6f;           // drop segments that are mostly silence
    wp.beam_search.beam_size = o.beam_size;
    if (!o.prompt.empty()) wp.initial_prompt = o.prompt.c_str();

    if (!o.vad_model.empty()) {
        // Only speech regions are decoded; whisper.cpp maps the segment and
        // token times back to the original recording timeline.
        wp.vad            = true;
        wp.vad_model_path = o.vad_model.c_str();
        wp.vad_params     = vad_params(o);
    }

    if (whisper_full(ctx, wp, pcm.data(), static_cast<int>(pcm.size())) != 0)
        throw std::runtime_error("whisper_full failed");

    const int n = whisper_full_n_segments(ctx);
    const int lang_id = n > 0 ? whisper_full_lang_id(ctx) : -1;
    const char* lang = lang_id >= 0 ? whisper_lang_str(lang_id) : nullptr;
    detected_lang = lang ? lang : "";

    const whisper_token eot = whisper_token_eot(ctx);  // ids >= eot are special tokens
    std::vector<Segment> segs;
    for (int i = 0; i < n; ++i) {
        Segment s;
        s.t0_ms   = whisper_full_get_segment_t0(ctx, i) * 10;  // centiseconds -> ms
        s.t1_ms   = whisper_full_get_segment_t1(ctx, i) * 10;
        s.text    = trim(whisper_full_get_segment_text(ctx, i));
        s.speaker = speaker;
        if (s.text.empty()) continue;

        double psum = 0.0;
        int pcount = 0;
        const int nt = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < nt; ++j) {
            const whisper_token_data td = whisper_full_get_token_data(ctx, i, j);
            if (td.id >= eot) continue;
            psum += td.p;
            ++pcount;

            if (!o.words) continue;
            // whisper_full_get_token_t0/t1 are on the original timeline even with VAD
            // (td.t0/t1 would be in VAD-trimmed time).
            const int64_t w0 = whisper_full_get_token_t0(ctx, i, j) * 10;
            const int64_t w1 = whisper_full_get_token_t1(ctx, i, j) * 10;
            const std::string piece = whisper_full_get_token_text(ctx, i, j);
            // A leading space marks the start of a new word; otherwise it's a sub-word piece.
            const bool starts_word = s.words.empty() || (!piece.empty() && piece[0] == ' ');
            if (starts_word) {
                s.words.push_back({w0, w1, trim(piece), td.p});
            } else {
                Word& w = s.words.back();
                w.text += piece;
                w.t1_ms = w1;
                w.prob  = std::min(w.prob, td.p);
            }
        }
        s.words.erase(std::remove_if(s.words.begin(), s.words.end(),
                                     [](const Word& w) { return w.text.empty(); }),
                      s.words.end());
        s.confidence = pcount ? static_cast<float>(psum / pcount) : 0.0f;
        segs.push_back(std::move(s));
    }
    return segs;
}

// -------------------------------------------------------------- dead air ---

static GapSummary add_gaps(std::vector<Segment>& segs, const std::vector<deadair::Gap>& gaps,
                           const std::string& method) {
    GapSummary sum;
    sum.method = method;
    for (const auto& g : gaps) {
        Segment s;
        s.type     = deadair::json_type(g.kind);
        s.t0_ms    = g.t0_ms;
        s.t1_ms    = g.t1_ms;
        s.level_db = g.level_db;
        segs.push_back(s);
        if (g.kind == deadair::Kind::DeadAir) {
            ++sum.dead_air_count;
            sum.dead_air_ms += g.duration_ms();
            sum.longest_dead_air_ms = std::max(sum.longest_dead_air_ms, g.duration_ms());
        } else {
            ++sum.non_speech_count;
            sum.non_speech_ms += g.duration_ms();
        }
    }
    std::stable_sort(segs.begin(), segs.end(),
                     [](const Segment& a, const Segment& b) { return a.t0_ms < b.t0_ms; });
    return sum;
}

// --------------------------------------------------------------- writers ---

static void write_txt(const fs::path& p, const std::vector<Segment>& segs) {
    std::ofstream f(p);
    for (const auto& s : segs)
        f << "[" << format_ts(s.t0_ms) << " --> " << format_ts(s.t1_ms) << "] " << display_text(s) << "\n";
}

static void write_srt(const fs::path& p, const std::vector<Segment>& segs) {
    std::ofstream f(p);
    int idx = 1;
    for (const auto& s : segs) {
        f << idx++ << "\n"
          << format_ts(s.t0_ms, ',') << " --> " << format_ts(s.t1_ms, ',') << "\n"
          << display_text(s) << "\n\n";
    }
}

static void write_json(const fs::path& p, const fs::path& src, double duration,
                       const std::string& lang, const Options& o,
                       const GapSummary& gs, const std::vector<Segment>& segs) {
    std::ofstream f(p);
    f << "{\n"
      << "  \"file\": \"" << json_escape(src.string()) << "\",\n"
      << "  \"duration_sec\": " << duration << ",\n"
      << "  \"language\": \"" << json_escape(lang) << "\",\n"
      << "  \"model\": \"" << json_escape(fs::path(o.model).filename().string()) << "\",\n"
      << "  \"vad\": " << (o.vad_model.empty() ? "false" : "true") << ",\n";
    if (gs.method != "off") {
        f << "  \"dead_air\": {\"method\": \"" << gs.method << "\""
          << ", \"min_gap_sec\": " << o.min_gap_sec
          << ", \"silence_db\": " << o.silence_db
          << ", \"count\": " << gs.dead_air_count
          << ", \"total_sec\": " << secs(gs.dead_air_ms)
          << ", \"longest_sec\": " << secs(gs.longest_dead_air_ms)
          << ", \"non_speech_count\": " << gs.non_speech_count
          << ", \"non_speech_total_sec\": " << secs(gs.non_speech_ms) << "},\n";
    }
    f << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        f << "    {\"type\": \"" << s.type << "\""
          << ", \"start_ms\": " << s.t0_ms << ", \"end_ms\": " << s.t1_ms
          << ", \"start\": \"" << format_ts(s.t0_ms) << "\", \"end\": \"" << format_ts(s.t1_ms) << "\"";
        if (s.is_speech()) {
            if (!s.speaker.empty()) f << ", \"speaker\": \"" << json_escape(s.speaker) << "\"";
            f << ", \"confidence\": " << s.confidence
              << ", \"text\": \"" << json_escape(s.text) << "\"";
            if (!s.words.empty()) {
                f << ", \"words\": [";
                for (size_t j = 0; j < s.words.size(); ++j) {
                    const auto& w = s.words[j];
                    f << (j ? ", " : "") << "{\"start_ms\": " << w.t0_ms << ", \"end_ms\": " << w.t1_ms
                      << ", \"p\": " << w.prob << ", \"text\": \"" << json_escape(w.text) << "\"}";
                }
                f << "]";
            }
        } else {
            f << ", \"duration_sec\": " << secs(s.t1_ms - s.t0_ms)
              << ", \"level_db\": " << fmt1(s.level_db);
        }
        f << "}" << (i + 1 < segs.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

// ------------------------------------------------------------------ main ---

int main(int argc, char** argv) {
    Options o;
    try {
        if (!parse_args(argc, argv, o)) return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    if (!o.verbose) whisper_log_set(quiet_log, nullptr);

    const auto files = collect_inputs(o.inputs);
    if (files.empty()) { std::cerr << "no .wav/.mp3 inputs found\n"; return 1; }
    fs::create_directories(o.out_dir);

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = o.use_gpu;
    whisper_context* ctx = whisper_init_from_file_with_params(o.model.c_str(), cp);
    if (!ctx) { std::cerr << "error: failed to load model " << o.model << "\n"; return 1; }

    // Separate VAD context for the speech map used by dead-air detection.
    whisper_vad_context* vctx = nullptr;
    if (!o.vad_model.empty()) {
        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = o.threads;
        vcp.use_gpu   = false;  // tiny model, CPU is fine
        vctx = whisper_vad_init_from_file_with_params(o.vad_model.c_str(), vcp);
        if (!vctx) {
            std::cerr << "error: failed to load VAD model " << o.vad_model << "\n";
            whisper_free(ctx);
            return 1;
        }
    } else if (o.dead_air) {
        std::cerr << "note: no --vad-model, dead air is detected by loudness only "
                     "(hold music is not separated)\n";
    }

    deadair::Params dp;
    dp.min_gap_ms = static_cast<int64_t>(o.min_gap_sec * 1000.0);
    dp.silence_db = o.silence_db;

    int failures = 0;
    for (const auto& input : files) {
        const fs::path& file = input.path;
        const auto t_start = std::chrono::steady_clock::now();
        std::cerr << "==> " << file.string() << "\n";

        AudioData audio;
        std::string err;
        if (!load_audio(file.string(), audio, err)) {
            std::cerr << "    error: " << err << "\n";
            ++failures;
            continue;
        }
        const int64_t duration_ms = static_cast<int64_t>(audio.duration_sec() * 1000.0);

        std::vector<Segment> segs;
        std::vector<deadair::Interval> speech;  // VAD speech map, all channels
        GapSummary gs;
        gs.method = "off";
        std::string lang;
        try {
            const bool split = o.split_channels && audio.channels.size() >= 2;
            if (o.split_channels && !split)
                std::cerr << "    note: file is mono, --split-channels ignored\n";

            // Channels to transcribe: L/R separately, or one mono mix.
            std::vector<std::vector<float>> mono_holder;
            const std::vector<std::vector<float>>* tracks = &audio.channels;
            if (!split) {
                mono_holder.push_back(audio.mono());
                tracks = &mono_holder;
            }
            const char* labels[2] = {"Speaker A", "Speaker B"};

            for (size_t c = 0; c < (split ? 2u : 1u); ++c) {
                const auto& pcm = (*tracks)[c];
                std::string l;
                auto part = transcribe(ctx, pcm, o, split ? labels[c] : "", l);
                if (lang.empty()) lang = l;
                segs.insert(segs.end(), part.begin(), part.end());
                if (vctx && o.dead_air) {
                    auto sp = detect_speech(vctx, pcm, o);
                    speech.insert(speech.end(), sp.begin(), sp.end());
                }
            }
            std::stable_sort(segs.begin(), segs.end(),
                             [](const Segment& a, const Segment& b) { return a.t0_ms < b.t0_ms; });

            if (o.dead_air) {
                // Stereo: a frame is quiet only if both channels are quiet.
                const auto levels = deadair::frame_levels(audio.channels, AudioData::kSampleRate);
                const auto gaps = vctx ? deadair::gaps_from_speech(speech, duration_ms, levels, dp)
                                       : deadair::gaps_from_energy(duration_ms, levels, dp);
                gs = add_gaps(segs, gaps, vctx ? "vad" : "energy");
            }
        } catch (const std::exception& e) {
            std::cerr << "    error: " << e.what() << "\n";
            ++failures;
            continue;
        }

        const fs::path base = fs::path(o.out_dir) / input.out_base;
        fs::create_directories(base.parent_path());
        if (o.formats.count("txt"))  write_txt(base.string() + ".txt", segs);
        if (o.formats.count("srt"))  write_srt(base.string() + ".srt", segs);
        if (o.formats.count("json")) write_json(base.string() + ".json", file, audio.duration_sec(),
                                                lang, o, gs, segs);

        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::fprintf(stderr, "    %zu entries, lang=%s, audio %.1fs, took %.1fs (%.1fx realtime)\n",
                     segs.size(), lang.empty() ? "-" : lang.c_str(), audio.duration_sec(), took,
                     took > 0 ? audio.duration_sec() / took : 0.0);
        std::string exts;
        for (const auto& f : o.formats) exts += (exts.empty() ? "" : ",") + f;
        std::fprintf(stderr, "    -> %s.{%s}\n", base.generic_string().c_str(), exts.c_str());
        if (gs.method != "off")
            std::fprintf(stderr, "    dead air: %d gaps, %ss total (longest %ss); non-speech: %d, %ss\n",
                         gs.dead_air_count, secs(gs.dead_air_ms).c_str(),
                         secs(gs.longest_dead_air_ms).c_str(), gs.non_speech_count,
                         secs(gs.non_speech_ms).c_str());
    }

    if (vctx) whisper_vad_free(vctx);
    whisper_free(ctx);
    return failures ? 2 : 0;
}
