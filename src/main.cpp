// calltx - call recording transcriber (WAV/MP3 -> timestamped transcript)
// Engine: whisper.cpp. Decoding/resampling: miniaudio.

#include "audio.hpp"
#include "whisper.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
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
};

struct Word {
    int64_t t0_ms = 0, t1_ms = 0;
    std::string text;
    float prob = 1.0f;
};

struct Segment {
    int64_t t0_ms = 0, t1_ms = 0;
    std::string speaker;  // empty unless --split-channels
    std::string text;
    float confidence = 0.0f;  // mean token probability
    std::vector<Word> words;
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

static std::vector<fs::path> collect_inputs(const std::vector<std::string>& args) {
    std::vector<fs::path> files;
    for (const auto& a : args) {
        fs::path p(a);
        if (fs::is_directory(p)) {
            for (const auto& e : fs::recursive_directory_iterator(p))
                if (e.is_regular_file() && is_audio(e.path())) files.push_back(e.path());
        } else if (fs::is_regular_file(p)) {
            if (is_audio(p)) files.push_back(p);
            else std::cerr << "skip (not .wav/.mp3): " << p << "\n";
        } else {
            std::cerr << "skip (not found): " << p << "\n";
        }
    }
    std::sort(files.begin(), files.end());
    return files;
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

    if (whisper_full(ctx, wp, pcm.data(), static_cast<int>(pcm.size())) != 0)
        throw std::runtime_error("whisper_full failed");

    detected_lang = whisper_lang_str(whisper_full_lang_id(ctx));

    const whisper_token eot = whisper_token_eot(ctx);  // ids >= eot are special tokens
    std::vector<Segment> segs;
    const int n = whisper_full_n_segments(ctx);
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
            const std::string piece = whisper_full_get_token_text(ctx, i, j);
            // A leading space marks the start of a new word; otherwise it's a sub-word piece.
            const bool starts_word = s.words.empty() || (!piece.empty() && piece[0] == ' ');
            if (starts_word) {
                s.words.push_back({td.t0 * 10, td.t1 * 10, trim(piece), td.p});
            } else {
                Word& w = s.words.back();
                w.text += piece;
                w.t1_ms = td.t1 * 10;
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

// --------------------------------------------------------------- writers ---

static void write_txt(const fs::path& p, const std::vector<Segment>& segs) {
    std::ofstream f(p);
    for (const auto& s : segs) {
        f << "[" << format_ts(s.t0_ms) << " --> " << format_ts(s.t1_ms) << "] ";
        if (!s.speaker.empty()) f << s.speaker << ": ";
        f << s.text << "\n";
    }
}

static void write_srt(const fs::path& p, const std::vector<Segment>& segs) {
    std::ofstream f(p);
    int idx = 1;
    for (const auto& s : segs) {
        f << idx++ << "\n"
          << format_ts(s.t0_ms, ',') << " --> " << format_ts(s.t1_ms, ',') << "\n";
        if (!s.speaker.empty()) f << s.speaker << ": ";
        f << s.text << "\n\n";
    }
}

static void write_json(const fs::path& p, const fs::path& src, double duration,
                       const std::string& lang, const std::string& model,
                       const std::vector<Segment>& segs) {
    std::ofstream f(p);
    f << "{\n"
      << "  \"file\": \"" << json_escape(src.string()) << "\",\n"
      << "  \"duration_sec\": " << duration << ",\n"
      << "  \"language\": \"" << json_escape(lang) << "\",\n"
      << "  \"model\": \"" << json_escape(fs::path(model).filename().string()) << "\",\n"
      << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        f << "    {\"start_ms\": " << s.t0_ms << ", \"end_ms\": " << s.t1_ms
          << ", \"start\": \"" << format_ts(s.t0_ms) << "\", \"end\": \"" << format_ts(s.t1_ms) << "\"";
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

    int failures = 0;
    for (const auto& file : files) {
        const auto t_start = std::chrono::steady_clock::now();
        std::cerr << "==> " << file.string() << "\n";

        AudioData audio;
        std::string err;
        if (!load_audio(file.string(), audio, err)) {
            std::cerr << "    error: " << err << "\n";
            ++failures;
            continue;
        }

        std::vector<Segment> segs;
        std::string lang;
        try {
            if (o.split_channels && audio.channels.size() >= 2) {
                const char* labels[2] = {"Speaker A", "Speaker B"};
                for (int c = 0; c < 2; ++c) {
                    std::string l;
                    auto part = transcribe(ctx, audio.channels[c], o, labels[c], l);
                    if (lang.empty()) lang = l;
                    segs.insert(segs.end(), part.begin(), part.end());
                }
                std::stable_sort(segs.begin(), segs.end(), [](const Segment& a, const Segment& b) {
                    return a.t0_ms < b.t0_ms;
                });
            } else {
                if (o.split_channels)
                    std::cerr << "    note: file is mono, --split-channels ignored\n";
                segs = transcribe(ctx, audio.mono(), o, "", lang);
            }
        } catch (const std::exception& e) {
            std::cerr << "    error: " << e.what() << "\n";
            ++failures;
            continue;
        }

        const fs::path base = fs::path(o.out_dir) / file.stem();
        if (o.formats.count("txt"))  write_txt(base.string() + ".txt", segs);
        if (o.formats.count("srt"))  write_srt(base.string() + ".srt", segs);
        if (o.formats.count("json")) write_json(base.string() + ".json", file, audio.duration_sec(),
                                                lang, o.model, segs);

        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::fprintf(stderr, "    %zu segments, lang=%s, audio %.1fs, took %.1fs (%.1fx realtime)\n",
                     segs.size(), lang.c_str(), audio.duration_sec(), secs,
                     secs > 0 ? audio.duration_sec() / secs : 0.0);
    }

    whisper_free(ctx);
    return failures ? 2 : 0;
}
