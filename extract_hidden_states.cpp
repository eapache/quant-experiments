#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::string tokens_path;
    std::string output_path;
    int gpu_layers = 999;
    int threads = 12;
    int chunk_limit = -1;
    std::string device = "CUDA0";
};

template <typename T>
T read_value(std::ifstream & stream, const char * label) {
    T value{};
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error(std::string("failed to read ") + label);
    }
    return value;
}

void write_bytes(std::ofstream & stream, const void * data, std::size_t size,
                 const char * label) {
    stream.write(reinterpret_cast<const char *>(data), size);
    if (!stream) {
        throw std::runtime_error(std::string("failed to write ") + label);
    }
}

int parse_int(const char * value, const char * option) {
    try {
        std::size_t used = 0;
        const long parsed = std::stol(value, &used);
        if (used != std::strlen(value) || parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range("integer range");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + option);
        }
        const char * value = argv[++index];
        if (option == "--model" || option == "-m") {
            options.model_path = value;
        } else if (option == "--tokens") {
            options.tokens_path = value;
        } else if (option == "--output" || option == "-o") {
            options.output_path = value;
        } else if (option == "--gpu-layers" || option == "-ngl") {
            options.gpu_layers = parse_int(value, option.c_str());
        } else if (option == "--threads" || option == "-t") {
            options.threads = parse_int(value, option.c_str());
        } else if (option == "--chunks") {
            options.chunk_limit = parse_int(value, option.c_str());
        } else if (option == "--device" || option == "-dev") {
            options.device = value;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (options.model_path.empty() || options.tokens_path.empty() || options.output_path.empty()) {
        throw std::runtime_error(
            "usage: extract_hidden_states --model MODEL --tokens CAPTURE.kld --output FILE "
            "[--gpu-layers N] [--device CUDA0] [--threads N] [--chunks N]");
    }
    if (options.threads <= 0 || options.chunk_limit == 0 || options.chunk_limit < -1) {
        throw std::runtime_error("threads must be positive and chunks must be positive or -1");
    }
    return options;
}

struct CaptureTokens {
    uint32_t n_ctx;
    int32_t n_vocab;
    int32_t n_chunks;
    std::vector<llama_token> tokens;
};

CaptureTokens read_capture_tokens(const Options & options) {
    std::ifstream stream(options.tokens_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open token capture: " + options.tokens_path);
    }
    char magic[8];
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, "_logits_", sizeof(magic)) != 0) {
        throw std::runtime_error(options.tokens_path + " is not a llama.cpp logits capture");
    }
    CaptureTokens result;
    result.n_ctx = read_value<uint32_t>(stream, "context length");
    result.n_vocab = read_value<int32_t>(stream, "vocabulary size");
    result.n_chunks = read_value<int32_t>(stream, "chunk count");
    if (result.n_ctx < 4 || result.n_chunks <= 0 || result.n_vocab <= 0) {
        throw std::runtime_error("invalid logits capture header");
    }
    if (options.chunk_limit > 0) {
        result.n_chunks = std::min(result.n_chunks, options.chunk_limit);
    }
    result.tokens.resize(static_cast<std::size_t>(result.n_ctx) * result.n_chunks);
    stream.read(reinterpret_cast<char *>(result.tokens.data()),
                result.tokens.size() * sizeof(result.tokens[0]));
    if (!stream) {
        throw std::runtime_error("failed to read capture tokens");
    }
    return result;
}

void extract(const Options & options) {
    const CaptureTokens capture = read_capture_tokens(options);

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    std::vector<ggml_backend_dev_t> devices;
    if (options.gpu_layers != 0) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name(options.device.c_str());
        if (device == nullptr) {
            throw std::runtime_error("backend device is unavailable: " + options.device);
        }
        devices = {device, nullptr};
        model_params.devices = devices.data();
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
    }
    llama_model * model = llama_model_load_from_file(options.model_path.c_str(), model_params);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model: " + options.model_path);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_vocab_n_tokens(vocab) != capture.n_vocab) {
        llama_model_free(model);
        throw std::runtime_error("model vocabulary does not match capture");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = capture.n_ctx;
    context_params.n_batch = capture.n_ctx;
    context_params.n_ubatch = capture.n_ctx;
    context_params.n_seq_max = 1;
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads;
    context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create model context");
    }

    // Qwen 3.5 exposes this as its post-final-norm state immediately before the LM head.
    // Masking keeps only rows that are also requested as logit outputs.
    llama_set_embeddings_nextn(context, true, true);

    const uint32_t first = capture.n_ctx / 2;
    const uint32_t rows_per_chunk = capture.n_ctx - first - 1;
    const uint32_t n_embd = llama_model_n_embd_out(model);
    const uint32_t version = 1;
    const uint32_t n_chunks = static_cast<uint32_t>(capture.n_chunks);

    std::ofstream output(options.output_path, std::ios::binary);
    if (!output) {
        llama_free(context);
        llama_model_free(model);
        throw std::runtime_error("failed to open output: " + options.output_path);
    }
    write_bytes(output, "_hidden_", 8, "magic");
    write_bytes(output, &version, sizeof(version), "version");
    write_bytes(output, &capture.n_ctx, sizeof(capture.n_ctx), "context length");
    write_bytes(output, &n_embd, sizeof(n_embd), "embedding size");
    write_bytes(output, &n_chunks, sizeof(n_chunks), "chunk count");
    write_bytes(output, &first, sizeof(first), "first row");
    write_bytes(output, &rows_per_chunk, sizeof(rows_per_chunk), "rows per chunk");
    write_bytes(output, capture.tokens.data(), capture.tokens.size() * sizeof(llama_token),
                "tokens");

    llama_batch batch = llama_batch_init(capture.n_ctx, 0, 1);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    for (int32_t chunk = 0; chunk < capture.n_chunks; ++chunk) {
        llama_memory_clear(llama_get_memory(context), true);
        batch.n_tokens = static_cast<int32_t>(capture.n_ctx);
        for (uint32_t position = 0; position < capture.n_ctx; ++position) {
            const std::size_t source = static_cast<std::size_t>(chunk) * capture.n_ctx + position;
            batch.token[position] = add_bos && position == 0 ? bos : capture.tokens[source];
            batch.pos[position] = static_cast<llama_pos>(position);
            batch.n_seq_id[position] = 1;
            batch.seq_id[position][0] = 0;
            batch.logits[position] = position >= first ? 1 : 0;
        }
        if (llama_decode(context, batch) != 0) {
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("decode failed at chunk " + std::to_string(chunk));
        }
        llama_synchronize(context);
        for (uint32_t position = first; position + 1 < capture.n_ctx; ++position) {
            const float * hidden = llama_get_embeddings_nextn_ith(
                context, static_cast<int32_t>(position));
            if (hidden == nullptr) {
                llama_batch_free(batch);
                llama_free(context);
                llama_model_free(model);
                throw std::runtime_error("hidden state unavailable at chunk " +
                                         std::to_string(chunk));
            }
            write_bytes(output, hidden, static_cast<std::size_t>(n_embd) * sizeof(float),
                        "hidden state");
        }
        std::cerr << "\rhidden-state chunks: " << (chunk + 1) << "/" << capture.n_chunks
                  << std::flush;
    }
    std::cerr << "\n";

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        extract(parse_options(argc, argv));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "extract_hidden_states: " << error.what() << "\n";
        return 1;
    }
}
