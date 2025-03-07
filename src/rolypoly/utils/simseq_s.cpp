#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/functional/hash.hpp>
#include <queue>
#include <functional>
#include <condition_variable>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

std::mutex output_mutex;

// Function to generate a simple hash from a string
std::string generate_hash(const std::string& input) {
    boost::hash<std::string> string_hash;
    std::size_t hash = string_hash(input);
    std::stringstream ss;
    ss << std::hex << hash;
    return ss.str().substr(0, 8);  // Return first 8 characters of the hash
}

// Function to generate a range of values
std::vector<double> generate_range(double start, double end, double step) {
    std::vector<double> range;
    for (double val = start; val <= end; val += step) {
        range.push_back(val);
    }
    return range;
}

// Function to parse size_t ranges from a string
std::vector<size_t> parse_size_t_range(const std::string& range_str) {
    std::vector<size_t> range;
    std::vector<std::string> parts;
    boost::split(parts, range_str, boost::is_any_of(","));
    if (parts.size() == 3) {
        double start = std::stod(parts[0]);
        double end = std::stod(parts[1]);
        double step = std::stod(parts[2]);
        for (double val = start; val <= end; val += step) {
            range.push_back(static_cast<size_t>(val));
        }
    }
    return range;
}

// Function to parse double ranges from a string
std::vector<double> parse_double_range(const std::string& range_str, double min_val = 0.0, double max_val = 1.0) {
    std::vector<double> range;
    std::vector<std::string> parts;
    boost::split(parts, range_str, boost::is_any_of(","));
    if (parts.size() == 3) {
        double start = std::max(min_val, std::min(max_val, std::stod(parts[0])));
        double end = std::max(min_val, std::min(max_val, std::stod(parts[1])));
        double step = std::stod(parts[2]);
        range = generate_range(start, end, step);
    }
    return range;
}

// Function to introduce variants based on identity
std::string introduce_variants(const std::string& sequence, double identity) {
    std::string result = sequence;
    size_t seq_length = sequence.size();
    size_t num_variants = static_cast<size_t>(seq_length * (1 - identity));
    
    if (num_variants == 0) {
        return result;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, seq_length - 1);
    std::uniform_int_distribution<> nuc_dis(0, 3);

    for (size_t i = 0; i < num_variants; ++i) {
        size_t idx = dis(gen);
        char original_nuc = result[idx];
        char new_nuc;
        do {
            new_nuc = "ATCG"[nuc_dis(gen)];
        } while (new_nuc == original_nuc);
        result[idx] = new_nuc;
    }
    
    return result;
}

// Function to subset the sequence based on coverage
std::string subset_sequence(const std::string& sequence, double coverage) {
    size_t seq_length = sequence.size();
    size_t subset_length = static_cast<size_t>(seq_length * coverage);
    std::vector<size_t> subset_indices(subset_length);
    
    std::iota(subset_indices.begin(), subset_indices.end(), 0);
    std::shuffle(subset_indices.begin(), subset_indices.end(), std::mt19937{std::random_device{}()});
    subset_indices.resize(subset_length);
    std::sort(subset_indices.begin(), subset_indices.end());
    
    std::string subset_seq;
    for (size_t idx : subset_indices) {
        subset_seq += sequence[idx];
    }
    
    return subset_seq;
}

// Function to process a single combination of coverage, identity, and rotation
void process_combination(const std::string& sequence, double coverage, double identity, double rotation_proportion, const std::string& base_output_folder, bool multifasta_output, std::ofstream* multifasta_file) {
    size_t actual_length = sequence.length();
    
    // Calculate the actual rotation based on the proportion
    size_t rotation = static_cast<size_t>(rotation_proportion * actual_length) % actual_length;
    
    // Rotate the sequence
    std::string rotated_seq = sequence.substr(rotation) + sequence.substr(0, rotation);
    
    std::string output_folder = base_output_folder + "/Len_" + std::to_string(actual_length);
    
    if (!multifasta_output) {
        if (!fs::exists(output_folder)) {
            fs::create_directory(output_folder);
            
            // Write the original sequence to a file
            std::string hash_suffix = generate_hash(sequence);
            std::string original_file = output_folder + "/original_sequence_" + hash_suffix + ".fasta";
            std::ofstream original_ofs(original_file);
            if (original_ofs.is_open()) {
                original_ofs << ">original_sequence_length_" << actual_length << "_" << hash_suffix << "\n" << sequence << "\n";
                original_ofs.close();
            } else {
                std::cerr << "Error opening file: " << original_file << "\n";
            }
        }
    }
    
    std::string subset_seq = subset_sequence(rotated_seq, coverage);
    std::string variant_seq = introduce_variants(subset_seq, identity);
    
    std::string variant_id = "original_length_" + std::to_string(actual_length) +
                             "_coverage_" + std::to_string(coverage) +
                             "_identity_" + std::to_string(identity) +
                             "_rotation_" + std::to_string(rotation_proportion);
    
    if (multifasta_output && multifasta_file != nullptr) {
        std::lock_guard<std::mutex> lock(output_mutex);
        (*multifasta_file) << "> " << variant_id << "\n" << variant_seq << "\n";
    } else if (!multifasta_output) {
        std::string output_file = output_folder + "/" + variant_id + ".fasta";
        std::ofstream ofs(output_file);
        if (ofs.is_open()) {
            ofs << ">" << variant_id << "\n" << variant_seq << "\n";
            ofs.close();
        } else {
            std::cerr << "Error opening file: " << output_file << "\n";
        }
    }
}

// Function to read sequences from a FASTA file .. TODO:move to seqan
std::vector<std::string> read_fasta(const std::string& filename) {
    std::vector<std::string> sequences;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return sequences;
    }
    
    std::string line;
    std::string current_seq;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!current_seq.empty()) {
                sequences.push_back(current_seq);
                current_seq.clear();
            }
        } else {
            current_seq += line;
        }
    }
    
    if (!current_seq.empty()) {
        sequences.push_back(current_seq);
    }
    
    return sequences;
}

// Lock-free progress counter
std::atomic<size_t> progress(0);

// Function to update progress bar less frequently
void update_progress_bar(const std::atomic<size_t>& progress, size_t total) {
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() < 100) {
        return;  // Update at most every 100ms
    }
    last_update = now;

    const int bar_width = 70;
    float progress_ratio = static_cast<float>(progress) / total;
    int pos = bar_width * progress_ratio;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress_ratio * 100.0) << "%";
    std::cout.flush();
}

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop(false), active_tasks(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                        active_tasks++;
                    }
                    task();
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        active_tasks--;
                        if (tasks.empty() && active_tasks == 0) {
                            completion.notify_all();
                        }
                    }
                    producer_condition.notify_one();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        producer_condition.wait(lock, [this] { return tasks.size() < max_queue_size; });
        tasks.emplace(std::forward<F>(f));
        condition.notify_one();
    }

    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        completion.wait(lock, [this] { return tasks.empty() && active_tasks == 0; });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable completion;
    std::condition_variable producer_condition;
    bool stop;
    std::atomic<int> active_tasks;
    const size_t max_queue_size = 1000; // Adjust this value as needed
};


#if __cplusplus < 201402L  // C++14
namespace std {
    template<typename T, typename     Args>
    std::unique_ptr<T> make_unique(Args&&     args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)    ));
    }
}
#endif

// Main function
int main(int argc, char** argv) {
    // Command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("coverage", po::value<std::string>()->default_value("0.1,1,0.01"), "Coverage range as 'start,end,step'")
        ("identity", po::value<std::string>()->default_value("0.5,1,0.01"), "Identity range as 'start,end,step'")
        ("rotations", po::value<std::string>()->default_value("0,1,0.1"), "Rotation proportion range as 'start,end,step' (between 0 and 1)")
        ("output_folder", po::value<std::string>()->default_value("./output"), "Output folder")
        ("seq_length_min", po::value<int>()->default_value(500), "Minimum length of simulated sequences")
        ("seq_length_max", po::value<int>()->default_value(80000), "Maximum length of simulated sequences")
        ("seq_length_step", po::value<int>()->default_value(700), "Step size for sequence lengths")
        ("threads", po::value<int>()->default_value(std::thread::hardware_concurrency()), "Number of threads (default: hardware concurrency)")
        ("fasta_file", po::value<std::string>(), "Input FASTA file with initial sequences")
        ("multifasta-output", po::bool_switch()->default_value(false), "Output all sequences to a single multifasta file");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

// Print help message if no arguments are provided or help is requested
if (vm.count("help") || argc == 1) {
    std::cout << "Sequence Variant Generator\n\n"
              << "This program generates sequence variants based on input parameters.\n\n"
              << "Usage: " << argv[0] << " [options]\n\n"
              << "Options:\n"
              << desc << "\n"
              << "Examples:\n"
              << "  Generate variants using default parameters:\n"
              << "    " << argv[0] << "\n\n"
              << "  Generate variants with custom parameters:\n"
              << "    " << argv[0] << " --coverage 0.5,1,0.1 --identity 0.8,1,0.05 --rotations 0,0.5,0.1 --seq_length_min 1000 --seq_length_max 5000 --seq_length_step 1000\n\n"
              << "  Generate variants from a FASTA file:\n"
              << "    " << argv[0] << " --fasta_file input_sequences.fasta\n\n"
              << "  Generate variants and output to a single multifasta file:\n"
              << "    " << argv[0] << " --multifasta-output\n\n"
              << "Default values:\n"
              << "  Coverage: 0.1 to 1 with step 0.01\n"
              << "  Identity: 0.5 to 1 with step 0.01\n"
              << "  Rotations: 0 to 1 with step 0.1\n"
              << "  Sequence length: 500 to 80000 with step 700\n"
              << "  Output folder: ./output\n"
              << "  Threads: " << std::thread::hardware_concurrency() << " (hardware concurrency)\n"
              << "  Multifasta output: disabled\n\n"
              << "Notes:\n"
              << "  - The --multifasta-output option writes all sequences (original and variants) to a single file named 'all_sequences.fasta' in the output folder.\n"
              << "  - Original sequences are given unique identifiers using a hash suffix.\n"
              << "  - When --multifasta-output is not used, sequences are written to separate files in subdirectories based on their length.\n";
    return 0;
}


    // Parse range arguments
    std::vector<double> covs = parse_double_range(vm["coverage"].as<std::string>());
    std::vector<double> ANIs = parse_double_range(vm["identity"].as<std::string>());
    std::vector<double> rotations = parse_double_range(vm["rotations"].as<std::string>(), 0.0, 1.0);
    std::string base_output_folder = vm["output_folder"].as<std::string>();
    int seq_length_min = vm["seq_length_min"].as<int>();
    int seq_length_max = vm["seq_length_max"].as<int>();
    int seq_length_step = vm["seq_length_step"].as<int>();
    int num_threads = vm["threads"].as<int>();
    bool multifasta_output = vm["multifasta-output"].as<bool>();

    // Ensure the base output folder exists
    if (!fs::exists(base_output_folder)) {
        fs::create_directory(base_output_folder);
    }

       std::vector<std::string> sequences;
    if (vm.count("fasta_file")) {
        std::string fasta_file = vm["fasta_file"].as<std::string>();
        sequences = read_fasta(fasta_file);
        if (sequences.empty()) {
            std::cerr << "No sequences found in FASTA file or file could not be read.\n";
            return 1;
        }
    } else {
        // Generate simulated sequences
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> nuc_dis(0, 3);

        for (int len = seq_length_min; len <= seq_length_max; len += seq_length_step) {
            std::string sequence(len, ' ');
            for (int i = 0; i < len; ++i) {
                sequence[i] = "ATCG"[nuc_dis(gen)];
            }
            sequences.push_back(sequence);
        }
    }

    std::unique_ptr<std::ofstream> multifasta_file;
    if (multifasta_output) {
        std::string multifasta_filename = base_output_folder + "/all_sequences.fasta";
        multifasta_file = std::unique_ptr<std::ofstream>(new std::ofstream(multifasta_filename));
        if (!multifasta_file->is_open()) {
            std::cerr << "Error opening multifasta file: " << multifasta_filename << "\n";
            return 1;
        }
    }

    // Calculate total number of combinations
    size_t total_combinations = sequences.size() * covs.size() * ANIs.size() * rotations.size();
    std::cout << "Total number of combinations (sequences) to generate:  " << total_combinations << std::endl;

    std::atomic<size_t> progress(0);

    // Create thread pool
    ThreadPool pool(num_threads);

     // Progress update thread
    std::atomic<bool> processing_complete(false);
    std::thread progress_thread([&progress, total_combinations, &processing_complete]() {
        while (!processing_complete.load(std::memory_order_acquire)) {
            update_progress_bar(progress, total_combinations);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        update_progress_bar(total_combinations, total_combinations);
        std::cout << std::endl;
    });

    // Task producer thread
    std::thread producer_thread([&]() {
        for (const auto& seq : sequences) {
            for (double coverage : covs) {
                for (double identity : ANIs) {
                    for (double rotation : rotations) {
                        pool.enqueue([&, seq, coverage, identity, rotation]() {
                            process_combination(seq, coverage, identity, rotation, base_output_folder, multifasta_output, multifasta_file.get());
                            progress.fetch_add(1, std::memory_order_relaxed);
                        });
                    }
                }
            }
        }
    });

    // Wait for all tasks to be produced and completed 
    producer_thread.join();
    pool.wait_for_completion();

    // Signal that processing is complete
    processing_complete.store(true, std::memory_order_release);

    // Wait for the progress thread to finish
    progress_thread.join();
    
    // The multifasta option is so slow.
    if (multifasta_file) {
        multifasta_file->close();
    }

    std::cout << "Processing complete." << std::endl;

    return 0;
}