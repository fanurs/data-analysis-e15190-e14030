#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <json/json.h>

#include "TTreeReaderValue.h"

#include "ParamReader.h"

NWBPositionCalibParamReader::NWBPositionCalibParamReader() {
    // initialize paths
    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        exit(1);
    }
    this->pcalib_dir = PROJECT_DIR / this->pcalib_reldir;
    this->json_path = this->pcalib_dir / this->json_filename;

    // read in the final calibration file (JSON)
    std::ifstream file(this->json_path.string());
    if (!file.is_open()) {
        std::cerr << "Fail to open JSON file: " << this->json_path.string() << std::endl;
        exit(1);
    }
    Json::CharReaderBuilder builder;
    auto parse_success = parseFromStream(builder, file, &this->database, &this->_json_error);
    file.close();
}

NWBPositionCalibParamReader::~NWBPositionCalibParamReader() { }

long NWBPositionCalibParamReader::load_single(int run) {
    std::filesystem::path path = this->pcalib_dir / "calib_params";
    path /= Form("run-%04d-nw%c.dat", run, this->ab);
    return this->load_single(path.string());
}

long NWBPositionCalibParamReader::load_single(
    const std::string& filename,
    const std::string& branch_descriptor,
    bool set_index,
    int n_skip_rows,
    char delimiter
) {
    long result = this->load_from_txt(filename, branch_descriptor, n_skip_rows, delimiter);
    if (set_index) {
        this->set_index("bar");
    }
    return result;
}

bool NWBPositionCalibParamReader::load(int run, bool extrapolate) {
    bool all_bars_found = true;

    // get bar numbers and sort into ascending order
    std::vector<int> bars;
    for (auto& bar: this->database.getMemberNames()) {
        bars.push_back(std::stoi(bar));
    }
    std::sort(bars.begin(), bars.end());

    // loop over all bars
    for (int bar: bars) {
        std::string bar_str = Form("%d", bar);
        bool found = false;
        int closest_ibatch = -1;
        int closest_diff = int(1e8);
        
        // identify run_range and load parameters
        for (int ibatch = 0; ibatch < this->database[bar_str].size(); ++ibatch) {
            auto& batch = this->database[bar_str][ibatch];
            auto& run_range = batch["run_range"];
            if ((run >= run_range[0].asInt()) && run <= run_range[1].asInt()) {
                auto& param = batch["parameters"];
                this->run_param[std::make_pair(bar, "p0")] = param[0].asDouble();
                this->run_param[std::make_pair(bar, "p1")] = param[1].asDouble();
                found = true;
                break;
            }
            else {
                // if not in run_range, record down the run difference for later
                // use of finding the closest batch
                int diff0 = abs(run - run_range[0].asInt());
                int diff1 = abs(run - run_range[1].asInt());
                int diff = std::min(diff0, diff1);
                if (diff < closest_diff) {
                    closest_diff = diff;
                    closest_ibatch = ibatch;
                }
            }
        }

        if (found) {
            continue;
        }

        // fail to find run in all run ranges, may extrapolate from the closest batch
        if (extrapolate) {
            auto& closest_params = this->database[bar_str][closest_ibatch]["parameters"];
            this->run_param[std::make_pair(bar, "p0")] = closest_params[0].asDouble();
            this->run_param[std::make_pair(bar, "p1")] = closest_params[1].asDouble();
        }
        else {
            std::cerr << "ERROR: run " << run << " is out of range in " << this->json_path.string() << std::endl;
            exit(1);
        }
    }
    return all_bars_found;
}

void NWBPositionCalibParamReader::set_index(const std::string& index_name) {
    TTreeReaderValue<int> index(this->reader, index_name.c_str());
    int n_entries = this->tree->GetEntries();
    for (int i_entry = 0; i_entry < n_entries; ++i_entry) {
        this->reader.SetEntry(i_entry);
        this->index_map[*index] = i_entry;
    }
}

double NWBPositionCalibParamReader::get(int bar, const std::string& par) {
    return this->run_param[std::make_pair(bar, par)];
}