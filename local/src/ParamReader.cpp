#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "Math/Interpolator.h"
#include "Math/InterpolationTypes.h"
#include "TTreeReaderValue.h"

#include "ParamReader.h"

using Json = nlohmann::json;

/*************************************/
/*****NWBPositionCalibParamReader*****/
/*************************************/
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
    file >> this->database;
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
    for (auto& [bar_str, info]: this->database.items()) {
        bars.push_back(std::stoi(bar_str));
    }

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
            if ((run >= run_range[0].get<int>()) && run <= run_range[1].get<int>()) {
                auto& param = batch["parameters"];
                this->run_param[std::make_pair(bar, "p0")] = param[0].get<double>();
                this->run_param[std::make_pair(bar, "p1")] = param[1].get<double>();
                found = true;
                break;
            }
            else {
                // if not in run_range, record down the run difference for later
                // use of finding the closest batch
                int diff0 = abs(run - run_range[0].get<int>());
                int diff1 = abs(run - run_range[1].get<int>());
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
            this->run_param[std::make_pair(bar, "p0")] = closest_params[0].get<double>();
            this->run_param[std::make_pair(bar, "p1")] = closest_params[1].get<double>();
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


/***********************************************/
/*****NWPulseShapeDiscriminationParamReader*****/
/***********************************************/
NWPulseShapeDiscriminationParamReader::NWPulseShapeDiscriminationParamReader(const char AB) {
    this->AB = toupper(AB);
    this->ab = tolower(this->AB);

    for (int bar = 1; bar <= 24; ++bar) {
        this->bars.push_back(bar);
    }

    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        exit(1);
    }
    this->param_dir = PROJECT_DIR / this->param_reldir;
}

NWPulseShapeDiscriminationParamReader::~NWPulseShapeDiscriminationParamReader() { }

void NWPulseShapeDiscriminationParamReader::reconstruct_interpolators(int bar) {
    auto method = ROOT::Math::Interpolation::kAKIMA;

    /*****Fast-total interpolation*****/
    auto& fast_total = this->database[bar]["fast_total"];
    auto& totals = fast_total["totals"];
    std::vector<double> fasts;

    // gamma_L
    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = fast_total["center_line_L"][i].get<double>() + fast_total["gamma_cfasts_L"][i].get<double>();
        fasts.push_back(fast);
    }
    this->gamma_fast_total_L[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    // neutron_L
    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = fast_total["center_line_L"][i].get<double>() + fast_total["neutron_cfasts_L"][i].get<double>();
        fasts.push_back(fast);
    }
    this->neutron_fast_total_L[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    // gamma_R
    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = fast_total["center_line_R"][i].get<double>() + fast_total["gamma_cfasts_R"][i].get<double>();
        fasts.push_back(fast);
    }
    this->gamma_fast_total_R[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    // neutron_R
    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = fast_total["center_line_R"][i].get<double>() + fast_total["neutron_cfasts_R"][i].get<double>();
        fasts.push_back(fast);
    }
    this->neutron_fast_total_R[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    /*****VPSD centroids interpolation*****/
    auto& pos_correction = this->database[bar]["position_correction"];
    auto& centroids = pos_correction["centroid_curves"];
    auto& positions = centroids["positions"];
    std::vector<double> coords;

    // gamma_L
    coords.clear();
    for (int i = 0; i < positions.size(); ++i) {
        coords.push_back(centroids["gamma_centroids"][i][0].get<double>());
    }
    this->gamma_vpsd_L[bar] = new ROOT::Math::Interpolator(positions, coords, method);

    // neutron_L
    coords.clear();
    for (int i = 0; i < positions.size(); ++i) {
        coords.push_back(centroids["neutron_centroids"][i][0].get<double>());
    }
    this->neutron_vpsd_L[bar] = new ROOT::Math::Interpolator(positions, coords, method);

    // gamma_R
    coords.clear();
    for (int i = 0; i < positions.size(); ++i) {
        coords.push_back(centroids["gamma_centroids"][i][1].get<double>());
    }
    this->gamma_vpsd_R[bar] = new ROOT::Math::Interpolator(positions, coords, method);

    // neutron_R
    coords.clear();
    for (int i = 0; i < positions.size(); ++i) {
        coords.push_back(centroids["neutron_centroids"][i][1].get<double>());
    }
    this->neutron_vpsd_R[bar] = new ROOT::Math::Interpolator(positions, coords, method);
}

void NWPulseShapeDiscriminationParamReader::process_pca(int bar) {
    auto& pca = this->database[bar]["position_correction"]["pca"];
    this->pca_mean[bar] = {pca["mean"][0].get<double>(), pca["mean"][1].get<double>()};
    this->pca_components[bar][0] = {pca["components"][0][0].get<double>(), pca["components"][0][1].get<double>()};
    this->pca_components[bar][1] = {pca["components"][1][0].get<double>(), pca["components"][1][1].get<double>()};
    this->pca_xpeaks[bar] = {pca["xpeaks"][0].get<double>(), pca["xpeaks"][1].get<double>()};
}

bool NWPulseShapeDiscriminationParamReader::load_single_bar(int run, int bar) {
    bool found_run = false;
    for (auto& batch_dir: std::filesystem::directory_iterator(this->param_dir)) {
        std::filesystem::path filepath = batch_dir;
        filepath /= Form("NW%c-bar%02d.json", this->AB, bar);

        // read in JSON file
        std::ifstream file(filepath.string());
        if (!file.is_open()) {
            std::cerr << "Fail to open JSON file: " << filepath.string() << std::endl;
            exit(1);
        }
        Json json_buf;
        file >> json_buf;
        file.close();

        // check if run is in this batch
        auto& runs = json_buf["runs"];
        if (std::find(runs.begin(), runs.end(), run) != runs.end()) {
            found_run = true;

            // load in the parameters
            this->database[bar] = json_buf;
            this->reconstruct_interpolators(bar);
            this->process_pca(bar);

            break;
        }
    }
    return found_run;
}

bool NWPulseShapeDiscriminationParamReader::load(int run) {
    bool found_all_bars = true;
    for (int bar: this->bars) {
        if (!this->load_single_bar(run, bar)) {
            found_all_bars = false;
        }
    }
    return found_all_bars;
}
