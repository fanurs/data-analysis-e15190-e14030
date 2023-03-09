#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "TFolder.h"
#include "Math/Interpolator.h"
#include "Math/InterpolationTypes.h"
#include "TNamed.h"
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
    this->project_dir = PROJECT_DIR;
    this->pcalib_dir = this->project_dir / this->pcalib_reldir;
    this->json_path = this->pcalib_dir / this->json_filename;
    this->pca_path = this->project_dir / this->pca_reldir / this->dat_filename;

    // read in the final calibration file (JSON)
    std::ifstream file(this->json_path.string());
    if (!file.is_open()) {
        std::cerr << "Fail to open JSON file: " << this->json_path.string() << std::endl;
        exit(1);
    }
    file >> this->database;
    file.close();

    // read in the PCA.dat file
    int barno;
    std::string TP;
    double q1,q2,q3;
    std::ifstream datfile(this->pca_path.string());
    if (!datfile.is_open()) {
        std::cerr << "Fail to open NW_pca.dat file: " << this->pca_path.string() << std::endl;
        exit(1);
    }
    std::string line;
    while(getline(datfile,line)) {
        line.erase(line.begin(), find_if(line.begin(), line.end(), not1(std::ptr_fun<int, int>(isspace)))); 
        if(line[0]=='#') continue;
        std::istringstream ss(line);
        ss >> barno >> TP >> q1 >> q2 >> q3;

        if(TP=="L") {
            this->L[std::make_pair(barno, "L0")] = q1;
            this->L[std::make_pair(barno, "L1")] = q2;
            this->L[std::make_pair(barno, "L2")] = q3;
        }
        if(TP=="X") {
            this->X[std::make_pair(barno, "X0")] = q1;
            this->X[std::make_pair(barno, "X1")] = q2;
            this->X[std::make_pair(barno, "X2")] = q3;
        }
        if(TP=="Y") {
            this->Y[std::make_pair(barno, "Y0")] = q1;
            this->Y[std::make_pair(barno, "Y1")] = q2;
            this->Y[std::make_pair(barno, "Y2")] = q3;
        }
        if(TP=="Z") {
            this->Z[std::make_pair(barno, "Z0")] = q1;
            this->Z[std::make_pair(barno, "Z1")] = q2;
            this->Z[std::make_pair(barno, "Z2")] = q3;
        }
    }
    datfile.close();
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

double NWBPositionCalibParamReader::getL(int bar, const std::string& par) {
    return this->L[std::make_pair(bar, par)];
}
double NWBPositionCalibParamReader::getX(int bar, const std::string& par) {
    return this->X[std::make_pair(bar, par)];
}
double NWBPositionCalibParamReader::getY(int bar, const std::string& par) {
    return this->Y[std::make_pair(bar, par)];
}
double NWBPositionCalibParamReader::getZ(int bar, const std::string& par) {
    return this->Z[std::make_pair(bar, par)];
}

void NWBPositionCalibParamReader::write_metadata(TFolder* folder, bool relative_path) {
    std::filesystem::path base_dir = (relative_path) ? this->project_dir : "/";
    std::filesystem::path path;

    path = std::filesystem::proximate(this->json_path, base_dir);
    TNamed* json_path_data = new TNamed(path.string().c_str(), "");
    folder->Add(json_path_data);

    path = std::filesystem::proximate(this->pca_path, base_dir);
    TNamed* pca_path_data = new TNamed(path.string().c_str(), "");
    folder->Add(pca_path_data);

    return;
}



/****************************************/
/*****NWTimeOfFlightCalibParamReader*****/
/****************************************/
NWTimeOfFlightCalibParamReader::NWTimeOfFlightCalibParamReader(const char AB, bool load_params) {
    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        exit(1);
    }
    this->AB = toupper(AB);
    this->ab = tolower(this->AB);
    this->project_dir = PROJECT_DIR;
    this->calib_dir = this->project_dir / this->calib_dir;
    this->json_path = this->calib_dir / Form(this->json_filename.c_str(), this->ab);

    if (load_params) {
        this->load_tof_offset();
    }
}

NWTimeOfFlightCalibParamReader::~NWTimeOfFlightCalibParamReader() { }

void NWTimeOfFlightCalibParamReader::load_tof_offset() {
    /* Read in all TOF offset parameters from .json file to this->database */
    std::ifstream file(this->json_path.string());
    if (!file.is_open()) {
        std::cerr << "ERROR: failed to open " << this->json_path.string() << std::endl;
        exit(1);
    }
    this->database.clear();
    file >> this->database;
    file.close();
}

void NWTimeOfFlightCalibParamReader::load(int run) {
    /* Load TOF offset parameters for a given run
     * from this->database to this->tof_offset.
     */
    for (auto& [bar, bar_info] : this->database.items()) {
        bool found = false;
        for (auto& par_info : bar_info) {
            auto& run_range = par_info["run_range"];
            if (run < run_range[0].get<int>() || run > run_range[1].get<int>()) {
                continue;
            }
            this->tof_offset[std::stoi(bar)] = par_info["tof_offset"].get<double>();
            found = true;
            break;
        }
        if (!found) {
            std::cerr << Form(
                "ERROR: run-%04d is not found for NW%c bar%02d",
                run, this->AB, std::stoi(bar)
            ) << std::endl;
        }
    }
}

void NWTimeOfFlightCalibParamReader::write_metadata(TFolder* folder, bool relative_path) {
    std::filesystem::path base_dir = (relative_path) ? this->project_dir : "/";
    std::filesystem::path path;

    path = std::filesystem::proximate(this->json_path, base_dir);
    TNamed* path_data = new TNamed(path.string().c_str(), "");
    folder->Add(path_data);
}


/**************************************/
/*****NWADCPreprocessorParamReader*****/
/**************************************/
NWADCPreprocessorParamReader::NWADCPreprocessorParamReader(const char AB) {
    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        exit(1);
    }
    this->AB = toupper(AB);
    this->ab = tolower(this->AB);
    this->project_dir = PROJECT_DIR;
}

NWADCPreprocessorParamReader::~NWADCPreprocessorParamReader() { }

void NWADCPreprocessorParamReader::load(int run) {
    this->run = run;
    this->calib_reldir = this->project_dir / Form(this->calib_reldir.string().c_str(), this->run);
    this->load_fast_total('L');
    this->load_fast_total('R');
    this->load_log_ratio_total();
    return;
}

void NWADCPreprocessorParamReader::load_fast_total(char side) {
    auto filepath = this->calib_reldir / Form(this->filename.c_str(), Form("fast_total_%c", side));
    this->filepaths.push_back(filepath);
    std::ifstream file(filepath.string());
    if (!file.is_open()) {
        std::cerr << "ERROR: failed to open " << filepath.string() << std::endl;
        exit(1);
    }
    Json content;
    file >> content;
    file.close();

    auto& map = (side == 'L') ? this->fast_total_L : this->fast_total_R;
    for (int bar = 1; bar <= 24; ++bar) {
        Json& bar_content = content[std::to_string(bar)];
        Json info;
        for (auto& chunk : bar_content) {
            if (chunk["run_range"][0].get<int>() <= this->run && this->run <= chunk["run_range"][1].get<int>()) {
                info = chunk;
                break;
            }
        }
        if (info.is_null()) {
            std::cerr << Form("ERROR: run-%04d is not found for NW%c bar%02d", this->run, this->AB, bar) << std::endl;
            exit(1);
        }
        map[bar] = {
            {"nonlinear_fast_threshold", info["nonlinear_fast_threshold"].get<double>()},
            {"stationary_point_x", info["stationary_point_x"].get<double>()},
            {"stationary_point_y", info["stationary_point_y"].get<double>()},
            {"fit_params[0]", info["linear_fit_params"][0].get<double>() - info["quadratic_fit_params"][0].get<double>()},
            {"fit_params[1]", info["linear_fit_params"][1].get<double>() - info["quadratic_fit_params"][1].get<double>()},
            {"fit_params[2]", -info["quadratic_fit_params"][2].get<double>()},
        };
    }
    return;
}

void NWADCPreprocessorParamReader::load_log_ratio_total() {
    auto filepath = this->calib_reldir / Form(this->filename.c_str(), "log_ratio_total");
    this->filepaths.push_back(filepath);
    std::ifstream file(filepath.string());
    if (!file.is_open()) {
        std::cerr << "ERROR: failed to open " << filepath.string() << std::endl;
        exit(1);
    }
    Json content;
    file >> content;
    file.close();

    auto& map = this->log_ratio_total;
    for (int bar = 1; bar <= 24; ++bar) {
        Json& bar_content = content[std::to_string(bar)];
        Json info;
        for (auto& chunk : bar_content) {
            if (chunk["run_range"][0].get<int>() <= this->run && this->run <= chunk["run_range"][1].get<int>()) {
                info = chunk;
                break;
            }
        }
        if (info.is_null()) {
            std::cerr << Form("ERROR: run-%04d is not found for NW%c bar%02d", this->run, this->AB, bar) << std::endl;
            exit(1);
        }
        map[bar] = {
            {"attenuation_length", info["attenuation_length"].get<double>()},
            {"gain_ratio", info["gain_ratio"].get<double>()},
        };
    }
    return;
}

void NWADCPreprocessorParamReader::write_metadata(TFolder* folder, bool relative_path) {
    std::filesystem::path base_dir = (relative_path) ? this->project_dir : "/";
    std::filesystem::path path;
    for (auto& filepath : this->filepaths) {
        path = std::filesystem::proximate(filepath, base_dir);
        TNamed* named = new TNamed(path.string().c_str(), "");
        folder->Add(named);
    }
    return;
}


/***************************************/
/*****NWLightOutputCalibParamReader*****/
/***************************************/
NWLightOutputCalibParamReader::NWLightOutputCalibParamReader(const char AB) {
    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        exit(1);
    }
    this->AB = toupper(AB);
    this->ab = tolower(this->AB);
    this->project_dir = PROJECT_DIR;
    this->lcalib_reldir = this->project_dir / this->lcalib_reldir;
    this->pul_path = this->lcalib_reldir / Form(this->pul_filename.c_str(), this->ab);
}

NWLightOutputCalibParamReader::~NWLightOutputCalibParamReader() { }

void NWLightOutputCalibParamReader::load_pulse_height() {
    std::vector<std::string> keys = {"a", "b", "c", "d", "e"};
    std::ifstream infile(this->pul_path.c_str());
    std::string line;
    std::getline(infile, line);
    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.empty()) { continue; }

        std::stringstream ss(line);
        int bar;
        ss >> bar;
        for (std::string& key: keys) {
            ss >> this->run_param[bar][key];
        }
    }
    infile.close();
}

void NWLightOutputCalibParamReader::load(int run) {
    // run dependency not implemented
    this->load_pulse_height();
    return;
}

void NWLightOutputCalibParamReader::write_metadata(TFolder* folder, bool relative_path) {
    std::filesystem::path base_dir = (relative_path) ? this->project_dir : "/";
    std::filesystem::path path = std::filesystem::proximate(this->pul_path, base_dir);
    TNamed* pul_path_data = new TNamed(path.string().c_str(), "");
    folder->Add(pul_path_data);

    return;
}



/***********************************************/
/*****NWPulseShapeDiscriminationParamReader*****/
/***********************************************/
double NWPulseShapeDiscriminationParamReader::polynomial(double x, std::vector<double>& params) {
    double y = 0;
    for (int i = 0; i < params.size(); ++i) {
        y += params[i] * std::pow(x, i);
    }
    return y;
}

double NWPulseShapeDiscriminationParamReader::polynomial(double x, Json& params) {
    std::vector<double> params_vec;
    for (auto& param : params) {
        params_vec.push_back(param.get<double>());
    }
    return this->polynomial(x, params_vec);
}

std::vector<double> NWPulseShapeDiscriminationParamReader::get_neutron_linear_params(double x_switch_neutron, std::vector<double>& quadratic_params) {
    auto& quad = quadratic_params;
    double lin1 = quad[1] + 2 * quad[2] * x_switch_neutron;
    double lin0 = quad[0] + quad[1] * x_switch_neutron + quad[2] * std::pow(x_switch_neutron, 2) - lin1 * x_switch_neutron;
    return {lin0, lin1};
}

std::vector<double> NWPulseShapeDiscriminationParamReader::get_neutron_linear_params(double x_switch_neutron, Json& quadratic_params) {
    std::vector<double> quadratic_params_vec;
    for (auto& param : quadratic_params) {
        quadratic_params_vec.push_back(param.get<double>());
    }
    return this->get_neutron_linear_params(x_switch_neutron, quadratic_params_vec);
}

Json NWPulseShapeDiscriminationParamReader::get_bar_params(int run, int bar) {
    auto& bar_params = this->database[Form("%d", bar)];
    Json params;
    for (auto& run_range_params : bar_params) {
        auto& run_range = run_range_params["run_range"];
        int run_start = (int)run_range[0].get<double>();
        int run_stop = (int)run_range[1].get<double>();
        if (run >= run_start && run <= run_stop) {
            params = run_range_params;
            break;
        }
    }
    if (params.empty()) {
        std::cerr << Form("Cannot find run %04d for NW%c-bar%02d", run, this->AB, bar) << std::endl;
        exit(1);
    }
    return params;
}

void NWPulseShapeDiscriminationParamReader::fast_total_interpolation(int bar, Json& params) {
    auto method = ROOT::Math::Interpolation::kAKIMA;

    std::vector<double> totals;
    for (double total = -20.0; total <= 4020.0; total += 20.0) {
        totals.push_back(total);
    }
    std::vector<double> fasts;

    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = this->polynomial(totals[i], params["cline_L"]) + this->polynomial(totals[i], params["g_cfast_L"]);
        fasts.push_back(fast);
    }
    this->gamma_fast_total_L[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        auto& _params = params["n_cfast_L"];
        if (totals[i] >= params["x_switch_neutron"]) {
            _params = this->get_neutron_linear_params(params["x_switch_neutron"], _params);
        }
        double fast = this->polynomial(totals[i], params["cline_L"]) + this->polynomial(totals[i], _params);
        fasts.push_back(fast);
    }
    this->neutron_fast_total_L[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        double fast = this->polynomial(totals[i], params["cline_R"]) + this->polynomial(totals[i], params["g_cfast_R"]);
        fasts.push_back(fast);
    }
    this->gamma_fast_total_R[bar] = new ROOT::Math::Interpolator(totals, fasts, method);

    fasts.clear();
    for (int i = 0; i < totals.size(); ++i) {
        auto& _params = params["n_cfast_R"];
        if (totals[i] >= params["x_switch_neutron"]) {
            _params = this->get_neutron_linear_params(params["x_switch_neutron"], _params);
        }
        double fast = this->polynomial(totals[i], params["cline_R"]) + this->polynomial(totals[i], _params);
        fasts.push_back(fast);
    }
    this->neutron_fast_total_R[bar] = new ROOT::Math::Interpolator(totals, fasts, method);
}

void NWPulseShapeDiscriminationParamReader::centroid_interpolation(int bar, Json& params) {
    auto method = ROOT::Math::Interpolation::kAKIMA;

    std::vector<double> pos_x;
    for (auto& x: params["centroid_pos_x"]) {
        pos_x.push_back(x.get<double>());
    }
    std::vector<double> coords;

    coords.clear();
    for (int i = 0; i < pos_x.size(); ++i) {
        coords.push_back(params["g_centroid_L"][i].get<double>());
    }
    this->gamma_vpsd_L[bar] = new ROOT::Math::Interpolator(pos_x, coords, method);

    coords.clear();
    for (int i = 0; i < pos_x.size(); ++i) {
        coords.push_back(params["n_centroid_L"][i].get<double>());
    }
    this->neutron_vpsd_L[bar] = new ROOT::Math::Interpolator(pos_x, coords, method);

    coords.clear();
    for (int i = 0; i < pos_x.size(); ++i) {
        coords.push_back(params["g_centroid_R"][i].get<double>());
    }
    this->gamma_vpsd_R[bar] = new ROOT::Math::Interpolator(pos_x, coords, method);

    coords.clear();
    for (int i = 0; i < pos_x.size(); ++i) {
        coords.push_back(params["n_centroid_R"][i].get<double>());
    }
    this->neutron_vpsd_R[bar] = new ROOT::Math::Interpolator(pos_x, coords, method);
}

void NWPulseShapeDiscriminationParamReader::process_pca(int bar, Json& params) {
    this->pca_mean[bar] = {params["pca_mean"][0].get<double>(), params["pca_mean"][1].get<double>()};
    this->pca_components[bar][0] = {params["pca_components"][0][0].get<double>(), params["pca_components"][0][1].get<double>()};
    this->pca_components[bar][1] = {params["pca_components"][1][0].get<double>(), params["pca_components"][1][1].get<double>()};
    this->pca_xpeaks[bar] = {params["pca_xpeaks"][0].get<double>(), params["pca_xpeaks"][1].get<double>()};
}

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
    this->project_dir = std::filesystem::path(PROJECT_DIR);
    this->param_dir = this->project_dir / this->param_reldir;
}

NWPulseShapeDiscriminationParamReader::~NWPulseShapeDiscriminationParamReader() { }

void NWPulseShapeDiscriminationParamReader::read_in_calib_params() {
    this->param_path = this->param_dir / Form("calib_params_nw%c.json", this->ab);
    std::ifstream database_file(this->param_path.string());
    if (!database_file.is_open()) {
        std::cerr << "Failed to open database file: " << this->param_path << std::endl;
        exit(1);
    }
    database_file >> this->database;
    database_file.close();
}

void NWPulseShapeDiscriminationParamReader::load(int run) {
    this->read_in_calib_params();
    for (int bar: this->bars) {
        auto params = this->get_bar_params(run, bar);
        this->fast_total_interpolation(bar, params);
        this->centroid_interpolation(bar, params);
        this->process_pca(bar, params);
    }
}

void NWPulseShapeDiscriminationParamReader::write_metadata(TFolder* folder, bool relative_path) {
    std::filesystem::path base_dir = (relative_path) ? this->project_dir : "/";
    std::filesystem::path path = std::filesystem::proximate(this->param_path, base_dir);
    TNamed* data = new TNamed(path.string().c_str(), "PulseShapeDiscrimination_param_path");
    folder->Add(data);
    return;
}
