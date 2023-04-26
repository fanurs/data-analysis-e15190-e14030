#pragma once

#include <any>
#include <array>
#include <cctype>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <string>

#include <nlohmann/json.hpp>

#include "TFolder.h"
#include "Math/Interpolator.h"
#include "TTree.h"
#include "TTreeReader.h"

using Json = nlohmann::json;

template <typename index_t>
class ParamReader {
public:
    TTree* tree;
    TTreeReader reader;
    std::unordered_map<index_t, int> index_map;

    ParamReader(const std::string& tr_name="", const std::string& tr_title="");
    ~ParamReader();

    void initialize_tree(const std::string& tr_name, const std::string& tr_title);
    long load_from_txt(const std::string& filename, const std::string& branch_descriptor, int n_skip_rows=0, char delimiter=' ');

    template <typename val_t>
    val_t get_value(index_t index, const std::string& col_name);
};
#include "ParamReader.tpp"

class NWPositionCalibParamReader {
private:
    std::map<std::pair<int, std::string>, double> param; // (bar, par) -> value
    std::filesystem::path resolve_project_dir(const std::string& path);

public:
    const char AB;
    const char ab;
    std::string pcalib_filepath = "$PROJECT_DIR/database/neutron_wall/position_calibration/NW%c_calib_params.json";
    std::string pca_filepath = "$PROJECT_DIR/database/neutron_wall/geometry/NW%c_pca.dat";

    NWPositionCalibParamReader(const char AB);
    ~NWPositionCalibParamReader();

    bool load(int run);
    double get(int bar, const std::string& par);
    void write_metadata(TFolder* folder, bool relative_path=true);
};

class NWTimeOfFlightCalibParamReader : public ParamReader<int> {
public:
    char AB;
    char ab;
    std::filesystem::path project_dir;
    std::filesystem::path calib_dir = "database/neutron_wall/time_of_flight_calibration";
    std::string json_filename = "calib_params_nw%c.json";
    std::filesystem::path json_path;
    Json database; // all bars and runs
    std::unordered_map<int, double> tof_offset; // all bars for a given run; <bar, tof_offset>

    NWTimeOfFlightCalibParamReader(const char AB, bool load_params=true);
    ~NWTimeOfFlightCalibParamReader();

    void load_tof_offset();
    void load(int run);
    void write_metadata(TFolder* folder, bool relative_path=true);
};

class NWADCPreprocessorParamReader {
public:
    char AB;
    char ab;
    int run;
    std::filesystem::path project_dir = "";
    std::filesystem::path calib_reldir = "database/neutron_wall/adc_preprocessing/";
    std::string filename = "calib_params_%s.json";
    std::vector<std::filesystem::path> filepaths; // to be written as metadata
    std::unordered_map<int, std::unordered_map<std::string, double> > fast_total_L;
    std::unordered_map<int, std::unordered_map<std::string, double> > fast_total_R;
    std::unordered_map<int, std::unordered_map<std::string, double> > log_ratio_total;

    NWADCPreprocessorParamReader(const char AB);
    ~NWADCPreprocessorParamReader();

    void load(int run);
    void load_fast_total(char side);
    void load_log_ratio_total();
    void write_metadata(TFolder* folder, bool relative_path=true);
};

class NWLightOutputCalibParamReader : public ParamReader<int> {
public:
    char AB;
    char ab;
    std::filesystem::path project_dir = "";
    std::filesystem::path lcalib_reldir = "database/neutron_wall/light_output_calibration";
    std::string pul_filename = "nw%c_pulse_height_calibration.dat";
    std::filesystem::path pul_path;
    std::unordered_map<int, std::unordered_map<std::string, double>> run_param;

    NWLightOutputCalibParamReader(const char AB);
    ~NWLightOutputCalibParamReader();

    void load_pulse_height();
    void load(int run);
    void write_metadata(TFolder* folder, bool relative_path=true);
};

class NWPulseShapeDiscriminationParamReader : public ParamReader<int> {
private:
    std::vector<int> not_found_bars;
    double polynomial(double x, std::vector<double>& params);
    double polynomial(double x, Json& params);
    std::vector<double> get_neutron_linear_params(double x_switch_neutron, std::vector<double>& quadratic_params);
    std::vector<double> get_neutron_linear_params(double x_switch_neutron, Json& quadratic_params);
    Json get_bar_params(int run, int bar);
    void fast_total_interpolation(int bar, Json& params);
    void centroid_interpolation(int bar, Json& params);
    void process_pca(int bar, Json& params);

public:
    char AB;
    char ab;
    std::vector<int> bars;
    std::filesystem::path project_dir = "";
    std::filesystem::path param_reldir = "database/neutron_wall/pulse_shape_discrimination/";
    std::filesystem::path param_dir; // PROJECT_DIR / param_reldir
    std::filesystem::path param_path;

    // std::unordered_map<int, Json> database; // bar -> json
    // std::unordered_map<int, std::unordered_map<std::string, double>> database; // bar -> <param, value>
    Json database;

    std::unordered_map<int, ROOT::Math::Interpolator*> gamma_fast_total_L; // bar -> interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> neutron_fast_total_L; // bar -> interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> gamma_fast_total_R; // bar -> interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> neutron_fast_total_R; // bar -> interpolator

    std::unordered_map<int, ROOT::Math::Interpolator*> gamma_vpsd_L; // bar - > interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> neutron_vpsd_L; // bar - > interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> gamma_vpsd_R; // bar - > interpolator
    std::unordered_map<int, ROOT::Math::Interpolator*> neutron_vpsd_R; // bar - > interpolator

    std::unordered_map<int, std::array<double, 2> > pca_mean; // bar -> (vpsd_L, vpsd_R)
    std::unordered_map<int, std::array<std::array<double, 2>, 2> > pca_components; // bar -> 2x2 matrix
    std::unordered_map<int, std::array<double, 2> > pca_xpeaks; // bar -> (g_xpeak, n_xpeak)

    NWPulseShapeDiscriminationParamReader(const char AB);
    ~NWPulseShapeDiscriminationParamReader();

    void load(int run);
    void read_in_calib_params();
    void write_metadata(TFolder* folder, bool relative_path=true);
};
