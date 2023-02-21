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

class NWBPositionCalibParamReader : public ParamReader<int> {
public:
    const char AB = 'B';
    const char ab = tolower(AB);
    std::filesystem::path project_dir = "";
    std::filesystem::path pcalib_reldir = "database/neutron_wall/position_calibration";
    std::string json_filename = "calib_params.json";
    std::filesystem::path pcalib_dir;
    std::filesystem::path json_path;
    Json database;
    std::map<std::pair<int, std::string>, double> run_param;

    std::filesystem::path pca_reldir = "database/neutron_wall/geometry";
    std::string dat_filename = "NWB_pca.dat";
    std::filesystem::path pca_path;
    std::map<std::pair<int, std::string>, double> L; // center of the NW bar
    std::map<std::pair<int, std::string>, double> X; // NW's principal components w.r.t L
    std::map<std::pair<int, std::string>, double> Y;
    std::map<std::pair<int, std::string>, double> Z;

    NWBPositionCalibParamReader();
    ~NWBPositionCalibParamReader();

    long load_single(int run);
    long load_single(
        const std::string& filename,
        const std::string& branch_descriptor="bar/I:p0/D:p1/D",
        bool set_index=true,
        int n_skip_rows=1,
        char delimiter=' '
    );
    bool load(int run, bool extrapolate=true);
    void set_index(const std::string& index_name="bar");
    double get(int bar, const std::string& par);
    double getL(int bar, const std::string& par);
    double getX(int bar, const std::string& par);
    double getY(int bar, const std::string& par);
    double getZ(int bar, const std::string& par);
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

class NWLightOutputCalibParamReader : public ParamReader<int> {
public:
    char AB;
    char ab;
    std::filesystem::path project_dir = "";
    std::filesystem::path lcalib_reldir = "database/neutron_wall/light_output_calibration";
    std::string sat_filename = "nw%c_saturation_recovery.csv";
    std::string pul_filename = "nw%c_pulse_height_calibration.dat";
    std::filesystem::path sat_path, pul_path;
    std::unordered_map<int, std::unordered_map<std::string, double>> run_param;

    NWLightOutputCalibParamReader(const char AB, bool load_params=true);
    ~NWLightOutputCalibParamReader();

    long load_saturation();
    void load_pulse_height();
    bool load(int run);
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
