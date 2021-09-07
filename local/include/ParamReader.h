#pragma once

#include <any>
#include <array>
#include <cctype>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <string>

#include <nlohmann/json.hpp>

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
    std::filesystem::path pcalib_reldir = "database/neutron_wall/position_calibration";
    std::string json_filename = "calib_params.json";
    std::filesystem::path pcalib_dir;
    std::filesystem::path json_path;
    Json database;
    std::map<std::pair<int, std::string>, double> run_param;

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
};

class NWPulseShapeDiscriminationParamReader : public ParamReader<int> {
public:
    char AB;
    char ab;
    std::vector<int> bars;
    std::filesystem::path param_reldir = "database/neutron_wall/pulse_shape_discrimination/calib_params";
    std::filesystem::path param_dir;

    std::unordered_map<int, Json> database; // bar -> json

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

    void reconstruct_interpolators(int bar);
    void process_pca(int bar);
    bool load_single_bar(int run, int bar);
    bool load(int run);
};