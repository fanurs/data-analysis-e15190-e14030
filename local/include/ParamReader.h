#pragma once

#include <any>
#include <cctype>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <string>

#include <nlohmann/json.hpp>

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