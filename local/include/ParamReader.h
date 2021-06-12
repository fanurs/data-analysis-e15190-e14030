#pragma once

#include <any>
#include <map>
#include <string>

#include "TTree.h"
#include "TTreeReader.h"

template <typename index_t>
class ParamReader {
public:
    TTree* tree;
    TTreeReader reader;
    std::map<index_t, int> index_map;

    ParamReader(const std::string& tr_name="", const std::string& tr_title="");
    ~ParamReader() { }

    void initialize_tree(const std::string& tr_name, const std::string& tr_title);
    long load_from_txt(const std::string& filename, const std::string& branch_descriptor, int n_skip_rows=0, char delimiter=' ');

    template <typename val_t>
    val_t get(index_t index, const std::string& col_name);
};
#include "ParamReader.tpp"

class NWBPositionCalibParamReader : public ParamReader<int> {
public:
    long load(
        const std::string& filename,
        const std::string& branch_descriptor="bar/I:p0/D:p1/D",
        int n_skip_rows=1,
        char delimiter=' ');
    void set_index(const std::string& index_name="bar");
};