#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "TTreeReaderValue.h"

#include "ParamReader.h"

template <typename index_t>
ParamReader<index_t>::ParamReader(const std::string& tr_name, const std::string& tr_title) {
    this->initialize_tree(tr_name, tr_title);
}

template <typename index_t>
ParamReader<index_t>::~ParamReader() { }

template <typename index_t>
void ParamReader<index_t>::initialize_tree(const std::string& tr_name, const std::string& tr_title) {
    if (tr_name == "") {
        const auto* ptr = &this->tree;
        std::ostringstream oss;
        oss << ptr;
        std::string rand_tr_name = oss.str();
        this->tree = new TTree(rand_tr_name.c_str(), tr_title.c_str());
    }
    else {
        this->tree = new TTree(tr_name.c_str(), tr_title.c_str());
    }
}

template <typename index_t>
long ParamReader<index_t>::load_from_txt(const std::string& filename, const std::string& branch_descriptor, int n_skip_rows, char delimiter) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Fail to find/open " << filename << std::endl;
        exit(1);
    }
    std::string content = "";
    std::string buffer;
    int i_row = 0;
    while (infile.is_open() && std::getline(infile, buffer)) {
        ++i_row;
        if (i_row <= n_skip_rows || buffer.size() < 2) continue;
        content += buffer + '\n';
    }
    std::istringstream iss(content);
    long n_lines = this->tree->ReadStream(iss, branch_descriptor.c_str(), delimiter);
    this->reader.SetTree(this->tree);
    infile.close();
    return n_lines;
}

template <typename index_t>
template <typename val_t>
val_t ParamReader<index_t>::get_value(index_t index, const std::string& col_name) {
    this->reader.Restart();
    TTreeReaderValue<val_t> value(this->reader, col_name.c_str());
    this->reader.SetEntry(this->index_map[index]);
    return (val_t)*value;
}