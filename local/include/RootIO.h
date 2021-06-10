#pragma once

#include <any>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "TChain.h"

struct Branch {
    std::string fullname, type;
    bool status = true;
    int index = -1;
    void* value;
};

class RootReader {
public:
    std::filesystem::path path;
    TChain* tree;
    std::map<std::string, Branch>* branches;

    RootReader();
    RootReader(const std::string& tr_name, const std::string& path);
    ~RootReader();

    void read_in(const std::string& tr_name, const std::string& path);
    void set_branches(std::map<std::string, Branch>& branches);
    std::map<std::string, std::any> get_entry(int i_netry);
};