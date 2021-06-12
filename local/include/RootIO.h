#pragma once

#include <any>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

struct Branch {
    std::string name, fullname, type;
    bool status = true;
    int index = -1;
    void* value;
};

class RootReader {
protected:
    static const int MAX_MULTI = 1024;
    std::vector<int> addr_int;
    std::vector<double> addr_double;
    std::vector< std::array<int, MAX_MULTI> > addr_aint;
    std::vector< std::array<double, MAX_MULTI> > addr_adouble;

public:
    std::filesystem::path path;
    TChain* tree;
    std::vector<std::string> branch_names;
    std::unordered_map<std::string, Branch> branches;

    RootReader();
    RootReader(const std::string& path, const std::string& tr_name);
    ~RootReader();

    void read_in(const std::string& path, const std::string& tr_name);
    void set_branches(std::vector<Branch>& branches);
    std::map<std::string, std::any> get_entry(int i_netry);
};