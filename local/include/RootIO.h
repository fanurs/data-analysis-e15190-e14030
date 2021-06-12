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
protected:
    static const int MAX_MULTI = 1024;
    std::vector<int> addr_int;
    std::vector<double> addr_double;
    std::vector< std::array<int, MAX_MULTI> > addr_aint;
    std::vector< std::array<double, MAX_MULTI> > addr_adouble;

public:
    std::filesystem::path path;
    TChain* tree;
    std::map<std::string, Branch> branches;

    RootReader();
    RootReader(const std::string& tr_name, const std::string& path);
    ~RootReader();

    void read_in(const std::string& tr_name, const std::string& path);
    void set_branches(std::map<std::string, Branch>& branches);
    std::map<std::string, std::any> get_entry(int i_netry);
};