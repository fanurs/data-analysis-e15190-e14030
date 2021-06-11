#include <any>
#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "TChain.h"

#include "../include/RootIO.h"

RootReader::RootReader() { }

RootReader::RootReader(const std::string& tr_name, const std::string& path) {
    this->read_in(tr_name, path);
}

RootReader::~RootReader() { }

void RootReader::read_in(const std::string& tr_name, const std::string& path) {
    this->path = std::filesystem::path(path);
    this->tree = new TChain(tr_name.c_str());
    this->tree->Add(this->path.string().c_str());

    return;
}

void RootReader::set_branches(std::map<std::string, Branch>& branches) {
    this->branches = &branches;
    auto resize = [branches](auto&&... args) {
        (args.resize(branches.size()), ...);
    };
    resize(this->addr_int, this->addr_double, this->addr_aint, this->addr_adouble);

    int index = 0;
    for (auto& [name, branch]: branches) {
        branch.index = index;

        if (branch.type == "int") {
            this->tree->SetBranchAddress(branch.fullname.c_str(), &this->addr_int[branch.index]);
            branch.value = &this->addr_int[branch.index];
        }
        else if (branch.type == "double") {
            this->tree->SetBranchAddress(branch.fullname.c_str(), &this->addr_double[branch.index]);
            branch.value = &this->addr_double[branch.index];
        }
        else if (branch.type == "aint") {
            this->tree->SetBranchAddress(branch.fullname.c_str(), &this->addr_aint[branch.index][0]);
            branch.value = &this->addr_aint[branch.index][0];
        }
        else if (branch.type == "adouble") {
            this->tree->SetBranchAddress(branch.fullname.c_str(), &this->addr_adouble[branch.index][0]);
            branch.value = &this->addr_adouble[branch.index][0];
        }

        ++index;
    }

    return;
}

std::map<std::string, std::any> RootReader::get_entry(int i_entry) {
    this->tree->GetEntry(i_entry);

    std::map<std::string, std::any> buffer;
    for (auto& [name, branch]: *this->branches) {
        if (branch.type == "int") {
            buffer[name] = *static_cast<int*>((*this->branches)[name].value);
        }
        else if (branch.type == "double") {
            buffer[name] = *static_cast<double*>((*this->branches)[name].value);
        }
        else if (branch.type == "aint") {
            buffer[name] = static_cast<int*>((*this->branches)[name].value);
        }
        else if (branch.type == "adouble") {
            buffer[name] = static_cast<double*>((*this->branches)[name].value);
        }
    }
    return buffer;
}