#include <any>
#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "TChain.h"

#include "../include/RootIO.h"

RootReader::RootReader() { }

RootReader::RootReader(const std::string& path, const std::string& tr_name) {
    this->read_in(path, tr_name);
}

RootReader::~RootReader() { }

void RootReader::read_in(const std::string& path, const std::string& tr_name) {
    this->path = std::filesystem::path(path);
    this->tree = new TChain(tr_name.c_str());
    this->tree->Add(this->path.string().c_str());
    return;
}

void RootReader::set_branches(std::vector<Branch>& branches) {
    for (auto& branch: branches) {
        this->branch_names.push_back(branch.name);
        this->branches[branch.name] = branch;
    }

    auto resize = [this](auto&&... args) {
        (args.resize(this->branches.size()), ...);
    };
    resize(this->addr_int, this->addr_double, this->addr_aint, this->addr_adouble);

    int index = 0;
    for (auto& br_name: this->branch_names) {
        Branch* branch = &this->branches[br_name];
        branch->index = index;
        const char* fullname = branch->fullname.c_str();

        if (branch->type == "int") {
            this->tree->SetBranchAddress(fullname, &this->addr_int[branch->index]);
            branch->value = &this->addr_int[branch->index];
        }
        else if (branch->type == "double") {
            this->tree->SetBranchAddress(fullname, &this->addr_double[branch->index]);
            branch->value = &this->addr_double[branch->index];
        }
        else if (branch->type == "aint") {
            this->tree->SetBranchAddress(fullname, &this->addr_aint[branch->index][0]);
            branch->value = &this->addr_aint[branch->index][0];
        }
        else if (branch->type == "adouble") {
            this->tree->SetBranchAddress(fullname, &this->addr_adouble[branch->index][0]);
            branch->value = &this->addr_adouble[branch->index][0];
        }

        ++index;
    }

    return;
}

std::map<std::string, std::any> RootReader::get_entry(int i_entry) {
    this->tree->GetEntry(i_entry);

    std::map<std::string, std::any> buffer;
    for (auto& [name, branch]: this->branches) {
        if (branch.type == "int") {
            buffer[name] = *static_cast<int*>(this->branches[name].value);
        }
        else if (branch.type == "double") {
            buffer[name] = *static_cast<double*>(this->branches[name].value);
        }
        else if (branch.type == "aint") {
            buffer[name] = static_cast<int*>(this->branches[name].value);
        }
        else if (branch.type == "adouble") {
            buffer[name] = static_cast<double*>(this->branches[name].value);
        }
    }
    return buffer;
}

RootWriter::RootWriter() { }

RootWriter::RootWriter(const std::string& path, const std::string& tr_name, const std::string& file_option) {
    this->initialize(path, tr_name, file_option);
}

RootWriter::~RootWriter() {
    this->file->Close();
}

void RootWriter::initialize(const std::string& path, const std::string& tr_name, const std::string& file_option) {
    this->path = std::filesystem::path(path);
    this->file = new TFile(this->path.string().c_str(), file_option.c_str());
    this->tree = new TTree(tr_name.c_str(), "");
    return;
}

void RootWriter::set_branches(std::vector<Branch>& branches) {
    for (auto& branch: branches) {
        this->branch_names.push_back(branch.name);
        this->branches[branch.name] = branch;
    }

    auto resize = [this](auto&&... args) {
        (args.resize(this->branches.size()), ...);
    };
    resize(this->addr_int, this->addr_double, this->addr_aint, this->addr_adouble);


    int index = 0;
    for (auto& br_name: this->branch_names) {
        Branch* branch = &this->branches[br_name];
        branch->index = index;
        const char* fullname = branch->fullname.c_str();
        const char* leaflist = branch->leaflist.c_str();

        if (branch->type == "int") {
            this->tree->Branch(fullname, &this->addr_int[branch->index], leaflist);
            branch->value = &this->addr_int[branch->index];
        }
        else if (branch->type == "double") {
            this->tree->Branch(fullname, &this->addr_double[branch->index], leaflist);
            branch->value = &this->addr_double[branch->index];
        }
        else if (branch->type == "aint") {
            this->tree->Branch(fullname, &this->addr_aint[branch->index][0], leaflist);
            branch->value = &this->addr_aint[branch->index][0];
        }
        else if (branch->type == "adouble") {
            this->tree->Branch(fullname, &this->addr_adouble[branch->index][0], leaflist);
            branch->value = &this->addr_adouble[branch->index][0];
        }

        ++index;
    }

    return;
}

void RootWriter::set(const std::string& br_name, const void* source, std::size_t nbytes) {
    std::memcpy(this->branches[br_name].value, source, nbytes);
}

void RootWriter::set(const std::string& br_name, int source) {
    this->set(br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& br_name, double source) {
    this->set(br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& br_name, std::vector<int>& source) {
    this->set(br_name, &source[0], sizeof(source[0]) * source.size());
}

void RootWriter::set(const std::string& br_name, std::vector<double>& source) {
    this->set(br_name, &source[0], sizeof(source[0]) * source.size());
}

int RootWriter::fill() {
    return this->tree->Fill();
}