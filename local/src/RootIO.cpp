#include <algorithm>
#include <any>
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TChain.h"

#include "../include/RootIO.h"

void Branch::autofill(const std::vector<std::string>& scalar_types, const std::vector<std::string>& array_types) {
    auto is_contain = [](auto& arr, const auto& ele) -> bool {
        return std::find(arr.begin(), arr.end(), ele) != arr.end();
    };

    // autofill fullname
    if (this->fullname == "") {
        this->fullname = this->name;
    }

    // autofill leaflists

    // leaflist has been specified by user explicitly
    if (is_contain(this->leaflist, '/')) {
        // nothing to do
    }
    // empty leaflist with scalar type
    else if (this->leaflist == "" && is_contain(scalar_types, this->type)) {
        char type_char = toupper(this->type[0]);
        this->leaflist = Form("%s/%c", this->fullname.c_str(), type_char);
    }
    // leaflist specifying only the array size, with array type
    else if (this->leaflist.front() == '[' && this->leaflist.back() == ']' && is_contain(array_types, this->type)) {
        char type_char = toupper(this->type[0]);
        std::string size_str = this->leaflist.substr(1, this->leaflist.length() - 2);
        this->leaflist = Form("%s[%s]/%c", this->fullname.c_str(), size_str.c_str(), type_char);
    }
    // empty leaflist, with type that is NOT "br_name[]", i.e. could be "br_name[multi]"
    else if (this->leaflist == "" && this->type[this->type.length() - 2] != '[') {
        char type_char = toupper(this->type[0]);
        int pos = this->type.find('[');
        std::string size_str(this->type.begin() + pos + 1, this->type.end() - 1);
        this->leaflist = Form("%s[%s]/%c", this->fullname.c_str(), size_str.c_str(), type_char);
        this->type = this->type.substr(0, pos) + "[]";
    }
    else if (is_contain(array_types, this->type)) {
        throw std::invalid_argument("unrecognized leaflist for array-like branch.");
    }
}

RootReader::RootReader() { }

RootReader::RootReader(const std::string& path, const std::string& tr_name) {
    this->read_in(path, tr_name);
}

RootReader::~RootReader() { }

void RootReader::read_in(const std::string& path, const std::string& tr_name) {
    this->path = std::filesystem::path(path);
    this->tree = new TChain(tr_name.c_str());
    this->tree->Add(this->path.string().c_str());
    if (this->tree->GetEntries() <= 0) {
        std::cerr << "ERROR: Fail to open " << this->path.string() << std::endl;
        exit(1);
    }
    return;
}

void RootReader::set_branches(std::vector<Branch>& branches) {
    for (auto& branch: branches) {
        this->branch_names.push_back(branch.name);
        this->branches[branch.name] = branch;
    }

    // some autofills
    for (auto& [name, branch]: this->branches) {
        branch.autofill(this->scalar_types, this->array_types);
    }

    auto resize = [this](auto&&... args) {
        (args.resize(this->branches.size()), ...);
    };
    resize(
        this->addr_short,   this->addr_int,   this->addr_float,   this->addr_double,
        this->addr_ashort,  this->addr_aint,  this->addr_afloat,  this->addr_adouble
    );

    this->tree->SetBranchStatus("*", false);

    int index = 0;
    for (auto& br_name: this->branch_names) {
        Branch* branch = &this->branches[br_name];
        branch->index = index;
        const char* fullname = branch->fullname.c_str();
        this->tree->SetBranchStatus(fullname, true);

        // scalar types
        if (branch->type == "short") {
            this->tree->SetBranchAddress(fullname, &this->addr_short[branch->index]);
            branch->value = &this->addr_short[branch->index];
        }
        else if (branch->type == "int") {
            this->tree->SetBranchAddress(fullname, &this->addr_int[branch->index]);
            branch->value = &this->addr_int[branch->index];
        }
        else if (branch->type == "float") {
            this->tree->SetBranchAddress(fullname, &this->addr_float[branch->index]);
            branch->value = &this->addr_float[branch->index];
        }
        else if (branch->type == "double") {
            this->tree->SetBranchAddress(fullname, &this->addr_double[branch->index]);
            branch->value = &this->addr_double[branch->index];
        }
        // onward, array types
        else if (branch->type == "short[]") {
            this->tree->SetBranchAddress(fullname, &this->addr_ashort[branch->index][0]);
            branch->value = &this->addr_ashort[branch->index][0];
        }
        else if (branch->type == "int[]") {
            this->tree->SetBranchAddress(fullname, &this->addr_aint[branch->index][0]);
            branch->value = &this->addr_aint[branch->index][0];
        }
        else if (branch->type == "float[]") {
            this->tree->SetBranchAddress(fullname, &this->addr_afloat[branch->index][0]);
            branch->value = &this->addr_afloat[branch->index][0];
        }
        else if (branch->type == "double[]") {
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
        // scalar types
        if (branch.type == "short") {
            buffer[name] = *static_cast<short*>(this->branches[name].value);
        }
        else if (branch.type == "int") {
            buffer[name] = *static_cast<int*>(this->branches[name].value);
        }
        else if (branch.type == "float") {
            buffer[name] = *static_cast<float*>(this->branches[name].value);
        }
        else if (branch.type == "double") {
            buffer[name] = *static_cast<double*>(this->branches[name].value);
        }
        // onward, array types
        else if (branch.type == "short[]") {
            buffer[name] = static_cast<short*>(this->branches[name].value);
        }
        else if (branch.type == "int[]") {
            buffer[name] = static_cast<int*>(this->branches[name].value);
        }
        else if (branch.type == "float[]") {
            buffer[name] = static_cast<float*>(this->branches[name].value);
        }
        else if (branch.type == "double[]") {
            buffer[name] = static_cast<double*>(this->branches[name].value);
        }
    }
    return buffer;
}

RootWriter::RootWriter(const std::string& path, const std::string& tr_name, const std::string& file_option) {
    this->path = std::filesystem::path(path);
    this->file = new TFile(this->path.string().c_str(), file_option.c_str());
    this->trees[tr_name].ttree = new TTree(tr_name.c_str(), "");
}

RootWriter::RootWriter(const std::string& path, const std::initializer_list<std::string>& tr_names, const std::string& file_option) {
    this->path = std::filesystem::path(path);
    this->file = new TFile(this->path.string().c_str(), file_option.c_str());
    for (auto& tr_name: tr_names) {
        this->trees[tr_name].ttree = new TTree(tr_name.c_str(), "");
    }
}

RootWriter::~RootWriter() {
    this->file->Close();
}

std::string RootWriter::get_tr_name() {
    if (this->trees.size() != 1) {
        throw std::invalid_argument("There are more than one TTree objects.");
    }
    return this->trees.begin()->first;
}

void RootWriter::set_branches(const std::string& tr_name, std::vector<Branch>& branches) {
    Tree* tree = &this->trees[tr_name];
    for (auto& branch: branches) {
        tree->branch_names.push_back(branch.name);
        tree->branches[branch.name] = branch;
    }

    // some autofills
    auto is_contain = [](auto& arr, const auto& ele) -> bool {
        return std::find(arr.begin(), arr.end(), ele) != arr.end();
    };
    for (auto& [name, branch]: tree->branches) {
        branch.autofill(this->scalar_types, this->array_types);
    }

    auto resize = [tree](auto&&... args) {
        (args.resize(tree->branches.size()), ...);
    };
    resize(
        this->addr_short,   this->addr_int,   this->addr_float,   this->addr_double,
        this->addr_ashort,  this->addr_aint,  this->addr_afloat,  this->addr_adouble
    );

    // define branches and their addresses
    int index = 0;
    for (auto& br_name: tree->branch_names) {
        Branch* branch = &tree->branches[br_name];
        branch->index = index;
        const char* fullname = branch->fullname.c_str();
        const char* leaflist = branch->leaflist.c_str();

        // scalar types
        if (branch->type == "short") {
            tree->ttree->Branch(fullname, &this->addr_short[branch->index], leaflist);
            branch->value = &this->addr_short[branch->index];
        }
        else if (branch->type == "int") {
            tree->ttree->Branch(fullname, &this->addr_int[branch->index], leaflist);
            branch->value = &this->addr_int[branch->index];
        }
        else if (branch->type == "float") {
            tree->ttree->Branch(fullname, &this->addr_float[branch->index], leaflist);
            branch->value = &this->addr_float[branch->index];
        }
        else if (branch->type == "double") {
            tree->ttree->Branch(fullname, &this->addr_double[branch->index], leaflist);
            branch->value = &this->addr_double[branch->index];
        }
        // onward, array types
        else if (branch->type == "short[]") {
            tree->ttree->Branch(fullname, &this->addr_ashort[branch->index][0], leaflist);
            branch->value = &this->addr_ashort[branch->index][0];
        }
        else if (branch->type == "int[]") {
            tree->ttree->Branch(fullname, &this->addr_aint[branch->index][0], leaflist);
            branch->value = &this->addr_aint[branch->index][0];
        }
        else if (branch->type == "float[]") {
            tree->ttree->Branch(fullname, &this->addr_afloat[branch->index][0], leaflist);
            branch->value = &this->addr_afloat[branch->index][0];
        }
        else if (branch->type == "double[]") {
            tree->ttree->Branch(fullname, &this->addr_adouble[branch->index][0], leaflist);
            branch->value = &this->addr_adouble[branch->index][0];
        }

        ++index;
    }

    return;
}

void RootWriter::set_branches(std::vector<Branch>& branches) {
    this->set_branches(this->get_tr_name(), branches);
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, const void* source, std::size_t nbytes) {
    std::memcpy(this->trees[tr_name].branches[br_name].value, source, nbytes);
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, short source) {
    this->set(tr_name, br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, int source) {
    this->set(tr_name, br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, float source) {
    this->set(tr_name, br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, double source) {
    this->set(tr_name, br_name, &source, sizeof(source));
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, std::vector<short>& source) {
    this->set(tr_name, br_name, &source[0], sizeof(source[0]) * source.size());
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, std::vector<int>& source) {
    this->set(tr_name, br_name, &source[0], sizeof(source[0]) * source.size());
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, std::vector<float>& source) {
    this->set(tr_name, br_name, &source[0], sizeof(source[0]) * source.size());
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, std::vector<double>& source) {
    this->set(tr_name, br_name, &source[0], sizeof(source[0]) * source.size());
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, int size, short* source) {
    this->set(tr_name, br_name, source, sizeof(source[0]) * size);
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, int size, int* source) {
    this->set(tr_name, br_name, source, sizeof(source[0]) * size);
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, int size, float* source) {
    this->set(tr_name, br_name, source, sizeof(source[0]) * size);
}

void RootWriter::set(const std::string& tr_name, const std::string& br_name, int size, double* source) {
    this->set(tr_name, br_name, source, sizeof(source[0]) * size);
}

void RootWriter::set(const std::string& br_name, short source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, int source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, float source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, double source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, std::vector<short>& source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, std::vector<int>& source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, std::vector<float>& source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, std::vector<double>& source) {
    this->set(this->get_tr_name(), br_name, source);
}

void RootWriter::set(const std::string& br_name, int size, short* source) {
    this->set(this->get_tr_name(), br_name, size, source);
}

void RootWriter::set(const std::string& br_name, int size, int* source) {
    this->set(this->get_tr_name(), br_name, size, source);
}

void RootWriter::set(const std::string& br_name, int size, float* source) {
    this->set(this->get_tr_name(), br_name, size, source);
}

void RootWriter::set(const std::string& br_name, int size, double* source) {
    this->set(this->get_tr_name(), br_name, size, source);
}

void RootWriter::fill() {
    for (auto& [tr_name, tree]: this->trees) {
        this->fill(tr_name);
    }
}

int RootWriter::fill(const std::string& tr_name) {
    return this->trees[tr_name].ttree->Fill();
}

void RootWriter::write() {
    for (auto& [tr_name, tree]: this->trees) {
        this->write(tr_name);
    }
}

int RootWriter::write(const std::string& tr_name) {
    this->file->cd();
    return this->trees[tr_name].ttree->Write();
}