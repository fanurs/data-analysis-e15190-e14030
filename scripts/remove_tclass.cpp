#include <any>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "TBranch.h"
#include "TChain.h"
#include "TError.h"
#include "TFile.h"
#include "TTree.h"

const int ARRAY_MULTI = 64;
struct Container {
    std::map<char, int> index;
    std::vector<Short_t> S;
    std::vector<UShort_t> s;
    std::vector<Int_t> I;
    std::vector<Double_t> D;

    Container();
    void resize(const std::vector<std::string>& leaflists);
    void* assign_address(std::string leaflist);
} container;

std::string get_cmd_output(const char* cmd);
std::filesystem::path get_local_path(const std::filesystem::path database_dir, const std::string key);
std::vector<std::string> get_leaflists(const std::string path, const std::string tree_name);
std::vector<std::string> get_leaflists(const std::string path);
std::string extract_branch_name(std::string leaflist);
char extract_branch_type(std::string leaflist);
bool is_array(std::string leaflist);
void* assign_address(std::string leaflist);

int main(int argc, char* argv[]) {
    int run = std::stoi(argv[1]);

    gErrorIgnoreLevel = kError;
    std::filesystem::path DATABASE_DIR = get_cmd_output("echo $DATABASE_DIR");
    std::filesystem::path inroot_dir = get_local_path(DATABASE_DIR, "daniele_root_files_dir");
    std::string filename = Form("CalibratedData_%04d.root", run);
    std::filesystem::path inpath = inroot_dir / filename;
    std::filesystem::path outpath = DATABASE_DIR / "root_files_daniele" / inpath.filename();

    std::map<std::string, void*> addr_map;
    auto leaflists = get_leaflists(inpath.string());
    container.resize(leaflists);

    std::string tree_name = get_cmd_output(Form("modular_scripts/infer_tree_name.exe %s", inpath.string().c_str()));
    auto intree = new TChain(tree_name.c_str());
    intree->Add(inpath.string().c_str());
    auto outroot = new TFile(outpath.string().c_str(), "RECREATE");
    auto outtree = new TTree(tree_name.c_str(), outpath.filename().string().c_str());

    intree->SetMakeClass(1);
    intree->SetBranchStatus("*", false);
    for (auto& leaflist : leaflists) {
        auto name = extract_branch_name(leaflist);

        intree->SetBranchStatus(name.c_str(), true);
        addr_map[name] = container.assign_address(leaflist);
        intree->SetBranchAddress(name.c_str(), addr_map[name]);

        outtree->Branch(name.c_str(), addr_map[name], leaflist.c_str());
    }

    int n_entries = intree->GetEntries();
    for (int i_entry = 0; i_entry < n_entries; i_entry++) {
        if (i_entry % 2357 == 0) {
            std::cout << Form("\r> %6.2f", 100.0 * i_entry / n_entries) << "%" << std::flush;
        }
        intree->GetEntry(i_entry);
        outtree->Fill();
    }
    std::cout << Form("\r> %6.2f", 100.0) << "%" << std::endl;

    outroot->cd();
    outtree->Write();
    outroot->Close();

    return 0;
}

std::string get_cmd_output(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    auto pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe)) {
        if (fgets(buffer.data(), 128, pipe) != nullptr)
            result += buffer.data();
    }
    result.erase(std::remove(result.end() - 1, result.end(), '\n'), result.end());
    return result;
}

std::filesystem::path get_local_path(const std::filesystem::path database_dir, const std::string key) {
    std::filesystem::path path = database_dir / "local_paths.json";
    std::ifstream file(path.string());
    nlohmann::json local_paths_json;
    file >> local_paths_json;
    file.close();
    return std::filesystem::path(local_paths_json[key]);
}

std::vector<std::string> get_leaflists(const std::string path, const std::string tree_name) {
    std::string output = get_cmd_output(Form("modular_scripts/output_leaflists.exe %s %s", path.c_str(), tree_name.c_str()));
    std::vector<std::string> leaflists;
    std::stringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        leaflists.push_back(line);
    }
    return leaflists;
}

std::vector<std::string> get_leaflists(const std::string path) {
    std::string tree_name = get_cmd_output(Form("modular_scripts/infer_tree_name.exe %s", path.c_str()));
    if (tree_name.substr(0, 5) == "ERROR") {
        throw std::runtime_error(tree_name);
    }
    return get_leaflists(path, tree_name);
}

std::string extract_branch_name(std::string leaflist) {
    auto pos_slash = leaflist.find("/");
    auto str = leaflist.substr(0, pos_slash);
    auto square_bracket_pos = str.find("[");
    return str.substr(0, square_bracket_pos);
}

char extract_branch_type(std::string leaflist) {
    auto pos_slash = leaflist.find("/");
    std::string type = leaflist.substr(pos_slash + 1);
    if (type.size() != 1) {
        throw std::runtime_error("leaflist has more than one character");
    }
    return type[0];
}

bool is_array(std::string leaflist) {
    return (leaflist.find("[") != std::string::npos);
}

Container::Container() {
    this->index['S'] = 0;
    this->index['s'] = 0;
    this->index['I'] = 0;
    this->index['D'] = 0;
}

void Container::resize(const std::vector<std::string>& leaflists) {
    std::map<char, int> sizes;
    for (const auto& leaflist : leaflists) {
        auto type = extract_branch_type(leaflist);
        sizes[type] += (is_array(leaflist) ? ARRAY_MULTI : 1);
    }

    this->S.resize(sizes['S']);
    this->s.resize(sizes['s']);
    this->I.resize(sizes['I']);
    this->D.resize(sizes['D']);
}

void* Container::assign_address(std::string leaflist) {
    void* result;
    char type = extract_branch_type(leaflist);
    int i = this->index[type];
    if      (type == 'S') result = &this->S[i];
    else if (type == 's') result = &this->s[i];
    else if (type == 'I') result = &this->I[i];
    else if (type == 'D') result = &this->D[i];
    this->index[type] += (is_array(leaflist) ? ARRAY_MULTI : 1);
    return result;
}