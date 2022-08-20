#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "TBranch.h"
#include "TChain.h"
#include "TError.h"
#include "TLeaf.h"
#include "TTree.h"

template<typename T>
std::vector<std::string> get_all_branch_names(T* obj);
std::string get_leaflist(TBranch* branch);
std::string get_leaflist(TBranch* branch, std::vector<std::string>& branch_names);

int main(int argc, char* argv[]) {
    gErrorIgnoreLevel = kError;

    std::string file_path = argv[1];
    std::string tree_name = argv[2];

    TChain* intree = new TChain(tree_name.c_str());
    intree->Add(file_path.c_str());
    auto branch_names = get_all_branch_names(intree);
    for (auto& branch_name : branch_names) {
        std::string leaflist = get_leaflist(intree->GetBranch(branch_name.c_str()));
        std::cout << leaflist << std::endl;
    }
    return 0;
}

template<typename T>
std::vector<std::string> get_all_branch_names(T* obj) {
    const bool is_ttree = std::is_same<T, TTree>::value;
    const bool is_tchain = std::is_same<T, TChain>::value;
    const bool is_tbranch = std::is_same<T, TBranch>::value;
    static_assert(
        is_ttree || is_tchain || is_tbranch,
        "Argument type must be a TTree, TChain, or TBranch."
    );

    auto branches = obj->GetListOfBranches();
    auto n_branches = branches->GetEntries();

    // terminate recursion
    if (n_branches == 0 && is_tbranch) return {obj->GetName()};
    if (n_branches == 0) return {}; // empty TTree or TChain

    std::vector<std::string> result = {};
    for (int i = 0; i < n_branches; i++) {
        auto branch = (TBranch*)branches->At(i);
        auto names = get_all_branch_names(branch); // recursion
        result.insert(result.end(), names.begin(), names.end());
    }
    return result;
}

std::string get_leaflist(TBranch* branch) {
    auto branch_names = get_all_branch_names(branch->GetTree());
    return get_leaflist(branch, branch_names);
}

// Only supports scalar and 1D-array
std::string get_leaflist(TBranch* branch, std::vector<std::string>& branch_names) {
    std::string result_fmt = "%s%s/%c";

    std::string title = branch->GetTitle();
    std::string branch_name = branch->GetName();
    std::string type_name = branch->FindLeaf(branch->GetName())->GetTypeName();

    char type_char = 'C'; // default
    std::map<std::string, char> type_char_map = {
        {"Char_t",     'B'},
        {"UChar_t",    'b'},
        {"Short_t",    'S'},
        {"UShort_t",   's'},
        {"Int_t",      'I'},
        {"UInt_t",     'i'},
        {"Float_t",    'F'},
        {"Float16_t",  'f'},
        {"Double_t",   'D'},
        {"Double32_t", 'd'},
        {"Long64_t",   'L'},
        {"ULong64_t",  'l'},
        {"Long_t",     'G'},
        {"ULong_t",    'g'},
        {"Bool_t",     'O'}
    };
    if (type_char_map.find(type_name) != type_char_map.end()) {
        type_char = type_char_map[type_name];
    }

    // check branch is array or scalar
    auto left_brac_pos = title.find('[');
    auto right_brac_pos = title.find(']');
    if (left_brac_pos == std::string::npos || right_brac_pos == std::string::npos) { // scalar
        return Form(result_fmt.c_str(), branch_name.c_str(), "", type_char); // scalar
    }
    std::string size_str = title.substr(left_brac_pos + 1, right_brac_pos - left_brac_pos - 1);

    // if size_str is an integer
    bool is_integer = true;
    for (char c : size_str) {
        if (!isdigit(c)) {
            is_integer = false;
            break;
        }
    }
    if (is_integer) { // array with fixed size
        return Form(result_fmt.c_str(), branch_name.c_str(), ("[" + size_str + "]").c_str(), type_char);
    }

    // otherwise, size_str is alphanumeric (i.e. another branch)
    if (std::find(branch_names.begin(), branch_names.end(), size_str) != branch_names.end()) { // array with variable size
        return Form(result_fmt.c_str(), branch_name.c_str(), ("[" + size_str + "]").c_str(), type_char);
    }

    // branch name incomplete, attempts to add prefix
    auto dot_pos = branch_name.find('.');
    std::string prefix = branch_name.substr(0, dot_pos);
    size_str = prefix + "." + size_str;
    if (std::find(branch_names.begin(), branch_names.end(), size_str) != branch_names.end()) { // array with variable size
        return Form(result_fmt.c_str(), branch_name.c_str(), ("[" + size_str + "]").c_str(), type_char);
    }
    else {
        throw std::runtime_error("Cannot determine size variable for branch \"" + title + "\".");
    }

    return "";
}
