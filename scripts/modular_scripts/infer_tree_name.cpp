#include <iostream>
#include <string>
#include <vector>

#include "TError.h"
#include "TFile.h"
#include "TTree.h"

int main(int argc, char* argv[]) {
    gErrorIgnoreLevel = kError;
    TFile* inroot = new TFile(argv[1], "READ");
    auto keys = gDirectory->GetListOfKeys();
    std::vector<std::string> tree_names;
    for (const auto&& key : *keys) {
        std::string name = key->GetName();
        auto obj = inroot->Get(name.c_str());
        if (obj->IsA()->InheritsFrom("TTree")) {
            tree_names.push_back(name);
        }
    }
    inroot->Close();

    if (tree_names.size() == 1) {
        std::cout << tree_names[0] << std::endl;
    }
    else {
        std::cout << Form("ERROR: Found %lu trees.", tree_names.size()) << std::endl;
    }

    return 0;
}