#include <iostream>
#include <string>

#include "TTreeReaderValue.h"

#include "ParamReader.h"

long NWBPositionCalibParamReader::load(
    const std::string& filename,
    const std::string& branch_descriptor,
    int n_skip_rows,
    char delimiter
) {
    return this->load_from_txt(filename, branch_descriptor, n_skip_rows, delimiter);
}

void NWBPositionCalibParamReader::set_index(const std::string& index_name) {
    TTreeReaderValue<int> index(this->reader, index_name.c_str());
    int n_entries = this->tree->GetEntries();
    for (int i_entry = 0; i_entry < n_entries; ++i_entry) {
        this->reader.SetEntry(i_entry);
        this->index_map[*index] = i_entry;
    }
}
