// standard libraries
#include <any>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// third-party libraries
#include <json/json.h>

// CERN ROOT libraries
#include "TError.h"

// local libraries
#include "ParamReader.h"
#include "RootIO.h"

struct ArgumentParser {
    int run_num = 0;
    std::string outroot_path = "";

    ArgumentParser(int argc, char* argv[]);
    void print_help();
};

int main(int argc, char* argv[]) {
    // initialization and argument parsing
    gErrorIgnoreLevel = kError; // ignore warnings
    const char* PROJECT_DIR = getenv("PROJECT_DIR");
    if (PROJECT_DIR == nullptr) {
        std::cerr << "Environment variable $PROJECT_DIR is not defined in current session" << std::endl;
        return 1;
    }
    std::filesystem::path project_dir(PROJECT_DIR);
    ArgumentParser argparser(argc, argv);

    // read in position calibration parameters
    std::filesystem::path pos_calib_path = project_dir;
    pos_calib_path /= "database/neutron_wall/position_calibration/calib_params";
    pos_calib_path /= Form("run-%04d-nwb.dat", argparser.run_num);
    NWBPositionCalibParamReader nwb_pcalib;
    nwb_pcalib.load(pos_calib_path.string());
    nwb_pcalib.set_index("bar");

    // read in Daniele's ROOT files
    std::ifstream local_paths_json(project_dir / "database/local_paths.json");
    Json::CharReaderBuilder json_builder;
    Json::Value json_value;
    auto parse_success = parseFromStream(json_builder, local_paths_json, &json_value, NULL);
    local_paths_json.close();
    if (!parse_success) {
        std::cerr << "Failed to read in $PROJECT_DIR/database/local_paths.json" << std::endl;
        return 1;
    }
    std::filesystem::path inroot_path = json_value["daniele_root_files_dir"].asString();
    inroot_path /= Form("CalibratedData_%04d.root", argparser.run_num);
    RootReader reader(inroot_path.string(), "E15190");
    std::vector<Branch> in_branches {
        {"NWB_multi",     "NWB.fmulti",                       "int"},
        {"NWB_bar",       "NWB.fnumbar",                      "int[NWB_multi]"},
        {"NWB_time_L",    "NWB.fTimeLeft",                    "double[NWB_multi]"},
        {"NWB_time_R",    "NWB.fTimeRight",                   "double[NWB_multi]"},
        {"NWB_total_L",   "NWB.fLeft",                        "short[NWB_multi]"},
        {"NWB_total_R",   "NWB.fRight",                       "short[NWB_multi]"},
        {"NWB_fast_L",    "NWB.ffastLeft",                    "short[NWB_multi]"},
        {"NWB_fast_R",    "NWB.ffastRight",                   "short[NWB_multi]"},
        {"NWB_light_GM",  "NWB.fGeoMeanSaturationCorrected",  "double[NWB_multi]"},
    };
    reader.set_branches(in_branches);

    // prepare output (calibrated) ROOT files
    RootWriter writer(argparser.outroot_path, "tree");
    std::vector<Branch> out_branches {
        {"NWB_multi",     "",  "int"},
        {"NWB_bar",       "",  "int[NWB_multi]"},
        {"NWB_time_L",    "",  "double[NWB_multi]"},
        {"NWB_time_R",    "",  "double[NWB_multi]"},
        {"NWB_total_L",   "",  "short[NWB_multi]"},
        {"NWB_total_R",   "",  "short[NWB_multi]"},
        {"NWB_fast_L",    "",  "short[NWB_multi]"},
        {"NWB_fast_R",    "",  "short[NWB_multi]"},
        {"NWB_light_GM",  "",  "double[NWB_multi]"},
        // new (calibrated) branches
        {"NWB_pos",       "",  "double[NWB_multi]"},
        {"NWB_psd",       "",  "double[NWB_multi]"},
    };
    writer.set_branches(out_branches);

    // main loop
    int n_entries = reader.tree->GetEntries();
    for (int ievt = 0; ievt < n_entries; ++ievt) {
        if (ievt % 4321 == 0) {
            std::cout << Form("\r> %6.2f", 1e2 * ievt / n_entries) << "%" << std::flush;
            std::cout << Form("%24s", Form("(%d/%d)", ievt, n_entries)) << std::flush;
        }

        auto buffer = reader.get_entry(ievt);
        auto nwb_multi    = std::any_cast<int>    (buffer["NWB_multi"]);
        if (nwb_multi == 0) continue;
        auto nwb_bar      = std::any_cast<int*>   (buffer["NWB_bar"]);
        auto nwb_time_L   = std::any_cast<double*>(buffer["NWB_time_L"]);
        auto nwb_time_R   = std::any_cast<double*>(buffer["NWB_time_R"]);
        auto nwb_total_L  = std::any_cast<short*> (buffer["NWB_total_L"]);
        auto nwb_total_R  = std::any_cast<short*> (buffer["NWB_total_R"]);
        auto nwb_fast_L   = std::any_cast<short*> (buffer["NWB_fast_L"]);
        auto nwb_fast_R   = std::any_cast<short*> (buffer["NWB_fast_R"]);
        auto nwb_light_GM = std::any_cast<double*>(buffer["NWB_light_GM"]);

        // calibrations
        std::vector<double> nwb_positions;
        std::vector<double> nwb_psd;
        for (int m = 0; m < nwb_multi; ++m) {
            // apply position calibration
            int p0 = nwb_pcalib.get<double>(nwb_bar[m], "p0");
            int p1 = nwb_pcalib.get<double>(nwb_bar[m], "p1");

            double pos = p0 + p1 * (nwb_time_L[m] - nwb_time_R[m]);
            nwb_positions.push_back(pos);

            // apply pulse shape discrimination
            nwb_psd.push_back(0.0);
        }

        writer.set("NWB_multi", nwb_multi);
        writer.set("NWB_bar", nwb_multi, nwb_bar);
        writer.set("NWB_time_L", nwb_multi, nwb_time_L);
        writer.set("NWB_time_R", nwb_multi, nwb_time_R);
        writer.set("NWB_total_L", nwb_multi, nwb_total_L);
        writer.set("NWB_total_R", nwb_multi, nwb_total_R);
        writer.set("NWB_fast_L", nwb_multi, nwb_fast_L);
        writer.set("NWB_fast_R", nwb_multi, nwb_fast_R);
        writer.set("NWB_light_GM", nwb_multi, nwb_light_GM);
        // new (calibrated) branches
        writer.set("NWB_pos", nwb_positions);
        writer.set("NWB_psd", nwb_psd);

        writer.fill();
    }
    std::cout << "\r> 100.00%" << Form("%24s", Form("(%d/%d)", n_entries, n_entries)) << std::endl;

    writer.write();

    return 0;
}

ArgumentParser::ArgumentParser(int argc, char* argv[]) {
    opterr = 0; // getopt() return '?' when getting errors

    int opt;
    while((opt = getopt(argc, argv, "hr:o:")) != -1) {
        switch (opt) {
            case 'h':
                this->print_help();
                exit(0);
            case 'r':
                this->run_num = std::stoi(optarg);
                break;
            case 'o':
                this->outroot_path = optarg;
                break;
            case '?':
                if (optopt == 'r') {
                    std::cerr << "Option -" << char(optopt) << " requires an argument" << std::endl;
                }
                else {
                    std::cerr << "Unknown option -" << char(optopt) << std::endl;
                }
                exit(1);
        }
    }

    // check for mandatory arguments
    if (this->run_num == 0) {
        std::cerr << "Option -r is mandatory" << std::endl;
        exit(1);
    }
    if (this->outroot_path == "") {
        std::cerr << "Option -o is mandatory" << std::endl;
        exit(1);
    }
}

void ArgumentParser::print_help() {
    const char* msg = R"(
    Mandatory arguments:
        -r      HiRA run number (four-digit)
        -o      ROOT file output path

    Optional arguments:
        -h      Print help message
    )";
    std::cout << msg << std::endl;
}