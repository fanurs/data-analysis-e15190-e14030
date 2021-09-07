// standard libraries
#include <any>
#include <array>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// third-party libraries
#include <nlohmann/json.hpp>

// CERN ROOT libraries
#include "TError.h"

// local libraries
#include "ParamReader.h"
#include "RootIO.h"

using Json = nlohmann::json;

struct ArgumentParser {
    int run_num = 0;
    std::string outroot_path = "";
    int first_entry = 0;
    int n_entries = -1; // negative value means all entries

    ArgumentParser(int argc, char* argv[]);
    void print_help();
};

std::array<double, 2> get_psd(
    NWPulseShapeDiscriminationParamReader& psd_reader,
    int bar, double total_L, double total_R, double fast_L, double fast_R, double pos
);

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
    NWBPositionCalibParamReader nwb_pcalib;
    nwb_pcalib.load(argparser.run_num);

    // read in pulse shape discrimination parameters
    NWPulseShapeDiscriminationParamReader nwb_psd_reader('B');
    nwb_psd_reader.load(argparser.run_num);

    // read in Daniele's ROOT files
    std::ifstream local_paths_json_file(project_dir / "database/local_paths.json");
    Json local_paths_json;
    try {
        local_paths_json_file >> local_paths_json;
        local_paths_json_file.close();
    }
    catch (...) {
        std::cerr << "Failed to read in $PROJECT_DIR/database/local_paths.json" << std::endl;
        return 1;
    }
    std::filesystem::path inroot_path = local_paths_json["daniele_root_files_dir"].get<std::string>();
    inroot_path /= Form("CalibratedData_%04d.root", argparser.run_num);
    RootReader reader(inroot_path.string(), "E15190");
    std::vector<Branch> in_branches {
        {"VW_multi",      "VetoWall.fmulti",                  "int"},
        {"VW_bar",        "VetoWall.fnumbar",                 "int[VW_multi]"},
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
        {"VW_multi",      "",  "int"},
        {"VW_bar",        "",  "int[VW_multi]"},
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
        {"NWB_psd_perp",  "",  "double[NWB_multi]"},
    };
    writer.set_branches(out_branches);

    // main loop
    std::setlocale(LC_NUMERIC, "");
    int total_n_entries = reader.tree->GetEntries();
    int last_entry;
    if (argparser.n_entries < 0) {
        last_entry = total_n_entries - 1;
    }
    else {
        last_entry = std::min(total_n_entries - 1, argparser.first_entry + argparser.n_entries - 1);
    }
    int n_entries = last_entry - argparser.first_entry + 1;
    int ievt;
    for (ievt = argparser.first_entry; ievt <= last_entry; ++ievt) {
        int iprogress = ievt - argparser.first_entry;
        if (iprogress % 4321 == 0) {
            std::cout << Form("\r> %6.2f", 1e2 * iprogress / n_entries) << "%" << std::flush;
            std::cout << Form("%28s", Form("(%'d/%'d)", ievt, total_n_entries - 1)) << std::flush;
        }

        auto buffer = reader.get_entry(ievt);
        auto vw_multi     = std::any_cast<int>    (buffer["VW_multi"]);
        auto vw_bar       = std::any_cast<int*>   (buffer["VW_bar"]);
        auto nwb_multi    = std::any_cast<int>    (buffer["NWB_multi"]);
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
        std::vector<double> nwb_psd_perp;
        for (int m = 0; m < nwb_multi; ++m) {
            // apply position calibration
            int p0 = nwb_pcalib.get(nwb_bar[m], "p0");
            int p1 = nwb_pcalib.get(nwb_bar[m], "p1");
            double pos = p0 + p1 * (nwb_time_L[m] - nwb_time_R[m]);
            nwb_positions.push_back(pos);

            // apply pulse shape discrimination
            std::array<double, 2> psd = get_psd(
                nwb_psd_reader,
                nwb_bar[m],
                double(nwb_total_L[m]), double(nwb_total_R[m]),
                double(nwb_fast_L[m]), double(nwb_fast_R[m]),
                pos
            );
            nwb_psd.push_back(psd[0]);
            nwb_psd_perp.push_back(psd[1]);
        }

        writer.set("VW_multi", vw_multi);
        writer.set("VW_bar", vw_multi, vw_bar);
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
        writer.set("NWB_psd_perp", nwb_psd_perp);

        writer.fill();
    }
    std::cout << "\r> 100.00%" << Form("%28s", Form("(%'d/%'d)", ievt - 1, total_n_entries - 1)) << std::endl;

    writer.write();

    return 0;
}

ArgumentParser::ArgumentParser(int argc, char* argv[]) {
    opterr = 0; // getopt() return '?' when getting errors

    int opt;
    while((opt = getopt(argc, argv, "hr:o:i:n:")) != -1) {
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
            case 'i':
                this->first_entry = std::stoi(optarg);
                break;
            case 'n':
                this->n_entries = std::stoi(optarg);
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
        -r      HiRA run number (four-digit).
        -o      ROOT file output path.

    Optional arguments:
        -h      Print help message.
        -i      First entry to process. Default is 0.
        -n      Number of entries to process. Default is all.
                If `n + i` is greater than the total number of entries, the
                program will safely stop after the last entry.
    )";
    std::cout << msg << std::endl;
}

std::array<double, 2> get_psd(
    NWPulseShapeDiscriminationParamReader& psd_reader,
    int bar,
    double total_L,
    double total_R,
    double fast_L,
    double fast_R,
    double pos
) {
    /*****eliminate bad data*****/
    if (total_L < 0 || total_R < 0 || fast_L < 0 || fast_R < 0
        || total_L > 4097 || total_R > 4097 || fast_L > 4097 || fast_R > 4097
        || pos < -120 || pos > 120
    ) {
        return {-9999.0, -9999.0};
    }

    /*****value assigning*****/
    double gamma_L = psd_reader.gamma_fast_total_L[bar]->Eval(total_L);
    double neutron_L = psd_reader.neutron_fast_total_L[bar]->Eval(total_L);
    double vpsd_L = (fast_L - gamma_L) / (neutron_L - gamma_L);

    double gamma_R = psd_reader.gamma_fast_total_R[bar]->Eval(total_R);
    double neutron_R = psd_reader.neutron_fast_total_R[bar]->Eval(total_R);
    double vpsd_R = (fast_R - gamma_R) / (neutron_R - gamma_R);

    /*****position correction*****/
    gamma_L = psd_reader.gamma_vpsd_L[bar]->Eval(pos);
    neutron_L = psd_reader.neutron_vpsd_L[bar]->Eval(pos);
    gamma_R = psd_reader.gamma_vpsd_R[bar]->Eval(pos);
    neutron_R = psd_reader.neutron_vpsd_R[bar]->Eval(pos);

    std::array<double, 2> xy = {vpsd_L - gamma_L, vpsd_R - gamma_R};
    std::array<double, 2> gn_vec = {neutron_L - gamma_L, neutron_R - gamma_R};
    std::array<double, 2> gn_rot90 = {-gn_vec[1], gn_vec[0]};

    // project to gn_vec and gn_rot90
    double x = (xy[0] * gn_vec[0] + xy[1] * gn_vec[1]);
    x /= (gn_vec[0] * gn_vec[0] + gn_vec[1] * gn_vec[1]);
    double y = (xy[0] * gn_rot90[0] + xy[1] * gn_rot90[1]);
    y /= (gn_rot90[0] * gn_rot90[0] + gn_rot90[1] * gn_rot90[1]);

    // PCA transform
    x -= psd_reader.pca_mean[bar][0];
    y -= psd_reader.pca_mean[bar][1];
    auto& pca_matrix = psd_reader.pca_components[bar];
    double pca_x = pca_matrix[0][0] * x + pca_matrix[0][1] * y;
    double pca_y = pca_matrix[1][0] * x + pca_matrix[1][1] * y;

    // normalization
    auto& xpeaks = psd_reader.pca_xpeaks[bar];
    double ppsd = (x - xpeaks[0]) / (xpeaks[1] - xpeaks[0]);
    double ppsd_perp = y;

    return {ppsd, ppsd_perp};
}