#pragma once

// standard libraries
#include <array>
#include <clocale>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// CERN ROOT libraries
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

struct Container {
    static constexpr int max_multi = 128;

    // TDC triggers
    double TDC_hira_ds_nwtdc;
    double TDC_hira_live;
    double TDC_master;
    double TDC_master_nw;
    double TDC_master_vw;
    double TDC_nw_ds;
    double TDC_nw_ds_nwtdc;
    double TDC_rf_nwtdc;
    double TDC_mb_hira;
    double TDC_mb_hira_nwtdc;
    double TDC_mb_nw;
    double TDC_mb_nw_nwtdc;
    double TDC_mb_ds;

    // Microball
    int MB_multi;
    std::array<int, max_multi> MB_ring; // ring number
    std::array<int, max_multi> MB_det; // detector (crystal) number
    std::array<short, max_multi> MB_tail;
    std::array<short, max_multi> MB_fast;
    std::array<short, max_multi> MB_time;

    // Forward Array
    int FA_multi;
    double FA_time_min;
    double FA_time_mean;
    std::array<int, max_multi> FA_det; // detector (crystal) number
    std::array<short, max_multi> FA_total;
    std::array<double, max_multi> FA_time;

    // Veto Wall
    int VW_multi;
    std::array<int, max_multi> VW_bar;
    std::array<short, max_multi> VW_total_T; // top
    std::array<short, max_multi> VW_total_B; // bottom
    std::array<double, max_multi> VW_time_T;
    std::array<double, max_multi> VW_time_B;
    std::array<double, max_multi> VW_pos_y;
    std::array<double, max_multi> VW_theta;
    std::array<double, max_multi> VW_phi;
    std::array<double, max_multi> VW_distance;

    // Neutron Wall A
    int NWA_multi;
    std::array<int, max_multi> NWA_bar;

    // Neutron Wall B
    int NWB_multi;
    std::array<int, max_multi> NWB_bar;
    std::array<short, max_multi> NWB_total_L;
    std::array<short, max_multi> NWB_total_R;
    std::array<short, max_multi> NWB_fast_L;
    std::array<short, max_multi> NWB_fast_R;
    std::array<double, max_multi> NWB_time;
    std::array<double, max_multi> NWB_time_L;
    std::array<double, max_multi> NWB_time_R;
    std::array<double, max_multi> NWB_tof; /*updated*/
    std::array<double, max_multi> NWB_pos_x; /*updated*/
    std::array<double, max_multi> NWB_pos_y; /*updated*/
    std::array<double, max_multi> NWB_pos_z; /*updated*/
    std::array<double, max_multi> NWB_distance; /*updated*/
    std::array<double, max_multi> NWB_theta; /*updated*/
    std::array<double, max_multi> NWB_phi; /*updated*/
    std::array<double, max_multi> NWB_distance_c; /*updated*/
    std::array<double, max_multi> NWB_theta_c; /*updated*/
    std::array<double, max_multi> NWB_phi_c; /*updated*/
    std::array<double, max_multi> NWB_light_GM; /*updated*/
    std::array<double, max_multi> NWB_psd; /*updated*/
    std::array<double, max_multi> NWB_psd_perp; /*updated*/
};
Container container;

class ArgumentParser {
public:
    int run_num = 0;
    std::string outroot_path = "";
    int first_entry = 0;
    int n_entries = -1; // negative value means all entries

    ArgumentParser(int argc, char* argv[]) {
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

    void print_help() {
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
};

class ProgressBar {
public:
    int total_n_entries; // total number of entries in the input ROOT file
    int last_entry; // last entry number to process
    int n_entries; // number of entries to process
    ArgumentParser* argparser;

    ProgressBar(ArgumentParser& argparser, int total_n_entries) {
        this->argparser = &argparser;
        this->total_n_entries = total_n_entries;
        std::setlocale(LC_NUMERIC, "");
        if (this->argparser->n_entries < 0) {
            this->last_entry = this->total_n_entries - 1;
        }
        else {
            this->last_entry = std::min(
                this->total_n_entries - 1,
                this->argparser->first_entry + this->argparser->n_entries - 1
            );
        }
        this->n_entries = this->last_entry - this->argparser->first_entry + 1;
    }

    void show(int i_evt, int step=4321) {
        int i_progress = i_evt - this->argparser->first_entry;
        if (i_progress % step == 0) {
            std::cout << Form("\r> %6.2f", 1e2 * i_progress / this->n_entries) << "%";
            std::cout << Form("%30s", Form("i_evt: %'d/%'d", i_evt, this->last_entry));
            std::cout << Form("%30s", Form("(total_nevts: %'d)", this->total_n_entries));
            std::cout << std::flush;
        }
    }

    void terminate() {
        std::cout << "\r> 100.00%";
        std::cout << Form("%30s", Form("i_evt: %'d/%'d", this->last_entry, this->last_entry));
        std::cout << Form("%30s", Form("(total_nevts: %'d)", this->total_n_entries));
        std::cout << std::endl << std::flush;
    }
};

std::filesystem::path get_project_dir(std::string env_var="PROJECT_DIR") {
    const char* project_dir_c_str = getenv(env_var.c_str());
    if (project_dir_c_str == nullptr) {
        std::cerr << "Environment variable $";
        std::cerr << env_var;
        std::cerr << " is not defined in current session" << std::endl;
        exit(1);
    }
    std::filesystem::path project_dir(project_dir_c_str);
    return project_dir;
}

std::filesystem::path get_input_root_path(
    std::filesystem::path& project_dir,
    ArgumentParser& argparser
) {
    /* Return input ROOT file path under database (tclass removed) */
    auto path = project_dir / "database" / "root_files_daniele";
    path /= Form("CalibratedData_%04d.root", argparser.run_num);
    return path;
}

std::filesystem::path get_input_root_path(
    std::filesystem::path& project_dir,
    ArgumentParser& argparser,
    std::string json_key
) {
    /* Return input ROOT file path from original Daniele's framework */
    std::ifstream local_paths_json_file(project_dir / "database/local_paths.json");
    Json local_paths_json;
    try {
        local_paths_json_file >> local_paths_json;
        local_paths_json_file.close();
    }
    catch (...) {
        std::cerr << "Failed to read in \"database/local_paths.json\"" << std::endl;
        exit(1);
    }
    std::filesystem::path inroot_path = local_paths_json[json_key].get<std::string>();
    inroot_path /= Form("CalibratedData_%04d.root", argparser.run_num);
    return inroot_path;
}

TChain* get_input_tree(const std::string& path, const std::string& tree_name) {
    TChain* chain = new TChain(tree_name.c_str());
    chain->Add(path.c_str());

    //TDC triggers
    chain->SetBranchAddress("TDCTriggers.HiRA_DS_TRG_NWTDC",    &container.TDC_hira_ds_nwtdc);
    chain->SetBranchAddress("TDCTriggers.HiRA_LIVE",            &container.TDC_hira_live);
    chain->SetBranchAddress("TDCTriggers.MASTER_TRG",           &container.TDC_master);
    chain->SetBranchAddress("TDCTriggers.MASTER_TRG_NWTDC",     &container.TDC_master_nw);
    chain->SetBranchAddress("TDCTriggers.MASTER_TRG_VWTDC",     &container.TDC_master_vw);
    chain->SetBranchAddress("TDCTriggers.NW_DS_TRG",            &container.TDC_nw_ds);
    chain->SetBranchAddress("TDCTriggers.NW_DS_TRG_NWTDC",      &container.TDC_nw_ds_nwtdc);
    chain->SetBranchAddress("TDCTriggers.RF_TRG_NWTDC",         &container.TDC_rf_nwtdc);
    chain->SetBranchAddress("TDCTriggers.uBallHiRA_TRG",        &container.TDC_mb_hira);
    chain->SetBranchAddress("TDCTriggers.uBallHiRA_TRG_NWTDC",  &container.TDC_mb_hira_nwtdc);
    chain->SetBranchAddress("TDCTriggers.uBallNW_TRG",          &container.TDC_mb_nw);
    chain->SetBranchAddress("TDCTriggers.uBallNW_TRG_NWTDC",    &container.TDC_mb_nw_nwtdc);
    chain->SetBranchAddress("TDCTriggers.uBall_DS_TRG",         &container.TDC_mb_ds);
    // Microball
    chain->SetBranchAddress("uBall.fmulti",                     &container.MB_multi);
    chain->SetBranchAddress("uBall.fnumring",                   &container.MB_ring[0]);
    chain->SetBranchAddress("uBall.fnumdet",                    &container.MB_det[0]);
    chain->SetBranchAddress("uBall.fTail",                      &container.MB_tail[0]);
    chain->SetBranchAddress("uBall.fFast",                      &container.MB_fast[0]);
    chain->SetBranchAddress("uBall.fTime",                      &container.MB_time[0]);
    // Forward Array
    chain->SetBranchAddress("ForwardArray.fmulti",              &container.FA_multi);
    chain->SetBranchAddress("ForwardArray.fTimeMin",            &container.FA_time_min);
    chain->SetBranchAddress("ForwardArray.fTimeMean",           &container.FA_time_mean);
    chain->SetBranchAddress("ForwardArray.fnumdet",             &container.FA_det[0]);
    chain->SetBranchAddress("ForwardArray.fE",                  &container.FA_total[0]);
    chain->SetBranchAddress("ForwardArray.fTime",               &container.FA_time[0]);
    // Veto Wall
    chain->SetBranchAddress("VetoWall.fmulti",                  &container.VW_multi);
    chain->SetBranchAddress("VetoWall.fnumbar",                 &container.VW_bar[0]);
    chain->SetBranchAddress("VetoWall.fTop",                    &container.VW_total_T[0]);
    chain->SetBranchAddress("VetoWall.fBottom",                 &container.VW_total_B[0]);
    chain->SetBranchAddress("VetoWall.fTimeTop",                &container.VW_time_T[0]);
    chain->SetBranchAddress("VetoWall.fTimeBottom",             &container.VW_time_B[0]);
    chain->SetBranchAddress("VetoWall.fYcm",                    &container.VW_pos_y[0]);
    chain->SetBranchAddress("VetoWall.fThetaRan",               &container.VW_theta[0]);
    chain->SetBranchAddress("VetoWall.fPhiRan",                 &container.VW_phi[0]);
    chain->SetBranchAddress("VetoWall.fDistRancm",              &container.VW_distance[0]);
    // Neutron Wall A
    chain->SetBranchAddress("NWA.fmulti",                       &container.NWA_multi);
    chain->SetBranchAddress("NWA.fnumbar",                      &container.NWA_bar[0]);
    // Neutron Wall B
    chain->SetBranchAddress("NWB.fmulti",                       &container.NWB_multi);
    chain->SetBranchAddress("NWB.fnumbar",                      &container.NWB_bar[0]);
    chain->SetBranchAddress("NWB.fLeft",                        &container.NWB_total_L[0]);
    chain->SetBranchAddress("NWB.fRight",                       &container.NWB_total_R[0]);
    chain->SetBranchAddress("NWB.ffastLeft",                    &container.NWB_fast_L[0]);
    chain->SetBranchAddress("NWB.ffastRight",                   &container.NWB_fast_R[0]);
    chain->SetBranchAddress("NWB.fTimeMean",                    &container.NWB_time[0]);
    chain->SetBranchAddress("NWB.fTimeLeft",                    &container.NWB_time_L[0]);
    chain->SetBranchAddress("NWB.fTimeRight",                   &container.NWB_time_R[0]);

    // enable class objects
    chain->SetMakeClass(1);

    // set branch status
    chain->SetBranchStatus("*", false);
    // TDC triggers
    chain->SetBranchStatus("TDCTriggers.HiRA_DS_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.HiRA_LIVE", true);
    chain->SetBranchStatus("TDCTriggers.MASTER_TRG", true);
    chain->SetBranchStatus("TDCTriggers.MASTER_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.MASTER_TRG_VWTDC", true);
    chain->SetBranchStatus("TDCTriggers.NW_DS_TRG", true);
    chain->SetBranchStatus("TDCTriggers.NW_DS_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.RF_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.uBallHiRA_TRG", true);
    chain->SetBranchStatus("TDCTriggers.uBallHiRA_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.uBallNW_TRG", true);
    chain->SetBranchStatus("TDCTriggers.uBallNW_TRG_NWTDC", true);
    chain->SetBranchStatus("TDCTriggers.uBall_DS_TRG", true);
    // Microball
    chain->SetBranchStatus("uBall.fmulti", true);
    chain->SetBranchStatus("uBall.fnumring", true);
    chain->SetBranchStatus("uBall.fnumdet", true);
    chain->SetBranchStatus("uBall.fTail", true);
    chain->SetBranchStatus("uBall.fFast", true);
    chain->SetBranchStatus("uBall.fTime", true);
    // Forward Array
    chain->SetBranchStatus("ForwardArray.fmulti", true);
    chain->SetBranchStatus("ForwardArray.fTimeMin", true);
    chain->SetBranchStatus("ForwardArray.fTimeMean", true);
    chain->SetBranchStatus("ForwardArray.fnumdet", true);
    chain->SetBranchStatus("ForwardArray.fE", true);
    chain->SetBranchStatus("ForwardArray.fTime", true);
    // Veto Wall
    chain->SetBranchStatus("VetoWall.fmulti", true);
    chain->SetBranchStatus("VetoWall.fnumbar", true);
    chain->SetBranchStatus("VetoWall.fTop", true);
    chain->SetBranchStatus("VetoWall.fBottom", true);
    chain->SetBranchStatus("VetoWall.fTimeTop", true);
    chain->SetBranchStatus("VetoWall.fTimeBottom", true);
    chain->SetBranchStatus("VetoWall.fYcm", true);
    chain->SetBranchStatus("VetoWall.fThetaRan", true);
    chain->SetBranchStatus("VetoWall.fPhiRan", true);
    chain->SetBranchStatus("VetoWall.fDistRancm", true);
    // Neutron Wall A
    chain->SetBranchStatus("NWA.fmulti", true);
    chain->SetBranchStatus("NWA.fnumbar", true);
    // Neutron Wall B
    chain->SetBranchStatus("NWB.fmulti", true);
    chain->SetBranchStatus("NWB.fnumbar", true);
    chain->SetBranchStatus("NWB.fLeft", true);
    chain->SetBranchStatus("NWB.fRight", true);
    chain->SetBranchStatus("NWB.ffastLeft", true);
    chain->SetBranchStatus("NWB.ffastRight", true);
    chain->SetBranchStatus("NWB.fTimeMean", true);
    chain->SetBranchStatus("NWB.fTimeLeft", true);
    chain->SetBranchStatus("NWB.fTimeRight", true);

    return chain;
}

TTree* get_output_tree(TFile*& outroot, const std::string& tree_name) {
    outroot->cd();
    TTree* tree = new TTree(tree_name.c_str(), "");

    //   TDC triggers
    tree->Branch("TDC_hira_ds_nwtdc",  &container.TDC_hira_ds_nwtdc,  "TDC_hira_ds_nwtdc/D");
    tree->Branch("TDC_hira_live",      &container.TDC_hira_live,      "TDC_hira_live/D");
    tree->Branch("TDC_master",         &container.TDC_master,         "TDC_master/D");
    tree->Branch("TDC_master_nw",      &container.TDC_master_nw,      "TDC_master_nw/D");
    tree->Branch("TDC_master_vw",      &container.TDC_master_vw,      "TDC_master_vw/D");
    tree->Branch("TDC_nw_ds",          &container.TDC_nw_ds,          "TDC_nw_ds/D");
    tree->Branch("TDC_nw_ds_nwtdc",    &container.TDC_nw_ds_nwtdc,    "TDC_nw_ds_nwtdc/D");
    tree->Branch("TDC_rf_nwtdc",       &container.TDC_rf_nwtdc,       "TDC_rf_nwtdc/D");
    tree->Branch("TDC_mb_hira",        &container.TDC_mb_hira,        "TDC_mb_hira/D");
    tree->Branch("TDC_mb_hira_nwtdc",  &container.TDC_mb_hira_nwtdc,  "TDC_mb_hira_nwtdc/D");
    tree->Branch("TDC_mb_nw",          &container.TDC_mb_nw,          "TDC_mb_nw/D");
    tree->Branch("TDC_mb_nw_nwtdc",    &container.TDC_mb_nw_nwtdc,    "TDC_mb_nw_nwtdc/D");
    tree->Branch("TDC_mb_ds",          &container.TDC_mb_ds,          "TDC_mb_ds/D");

    // Microball
    tree->Branch("MB_multi",        &container.MB_multi,          "MB_multi/I");
    tree->Branch("MB_ring",         &container.MB_ring[0],        "MB_ring[MB_multi]/I");
    tree->Branch("MB_det",          &container.MB_det[0],         "MB_det[MB_multi]/I");
    tree->Branch("MB_tail",         &container.MB_tail[0],        "MB_tail[MB_multi]/S");
    tree->Branch("MB_fast",         &container.MB_fast[0],        "MB_fast[MB_multi]/S");
    tree->Branch("MB_time",         &container.MB_time[0],        "MB_time[MB_multi]/S");

    // Forward Array
    tree->Branch("FA_multi",        &container.FA_multi,          "FA_multi/I");
    tree->Branch("FA_time_min",     &container.FA_time_min,       "FA_time_min/D");
    tree->Branch("FA_time_mean",    &container.FA_time_mean,      "FA_time_mean/D");
    tree->Branch("FA_det",          &container.FA_det[0],         "FA_det[FA_multi]/I");
    tree->Branch("FA_total",        &container.FA_total[0],       "FA_total[FA_multi]/S");
    tree->Branch("FA_time",         &container.FA_time[0],        "FA_time[FA_multi]/D");

    // Veto Wall
    tree->Branch("VW_multi",        &container.VW_multi,          "VW_multi/I");
    tree->Branch("VW_bar",          &container.VW_bar[0],         "VW_bar[VW_multi]/I");
    tree->Branch("VW_total_T",      &container.VW_total_T[0],     "VW_total_T[VW_multi]/S");
    tree->Branch("VW_total_B",      &container.VW_total_B[0],     "VW_total_B[VW_multi]/S");
    tree->Branch("VW_time_T",       &container.VW_time_T[0],      "VW_time_T[VW_multi]/D");
    tree->Branch("VW_time_B",       &container.VW_time_B[0],      "VW_time_B[VW_multi]/D");
    tree->Branch("VW_pos_y",        &container.VW_pos_y[0],       "VW_pos_y[VW_multi]/D");
    tree->Branch("VW_theta",        &container.VW_theta[0],       "VW_theta[VW_multi]/D");
    tree->Branch("VW_phi",          &container.VW_phi[0],         "VW_phi[VW_multi]/D");
    tree->Branch("VW_distance",     &container.VW_distance[0],    "VW_distance[VW_multi]/D");

    // Neutron Wall A
    tree->Branch("NWA_multi",       &container.NWA_multi,         "NWA_multi/I");
    tree->Branch("NWA_bar",         &container.NWA_bar[0],        "NWA_bar[NWA_multi]/I");

    // Neutron Wall B
    tree->Branch("NWB_multi",       &container.NWB_multi,         "NWB_multi/I");
    tree->Branch("NWB_bar",         &container.NWB_bar[0],        "NWB_bar[NWB_multi]/I");
    tree->Branch("NWB_total_L",     &container.NWB_total_L[0],    "NWB_total_L[NWB_multi]/S");
    tree->Branch("NWB_total_R",     &container.NWB_total_R[0],    "NWB_total_R[NWB_multi]/S");
    tree->Branch("NWB_fast_L",      &container.NWB_fast_L[0],     "NWB_fast_L[NWB_multi]/S");
    tree->Branch("NWB_fast_R",      &container.NWB_fast_R[0],     "NWB_fast_R[NWB_multi]/S");
    tree->Branch("NWB_time",        &container.NWB_time[0],       "NWB_time[NWB_multi]/D");
    tree->Branch("NWB_time_L",      &container.NWB_time_L[0],     "NWB_time_L[NWB_multi]/D");
    tree->Branch("NWB_time_R",      &container.NWB_time_R[0],     "NWB_time_R[NWB_multi]/D");
    tree->Branch("NWB_tof",         &container.NWB_tof[0],        "NWB_tof[NWB_multi]/D");
    tree->Branch("NWB_pos_x",       &container.NWB_pos_x[0],      "NWB_pos_x[NWB_multi]/D");
    tree->Branch("NWB_pos_y",       &container.NWB_pos_y[0],      "NWB_pos_y[NWB_multi]/D");
    tree->Branch("NWB_pos_z",       &container.NWB_pos_z[0],      "NWB_pos_z[NWB_multi]/D");
    tree->Branch("NWB_distance",    &container.NWB_distance[0],   "NWB_distance[NWB_multi]/D");
    tree->Branch("NWB_theta",       &container.NWB_theta[0],      "NWB_theta[NWB_multi]/D");
    tree->Branch("NWB_phi",         &container.NWB_phi[0],        "NWB_phi[NWB_multi]/D");
    tree->Branch("NWB_distance_c",  &container.NWB_distance_c[0], "NWB_distance_c[NWB_multi]/D");
    tree->Branch("NWB_theta_c",     &container.NWB_theta_c[0],    "NWB_theta_c[NWB_multi]/D");
    tree->Branch("NWB_phi_c",       &container.NWB_phi_c[0],      "NWB_phi_c[NWB_multi]/D");
    tree->Branch("NWB_light_GM",    &container.NWB_light_GM[0],   "NWB_light_GM[NWB_multi]/D");
    tree->Branch("NWB_psd",         &container.NWB_psd[0],        "NWB_psd[NWB_multi]/D");
    tree->Branch("NWB_psd_perp",    &container.NWB_psd_perp[0],   "NWB_psd_perp[NWB_multi]/D");

    return tree;
}
