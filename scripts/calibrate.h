#pragma once

// standard libraries
#include <array>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// CERN ROOT libraries
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

struct Container {
    static constexpr int max_multi = 1024;

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

    // Forward Array
    int FA_multi;
    double FA_time_min;

    // Veto Wall
    int VW_multi;
    std::array<int, max_multi> VW_bar;;

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
    std::array<double, max_multi> NWB_light_GM;
    std::array<double, max_multi> NWB_theta;
    std::array<double, max_multi> NWB_phi;
    std::array<double, max_multi> NWB_distance;

    // New branches
    std::array<double, max_multi> NWB_pos;
    std::array<double, max_multi> NWB_psd;
    std::array<double, max_multi> NWB_psd_perp;
};

Container container;

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
    // Forward Array
    chain->SetBranchAddress("ForwardArray.fmulti",              &container.FA_multi);
    chain->SetBranchAddress("ForwardArray.fTimeMin",            &container.FA_time_min);
    // Veto Wall
    chain->SetBranchAddress("VetoWall.fmulti",                  &container.VW_multi);
    chain->SetBranchAddress("VetoWall.fnumbar",                 &container.VW_bar[0]);
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
    chain->SetBranchAddress("NWB.fGeoMeanSaturationCorrected",  &container.NWB_light_GM[0]);
    chain->SetBranchAddress("NWB.fThetaRan",                    &container.NWB_theta[0]);
    chain->SetBranchAddress("NWB.fPhiRan",                      &container.NWB_phi[0]);
    chain->SetBranchAddress("NWB.fDistRancm",                   &container.NWB_distance[0]);

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
    // Forward Array
    chain->SetBranchStatus("ForwardArray.fmulti", true);
    chain->SetBranchStatus("ForwardArray.fTimeMin", true);
    // Veto Wall
    chain->SetBranchStatus("VetoWall.fmulti", true);
    chain->SetBranchStatus("VetoWall.fnumbar", true);
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
    chain->SetBranchStatus("NWB.fGeoMeanSaturationCorrected", true);
    chain->SetBranchStatus("NWB.fThetaRan", true);
    chain->SetBranchStatus("NWB.fPhiRan", true);
    chain->SetBranchStatus("NWB.fDistRancm", true);

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
    tree->Branch("MB_multi",      &container.MB_multi,        "MB_multi/I");

    // Forward Array
    tree->Branch("FA_multi",      &container.FA_multi,        "FA_multi/I");
    tree->Branch("FA_time_min",   &container.FA_time_min,     "FA_time_min/D");

    // Veto Wall
    tree->Branch("VW_multi",      &container.VW_multi,        "VW_multi/I");
    tree->Branch("VW_bar",        &container.VW_bar[0],       "VW_bar[VW_multi]/I");

    // Neutron Wall B
    tree->Branch("NWB_multi",     &container.NWB_multi,       "NWB_multi/I");
    tree->Branch("NWB_bar",       &container.NWB_bar[0],      "NWB_bar[NWB_multi]/I");
    tree->Branch("NWB_total_L",   &container.NWB_total_L[0],  "NWB_total_L[NWB_multi]/S");
    tree->Branch("NWB_total_R",   &container.NWB_total_R[0],  "NWB_total_R[NWB_multi]/S");
    tree->Branch("NWB_fast_L",    &container.NWB_fast_L[0],   "NWB_fast_L[NWB_multi]/S");
    tree->Branch("NWB_fast_R",    &container.NWB_fast_R[0],   "NWB_fast_R[NWB_multi]/S");
    tree->Branch("NWB_time",      &container.NWB_time[0],     "NWB_time[NWB_multi]/D");
    tree->Branch("NWB_time_L",    &container.NWB_time_L[0],   "NWB_time_L[NWB_multi]/D");
    tree->Branch("NWB_time_R",    &container.NWB_time_R[0],   "NWB_time_R[NWB_multi]/D");
    tree->Branch("NWB_light_GM",  &container.NWB_light_GM[0], "NWB_light_GM[NWB_multi]/D");
    tree->Branch("NWB_theta",     &container.NWB_theta[0],    "NWB_theta[NWB_multi]/D");
    tree->Branch("NWB_phi",       &container.NWB_phi[0],      "NWB_phi[NWB_multi]/D");
    tree->Branch("NWB_distance",  &container.NWB_distance[0], "NWB_distance[NWB_multi]/D");
    // new branches
    tree->Branch("NWB_pos",       &container.NWB_pos[0],      "NWB_pos[NWB_multi]/D");
    tree->Branch("NWB_psd",       &container.NWB_psd[0],      "NWB_psd[NWB_multi]/D");
    tree->Branch("NWB_psd_perp",  &container.NWB_psd_perp[0], "NWB_psd_perp[NWB_multi]/D");

    return tree;
}

struct ArgumentParser {
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