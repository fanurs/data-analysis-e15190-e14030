// standard libraries
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

// third-party libraries
#include <nlohmann/json.hpp>

// CERN ROOT libraries
#include "TError.h"
#include "TMath.h"
#include "TNamed.h"
#include "TRandom.h"
#include "TROOT.h"

// local libraries
#include "ParamReader.h"
#include "calibrate.h"

using Json = nlohmann::json;

// forward declarations of calibration functions
double get_position(NWBPositionCalibParamReader& nw_pcalib, int bar, double time_L, double time_R);
double get_time_of_flight(NWTimeOfFlightCalibParamReader& nw_tcalib, int bar, double time_L, double time_R, double fa_time);
double get_light_output(
    NWLightOutputCalibParamReader& nw_pcalib,
    int bar, double total_L, double total_R, double pos_x
);
std::array<double, 2> get_psd(
    NWPulseShapeDiscriminationParamReader& psd_reader,
    int bar, double total_L, double total_R, double fast_L, double fast_R, double pos_x
);
std::array<double, 3> randomize_position(double pos_x);
std::array<double, 3> get_spherical_coordinates(NWBPositionCalibParamReader& nw_pcalib, int bar, std::array<double, 3>& position);

int main(int argc, char* argv[]) {
    // initialization and argument parsing
    gErrorIgnoreLevel = kError; // ignore warnings
    std::filesystem::path project_dir = get_project_dir();
    ArgumentParser argparser(argc, argv);

    // read in parameter readers
    NWBPositionCalibParamReader nwb_pcalib;
    NWTimeOfFlightCalibParamReader nwb_tcalib('B');
    NWLightOutputCalibParamReader nwb_lcalib('B');
    NWPulseShapeDiscriminationParamReader nwb_psd_reader('B');
    nwb_pcalib.load(argparser.run_num);
    nwb_tcalib.load(argparser.run_num);
    nwb_lcalib.load(argparser.run_num);
    nwb_psd_reader.load(argparser.run_num);

    // read in Daniele's ROOT files (Kuan's version)
    // std::filesystem::path inroot_path = get_input_root_path(project_dir, argparser, "daniele_root_files_dir");
    std::filesystem::path inroot_path = get_input_root_path(project_dir, argparser);
    TChain* intree = get_input_tree(inroot_path.string(), "E15190");

    // prepare output (calibrated) ROOT files
    TFile* outroot = new TFile(argparser.outroot_path.c_str(), "RECREATE");
    TTree* outtree = get_output_tree(outroot, "tree");

    // save metadata into TFolder
    TFolder* metadata = gROOT->GetRootFolder()->AddFolder("metadata", "");
    metadata->Add(new TNamed(inroot_path.string().c_str(), "inroot_path"));
    TFolder* position_param_paths = metadata->AddFolder("position_param_paths", "");
    TFolder* time_of_fligh_param_paths = metadata->AddFolder("time_of_flight_param_paths", "");
    TFolder* light_param_paths = metadata->AddFolder("light_param_path", "");
    TFolder* psd_param_paths = metadata->AddFolder("psd_param_paths", "");
    nwb_pcalib.write_metadata(position_param_paths);
    nwb_tcalib.write_metadata(time_of_fligh_param_paths);
    nwb_lcalib.write_metadata(light_param_paths);
    nwb_psd_reader.write_metadata(psd_param_paths);

    // main loop
    gRandom->SetSeed((unsigned)time( NULL ));
    ProgressBar progress_bar(argparser, intree->GetEntries());
    Container& evt = container; // a shorter alias; see "calibrate.h"
    for (int ievt = argparser.first_entry; ievt <= progress_bar.last_entry; ++ievt) {
        progress_bar.show(ievt);
        intree->GetEntry(ievt);

        // calibrations
        for (int m = 0; m < evt.NWB_multi; ++m) {
            // position calibration
            evt.NWB_pos_x[m] = get_position(
                nwb_pcalib,
                evt.NWB_bar[m], evt.NWB_time_L[m], evt.NWB_time_R[m]
            );
            std::array<double, 3> bar_position = randomize_position(evt.NWB_pos_x[m]);
            evt.NWB_pos_y[m] = bar_position[1];
            evt.NWB_pos_z[m] = bar_position[2];
            std::array<double, 3> sph_coord = get_spherical_coordinates(nwb_pcalib, evt.NWB_bar[m], bar_position);
            evt.NWB_distance[m] = sph_coord[0];
            evt.NWB_theta[m] = sph_coord[1];
            evt.NWB_phi[m] = sph_coord[2];
            std::array<double, 3> bar_position_c = {evt.NWB_pos_x[m], 0.0, 0.0};
            std::array<double, 3> sph_coord_c = get_spherical_coordinates(nwb_pcalib, evt.NWB_bar[m], bar_position_c);
            evt.NWB_distance_c[m] = sph_coord_c[0];
            evt.NWB_theta_c[m] = sph_coord_c[1];
            evt.NWB_phi_c[m] = sph_coord_c[2];

            // time-of-flight calibration
            evt.NWB_tof[m] = get_time_of_flight(
                nwb_tcalib,
                evt.NWB_bar[m], evt.NWB_time_L[m], evt.NWB_time_R[m], evt.FA_time_mean
            );

            // light output calibration
            evt.NWB_light_GM[m] = get_light_output(
                nwb_lcalib,
                evt.NWB_bar[m],
                double(evt.NWB_total_L[m]), double(evt.NWB_total_R[m]),
                evt.NWB_pos_x[m]
            );
            
            // pulse shape discrimination
            std::array<double, 2> psd = get_psd(
                nwb_psd_reader,
                evt.NWB_bar[m],
                double(evt.NWB_total_L[m]), double(evt.NWB_total_R[m]),
                double(evt.NWB_fast_L[m]), double(evt.NWB_fast_R[m]),
                evt.NWB_pos_x[m]
            );
            evt.NWB_psd[m] = psd[0];
            evt.NWB_psd_perp[m] = psd[1];
        }

        outtree->Fill();
    }
    progress_bar.terminate();

    // save output to file
    outroot->cd();
    metadata->Write();
    outtree->Write();
    outroot->Close();

    return 0;
}

double get_position(NWBPositionCalibParamReader& nw_pcalib, int bar, double time_L, double time_R) {
    int p0 = nw_pcalib.get(bar, "p0");
    int p1 = nw_pcalib.get(bar, "p1");
    double pos = p0 + p1 * (time_L - time_R);
    return pos;
}

double get_time_of_flight(NWTimeOfFlightCalibParamReader& nw_tcalib, int bar, double time_L, double time_R, double fa_time) {
    return 0.5 * (time_L + time_R) - fa_time - nw_tcalib.tof_offset[bar];
}

double get_light_output(NWLightOutputCalibParamReader& nw_lcalib, int bar, double total_L, double total_R, double pos_x) {
    double light_L = total_L;
    double light_R = total_R;
    std::unordered_map<std::string, double> par = nw_lcalib.run_param.at(bar);

    // saturation recovery
    double scalar = exp((2 / par.at("att_length")) * pos_x + log(par.at("gain_ratio")));
    double threshold = 4090;
    if (light_L > threshold && light_R < threshold) { light_L = light_R / scalar; }
    if (light_R > threshold && light_L < threshold) { light_R = light_L * scalar; }

    // gain matching
    // Not needed for calibration using Geometric Mean
    // Gain matching factor will be cancelled out when taking the geometric mean

    // light output calibration
    double light_GM = sqrt(light_L * light_R);
    light_GM *= 4.196; // w.r.t. AmBe 4.196 MeVee
    light_GM /= par.at("a") + par.at("b") * pos_x + par.at("c") * pos_x * pos_x;
    light_GM = par.at("d") + light_GM * par.at("e");
    
    return light_GM;
}

std::array<double, 2> get_psd(
    NWPulseShapeDiscriminationParamReader& psd_reader,
    int bar,
    double total_L, double total_R,
    double fast_L, double fast_R,
    double pos_x
) {
    /*****eliminate bad data*****/
    if (total_L < 0 || total_R < 0 || fast_L < 0 || fast_R < 0
        || total_L > 4097 || total_R > 4097 || fast_L > 4097 || fast_R > 4097
        || pos_x < -120 || pos_x > 120
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
    gamma_L = psd_reader.gamma_vpsd_L[bar]->Eval(pos_x);
    neutron_L = psd_reader.neutron_vpsd_L[bar]->Eval(pos_x);
    gamma_R = psd_reader.gamma_vpsd_R[bar]->Eval(pos_x);
    neutron_R = psd_reader.neutron_vpsd_R[bar]->Eval(pos_x);

    std::array<double, 2> xy = {vpsd_L - gamma_L, vpsd_R - gamma_R};
    std::array<double, 2> gn_vec = {neutron_L - gamma_L, neutron_R - gamma_R};
    std::array<double, 2> gn_rot90 = {-gn_vec[1], gn_vec[0]};

    // project to gn_vec and gn_rot90
    double x = (xy[0] * gn_vec[0] + xy[1] * gn_vec[1]);
    x /= sqrt(gn_vec[0] * gn_vec[0] + gn_vec[1] * gn_vec[1]);
    double y = (xy[0] * gn_rot90[0] + xy[1] * gn_rot90[1]);
    y /= sqrt(gn_rot90[0] * gn_rot90[0] + gn_rot90[1] * gn_rot90[1]);

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

std::array<double, 3> randomize_position(double pos_x) {
    const double y_length = 3 * 2.54; // cm
    const double z_length = 2.5 * 2.54; // cm
    double pos_y = gRandom->Uniform(-0.5 * y_length, 0.5 * y_length);
    double pos_z = gRandom->Uniform(-0.5 * z_length, 0.5 * z_length);
    return {pos_x, pos_y, pos_z};
}

std::array<double, 3> get_spherical_coordinates(NWBPositionCalibParamReader& nw_pcalib, int bar, std::array<double, 3>& position) {
    std::array<double, 3> L = {
        nw_pcalib.getL(bar, "L0"), nw_pcalib.getL(bar, "L1"), nw_pcalib.getL(bar, "L2")
    };
    std::array<double, 3> X = {
        nw_pcalib.getX(bar, "X0"), nw_pcalib.getX(bar, "X1"), nw_pcalib.getX(bar, "X2")
    };
    std::array<double, 3> Y = {
        nw_pcalib.getY(bar, "Y0"), nw_pcalib.getY(bar, "Y1"), nw_pcalib.getY(bar, "Y2")
    };
    std::array<double, 3> Z = {
        nw_pcalib.getZ(bar, "Z0"), nw_pcalib.getZ(bar, "Z1"), nw_pcalib.getZ(bar, "Z2")
    };
    
    double lab_x = position[0] * X[0] + position[1] * Y[0] + position[2] * Z[0] + L[0];
    double lab_y = position[0] * X[1] + position[1] * Y[1] + position[2] * Z[1] + L[1];
    double lab_z = position[0] * X[2] + position[1] * Y[2] + position[2] * Z[2] + L[2];
    
    double rho = sqrt(lab_x * lab_x + lab_y * lab_y + lab_z * lab_z);
    double theta = acos(lab_z / rho) * TMath::RadToDeg();
    double phi = atan2(lab_y, lab_x) * TMath::RadToDeg();
    
    return {rho, theta, phi};
}
