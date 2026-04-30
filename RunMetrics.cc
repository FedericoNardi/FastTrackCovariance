// RunMetrics.C
#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TString.h>
#include "SolGeom.h"
#include "SolTrack.h"

void RunMetrics(const char *geoFile = "GeoOPT.txt",
                const char *outFile = "metrics.root")
{
    Bool_t Res = kTRUE; // include measurement resolutions
    Bool_t MS  = kTRUE; // include multiple scattering

    // Use the CLD version of SolGeom with text geometry
    SolGeom *G = new SolGeom((char*)geoFile);

    // Simple grid: pT in [1, 100] GeV, a few polar angles
    const int Npt = 15;
    double ptmin = 1.0;
    double ptmax = 100.0;
    double dpt = (ptmax - ptmin) / (Npt - 1);

    const int Nang = 8;
    double ang_deg[Nang] = {10., 20., 30., 40., 45., 60., 75., 90.};

    // Output file & tree
    TFile *fout = new TFile(outFile, "RECREATE");
    TTree *t = new TTree("metrics", "tracker resolutions");

    double pt, theta_deg, spt_rel, sd0_um, sz0_um;
    t->Branch("pt",        &pt,        "pt/D");
    t->Branch("theta_deg", &theta_deg, "theta_deg/D");
    t->Branch("spt_rel",   &spt_rel,   "spt_rel/D");   // σ(pT)/pT
    t->Branch("sd0_um",    &sd0_um,    "sd0_um/D");    // σ(d0) in μm
    t->Branch("sz0_um",    &sz0_um,    "sz0_um/D");    // σ(z0) in μm

    for (int ia = 0; ia < Nang; ++ia) {
        theta_deg = ang_deg[ia];
        double th = theta_deg * TMath::Pi() / 180.0;

        for (int k = 0; k < Npt; ++k) {
            pt = ptmin + k * dpt;

            double x[3] = {0.0, 0.0, 0.0};
            double p[3];
            p[0] = pt;
            p[1] = 0.0;
            p[2] = pt / TMath::Tan(th); // so that transverse momentum = pt

            SolTrack *tr = new SolTrack(x, p, G);
            tr->CovCalc(Res, MS);

            spt_rel = tr->s_pt();             // σ(pT)/pT
            sd0_um  = tr->s_D()  * 1e3;       // m -> μm
            sz0_um  = tr->s_z0() * 1e3;       // m -> μm
            t->Fill();
            //delete tr;
        }
    }

    fout->cd();
    t->Write();
    fout->Close();
    delete fout;
    // delete G;
}
