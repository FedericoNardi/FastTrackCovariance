// RunMetrics.C
#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TString.h>
#include "SolGeom.h"
#include "SolTrack.h"

// Evaluate track resolutions on a (theta, pT) grid:
//   theta: Nang values uniform in cos(theta) between thmin_deg and 90 deg
//          (uniform in solid angle; z>0 only, the geometry is symmetric)
//   pT:    Npt values log-spaced between ptmin and ptmax (GeV)
// For each track the indices of the measurement layers it crosses are stored
// (nmeas, mlay[nmeas]; indices follow the order of the geometry file), so that
// hit requirements and track-finding efficiency can be computed downstream.
void RunMetrics(const char *geoFile = "GeoOPT.txt",
                const char *outFile = "metrics.root",
                int Nang = 40, double thmin_deg = 10.,
                int Npt = 20, double ptmin = 0.5, double ptmax = 100.)
{
    Bool_t Res = kTRUE; // include measurement resolutions
    Bool_t MS  = kTRUE; // include multiple scattering

    // Use the CLD version of SolGeom with text geometry
    SolGeom *G = new SolGeom((char*)geoFile);

    double cmax = TMath::Cos(thmin_deg * TMath::Pi() / 180.0);

    // Output file & tree
    TFile *fout = new TFile(outFile, "RECREATE");
    TTree *t = new TTree("metrics", "tracker resolutions");

    double pt, theta_deg, spt_rel, sd0_um, sz0_um;
    const int kMaxHit = 500;
    int nmeas;
    int mlay[kMaxHit];
    t->Branch("pt",        &pt,        "pt/D");
    t->Branch("theta_deg", &theta_deg, "theta_deg/D");
    t->Branch("spt_rel",   &spt_rel,   "spt_rel/D");   // σ(pT)/pT
    t->Branch("sd0_um",    &sd0_um,    "sd0_um/D");    // σ(d0) in μm
    t->Branch("sz0_um",    &sz0_um,    "sz0_um/D");    // σ(z0) in μm
    t->Branch("nmeas",     &nmeas,     "nmeas/I");     // measurement layers crossed
    t->Branch("mlay",      mlay,       "mlay[nmeas]/I"); // their indices

    for (int ia = 0; ia < Nang; ++ia) {
        double c = (Nang > 1) ? cmax * (1.0 - double(ia) / (Nang - 1)) : 0.0;  // cmax -> 0
        double th = TMath::ACos(c);
        theta_deg = th * 180.0 / TMath::Pi();

        for (int k = 0; k < Npt; ++k) {
            pt = (Npt > 1) ? ptmin * TMath::Power(ptmax / ptmin, double(k) / (Npt - 1)) : ptmin;

            double x[3] = {0.0, 0.0, 0.0};
            double p[3];
            p[0] = pt;
            p[1] = 0.0;
            p[2] = pt / TMath::Tan(th); // so that transverse momentum = pt

            SolTrack *tr = new SolTrack(x, p, G);
            tr->CovCalc(Res, MS);

            // Measurement layers crossed by the track (same hit search as CovCalc)
            int nh = tr->nHit();
            Int_t    *ih = new Int_t[nh > 0 ? nh : 1];
            Double_t *rh = new Double_t[nh > 0 ? nh : 1];
            Double_t *zh = new Double_t[nh > 0 ? nh : 1];
            tr->HitList(ih, rh, zh);
            nmeas = 0;
            for (int ih_ = 0; ih_ < nh && nmeas < kMaxHit; ++ih_)
                if (G->isMeasure(ih[ih_])) mlay[nmeas++] = ih[ih_];
            delete[] ih; delete[] rh; delete[] zh;

            spt_rel = tr->s_pt();             // σ(pT)/pT
            sd0_um  = tr->s_D()  * 1e6;       // m -> μm
            sz0_um  = tr->s_z0() * 1e6;       // m -> μm
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
