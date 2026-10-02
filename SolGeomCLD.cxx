#include <TMath.h>
#include <TGraph.h>
#include <iostream>
#include <TCanvas.h>
#include <TPave.h>
#include <TLine.h>
#include <TPolyLine.h>
#include <TF1.h>
#include <TString.h>
#include "SolGeom.h"
#include "SolTrack.h"

SolGeom::SolGeom()
{
	SolGeoInit();
	for (Int_t i = 0; i < fNdet; i++)fEnable[i] = kTRUE;	// default is everything enabled
	SolGeoFill();
	TString OldLab = " "; Int_t k = 0;
	for (Int_t i = 0; i < fNlay; i++)
	{
		if (fLyLabl[i] != OldLab)
		{
			fDtype[k] = fLyLabl[i];
			fDfstLay[k] = i;
			OldLab = fLyLabl[i];
			k++;
			if (k > fNdty)
			{
				cout << "SolGeom::SolGeom : Too many detector types! Layer "<<i<<" reached" << endl;
				return;
			}
		}
	}
	for (Int_t i=0; i < k; i++)cout << "i = " << i << ", Detector = " << fDtype[i]<<endl;
}
//
SolGeom::SolGeom(Bool_t *OK)
{
	SolGeoInit();
	for (Int_t i = 0; i < fNdet; i++)fEnable[i] = OK[i];	// User defined list
	SolGeoFill();
	TString OldLab = " "; Int_t k = 0;
	for (Int_t i = 0; i < fNlay; i++)
	{
		if (fLyLabl[i] != OldLab)
		{
			fDtype[k] = fLyLabl[i];
			fDfstLay[k] = i;
			OldLab = fLyLabl[i];
			k++;
			if (k > fNdty)
			{
				cout << "SolGeom::SolGeom : Too many detector types! Layer " << i << " reached" << endl;
				return;
			}
		}
	}
	for (Int_t i=0; i < k; i++)cout << "i = " << i << ", Detector = " << fDtype[i]<<endl;
}
SolGeom::SolGeom(char *fname)
{
	SolGeoInit();
	for (Int_t i = 0; i < fNdet; i++)fEnable[i] = kTRUE;	// default is everything enabled
	GeoRead(fname);
	TString OldLab = " "; Int_t k = 0;
	for (Int_t i = 0; i < fNlay; i++)
	{
		if (fLyLabl[i] != OldLab)
		{
			fDtype[k] = fLyLabl[i];
			fDfstLay[k] = i;
			OldLab = fLyLabl[i];
			k++;
			if (k > fNdty)
			{
				cout << "SolGeom::SolGeom : Too many detector types! Layer " << i << " reached" << endl;
				return;
			}
		}
	}
	for (Int_t i = 0; i < k; i++)cout << "i = " << i << ", Detector = " << fDtype[i] << endl;
}

void SolGeom::SolGeoInit()
{
	//
	// Magnetic field
	//
	fB = 4;
	//
	// Create arrays
	//
	ftyLay = new Int_t[fNlMax];		// Layer type 1 = R (barrel) or 2 = z (forward/backward)
	fLyLabl = new TString[fNlMax];	// Layer label
	fxMin = new Double_t[fNlMax];	// Minimum dimension z for barrel  or R for forward
	fxMax = new Double_t[fNlMax];	// Maximum dimension z for barrel  or R for forward
	frPos = new Double_t[fNlMax];	// R/z location of layer
	fthLay = new Double_t[fNlMax];	// Thickness (meters)
	frlLay = new Double_t[fNlMax];	// Radiation length (meters)
	fnmLay = new Int_t[fNlMax];		// Number of measurements in layers (1D or 2D)
	fstLayU = new Double_t[fNlMax];	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
	fstLayL = new Double_t[fNlMax];	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
	fsgLayU = new Double_t[fNlMax];	// Resolution Upper side (meters) - 0 = no measurement
	fsgLayL = new Double_t[fNlMax];	// Resolution Lower side (meters) - 0 = no measurement
	fflLay = new Bool_t[fNlMax];	// measurement flag = T, scattering only = F
	fEnable = new Bool_t[fNdet];	// list of enabled detectors
	fDtype = new TString[fNdty];	// Array with layer labels 
	fDfstLay = new Int_t[fNdty];	// Array with start layer
	//
	// Load geometry info in SolGeom.h
	//
	fNlay = 0;	// Actual number of layers
	fBlay = 0;	// Nr. of barrel layers
	fFlay = 0;	// Nr. of forward/backward layers
	fNm = 0;	// Nr. of measuring layers
}
	//
void SolGeom::SolGeoFill()
{
	//===================================================================================
	//		BARREL REGION
	//===================================================================================
	//
	Double_t R12 = TMath::Sqrt(12);
	//
	// Beam pipe
	//
	if (fEnable[0])
	{
		ftyLay[fNlay] = 1;			// Layer type 1 = R (barrel) or 2 = z (forward/backward)
		fLyLabl[fNlay] = "PIPE";
		fxMin[fNlay] = -100.;		// Minimum dimension z for barrel  or R for forward
		fxMax[fNlay] = 100.;		// Maximum dimension z for barrel  or R for forward
		frPos[fNlay] = 0.015;		// R/z location of layer
		fthLay[fNlay] = 0.0012;		// Thickness (meters)
		frlLay[fNlay] = 35.276e-2;	// Radiation length (meters)
		fnmLay[fNlay] = 0;			// Number of measurements in layers (1D or 2D)
		fstLayU[fNlay] = 0;			// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
		fstLayL[fNlay] = 0;			// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
		fsgLayU[fNlay] = 0.;			// Resolution Upper side (meters) - 0 = no measurement
		fsgLayL[fNlay] = 0.;			// Resolution Lower side (meters) - 0 = no measurement
		fflLay[fNlay] = kFALSE;		// measurement flag = T, scattering only = F
		fNlay++; fBlay++;
	}
	//
	// Vertex  detector (inner)
	if (fEnable[1])
	{
		const Int_t NlVtx = 6;	// Assume 3 pixel double layers
		Double_t rVtx[NlVtx] = { 1.75, 1.85, 3.7, 3.8, 5.7, 5.8 };		// Vertex layer radii in cm
		Double_t lVtx[NlVtx] = { 12.5, 12.5, 12.5, 12.5, 12.5, 12.5};	// Vertex layer half length in cm
		for (Int_t i = 0; i < NlVtx; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "VTX";				// Layer label
			fxMin[fNlay] = -lVtx[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lVtx[i] * 1.e-2;		// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rVtx[i] * 1.e-2;		// R/z location of layer
			fthLay[fNlay] = 45.0E-6;			// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0;					// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
			fsgLayU[fNlay] = 3.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 3.E-6;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlVtxM = 6;
		Double_t rVtxM[NlVtxM] = { 1.8, 3.75, 5.75, 11.2, 11.1, 11.15 };		// Vertex layer radii in cm
		Double_t lVtxM[NlVtxM] = { 12.5, 12.5, 12.5, 50.0, 12.6, 50.0 };		// Vertex layer half length in cm
		Double_t lThkM[NlVtxM] = { 557., 557., 557., 337., 9., 9. };	// Layer thickness in um of Si
		for (Int_t i = 0; i < NlVtxM; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "VTX";				// Layer label
			fxMin[fNlay] = -lVtxM[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lVtxM[i] * 1.e-2;	// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rVtxM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThkM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0;					// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
			fstLayL[fNlay] = 0;	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
		}
	}
	//
	// Inner tracker
	if (fEnable[2])
	{
		const Int_t NlTrki = 3;	// Assume 3 long pixel layers
		Double_t rTrki[NlTrki] = { 12.7, 40., 67. };		// Tracker layer radii in cm
		Double_t lTrki[NlTrki] = { 48.16, 48.16, 69.23 };	// Tracker layer half length in cm
		for (Int_t i = 0; i < NlTrki; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "ITK";				// Layer label
			fxMin[fNlay] = -lTrki[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lTrki[i] * 1.e-2;	// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rTrki[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = 956.E-6;			// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 7.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 90.E-6;			// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlTrkiM = 5;	// Assume 5 material layers
		Double_t rTrkiM[NlTrkiM] = { 13.2, 40.5, 67.5, 68.6, 68.55 };	// Tracker layer radii in cm
		Double_t lTrkiM[NlTrkiM] = { 48.16, 48.16, 69.23, 230., 230. };	// Tracker layer half length in cm
		Double_t lThkiM[NlTrkiM] = { 159., 159., 159., 1171., 281. };	// Layer thickness in um of Si
		for (Int_t i = 0; i < NlTrkiM; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "ITK";				// Layer label
			fxMin[fNlay] = -lTrkiM[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lTrkiM[i] * 1.e-2;	// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rTrkiM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThkiM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = 0;				// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
		}
	}
	//
	// Outer tracker
	//
	if (fEnable[3])
	{
		const Int_t NlTrko = 3;	// Assume 3 long pixel layers
		Double_t rTrko[NlTrko] = { 100., 156.8, 213.6 };		// Tracker layer radii in cm
		Double_t lTrko[NlTrko] = { 126.42, 126.42, 126.42 };	// Tracker layer half length in cm
		for (Int_t i = 0; i < NlTrko; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "OTK";				// Layer label
			fxMin[fNlay] = -lTrko[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lTrko[i] * 1.e-2;	// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rTrko[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = 956.E-6;			// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 7.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 90.E-6;			// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlTrkoM = 3;	// Assume 3 material layers
		Double_t rTrkoM[NlTrkoM] = { 100.5, 157.8, 212.6 };			// Tracker layer radii in cm
		Double_t lTrkoM[NlTrkoM] = { 126.42, 126.42, 126.42 };		// Tracker layer half length in cm
		Double_t lThkoM[NlTrkoM] = { 244., 117., 117. };			// Layer thickness in um of Si
		for (Int_t i = 0; i < NlTrkoM; i++)
		{
			ftyLay[fNlay] = 1;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "OTK";				// Layer label
			fxMin[fNlay] = -lTrkoM[i] * 1.e-2;	// Minimum dimension z for barrel  or R for forward
			fxMax[fNlay] = lTrkoM[i] * 1.e-2;	// Maximum dimension z for barrel  or R for forward
			frPos[fNlay] = rTrkoM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThkoM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = 0;				// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fBlay++;
		}
	}
	//
	//================================================================================================
	//		FORWARD/BACKWARD
	//================================================================================================
	//
	// Vertex disks
	if (fEnable[4])
	{
		const Int_t NlVtxd = 12;	// Assume 12 pixel layers
		Double_t zVtxd[NlVtxd] = { -30.1, -29.9, -23.1, -22.9, -16.1, -15.9, 
			                        15.9, 16.1, 22.9, 23.1, 29.9, 30.1};		// Vertex layer z in cm
		Double_t riVtxd[NlVtxd] = { 4.5, 4.5, 3.45, 3.45, 2.4, 2.4, 
			                        2.4, 2.4, 3.45, 3.45, 4.5, 4.5 };	// Vertex layer R min in cm
		for (Int_t i = 0; i < NlVtxd; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "VTXDSK";			// Layer label
			fxMin[fNlay] = riVtxd[i] * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = 10.2 * 1.e-2;		// Maximum dimension R for forward disk
 			frPos[fNlay] = zVtxd[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = 44.E-6;				// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 3.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 3.E-6;			// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlVtxdM = 10;	// Assume 10 material layers
		Double_t zVtxdM[NlVtxdM] = { -50., -30., -23., -16., -12.6, 
			                          12.6, 16., 23., 30., 50. };	// Material layer z in cm
		Double_t riVtxdM[NlVtxdM] = { 7.5, 4.5, 3.45, 2.4, 1.75, 
			                          1.75, 2.4, 3.45, 4.5, 7.5 };	// Material layer R min in cm
		Double_t lThVdM[NlVtxdM] = { 337., 557., 557., 557., 9., 
								     9., 557., 557., 557., 337. };		// Layer thickness in um of Si
		for (Int_t i = 0; i < NlVtxdM; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "VTXDSK";			// Layer label
			fxMin[fNlay] = riVtxdM[i] * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = 10.2 * 1.e-2;		// Maximum dimension R for forward disk
			if(i==4 || i==5)fxMax[fNlay] = 11.2 * 1.e-2;
			frPos[fNlay] = zVtxdM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThVdM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = 0;				// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
		}
	}
	//
	// Inner tracker disks	
	//
	if (fEnable[5])
	{
		const Int_t NlItkd = 14;	// Assume 14 long pixel layers
		Double_t zItkd[NlItkd] = { -219., -194.6, -166.1, -137.7, -109.3, -80.8, -52.4,
									52.4, 80.8, 109.3, 137.7, 166.1, 194.6, 219. };		// Vertex layer z in cm
		Double_t riItkd[NlItkd] = { 33., 29.3, 24.95, 20.75, 16.6, 12.35, 7.95,
									7.95, 12.35, 16.6, 20.75, 24.95, 29.3, 33. };	// Vertex layer R min in cm
		Double_t roItkd[NlItkd] = { 64.7, 64., 65.7, 66.05, 66.3, 65.2, 45.7,
									45.7, 65.2, 66.3, 66.05, 65.7, 64., 64.7 };		// Vertex layer R max in cm
		for (Int_t i = 0; i < NlItkd; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "ITKDSK";			// Layer label
			fxMin[fNlay] = riItkd[i] * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = roItkd[i] * 1.e-2;	// Maximum dimension R for forward disk
			frPos[fNlay] = zItkd[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = 956.E-6;			// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 7.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 90.E-6;			// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlItkdM = 14;	// Assume 10 material layers
		Double_t zItkdM[NlItkdM] = { -218., -195.6, -165.1, -138.7, -108.3, -81.8, -51.4,
									51.4, 81.8, 108.3, 138.7, 165.1, 195.6, 218. };		// Vertex layer z in cm
		Double_t riItkdM[NlItkdM] = { 33., 29.3, 24.95, 20.75, 16.6, 12.35, 7.95,
									7.95, 12.35, 16.6, 20.75, 24.95, 29.3, 33. };		// Vertex layer R min in cm	
		Double_t roItkdM[NlItkdM] = { 64.8, 64.1, 65.8, 66.15, 66.4, 65.3, 45.6,
									  45.6, 65.3, 66.4, 66.15, 65.8, 64.1, 64.8 };		// Vertex layer R max in cm
		Double_t lThItdM[NlItkdM] = { 346., 346., 321., 321., 289., 289., 309.,
									309., 289., 289., 321., 321., 346., 346. };		// Layer thickness in um of Si
		for (Int_t i = 0; i < NlItkdM; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "ITKDSK";			// Layer label
			fxMin[fNlay] = riItkdM[i] * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = roItkdM[i] * 1.e-2;		// Maximum dimension R for forward disk
			frPos[fNlay] = zItkdM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThItdM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = 0;				// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
		}
	}
	//
	// Outer tracker disks
	//
	if (fEnable[6])
	{
		const Int_t NlOtkd = 8;	// Assume 8 long pixel layers
		Double_t zOtkd[NlOtkd] = { -219., -188.3, -161.7, -131.,
									131., 161.7, 188.3, 219.};		// Vertex layer z in cm
		for (Int_t i = 0; i < NlOtkd; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "OTKDSK";			// Layer label
			fxMin[fNlay] = 71.8 * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = 208. * 1.e-2;	// Maximum dimension R for forward disk
			frPos[fNlay] = zOtkd[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = 956.E-6;			// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 2;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = TMath::Pi() / 2.;	// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 7.E-6;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 90.E-6;			// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kTRUE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
			fNm++;
		}
		// Describe associated material
		const Int_t NlOtkdM = 14;	// Assume 10 material layers
		Double_t zOtkdM[NlOtkdM] = { -128.42, -71.23, -50.16, -218., -189.3, -160.7, -132.,
									132., 160.7, 189.3, 218., 50.16, 71.23, 128.42 };		// Vertex layer z in cm
		Double_t riOtkdM[NlOtkdM] = {99., 65.17, 12.7, 71.8, 71.8, 71.8, 71.8,
									71.8, 71.8, 71.8, 71.8, 12.7, 65.17, 99. };		// Vertex layer R min in cm	
		Double_t roOtkdM[NlOtkdM] = { 208., 68.6, 65., 208., 208., 208., 208.,
									  208., 208., 208., 208., 65., 68.6, 208. };		// Vertex layer R max in cm
		Double_t lThOtdM[NlOtkdM] = { 562., 562., 562., 342., 342., 342., 342.,
									 342., 342., 342., 342., 562., 562., 562.};		// Layer thickness in um of Si
		for (Int_t i = 0; i < NlOtkdM; i++)
		{
			ftyLay[fNlay] = 2;					// Layer type 1 = R (barrel) or 2 = z (forward/backward)
			fLyLabl[fNlay] = "OTKDSK";			// Layer label
			fxMin[fNlay] = riOtkdM[i] * 1.e-2;	// Minimum dimension R for forward disk
			fxMax[fNlay] = roOtkdM[i] * 1.e-2;		// Maximum dimension R for forward disk
			frPos[fNlay] = zOtkdM[i] * 1.e-2;	// R/z location of layer
			fthLay[fNlay] = lThOtdM[i] * 1.E-6;	// Thickness (meters)
			frlLay[fNlay] = 9.370e-2;			// Radiation length (meters)
			fnmLay[fNlay] = 0;					// Number of measurements in layers (1D or 2D)
			fstLayU[fNlay] = 0.0;				// Stereo angle (rad) - 0 = axial layer - Upper side
			fstLayL[fNlay] = 0;				// Stereo angle (rad) - pi/2 = z layer - Lower side
			fsgLayU[fNlay] = 0;				// Resolution Upper side (meters) - 0 = no measurement
			fsgLayL[fNlay] = 0;				// Resolution Lower side (meters) - 0 = no measurement
			fflLay[fNlay] = kFALSE;				// measurement flag = T, scattering only = F
			fNlay++; fFlay++;
		}
	}
	//
	//
	// Magnet
	//
	ftyLay[fNlay] = 1;			// Layer type 1 = R (barrel) or 2 = z (forward/backward)
	fLyLabl[fNlay] = "MAG";		// Layer label
	fxMin[fNlay] = -2.5;		// Minimum dimension z for barrel  or R for forward
	fxMax[fNlay] = 2.5;			// Maximum dimension z for barrel  or R for forward
	frPos[fNlay] = 2.25;		// R/z location of layer
	fthLay[fNlay] = 0.05;		// Thickness (meters)
	frlLay[fNlay] = 6.58e-2;	// Radiation length (meters)
	fnmLay[fNlay] = 0;			// Number of measurements in layers (1D or 2D)
	fstLayU[fNlay] = 0;			// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
	fstLayL[fNlay] = 0;			// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
	fsgLayU[fNlay] = 0.;		// Resolution Upper side (meters) - 0 = no measurement
	fsgLayL[fNlay] = 0.;		// Resolution Lower side (meters) - 0 = no measurement
	fflLay[fNlay] = kFALSE;		// measurement flag = T, scattering only = F
	fNlay++; fBlay++;

	cout << "Geometry created with " << fNlay << "/" << fNm << " layers" << endl;
}
//
// Print geometry
void SolGeom::GeoPrint(char *fname)
{
	FILE *fdata = fopen(fname, "w");
	if (!fdata)
	{
		cout << "SolGeom::GeoPrint - can't open output file" << endl;
		return;
	}
	for (Int_t l = 0; l < fNlay; l++)
	{
		fprintf(fdata, "%d %s %g %g %g %g %g %d %g %g %g %g %d\n",
		ftyLay[l], fLyLabl[l].Data(), fxMin[l], fxMax[l], frPos[l], fthLay[l],
		frlLay[l], fnmLay[l], fstLayU[l], fstLayL[l], fsgLayU[l], fsgLayL[l], fflLay[l]);
		//cout << strng << endl<< endl;
	}	
	fclose(fdata);
}
//
// Material counter (Units are fraction of X0)
Double_t *SolGeom::FracX0(Double_t theta)
{
	//
	// Calculates amount of material crossed by a straight track at a polar angle theta
	// for each subdetector:
	// 0: Pipe, 1: VTXLOW, 2: VTXHIGH, 3: DCHCANI, 4: DCH, 5: DCHCANO, 6: BSILWRP, 7: MAG,
	// 8: BPRESH, 9: VTXDSK, 10: DCHWALL, 11: FSILWRP, 12: FRAD, 13: FPRESH
	//
	Double_t *Mat;
	Mat = new Double_t[fNdty];
	for (Int_t i = 0; i < fNdty; i++)Mat[i] = 0;
	if (fNlay <= 0)
	{
		cout << "SolGeom::FracX0 : No geometry available. # layers = " << fNlay << endl;
		return Mat;
	}
	//
	// Loop over all layers
	Double_t lmb = 0.0;
	if (TMath::Abs(theta - TMath::PiOver2()) > 1.0e-10 && 
		TMath::Abs(theta)  > 1.0e-10) lmb = 1.0 / TMath::Tan(theta);	// Cot(theta)
	if (theta == 0.0) lmb = 1.e10;
	for (Int_t il = 0; il < fNlay; il++)
	{
		Int_t dNum;
		for (Int_t i = 0; i<fNdty; i++) if (fLyLabl[il] == fDtype[i])dNum = i;
		//cout << "dnum = " << dNum << ", detector: "<<fDtype[dNum]<<endl;
		if (ftyLay[il] == 1)		// Cylinder at constant R
		{
			Double_t R = frPos[il];
			Double_t z = lmb*R;
			//cout << "l num: " << il << ", R = " << R << ", z min: "<<fxMin[il]<<", z = " << z<<" , z max: "<<fxMax[il] << endl;
			if (z>fxMin[il] && z < fxMax[il])	// the layer is hit
			{
				Mat[dNum] += fthLay[il] / (TMath::Sin(theta)*frlLay[il]);
			}
		}
		else if (ftyLay[il] == 2) // disk at constant z
		{
			Double_t z = frPos[il];
			Double_t R = z / lmb;
			//cout << "l num: " << il << ", z = " << z << ", R min: " << fxMin[il] << ", R = " << R << " , R max: " << fxMax[il] << endl;
			if (R>fxMin[il] && R < fxMax[il])	// the layer is hit
			{
				Mat[dNum] += fthLay[il] / (TMath::Cos(theta)*frlLay[il]);
			}
		}
	}
	//
	return Mat;
}
//
// Read geometry
void SolGeom::GeoRead(char *fname)
{
	char strng[200];
	int nbytes = 200;
	FILE *fdata = fopen(fname, "r");
	if (!fdata)
	{
		cout << "SolGeom::GeoRead - can't open input file" << endl;
		return;
	}
	Int_t tyLay;
	char LyLabl[20];
	float xMin;
	float xMax;
	float rPos;
	float thLay;
	float rlLay;
	Int_t nmLay;
	float stLayU;
	float stLayL;
	float sgLayU;
	float sgLayL;
	Int_t flLay;
	//
	while (fgets(strng, nbytes, fdata) != NULL)
	{
		cout << strng;
		int status = sscanf(strng, "%d %s %g %g %g %g %g %d %g %g %g %g %d",
			&tyLay, LyLabl, &xMin, &xMax, &rPos, &thLay,
			&rlLay, &nmLay, &stLayU, &stLayL, &sgLayU, &sgLayL, &flLay);
		ftyLay[fNlay] = tyLay; 
		fLyLabl[fNlay] = LyLabl; 
		fxMin[fNlay] = (Double_t) xMin; 
		fxMax[fNlay] = (Double_t) xMax; 
		frPos[fNlay] = (Double_t) rPos; 
		fthLay[fNlay] = (Double_t) thLay; 
		frlLay[fNlay] = (Double_t) rlLay; 
		fnmLay[fNlay] = nmLay; 
		fstLayU[fNlay] = (Double_t) stLayU; 
		fstLayL[fNlay] = (Double_t) stLayL; 
		fsgLayU[fNlay] = (Double_t) sgLayU; 
		fsgLayL[fNlay] = (Double_t) sgLayL; 
		fflLay[fNlay] = (Bool_t) flLay; 
		//cout << "Layer # " << fNlay << ": " << fLyLabl[fNlay] << ", Position: " << frPos[fNlay]
		//	<< ", Measurement: " << fflLay[fNlay] << endl;
		
		fNlay++;
		if (tyLay == 1)fBlay++;
		if (flLay == 1)fNm++;

	}
	fclose(fdata);
	cout << "SolGeom::GeoRead completed with " << fNlay << " layers input" << endl;
}
//
// Destructor
SolGeom::~SolGeom()
{
	fNlay = 0;
	fBlay = 0;
	fNm = 0;

	delete[] & ftyLay;
	delete[] & fxMin;
	delete[] & fxMax;
	delete[] & frPos;
	delete[] & fthLay;
	delete[] & frlLay;
	delete[] & fnmLay;
	delete[] & fstLayU;
	delete[] & fstLayL;
	delete[] & fsgLayU;
	delete[] & fsgLayL;
	delete[] & fflLay;
	delete[] & fEnable;
}
//
// Draw the geometry (just a sketch)
//
void SolGeom::Draw()
{
	Double_t zMin = -2.75; Double_t zMax = 2.75;
	Double_t rMax = 2.6;
	fcnv = new TCanvas("cnv", "Geometry sketch", 10, 10, 950, 550);
	fcnv->Range(zMin, -0.1, zMax, rMax);
	// 
	// beam pipe
	if (fEnable[0])
	{
		TPave *pipe = new TPave(zMin, -frPos[0], zMax, frPos[0], 0, "");
		pipe->SetFillColor(kYellow);
		pipe->Draw();
	}
	// Beamline
	TLine *beam = new TLine(zMin, 0.0, zMax, 0.0);
	beam->SetLineColor(kBlack);
	beam->SetLineWidth(1);
	beam->SetLineStyle(9);
	beam->Draw("SAME");
	// Magnet
	TPave *sol = new TPave(-2.5, 2.2, 2.5, 2.3, 0, "");
	sol->SetFillColor(30);
	sol->Draw("SAME");
	//
	// Draw Calorimeter
	// Barrel
	const Int_t nP = 5;
	Double_t brCalX[nP] = { -2.6, 2.6, 4.6, -4.6, -2.6 };
	Double_t brCalY[nP] = { 2.5, 2.5, 4.5, 4.5, 2.5 };
	TPolyLine *brCalor = new TPolyLine(nP, brCalX, brCalY,"F");
	brCalor->SetFillColor(38);
	brCalor->SetLineColor(kBlack);
	brCalor->Draw("FSAME");
	// Backward
	Double_t bkCalX[nP] = { -4.6, -2.6, -2.6, -4.6, -4.6 };
	Double_t bkCalY[nP] = { 0.68, 0.39, 2.5, 4.5, 0.68 };
	TPolyLine *bkCalor = new TPolyLine(nP, bkCalX, bkCalY, "F");
	bkCalor->SetFillColor(38);
	bkCalor->SetLineColor(kBlack);
	bkCalor->Draw("FSAME");
	// Forward
	Double_t bfCalX[nP] = { 2.6, 4.6, 4.6, 2.6, 2.6 };
	Double_t bfCalY[nP] = { 0.39, 0.68, 4.5, 2.5, 0.39 };
	TPolyLine *bfCalor = new TPolyLine(nP, bfCalX, bfCalY, "F");
	bfCalor->SetFillColor(38);
	bfCalor->SetLineColor(kBlack);
	bfCalor->Draw("FSAME");
	// All other layers
	// Measurement silicon (red), blue (DCH), scattering black
	//
	const Int_t lMax = 200;
	TLine *ln[lMax];
	TF1   *fn[lMax];
	Int_t il = 0; 
	Int_t ig = 0;
	for (Int_t i = 0; i < fNlay; i++)
	{
		if (fLyLabl[i] == "DCH")		// Drift chamber layers (hypeboloids)
		{
				char lab[10]; 
				Int_t stat;
				stat = sprintf(lab, "fun%d", ig);
				fn[ig] = new TF1(lab, this, &SolGeom::StereoHyp, lxMin(i), lxMax(i), 3);
				fn[ig]->SetParameter(0, lPos(i));
				fn[ig]->SetParameter(1, lStU(i));
				fn[ig]->SetParameter(2, (Double_t) i);
				fn[ig]->SetLineColor(kBlue);
				fn[ig]->Draw("SAME");
				ig++;
		}
		else
		{
			if(ftyLay[i] == 1)ln[il] = new TLine(lxMin(i), lPos(i), lxMax(i), lPos(i));
			else ln[il] = new TLine(lPos(i), lxMin(i), lPos(i), lxMax(i));
			ln[il]->SetLineColor(kBlack);
			if (isMeasure(i))ln[il]->SetLineColor(kRed);
			ln[il]->Draw("SAME");
			il++;
		}
	}
}
//
Double_t SolGeom::StereoHyp(Double_t *x, Double_t *p)
{
	Double_t R   = p[0];
	Double_t tg  = TMath::Tan(p[1]);
	Int_t i = (Int_t)p[2];
	Double_t r = TMath::Sqrt(R*R - lxMax(i)*lxMax(i)*tg*tg);
	Double_t z   = x[0];
	//
	return TMath::Sqrt(r*r + z*z*tg*tg);
}
