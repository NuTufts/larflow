#!/usr/bin/env python3
"""
Test script for the enhanced NuVertexFlashPrediction class with real data.

This script demonstrates:
1. Loading NuVertexCandidate and ADC images from ROOT files
2. Using NuVertexFlashPrediction to predict optical flash
3. Analyzing individual particle contributions to the light prediction

Usage:
    python3 test_NuVertexFlashPrediction_example.py [--vertex-index INDEX] [--entry ENTRY] [--threshold THRESHOLD] [--save-output]
"""

import ROOT as rt
from ROOT import larflow, larlite, larcv, std, ublarcvapp
import numpy as np
import argparse
import sys
import os

# Set ROOT to batch mode to avoid GUI windows
rt.gROOT.SetBatch(True)
rt.gStyle.SetOptStat(0)

def load_data_from_files(larcv_file, kps_file, larlite_file, entry=0):
    """
    Load ADC images and NuVertexCandidate from separate ROOT files
    
    Args:
        larcv_file: Path to file containing larcv Image2D data
        kps_file: Path to file containing KPSRecoManager data with vertex candidates
        entry: Entry number to read (default: 0)
        
    Returns:
        tuple: (adc_images, vertex_candidates_list, opflashes)
    """
    print(f"Loading data from entry {entry}:")
    print(f"  LArCV file: {larcv_file}")
    print(f"  KPS file: {kps_file}")
    print(f"  larlite file: {larlite_file}")
    
    # Load ADC images from larcv file
    print("\nLoading ADC images...")
    ioman = larcv.IOManager(larcv.IOManager.kREAD)
    ioman.add_in_file(larcv_file)
    ioman.initialize()
    
    if entry >= ioman.get_n_entries():
        raise ValueError(f"Entry {entry} >= {ioman.get_n_entries()} available entries")
        
    ioman.read_entry(entry)
    
    ev_img = ioman.get_data(larcv.kProductImage2D, "wire")
    adc_images = std.vector('larcv::Image2D')()
    
    for p in range(3):
        img = ev_img.as_vector().at(p)
        adc_images.push_back(img)
        print(f"    Plane {p}: {img.meta().cols()} x {img.meta().rows()} pixels")
    
    # Load vertex candidates from KPS file
    print("\nLoading vertex candidates...")
    kps_file_handle = rt.TFile(kps_file, 'READ')
    kps_tree = kps_file_handle.Get("KPSRecoManagerTree")
    
    if not kps_tree:
        raise ValueError(f"Could not find KPSRecoManagerTree in {kps_file}")
        
    if entry >= kps_tree.GetEntries():
        raise ValueError(f"Entry {entry} >= {kps_tree.GetEntries()} available entries in KPS tree")
        
    kps_tree.GetEntry(entry)
    
    # Get vertex candidates
    vertex_candidates = std.vector('larflow::reco::NuVertexCandidate')()
    opflashes = std.vector('larlite::opflash')()
    
    if hasattr(kps_tree, 'nuvetoed_v') and kps_tree.nuvetoed_v.size() > 0:
        print(f"    Found {kps_tree.nuvetoed_v.size()} nu vertex candidates")
        for i in range(kps_tree.nuvetoed_v.size()):
            vtx_cand = kps_tree.nuvetoed_v.at(i)
            vertex_candidates.push_back(vtx_cand) # this is a copy, but needed because files will go out of scope
            print(f"      Candidate {i}: {vtx_cand.track_v.size()} tracks, {vtx_cand.shower_v.size()} showers")
    else:
        print("    Warning: No nuvetoed_v found in tree")
    
    # Try to get opflash data if available
    #try:
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(larlite_file)
    ioll.open()
    ioll.go_to(entry)
    
    ev_opflash = ioll.get_data(larlite.data.kOpFlash, "simpleFlashBeam")
    print(f"    Found {ev_opflash.size()} optical flashes")
    for i in range(ev_opflash.size()):
        opflashes.push_back(ev_opflash.at(i))
    #except:
    #    print("    Note: Could not load opflash data")
    
    ioman.finalize()
    kps_file_handle.Close()
    ioll.close()
    
    return adc_images, vertex_candidates, opflashes

def create_flash_visualization(predictor, contributions, opflashes, output_filename="nuvertex_flash_prediction.png"):
    """
    Create visualization comparing predicted and observed PMT responses
    
    Args:
        predictor: NuVertexFlashPrediction object with results
        contributions: List of ParticleContribution objects  
        opflashes: Vector of observed optical flashes
        output_filename: Name for output image file
    """
    print("\nCreating flash prediction visualization...")
    
    # Create stacked histogram for individual particle contributions
    h_pmt_pred_stack = rt.THStack("h_pmt_pred_stack", "Flash Prediction: Individual Particle Contributions;PMT ID;Photoelectrons")
    hists_v = []
    legend = rt.TLegend(0.65, 0.65, 0.89, 0.89)
    
    # Define colors for different particle types
    track_colors = [rt.kRed+1, rt.kRed-7, rt.kRed+3, rt.kRed-3, rt.kRed+2]
    shower_colors = [rt.kBlue+1, rt.kBlue-7, rt.kBlue+3, rt.kBlue-3, rt.kBlue+2]
    
    # Create histogram for each particle contribution
    for i, contrib in enumerate(contributions):
        if contrib.total_pe > 0.001:  # Only show particles with significant contribution
            hist_name = f"h_pmt_pred_{contrib.type}{contrib.index}"
            h_pmt_pred = rt.TH1F(hist_name, "", 32, 0, 32)
            
            # Choose color based on particle type
            if contrib.type == "track":
                color = track_colors[min(i, len(track_colors)-1)]
            else:
                color = shower_colors[min(i, len(shower_colors)-1)]
                
            h_pmt_pred.SetFillColor(color)
            h_pmt_pred.SetFillStyle(3001)
            h_pmt_pred.SetLineColor(color)
            h_pmt_pred.SetLineWidth(2)
            
            # Fill histogram with PE values - iterate through PMT IDs
            for pmt_id in range(32):
                if pmt_id in contrib.pe_per_pmt:
                    pe = contrib.pe_per_pmt[pmt_id]
                    if pe > 0:
                        h_pmt_pred.SetBinContent(pmt_id+1, pe)
            
            h_pmt_pred_stack.Add(h_pmt_pred)
            legend.AddEntry(h_pmt_pred, f"{contrib.type.capitalize()}[{contrib.index}]: {contrib.total_pe:.3f} PE", "f")
            hists_v.append(h_pmt_pred)
    
    # Create histogram for observed flash if available
    h_pmt_obs = None
    if len(opflashes) > 0:
        flash = opflashes[0]  # Use first flash
        h_pmt_obs = rt.TH1F("h_pmt_obs", "Observed Flash;PMT ID;Photoelectrons", 32, 0, 32)
        h_pmt_obs.SetLineColor(rt.kBlack)
        h_pmt_obs.SetLineWidth(3)
        h_pmt_obs.SetMarkerStyle(20)
        h_pmt_obs.SetMarkerColor(rt.kBlack)
        
        total_obs_pe = 0.0
        for pmt in range(32):
            pe = flash.PE(pmt)
            h_pmt_obs.SetBinContent(pmt+1, pe)
            total_obs_pe += pe
            
        legend.AddEntry(h_pmt_obs, f"Observed: {total_obs_pe:.1f} PE", "lp")
    
    # Create canvas with multiple plots
    canvas = rt.TCanvas("c1", "Flash Prediction Analysis", 1600, 1200)
    canvas.Divide(2, 2)
    
    # Plot 1: Stacked prediction with observed overlay
    canvas.cd(1)
    rt.gPad.SetLeftMargin(0.12)
    rt.gPad.SetBottomMargin(0.12)
    
    # Determine y-axis range
    pred_max = h_pmt_pred_stack.GetMaximum() if h_pmt_pred_stack.GetMaximum() > 0 else 1.0
    obs_max = h_pmt_obs.GetMaximum() if h_pmt_obs and h_pmt_obs.GetMaximum() > 0 else 1.0
    y_max = max(pred_max, obs_max) * 1.2
    
    # Draw prediction stack
    if h_pmt_pred_stack.GetMaximum() > 0:
        h_pmt_pred_stack.Draw("hist")
        h_pmt_pred_stack.GetYaxis().SetRangeUser(0, y_max)
        h_pmt_pred_stack.GetXaxis().SetTitle("PMT ID")
        h_pmt_pred_stack.GetYaxis().SetTitle("Photoelectrons")
    else:
        # Create dummy histogram if no predictions
        h_dummy = rt.TH1F("h_dummy", "Flash Prediction: Individual Particle Contributions;PMT ID;Photoelectrons", 32, 0, 32)
        h_dummy.SetMinimum(0)
        h_dummy.SetMaximum(y_max)
        h_dummy.Draw()
    
    # Overlay observed data
    if h_pmt_obs:
        h_pmt_obs.Draw("E1 SAME")
    
    legend.Draw()
    
    # Plot 2: Total prediction vs observation comparison
    canvas.cd(2)
    rt.gPad.SetLeftMargin(0.12)
    rt.gPad.SetBottomMargin(0.12)
    
    # Create total prediction histogram
    h_total_pred = rt.TH1F("h_total_pred", "Prediction vs Observation Comparison;PMT ID;Photoelectrons", 32, 0, 32)
    h_total_pred.SetLineColor(rt.kRed)
    h_total_pred.SetLineWidth(3)
    h_total_pred.SetMarkerStyle(22)
    h_total_pred.SetMarkerColor(rt.kRed)
    
    # Fill with total prediction
    pe_per_pmt = predictor.getPredictedPE()
    for pmt_id in range(32):
        if pmt_id in pe_per_pmt:
            h_total_pred.SetBinContent(pmt_id+1, pe_per_pmt[pmt_id])
    
    # Set axis range and draw
    y_max_comp = max(h_total_pred.GetMaximum(), obs_max if h_pmt_obs else 0) * 1.2
    h_total_pred.SetMaximum(y_max_comp)
    h_total_pred.SetMinimum(0)
    h_total_pred.Draw("E1")
    
    if h_pmt_obs:
        h_pmt_obs.Draw("E1 SAME")
    
    # Add comparison legend
    legend2 = rt.TLegend(0.65, 0.75, 0.89, 0.89)
    legend2.AddEntry(h_total_pred, f"Total Prediction: {predictor.getTotalPredictedPE():.2f} PE", "lp")
    if h_pmt_obs:
        legend2.AddEntry(h_pmt_obs, f"Observed: {total_obs_pe:.1f} PE", "lp")
    legend2.Draw()
    
    # Plot 3: Logarithmic scale comparison (if there's a large dynamic range)
    canvas.cd(3)
    rt.gPad.SetLeftMargin(0.12)
    rt.gPad.SetBottomMargin(0.12)
    rt.gPad.SetLogy()
    
    # Clone histograms for log plot
    h_total_pred_log = h_total_pred.Clone("h_total_pred_log")
    h_total_pred_log.SetTitle("Log Scale Comparison;PMT ID;Photoelectrons")
    h_total_pred_log.SetMinimum(0.001)
    
    # Add small offset to avoid log(0) issues
    for pmt in range(32):
        current_val = h_total_pred_log.GetBinContent(pmt+1)
        if current_val <= 0:
            h_total_pred_log.SetBinContent(pmt+1, 0.001)
    
    h_total_pred_log.Draw("E1")
    
    if h_pmt_obs:
        h_pmt_obs_log = h_pmt_obs.Clone("h_pmt_obs_log")
        # Add offset for log scale
        for pmt in range(32):
            current_val = h_pmt_obs_log.GetBinContent(pmt+1)
            if current_val <= 0:
                h_pmt_obs_log.SetBinContent(pmt+1, 0.001)
        h_pmt_obs_log.Draw("E1 SAME")
    
    legend3 = rt.TLegend(0.65, 0.75, 0.89, 0.89)
    legend3.AddEntry(h_total_pred_log, "Total Prediction", "lp")
    if h_pmt_obs:
        legend3.AddEntry(h_pmt_obs_log, "Observed", "lp")
    legend3.Draw()
    
    # Plot 4: Summary text and statistics
    canvas.cd(4)
    rt.gPad.SetLeftMargin(0.05)
    rt.gPad.SetBottomMargin(0.05)
    
    text = rt.TText()
    text.SetTextAlign(12)  # Left-center alignment
    text.SetTextSize(0.06)
    
    y_pos = 0.95
    dy = 0.08
    
    # Prediction summary
    text.DrawText(0.05, y_pos, f"Flash Prediction Summary:")
    y_pos -= dy
    text.DrawText(0.05, y_pos, f"Total Predicted PE: {predictor.getTotalPredictedPE():.3f}")
    y_pos -= dy
    text.DrawText(0.05, y_pos, f"Tracks processed: {predictor.getNumTracksProcessed()}")
    y_pos -= dy
    text.DrawText(0.05, y_pos, f"Showers processed: {predictor.getNumShowersProcessed()}")
    y_pos -= dy
    text.DrawText(0.05, y_pos, f"Total charge: {predictor.getTotalChargeCollected():.0f} ADC")
    y_pos -= dy
    text.DrawText(0.05, y_pos, f"Total photons: {predictor.getTotalPhotonsEmitted():.0f}")
    y_pos -= dy
    
    # Observation summary if available
    if h_pmt_obs and total_obs_pe > 0:
        text.DrawText(0.05, y_pos, f"Observed PE: {total_obs_pe:.1f}")
        y_pos -= dy
        ratio = predictor.getTotalPredictedPE() / total_obs_pe
        text.DrawText(0.05, y_pos, f"Pred/Obs ratio: {ratio:.4f}")
        y_pos -= dy
        text.DrawText(0.05, y_pos, f"Flash time: {flash.Time():.1f} μs")
        y_pos -= dy
    
    # Individual contributions
    text.DrawText(0.05, y_pos, f"Individual contributions:")
    y_pos -= dy
    for contrib in contributions:
        if contrib.total_pe > 0.001:
            text.SetTextSize(0.05)
            text.DrawText(0.1, y_pos, f"{contrib.type}[{contrib.index}]: {contrib.total_pe:.3f} PE")
            y_pos -= 0.06
    
    # Save the plot
    canvas.SaveAs(output_filename)
    print(f"Visualization saved to: {output_filename}")
    
    return canvas

def test_nuvertex_flash_prediction(larcv_file, kps_file, larlite_file,
        vertex_index=0, entry=0, threshold=10.0, save_output=False):
    """
    Test the enhanced NuVertexFlashPrediction class with real data
    """
    print("NuVertexFlashPrediction Test with Real Data")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(larcv_file):
        raise FileNotFoundError(f"LArCV file not found: {larcv_file}")
    if not os.path.exists(kps_file):
        raise FileNotFoundError(f"KPS file not found: {kps_file}")
    
    # Load data
    try:
        adc_images, vertex_candidates, opflashes \
            = load_data_from_files(larcv_file, kps_file, larlite_file, entry)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(vertex_candidates) == 0:
        print("No vertex candidates found in the data file!")
        return
        
    if vertex_index >= len(vertex_candidates):
        print(f"Vertex index {vertex_index} >= {len(vertex_candidates)} available candidates")
        print(f"Available candidates: 0 to {len(vertex_candidates)-1}")
        return
    
    vertex_candidate = vertex_candidates.at(vertex_index)
    print(f"\nUsing vertex candidate {vertex_index}:")
    print(f"  Tracks: {vertex_candidate.track_v.size()}")
    print(f"  Showers: {vertex_candidate.shower_v.size()}")
    
    # Create the flash predictor
    print("\nInitializing NuVertexFlashPrediction...")
    predictor = larflow.reco.NuVertexFlashPrediction()
    
    # Configure parameters
    predictor.setChargeToPhotonParams(
        200.0,    # adc_per_electron  
        23.6e-3,  # mev_per_electron (MeV)
        24000.0,  # photons_per_mev
        0.7       # recombination_factor
    )
    
    predictor.setTrackConversionParams(
        3,      # dcol (wire window)
        3,      # drow (tick window)  
        0.3,    # minstepsize (cm)
        0.5     # maxstepsize (cm)
    )
    
    predictor.setShowerConversionParams(
        3,      # dcol (wire window)
        3       # drow (tick window)
    )
    
    print("Flash predictor configured with:")
    print("  - Charge-to-photon conversion: 200 ADC/e-, 23.6 μeV/e-, 24k photons/MeV, 70% recombination")
    print("  - Track conversion: 3x3 pixel window, 0.3-0.5 cm steps") 
    print("  - Shower conversion: 3x3 pixel window")
    print(f"  - ADC threshold: {threshold}")
    
    # Make the prediction
    print(f"\nRunning flash prediction...")
    try:
        predicted_flash = predictor.predictFlash(
            vertex_candidate, 
            adc_images,
            threshold,  # ADC threshold
            True        # use trilinear interpolation
        )
        print("Flash prediction completed successfully!")
    except Exception as e:
        print(f"Error during flash prediction: {e}")
        return
    
    # Analyze results
    print(f"\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    # Overall results
    total_pe = predictor.getTotalPredictedPE()
    print(f"\nOverall Results:")
    print(f"  Total predicted PE: {total_pe:.2f}")
    print(f"  Tracks processed: {predictor.getNumTracksProcessed()}")
    print(f"  Showers processed: {predictor.getNumShowersProcessed()}")
    print(f"  Total charge collected: {predictor.getTotalChargeCollected():.1f} ADC")
    print(f"  Total photons emitted: {predictor.getTotalPhotonsEmitted():.0f}")
    
    # Individual particle contributions
    contributions = predictor.getParticleContributions()
    print(f"\nIndividual Particle Contributions ({len(contributions)} particles):")
    
    for i, contrib in enumerate(contributions):
        print(f"\n  Particle {i}: {contrib.type}[{contrib.index}]")
        print(f"    Total PE: {contrib.total_pe:.3f}")
        print(f"    Charge collected: {contrib.charge_collected:.1f} ADC")
        print(f"    Photons emitted: {contrib.photons_emitted:.0f}")
        print(f"    Space points: {contrib.num_spacepoints}")
        
        # Show significant PMT contributions
        significant_pmts = []
        pe_per_pmt_map = contrib.pe_per_pmt
        for pmt_id in range(32):  # MicroBooNE has 32 PMTs
            if pmt_id in pe_per_pmt_map and pe_per_pmt_map[pmt_id] > 0.01:
                significant_pmts.append((pmt_id, pe_per_pmt_map[pmt_id]))
        
        if significant_pmts:
            print(f"    Significant PMT contributions ({len(significant_pmts)} PMTs):")
            for pmt_id, pe in sorted(significant_pmts, key=lambda x: x[1], reverse=True)[:5]:
                print(f"      PMT {pmt_id:2d}: {pe:.3f} PE")
            if len(significant_pmts) > 5:
                print(f"      ... and {len(significant_pmts)-5} more")
        else:
            print(f"    No significant PMT contributions (all < 0.01 PE)")
    
    # Analysis
    print(f"\nAnalysis:")
    
    if len(contributions) > 0:
        # Find brightest particle
        max_contrib = max(contributions, key=lambda x: x.total_pe)
        print(f"  Brightest particle: {max_contrib.type}[{max_contrib.index}] with {max_contrib.total_pe:.3f} PE")
        
        # Track vs shower contributions
        track_pe = sum(c.total_pe for c in contributions if c.type == "track")
        shower_pe = sum(c.total_pe for c in contributions if c.type == "shower")
        
        if total_pe > 0:
            print(f"  Track contribution: {track_pe:.3f} PE ({track_pe/total_pe*100:.1f}%)")
            print(f"  Shower contribution: {shower_pe:.3f} PE ({shower_pe/total_pe*100:.1f}%)")
        
        # PMT distribution
        pe_per_pmt = predictor.getPredictedPE()
        bright_pmts = []
        for pmt_id in range(32):
            if pmt_id in pe_per_pmt and pe_per_pmt[pmt_id] > 0.1:
                bright_pmts.append((pmt_id, pe_per_pmt[pmt_id]))
        
        print(f"  PMTs with >0.1 PE: {len(bright_pmts)}")
        if bright_pmts:
            bright_pmts.sort(key=lambda x: x[1], reverse=True)
            print(f"  Brightest PMTs:")
            for pmt_id, pe in bright_pmts[:5]:
                print(f"    PMT {pmt_id:2d}: {pe:.3f} PE")
    
    # Compare with observed flash if available
    if len(opflashes) > 0:
        print(f"\nObserved Flash Comparison:")
        flash = opflashes[0]  # Use first flash
        observed_pe = flash.TotalPE()
        print(f"  Observed total PE: {observed_pe:.2f}")
        print(f"  Predicted total PE: {total_pe:.2f}")
        if observed_pe > 0:
            ratio = total_pe / observed_pe
            print(f"  Predicted/Observed ratio: {ratio:.3f}")
        
        print(f"  Observed flash time: {flash.Time():.1f} μs")
        print(f"  Observed flash channels with >1 PE:")
        for pmt in range(min(32, flash.nOpDets())):
            pe = flash.PE(pmt)
            if pe > 1.0:
                print(f"    PMT {pmt:2d}: {pe:.1f} PE")
    
    # Calculate Sinkhorn divergence between prediction and observation
    if len(opflashes) > 0:
        print(f"\nCalculating Sinkhorn divergence...")
        
        # Create Sinkhorn divergence calculator
        sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
        
        # Get predicted PE per PMT
        pe_per_pmt = predictor.getPredictedPE()
        predicted_pe = []
        for pmt_id in range(32):
            if pmt_id in pe_per_pmt:
                predicted_pe.append(pe_per_pmt[pmt_id])
            else:
                predicted_pe.append(0.0)
        
        # Get observed PE per PMT
        flash = opflashes[0]
        observed_pe = []
        for pmt_id in range(32):
            observed_pe.append(flash.PE(pmt_id))
        
        # Convert to std::vector<float>
        pred_vec = std.vector('float')(predicted_pe)
        obs_vec = std.vector('float')(observed_pe)
        
        # Calculate divergence with different regularization parameters
        regularizations = [0.1, 1.0, 10.0]
        print(f"  Sinkhorn divergence results:")
        for reg in regularizations:
            divergence = sinkhorn_calc.calculateDivergence(pred_vec, obs_vec, reg, 100, 1e-6)
            converged = sinkhorn_calc.getLastConverged()
            iterations = sinkhorn_calc.getLastIterations()
            print(f"    λ={reg:5.2f}: divergence={divergence:8.3f} (converged: {converged}, {iterations:2d} iter)")
    
    # Create visualization
    if save_output:
        print(f"\nCreating visualization and saving output...")
        
        # Create the flash prediction visualization
        plot_filename = f"nuvertex_flash_prediction_vtx{vertex_index}_entry{entry}.png"
        canvas = create_flash_visualization(predictor, contributions, opflashes, plot_filename)
        
        # Also save a text summary
        output_file = f"nuvertex_flash_prediction_vtx{vertex_index}_entry{entry}.txt"
        with open(output_file, 'w') as f:
            f.write(f"NuVertexFlashPrediction Results\n")
            f.write(f"Entry: {entry}, Vertex: {vertex_index}\n")
            f.write(f"Total PE: {total_pe:.3f}\n")
            f.write(f"Tracks: {predictor.getNumTracksProcessed()}\n")
            f.write(f"Showers: {predictor.getNumShowersProcessed()}\n")
            f.write(f"Individual contributions:\n")
            for i, contrib in enumerate(contributions):
                f.write(f"  {contrib.type}[{contrib.index}]: {contrib.total_pe:.3f} PE\n")
            
            # Add Sinkhorn divergence results if available
            if len(opflashes) > 0:
                f.write(f"\nSinkhorn divergence results:\n")
                pe_per_pmt = predictor.getPredictedPE()
                predicted_pe = [pe_per_pmt[pmt_id] if pmt_id in pe_per_pmt else 0.0 for pmt_id in range(32)]
                flash = opflashes[0]
                observed_pe = [flash.PE(pmt_id) for pmt_id in range(32)]
                pred_vec = std.vector('float')(predicted_pe)
                obs_vec = std.vector('float')(observed_pe)
                sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
                for reg in [0.1, 1.0, 10.0]:
                    divergence = sinkhorn_calc.calculateDivergence(pred_vec, obs_vec, reg, 100, 1e-6)
                    f.write(f"  λ={reg:5.2f}: {divergence:8.3f}\n")
                    
        print(f"Summary saved to: {output_file}")
        
        # Keep canvas alive for interactive viewing if desired
        print("Visualization completed.")
    else:
        # Even if not saving, create a simple visualization
        print(f"\nCreating quick visualization...")
        plot_filename = f"nuvertex_flash_prediction_quick.png"
        canvas = create_flash_visualization(predictor, contributions, opflashes, plot_filename)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test NuVertexFlashPrediction with real data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-lcv','--larcv-file',required=True,type=str,
                        help="Input LArCV file (output of run_kpsrecoman.py)")
    parser.add_argument('-r','--reco-file',required=True,type=str,
                        help="Input Reco File containing KPSRecoManagerTree TTree (output of run_kpsrecoman.py)")
    parser.add_argument('-ll','--larlite-file',required=True,type=str,
                        help='larlite input file (output of run_kpsrecoman.py) which will have target opflash')
    parser.add_argument("--vertex-index", type=int, default=0,
                       help="Index of vertex candidate to analyze")
    parser.add_argument("--entry", type=int, default=0,
                       help="Entry number to read from files")
    parser.add_argument("--threshold", type=float, default=10.0,
                       help="ADC threshold for pixel selection")
    parser.add_argument("--save-output", action="store_true",
                       help="Save output summary to file")

    
    args = parser.parse_args()

    # File paths (relative to test directory)
    larcv_file   = args.larcv_file
    larlite_file = args.larlite_file
    kps_file     = args.reco_file
    
    print(f"Running NuVertexFlashPrediction test with:")
    print(f"  Entry: {args.entry}")
    print(f"  Vertex index: {args.vertex_index}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Save output: {args.save_output}")
    
    try:
        test_nuvertex_flash_prediction(
            larcv_file,
            kps_file,
            larlite_file,
            vertex_index=args.vertex_index,
            entry=args.entry, 
            threshold=args.threshold,
            save_output=args.save_output
        )
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()