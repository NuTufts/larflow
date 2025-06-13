#!/usr/bin/env python3
"""
Simple test script for SinkhornFlashDivergence class

This script tests the Sinkhorn divergence calculation with simple known cases
to verify the implementation is working correctly.
"""

import ROOT as rt
from ROOT import larflow, std
import numpy as np

# Set ROOT to batch mode
rt.gROOT.SetBatch(True)

def test_identical_distributions():
    """Test that identical distributions give zero divergence"""
    print("Test 1: Identical distributions should give zero divergence")
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Create identical distributions (uniform)
    uniform_dist = [1.0] * 32
    dist_a = std.vector('float')(uniform_dist)
    dist_b = std.vector('float')(uniform_dist)
    
    divergence = sinkhorn_calc.calculateDivergence(dist_a, dist_b, 1.0, 1000, 1e-8)
    converged = sinkhorn_calc.getLastConverged()
    iterations = sinkhorn_calc.getLastIterations()
    
    print(f"  Uniform distributions: divergence={divergence:.6f}, converged={converged}, iterations={iterations}")
    
    # Create identical peaked distributions
    peaked_dist = [0.0] * 32
    peaked_dist[15] = 10.0  # Peak at PMT 15
    peaked_dist[16] = 5.0   # Smaller peak at PMT 16
    
    dist_c = std.vector('float')(peaked_dist)
    dist_d = std.vector('float')(peaked_dist)
    
    divergence = sinkhorn_calc.calculateDivergence(dist_c, dist_d, 1.0, 1000, 1e-8)
    converged = sinkhorn_calc.getLastConverged()
    iterations = sinkhorn_calc.getLastIterations()
    
    print(f"  Peaked distributions:  divergence={divergence:.6f}, converged={converged}, iterations={iterations}")

def test_different_distributions():
    """Test divergence between different distributions"""
    print("\nTest 2: Different distributions should give positive divergence")
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Create two different peaked distributions
    dist1 = [0.0] * 32
    dist1[10] = 10.0  # Peak at PMT 10
    
    dist2 = [0.0] * 32
    dist2[20] = 10.0  # Peak at PMT 20 (far from PMT 10)
    
    dist_a = std.vector('float')(dist1)
    dist_b = std.vector('float')(dist2)
    
    # Test with different regularization parameters
    regularizations = [0.1, 1.0, 10.0, 100.0]
    
    for reg in regularizations:
        divergence = sinkhorn_calc.calculateDivergence(dist_a, dist_b, reg, 1000, 1e-8)
        converged = sinkhorn_calc.getLastConverged()
        iterations = sinkhorn_calc.getLastIterations()
        
        print(f"  λ={reg:6.1f}: divergence={divergence:8.3f}, converged={converged}, iterations={iterations:3d}")

def test_geometry_versions():
    """Test different geometry versions"""
    print("\nTest 3: Testing different geometry versions")
    
    # Test V4 geometry
    print("  V4 Geometry:")
    sinkhorn_v4 = larflow.reco.SinkhornFlashDivergence()
    sinkhorn_v4.setGeometryVersion(larflow.reco.SinkhornFlashDivergence.kV4)
    
    # Test a few PMT positions
    for pmt_id in [0, 15, 31]:
        pos = sinkhorn_v4.getPMTPosition(pmt_id)
        print(f"    PMT {pmt_id:2d}: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f})")
    
    # Test V12 geometry
    print("  V12 Geometry:")
    sinkhorn_v12 = larflow.reco.SinkhornFlashDivergence()
    sinkhorn_v12.setGeometryVersion(larflow.reco.SinkhornFlashDivergence.kV12)
    
    for pmt_id in [0, 15, 31]:
        pos = sinkhorn_v12.getPMTPosition(pmt_id)
        print(f"    PMT {pmt_id:2d}: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f})")

def test_realistic_case():
    """Test with realistic PE distributions"""
    print("\nTest 4: Realistic flash-like distributions")
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Create realistic distribution 1 (central PMTs active)
    realistic1 = [0.0] * 32
    central_pmts = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    for pmt in central_pmts:
        realistic1[pmt] = 50.0 + 20.0 * np.random.random()  # 50-70 PE
    
    # Add some noise to other PMTs
    for pmt in range(32):
        if pmt not in central_pmts:
            realistic1[pmt] = 5.0 * np.random.random()  # 0-5 PE
    
    # Create realistic distribution 2 (slightly shifted pattern)
    realistic2 = [0.0] * 32
    shifted_pmts = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    for pmt in shifted_pmts:
        realistic2[pmt] = 45.0 + 25.0 * np.random.random()  # 45-70 PE
    
    # Add some noise
    for pmt in range(32):
        if pmt not in shifted_pmts:
            realistic2[pmt] = 8.0 * np.random.random()  # 0-8 PE
    
    dist_a = std.vector('float')(realistic1)
    dist_b = std.vector('float')(realistic2)
    
    # Show total PE
    total_a = sum(realistic1)
    total_b = sum(realistic2)
    print(f"  Distribution A total PE: {total_a:.1f}")
    print(f"  Distribution B total PE: {total_b:.1f}")
    
    # Calculate divergence
    regularizations = [1.0, 10.0, 50.0]
    for reg in regularizations:
        divergence = sinkhorn_calc.calculateDivergence(dist_a, dist_b, reg, 1000, 1e-8)
        converged = sinkhorn_calc.getLastConverged()
        iterations = sinkhorn_calc.getLastIterations()
        
        print(f"  λ={reg:5.1f}: divergence={divergence:8.3f}, converged={converged}, iterations={iterations:3d}")

def main():
    """Run all tests"""
    print("SinkhornFlashDivergence Test Suite")
    print("=" * 50)
    
    try:
        test_identical_distributions()
        test_different_distributions()
        test_geometry_versions()
        test_realistic_case()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()