#!/usr/bin/env python3
"""
Test script demonstrating optimal transport plan calculation

This script shows how to use the SinkhornFlashDivergence class to calculate
the optimal transport plan and analyze how mass flows between PMTs.
"""

import ROOT as rt
from ROOT import larflow, std
import numpy as np

# Set ROOT to batch mode
rt.gROOT.SetBatch(True)

def test_transport_plan_simple():
    """Test transport plan with simple peaked distributions"""
    print("=" * 60)
    print("OPTIMAL TRANSPORT PLAN DEMONSTRATION")
    print("=" * 60)
    print("\nTest: Transport plan between two peaked distributions")
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Create two peaked distributions
    dist1 = [0.0] * 32
    dist1[10] = 10.0  # Peak at PMT 10
    dist1[11] = 5.0   # Smaller peak at PMT 11
    
    dist2 = [0.0] * 32
    dist2[15] = 8.0   # Peak at PMT 15
    dist2[16] = 7.0   # Peak at PMT 16
    
    # Convert to C++ vectors
    dist_a = std.vector('float')(dist1)
    dist_b = std.vector('float')(dist2)
    
    # Calculate transport plan
    print("Calculating transport plan...")
    regularization = 10.0
    transport_plan = sinkhorn_calc.calculateTransportPlan(dist_a, dist_b, regularization, 1000, 1e-8)
    converged = sinkhorn_calc.getLastConverged()
    iterations = sinkhorn_calc.getLastIterations()
    
    print(f"  Regularization: λ = {regularization}")
    print(f"  Converged: {converged} ({iterations} iterations)")
    
    # Analyze transport plan
    print(f"\nTransport Plan Analysis:")
    print(f"  Source distribution: PMTs 10={dist1[10]:.1f}, 11={dist1[11]:.1f}")
    print(f"  Target distribution: PMTs 15={dist2[15]:.1f}, 16={dist2[16]:.1f}")
    
    # Show significant transport flows
    print(f"\nSignificant transport flows (> 0.01):")
    total_transport = 0.0
    for i in range(32):
        for j in range(32):
            flow = transport_plan[i][j]
            if flow > 0.01:
                total_transport += flow
                print(f"  PMT {i:2d} → PMT {j:2d}: {flow:.4f}")
    
    print(f"\nTotal transport: {total_transport:.4f}")
    
    # Analyze flows from specific PMTs
    print(f"\nFlows FROM PMT 10 (main source):")
    flows_from_10 = sinkhorn_calc.getFlowsFromPMT(transport_plan, 10)
    for j in range(32):
        if flows_from_10[j] > 0.01:
            print(f"  PMT 10 → PMT {j:2d}: {flows_from_10[j]:.4f}")
    
    print(f"\nFlows TO PMT 15 (main target):")
    flows_to_15 = sinkhorn_calc.getFlowsToPMT(transport_plan, 15)
    for i in range(32):
        if flows_to_15[i] > 0.01:
            print(f"  PMT {i:2d} → PMT 15: {flows_to_15[i]:.4f}")
    
    return transport_plan

def test_transport_plan_realistic():
    """Test transport plan with realistic flash-like distributions"""
    print("\n" + "=" * 60)
    print("REALISTIC FLASH TRANSPORT ANALYSIS")
    print("=" * 60)
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Create realistic predicted distribution (central PMTs)
    predicted = [0.0] * 32
    central_pmts = [10, 11, 12, 13, 14, 15, 16]
    for pmt in central_pmts:
        predicted[pmt] = 50.0 + 30.0 * np.random.random()  # 50-80 PE
    
    # Add noise
    for pmt in range(32):
        if pmt not in central_pmts:
            predicted[pmt] = 5.0 * np.random.random()  # 0-5 PE
    
    # Create observed distribution (slightly shifted and different intensities)
    observed = [0.0] * 32
    observed_pmts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for pmt in observed_pmts:
        observed[pmt] = 40.0 + 40.0 * np.random.random()  # 40-80 PE
    
    # Add realistic noise to all PMTs
    for pmt in range(32):
        observed[pmt] += 10.0 * np.random.random()  # 0-10 PE noise
    
    # Convert to C++ vectors
    pred_vec = std.vector('float')(predicted)
    obs_vec = std.vector('float')(observed)
    
    # Show totals
    pred_total = sum(predicted)
    obs_total = sum(observed)
    print(f"Predicted total PE: {pred_total:.1f}")
    print(f"Observed total PE:  {obs_total:.1f}")
    print(f"PE ratio (pred/obs): {pred_total/obs_total:.3f}")
    
    # Calculate divergence first
    regularization = 50.0  # Use larger regularization for realistic case
    divergence = sinkhorn_calc.calculateDivergence(pred_vec, obs_vec, regularization, 1000, 1e-8)
    print(f"\nSinkhorn divergence (λ={regularization}): {divergence:.3f}")
    
    # Calculate transport plan
    print(f"\nCalculating transport plan...")
    transport_plan = sinkhorn_calc.calculateTransportPlan(pred_vec, obs_vec, regularization, 1000, 1e-8)
    converged = sinkhorn_calc.getLastConverged()
    iterations = sinkhorn_calc.getLastIterations()
    print(f"  Converged: {converged} ({iterations} iterations)")
    
    # Find top transport flows
    print(f"\nTop 10 transport flows:")
    flows = []
    for i in range(32):
        for j in range(32):
            flow = transport_plan[i][j]
            if flow > 0.001:  # Only consider significant flows
                flows.append((i, j, flow))
    
    # Sort by flow magnitude
    flows.sort(key=lambda x: x[2], reverse=True)
    
    for i, (source, target, flow) in enumerate(flows[:10]):
        pred_pe = predicted[source]
        obs_pe = observed[target]
        # Get PMT positions for distance calculation
        pos_source = sinkhorn_calc.getPMTPosition(source)
        pos_target = sinkhorn_calc.getPMTPosition(target)
        distance = np.sqrt((pos_source[0]-pos_target[0])**2 + 
                          (pos_source[1]-pos_target[1])**2 + 
                          (pos_source[2]-pos_target[2])**2)
        
        print(f"  {i+1:2d}. PMT {source:2d} → PMT {target:2d}: {flow:.4f} "
              f"(pred={pred_pe:5.1f}, obs={obs_pe:5.1f}, dist={distance:5.1f}cm)")
    
    # Analyze mass conservation
    print(f"\nMass conservation check:")
    for check_pmt in [10, 12, 15]:
        if predicted[check_pmt] > 10.0:  # Only check PMTs with significant predicted PE
            # Sum outflows from this PMT
            outflows = sum(transport_plan[check_pmt])
            pred_mass = predicted[check_pmt] / pred_total  # Normalized mass
            print(f"  PMT {check_pmt:2d}: predicted mass = {pred_mass:.4f}, outflows = {outflows:.4f}")
    
    return transport_plan

def analyze_transport_patterns(transport_plan):
    """Analyze patterns in the transport plan"""
    print(f"\n" + "=" * 60)
    print("TRANSPORT PATTERN ANALYSIS")
    print("=" * 60)
    
    sinkhorn_calc = larflow.reco.SinkhornFlashDivergence()
    
    # Calculate transport statistics
    total_flows = 0.0
    local_flows = 0.0  # Flows to nearby PMTs (< 100 cm)
    long_flows = 0.0   # Flows to distant PMTs (> 300 cm)
    
    flow_distances = []
    
    for i in range(32):
        for j in range(32):
            flow = transport_plan[i][j]
            if flow > 0.001:
                total_flows += flow
                
                # Calculate distance between PMTs
                pos_i = sinkhorn_calc.getPMTPosition(i)
                pos_j = sinkhorn_calc.getPMTPosition(j)
                distance = np.sqrt((pos_i[0]-pos_j[0])**2 + 
                                 (pos_i[1]-pos_j[1])**2 + 
                                 (pos_i[2]-pos_j[2])**2)
                
                flow_distances.append((distance, flow))
                
                if distance < 100.0:
                    local_flows += flow
                elif distance > 300.0:
                    long_flows += flow
    
    print(f"Transport statistics:")
    print(f"  Total transport flows: {total_flows:.4f}")
    print(f"  Local flows (< 100cm):  {local_flows:.4f} ({local_flows/total_flows*100:.1f}%)")
    print(f"  Long flows (> 300cm):   {long_flows:.4f} ({long_flows/total_flows*100:.1f}%)")
    
    # Distance distribution
    if flow_distances:
        distances = [d for d, f in flow_distances]
        flows = [f for d, f in flow_distances]
        
        avg_distance = np.average(distances, weights=flows)
        print(f"  Average transport distance: {avg_distance:.1f} cm")
        print(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} cm")

def main():
    """Run all transport plan tests"""
    try:
        # Test 1: Simple peaked distributions
        transport_plan_simple = test_transport_plan_simple()
        
        # Test 2: Realistic flash distributions
        transport_plan_realistic = test_transport_plan_realistic()
        
        # Test 3: Pattern analysis
        analyze_transport_patterns(transport_plan_realistic)
        
        print("\n" + "=" * 60)
        print("TRANSPORT PLAN TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nKey insights:")
        print(f"• The transport plan π[i][j] shows how much mass moves from PMT i to PMT j")
        print(f"• Mass conservation: Σ_j π[i][j] = normalized_mass_at_PMT_i")
        print(f"• Target conservation: Σ_i π[i][j] = normalized_mass_at_PMT_j")
        print(f"• Nearby PMTs tend to exchange more mass (lower transport cost)")
        print(f"• The regularization parameter λ controls transport locality")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()