#ifndef __SINKHORN_FLASH_DIVERGENCE_H__
#define __SINKHORN_FLASH_DIVERGENCE_H__

#include <vector>
#include <map>
#include <string>
#include "larcv/core/Base/larcv_base.h"

namespace larflow {
namespace reco {

  /**
   * @brief Class to calculate Sinkhorn divergence between predicted and observed optical flash PE distributions
   *
   * The Sinkhorn divergence is a regularized optimal transport distance that measures how different
   * two probability distributions are, taking into account the spatial arrangement of PMTs.
   * It's particularly useful for comparing optical flash patterns while considering the 3D geometry.
   *
   * The symmetric Sinkhorn divergence is computed as:
   * S_λ(μ,ν) = OT_λ(μ,ν) - 0.5*OT_λ(μ,μ) - 0.5*OT_λ(ν,ν)
   * where OT_λ is the entropic regularized optimal transport cost.
   */
  class SinkhornFlashDivergence : public larcv::larcv_base {

  public:

    /// Geometry versions available for PMT positions
    enum GeometryVersion {
      kV4,   ///< Old geometry (v4) with PMTs incorrectly placed in TPC
      kV12   ///< Current geometry (v12) with correct PMT positions
    };

    SinkhornFlashDivergence();
    virtual ~SinkhornFlashDivergence();

    /**
     * @brief Set the geometry version for PMT positions
     * @param version Either kV4 or kV12
     */
    void setGeometryVersion(GeometryVersion version);

    /**
     * @brief Calculate symmetric Sinkhorn divergence between two PE distributions
     * @param predicted_pe Vector of predicted PE values per PMT channel (32 channels)
     * @param observed_pe Vector of observed PE values per PMT channel (32 channels)
     * @param regularization Regularization parameter λ (default: 0.1)
     * @param max_iterations Maximum number of Sinkhorn iterations (default: 100)
     * @param tolerance Convergence tolerance (default: 1e-6)
     * @return Symmetric Sinkhorn divergence value
     */
    float calculateDivergence(
      const std::vector<float>& predicted_pe,
      const std::vector<float>& observed_pe,
      float regularization = 0.1f,
      int max_iterations = 100,
      float tolerance = 1e-6f
    );

    /**
     * @brief Calculate entropic regularized optimal transport cost
     * @param distribution_a First distribution (normalized)
     * @param distribution_b Second distribution (normalized) 
     * @param regularization Regularization parameter λ
     * @param max_iterations Maximum Sinkhorn iterations
     * @param tolerance Convergence tolerance
     * @return Optimal transport cost
     */
    float calculateOptimalTransportCost(
      const std::vector<float>& distribution_a,
      const std::vector<float>& distribution_b,
      float regularization,
      int max_iterations,
      float tolerance
    );

    /**
     * @brief Get PMT position for a given channel
     * @param channel PMT channel number (0-31)
     * @return 3D position [x, y, z] in cm (TPC coordinates)
     */
    std::vector<float> getPMTPosition(int channel) const;

    /**
     * @brief Get the cost matrix between all PMT pairs
     * @return 32x32 cost matrix based on Euclidean distance
     */
    const std::vector<std::vector<float>>& getCostMatrix() const { return _cost_matrix; }

    /**
     * @brief Get number of iterations from last calculation
     * @return Number of iterations used in last Sinkhorn algorithm run
     */
    int getLastIterations() const { return _last_iterations; }

    /**
     * @brief Check if last calculation converged
     * @return True if last calculation converged within tolerance
     */
    bool getLastConverged() const { return _last_converged; }

    /**
     * @brief Calculate the optimal transport plan between two distributions
     * @param distribution_a First distribution (normalized)
     * @param distribution_b Second distribution (normalized)
     * @param regularization Regularization parameter λ
     * @param max_iterations Maximum Sinkhorn iterations
     * @param tolerance Convergence tolerance
     * @return 32x32 transport plan matrix π[i][j] = mass moved from PMT i to PMT j
     */
    std::vector<std::vector<float>> calculateTransportPlan(
      const std::vector<float>& distribution_a,
      const std::vector<float>& distribution_b,
      float regularization = 1.0f,
      int max_iterations = 100,
      float tolerance = 1e-6f
    );

    /**
     * @brief Get transport flows from a specific source PMT to all target PMTs
     * @param transport_plan The 32x32 transport plan matrix
     * @param source_pmt Source PMT index (0-31)
     * @return Vector of flows to each target PMT
     */
    std::vector<float> getFlowsFromPMT(
      const std::vector<std::vector<float>>& transport_plan,
      int source_pmt
    ) const;

    /**
     * @brief Get transport flows to a specific target PMT from all source PMTs
     * @param transport_plan The 32x32 transport plan matrix
     * @param target_pmt Target PMT index (0-31)
     * @return Vector of flows from each source PMT
     */
    std::vector<float> getFlowsToPMT(
      const std::vector<std::vector<float>>& transport_plan,
      int target_pmt
    ) const;

  protected:

    /**
     * @brief Initialize PMT positions for the selected geometry
     */
    void initializePMTPositions();

    /**
     * @brief Calculate cost matrix based on PMT positions
     */
    void calculateCostMatrix();

    /**
     * @brief Normalize a distribution to sum to 1
     * @param distribution Input distribution
     * @return Normalized distribution
     */
    std::vector<float> normalizeDistribution(const std::vector<float>& distribution) const;

    /**
     * @brief Calculate Euclidean distance between two 3D points
     * @param pos1 First position [x, y, z]
     * @param pos2 Second position [x, y, z]
     * @return Euclidean distance
     */
    float calculateDistance(const std::vector<float>& pos1, const std::vector<float>& pos2) const;

  private:

    GeometryVersion _geometry_version;           ///< Current geometry version
    std::vector<std::vector<float>> _pmt_positions; ///< PMT positions [channel][x,y,z]
    std::vector<std::vector<float>> _cost_matrix;    ///< Cost matrix between PMT pairs
    bool _initialized;                           ///< Whether positions and cost matrix are initialized
    float _dist_scale;                           ///< normalize distance to this scale to keep distance values below 1.0 for stability.

    // Results from last calculation
    int _last_iterations;                        ///< Number of iterations from last run
    bool _last_converged;                        ///< Whether last calculation converged

    // Hard-coded PMT positions from lardly/lardly/ubdl/pmtpos.py
    
    /// V12 geometry PMT positions (current geometry) - OpDet ID -> [x, y, z] in global coordinates
    static const std::map<int, std::vector<float>> _v12_opdet_positions;
    
    /// V4 geometry PMT positions (old geometry) - Channel -> [x, y, z] in TPC coordinates  
    static const std::map<int, std::vector<float>> _v4_channel_positions;
    
    /// OpDet to OpChannel mapping for V12 geometry
    static const std::map<int, int> _opdet_to_channel;

    /// TPC origin offset for coordinate conversion
    static const std::vector<float> _tpc_origin;

  };

}
}

#endif