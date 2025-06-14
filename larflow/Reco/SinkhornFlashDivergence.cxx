#include "SinkhornFlashDivergence.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace larflow {
namespace reco {

  // V12 geometry PMT positions (OpDet ID -> [x, y, z] in global coordinates)
  const std::map<int, std::vector<float>> SinkhornFlashDivergence::_v12_opdet_positions = {
    {0, {-11.4545f, -28.625f, 990.356f}},
    {1, {-11.4175f, 27.607f, 989.712f}},
    {2, {-11.7755f, -56.514f, 951.865f}},
    {3, {-11.6415f, 55.313f, 951.861f}},
    {4, {-12.0585f, -56.309f, 911.939f}},
    {5, {-11.8345f, 55.822f, 911.065f}},
    {6, {-12.1765f, -0.722f, 865.599f}},
    {7, {-12.3045f, -0.502f, 796.208f}},
    {8, {-12.6045f, -56.284f, 751.905f}},
    {9, {-12.5405f, 55.625f, 751.884f}},
    {10, {-12.6125f, -56.408f, 711.274f}},
    {11, {-12.6615f, 55.8f, 711.073f}},
    {12, {-12.6245f, -0.051f, 664.203f}},
    {13, {-12.6515f, -0.549f, 585.284f}},
    {14, {-12.8735f, 55.822f, 540.929f}},
    {15, {-12.6205f, -56.205f, 540.616f}},
    {16, {-12.5945f, -56.323f, 500.221f}},
    {17, {-12.9835f, 55.771f, 500.134f}},
    {18, {-12.6185f, -0.875f, 453.096f}},
    {19, {-13.0855f, -0.706f, 373.839f}},
    {20, {-12.6485f, -57.022f, 328.341f}},
    {21, {-13.1865f, 54.693f, 328.212f}},
    {22, {-13.4175f, 54.646f, 287.976f}},
    {23, {-13.0075f, -56.261f, 287.639f}},
    {24, {-13.1505f, -0.829f, 242.014f}},
    {25, {-13.4415f, -0.303f, 173.743f}},
    {26, {-13.3965f, 55.249f, 128.354f}},
    {27, {-13.2784f, -56.203f, 128.18f}},
    {28, {-13.2375f, -56.615f, 87.8695f}},
    {29, {-13.5415f, 55.249f, 87.7605f}},
    {30, {-13.4345f, 27.431f, 51.1015f}},
    {31, {-13.1525f, -28.576f, 50.4745f}}
  };

  // V4 geometry PMT positions (Channel -> [x, y, z] in TPC coordinates)
  const std::map<int, std::vector<float>> SinkhornFlashDivergence::_v4_channel_positions = {
    {0, {2.458f, 55.313f, 951.861f}},
    {1, {2.265f, 55.822f, 911.066f}},
    {2, {2.682f, 27.607f, 989.712f}},
    {3, {1.923f, -0.722f, 865.598f}},
    {4, {2.645f, -28.625f, 990.356f}},
    {5, {2.324f, -56.514f, 951.865f}},
    {6, {2.041f, -56.309f, 911.939f}},
    {7, {1.438f, 55.8f, 711.073f}},
    {8, {1.559f, 55.625f, 751.884f}},
    {9, {1.795f, -0.502f, 796.208f}},
    {10, {1.475f, -0.051f, 664.203f}},
    {11, {1.495f, -56.284f, 751.905f}},
    {12, {1.487f, -56.408f, 711.274f}},
    {13, {1.116f, 55.771f, 500.134f}},
    {14, {1.226f, 55.822f, 540.929f}},
    {15, {1.448f, -0.549f, 585.284f}},
    {16, {1.481f, -0.875f, 453.096f}},
    {17, {1.479f, -56.205f, 540.616f}},
    {18, {1.505f, -56.323f, 500.221f}},
    {19, {0.913f, 54.693f, 328.212f}},
    {20, {0.682f, 54.646f, 287.976f}},
    {21, {1.014f, -0.706f, 373.839f}},
    {22, {0.949f, -0.829f, 242.014f}},
    {23, {1.451f, -57.022f, 328.341f}},
    {24, {1.092f, -56.261f, 287.639f}},
    {25, {0.703f, 55.249f, 128.355f}},
    {26, {0.558f, 55.249f, 87.7605f}},
    {27, {0.665f, 27.431f, 51.1015f}},
    {28, {0.658f, -0.303f, 173.743f}},
    {29, {0.947f, -28.576f, 50.4745f}},
    {30, {0.8211f, -56.203f, 128.179f}},
    {31, {0.862f, -56.615f, 87.8695f}}
  };

  // OpDet to OpChannel mapping for V12 geometry (first channel of each OpDet)
  const std::map<int, int> SinkhornFlashDivergence::_opdet_to_channel = {
    {0, 29}, {1, 27}, {2, 31}, {3, 26}, {4, 30}, {5, 25}, {6, 28}, {7, 22},
    {8, 24}, {9, 20}, {10, 23}, {11, 19}, {12, 21}, {13, 16}, {14, 14}, {15, 18},
    {16, 17}, {17, 13}, {18, 15}, {19, 10}, {20, 12}, {21, 8}, {22, 7}, {23, 11},
    {24, 9}, {25, 3}, {26, 1}, {27, 6}, {28, 5}, {29, 0}, {30, 2}, {31, 4}
  };

  // TPC origin offset for coordinate conversion
  const std::vector<float> SinkhornFlashDivergence::_tpc_origin = {-1.825f, 0.97f, -4.0f};

  SinkhornFlashDivergence::SinkhornFlashDivergence()
    : larcv::larcv_base("SinkhornFlashDivergence"),
      _geometry_version(kV12),
      _initialized(false),
      _last_iterations(0),
      _dist_scale(1000.0), // 1000 cm, the length (and largest direction) of the microbone detector
      _last_converged(false)
  {
    // Initialize with default V12 geometry
    initializePMTPositions();
    calculateCostMatrix();
  }

  SinkhornFlashDivergence::~SinkhornFlashDivergence() {
  }

  void SinkhornFlashDivergence::setGeometryVersion(GeometryVersion version) {
    if (version != _geometry_version) {
      _geometry_version = version;
      _initialized = false;
      initializePMTPositions();
      calculateCostMatrix();
    }
  }

  void SinkhornFlashDivergence::initializePMTPositions() {
    _pmt_positions.clear();
    _pmt_positions.resize(32);

    if (_geometry_version == kV4) {
      // Use V4 positions directly (already in TPC coordinates)
      for (int channel = 0; channel < 32; channel++) {
        auto it = _v4_channel_positions.find(channel);
        if (it != _v4_channel_positions.end()) {
          _pmt_positions[channel] = it->second;
        } else {
          LARCV_WARNING() << "No V4 position found for channel " << channel << std::endl;
          _pmt_positions[channel] = {0.0f, 0.0f, 0.0f};
        }
      }
    } else {
      // Use V12 positions, convert from global to TPC coordinates
      for (int channel = 0; channel < 32; channel++) {
        // Find OpDet ID for this channel
        int opdet_id = -1;
        for (const auto& pair : _opdet_to_channel) {
          if (pair.second == channel) {
            opdet_id = pair.first;
            break;
          }
        }
        
        if (opdet_id >= 0) {
          auto it = _v12_opdet_positions.find(opdet_id);
          if (it != _v12_opdet_positions.end()) {
            // Convert from global to TPC coordinates
            std::vector<float> global_pos = it->second;
            _pmt_positions[channel] = {
              global_pos[0] - _tpc_origin[0],
              global_pos[1] - _tpc_origin[1], 
              global_pos[2] - _tpc_origin[2]
            };
          } else {
            LARCV_WARNING() << "No V12 position found for OpDet " << opdet_id << std::endl;
            _pmt_positions[channel] = {0.0f, 0.0f, 0.0f};
          }
        } else {
          LARCV_WARNING() << "No OpDet mapping found for channel " << channel << std::endl;
          _pmt_positions[channel] = {0.0f, 0.0f, 0.0f};
        }
      }
    }

    _initialized = true;
    
    LARCV_INFO() << "Initialized PMT positions for geometry version " 
                 << (_geometry_version == kV4 ? "V4" : "V12") << std::endl;
  }

  void SinkhornFlashDivergence::calculateCostMatrix() {
    if (!_initialized) {
      LARCV_ERROR() << "PMT positions not initialized!" << std::endl;
      return;
    }

    _cost_matrix.clear();
    _cost_matrix.resize(32, std::vector<float>(32, 0.0f));
    float scalefactor = 1.0/_dist_scale;
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        _cost_matrix[i][j] = scalefactor*calculateDistance(_pmt_positions[i], _pmt_positions[j]);
      }
    }

    LARCV_DEBUG() << "Calculated 32x32 cost matrix based on PMT distances" << std::endl;
  }

  float SinkhornFlashDivergence::calculateDistance(
    const std::vector<float>& pos1, 
    const std::vector<float>& pos2
  ) const {
    float dx = pos1[0] - pos2[0];
    float dy = pos1[1] - pos2[1]; 
    float dz = pos1[2] - pos2[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }

  std::vector<float> SinkhornFlashDivergence::normalizeDistribution(
    const std::vector<float>& distribution
  ) const {
    float sum = std::accumulate(distribution.begin(), distribution.end(), 0.0f);
    
    if (sum <= 0.0f) {
      LARCV_WARNING() << "Distribution sums to " << sum << ", using uniform distribution" << std::endl;
      return std::vector<float>(distribution.size(), 1.0f / distribution.size());
    }

    std::vector<float> normalized(distribution.size());
    for (size_t i = 0; i < distribution.size(); i++) {
      normalized[i] = std::max(0.0f, distribution[i]) / sum;
    }
    
    return normalized;
  }

  float SinkhornFlashDivergence::calculateOptimalTransportCost(
    const std::vector<float>& distribution_a,
    const std::vector<float>& distribution_b,
    float regularization,
    int max_iterations,
    float tolerance
  ) {
    if (distribution_a.size() != 32 || distribution_b.size() != 32) {
      LARCV_ERROR() << "Distributions must have exactly 32 elements (PMT channels)" << std::endl;
      return -1.0f;
    }

    // Normalize distributions
    std::vector<float> mu = normalizeDistribution(distribution_a);
    std::vector<float> nu = normalizeDistribution(distribution_b);

    // Initialize dual variables (log-domain for numerical stability)
    std::vector<float> log_u(32, 0.0f);
    std::vector<float> log_v(32, 0.0f);

    // Pre-compute exp(-C/λ) matrix for efficiency
    std::vector<std::vector<float>> K(32, std::vector<float>(32));
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        K[i][j] = std::exp(-_cost_matrix[i][j] / regularization);
      }
    }

    _last_iterations = 0;
    _last_converged = false;

    // Sinkhorn iterations
    for (int iter = 0; iter < max_iterations; iter++) {
      std::vector<float> old_log_u = log_u;

      // Update u: log_u[i] = log(mu[i]) - log(sum_j(K[i][j] * exp(log_v[j])))
      for (int i = 0; i < 32; i++) {
        float sum_kv = 0.0f;
        for (int j = 0; j < 32; j++) {
          sum_kv += K[i][j] * std::exp(log_v[j]);
        }
        if (sum_kv > 0.0f) {
          log_u[i] = std::log(mu[i] + 1e-12f) - std::log(sum_kv);
        }
      }

      // Update v: log_v[j] = log(nu[j]) - log(sum_i(K[i][j] * exp(log_u[i])))
      for (int j = 0; j < 32; j++) {
        float sum_ku = 0.0f;
        for (int i = 0; i < 32; i++) {
          sum_ku += K[i][j] * std::exp(log_u[i]);
        }
        if (sum_ku > 0.0f) {
          log_v[j] = std::log(nu[j] + 1e-12f) - std::log(sum_ku);
        }
      }

      // Check convergence
      float max_change = 0.0f;
      for (int i = 0; i < 32; i++) {
        max_change = std::max(max_change, std::abs(log_u[i] - old_log_u[i]));
      }

      _last_iterations = iter + 1;
      
      if (max_change < tolerance) {
        _last_converged = true;
        LARCV_DEBUG() << "Sinkhorn converged after " << _last_iterations << " iterations" << std::endl;
        break;
      }
    }

    if (!_last_converged) {
      LARCV_WARNING() << "Sinkhorn did not converge after " << max_iterations << " iterations" 
                      << " for transport plan "
                      << " ( with regularization par = " << regularization << ")" 
                      << std::endl;
    }

    // Calculate optimal transport cost
    float cost = 0.0f;
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        float pi_ij = std::exp(log_u[i] + log_v[j]) * K[i][j];
        cost += pi_ij * _cost_matrix[i][j]*_dist_scale;
      }
    }

    return cost;
  }

  float SinkhornFlashDivergence::calculateDivergence(
    const std::vector<float>& predicted_pe,
    const std::vector<float>& observed_pe,
    float regularization,
    int max_iterations,
    float tolerance
  ) {
    if (predicted_pe.size() != 32 || observed_pe.size() != 32) {
      LARCV_ERROR() << "PE vectors must have exactly 32 elements (PMT channels)" << std::endl;
      return -1.0f;
    }

    LARCV_DEBUG() << "Calculating Sinkhorn divergence with regularization=" << regularization
                  << ", max_iterations=" << max_iterations << ", tolerance=" << tolerance << std::endl;

    // Calculate three optimal transport costs for symmetric divergence
    float cost_pred_obs  = calculateOptimalTransportCost(predicted_pe, observed_pe, regularization, max_iterations, tolerance);
    float cost_pred_pred = calculateOptimalTransportCost(predicted_pe, predicted_pe, regularization, max_iterations, tolerance);
    float cost_obs_obs   = calculateOptimalTransportCost(observed_pe, observed_pe, regularization, max_iterations, tolerance);

    // Symmetric Sinkhorn divergence: S_λ(μ,ν) = OT_λ(μ,ν) - 0.5*OT_λ(μ,μ) - 0.5*OT_λ(ν,ν)
    float divergence = cost_pred_obs - 0.5f * cost_pred_pred - 0.5f * cost_obs_obs;

    LARCV_INFO() << "Sinkhorn divergence calculation: OT(pred,obs)=" << cost_pred_obs
                 << ", OT(pred,pred)=" << cost_pred_pred << ", OT(obs,obs)=" << cost_obs_obs
                 << " => divergence=" << divergence << std::endl;

    return divergence;
  }

  std::vector<std::vector<float>> SinkhornFlashDivergence::calculateTransportPlan(
    const std::vector<float>& distribution_a,
    const std::vector<float>& distribution_b,
    float regularization,
    int max_iterations,
    float tolerance
  ) {
    if (distribution_a.size() != 32 || distribution_b.size() != 32) {
      LARCV_ERROR() << "Distributions must have exactly 32 elements (PMT channels)" << std::endl;
      return std::vector<std::vector<float>>(32, std::vector<float>(32, 0.0f));
    }

    // Normalize distributions
    std::vector<float> mu = normalizeDistribution(distribution_a);
    std::vector<float> nu = normalizeDistribution(distribution_b);

    // Initialize dual variables (log-domain for numerical stability)
    std::vector<float> log_u(32, 0.0f);
    std::vector<float> log_v(32, 0.0f);

    // Pre-compute exp(-C/λ) matrix for efficiency
    std::vector<std::vector<float>> K(32, std::vector<float>(32));
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        K[i][j] = std::exp(-_cost_matrix[i][j] / regularization);
      }
    }

    _last_iterations = 0;
    _last_converged = false;

    // Sinkhorn iterations (same as in calculateOptimalTransportCost)
    for (int iter = 0; iter < max_iterations; iter++) {
      std::vector<float> old_log_u = log_u;

      // Update u: log_u[i] = log(mu[i]) - log(sum_j(K[i][j] * exp(log_v[j])))
      for (int i = 0; i < 32; i++) {
        float sum_kv = 0.0f;
        for (int j = 0; j < 32; j++) {
          sum_kv += K[i][j] * std::exp(log_v[j]);
        }
        if (sum_kv > 0.0f) {
          log_u[i] = std::log(mu[i] + 1e-12f) - std::log(sum_kv);
        }
      }

      // Update v: log_v[j] = log(nu[j]) - log(sum_i(K[i][j] * exp(log_u[i])))
      for (int j = 0; j < 32; j++) {
        float sum_ku = 0.0f;
        for (int i = 0; i < 32; i++) {
          sum_ku += K[i][j] * std::exp(log_u[i]);
        }
        if (sum_ku > 0.0f) {
          log_v[j] = std::log(nu[j] + 1e-12f) - std::log(sum_ku);
        }
      }

      // Check convergence
      float max_change = 0.0f;
      for (int i = 0; i < 32; i++) {
        max_change = std::max(max_change, std::abs(log_u[i] - old_log_u[i]));
      }

      _last_iterations = iter + 1;
      
      if (max_change < tolerance) {
        _last_converged = true;
        LARCV_DEBUG() << "Sinkhorn converged after " << _last_iterations << " iterations for transport plan" << std::endl;
        break;
      }
    }

    if (!_last_converged) {
      LARCV_WARNING() << "Sinkhorn did not converge after " << max_iterations << " iterations"
                      << " for transport plan "
                      << " ( with regularization par = " << regularization << ")" 
                      << std::endl;
    }

    // Construct transport plan π[i][j] = exp(log_u[i] + log_v[j]) * K[i][j]
    std::vector<std::vector<float>> transport_plan(32, std::vector<float>(32, 0.0f));
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        transport_plan[i][j] = std::exp(log_u[i] + log_v[j]) * K[i][j];
      }
    }

    LARCV_INFO() << "Calculated transport plan with " << _last_iterations << " iterations" << std::endl;

    return transport_plan;
  }

  std::vector<float> SinkhornFlashDivergence::getFlowsFromPMT(
    const std::vector<std::vector<float>>& transport_plan,
    int source_pmt
  ) const {
    if (source_pmt < 0 || source_pmt >= 32) {
      LARCV_ERROR() << "Invalid source PMT: " << source_pmt << " (must be 0-31)" << std::endl;
      return std::vector<float>(32, 0.0f);
    }

    if (transport_plan.size() != 32 || transport_plan[0].size() != 32) {
      LARCV_ERROR() << "Transport plan must be 32x32 matrix" << std::endl;
      return std::vector<float>(32, 0.0f);
    }

    // Return row source_pmt (flows from source_pmt to all targets)
    return transport_plan[source_pmt];
  }

  std::vector<float> SinkhornFlashDivergence::getFlowsToPMT(
    const std::vector<std::vector<float>>& transport_plan,
    int target_pmt
  ) const {
    if (target_pmt < 0 || target_pmt >= 32) {
      LARCV_ERROR() << "Invalid target PMT: " << target_pmt << " (must be 0-31)" << std::endl;
      return std::vector<float>(32, 0.0f);
    }

    if (transport_plan.size() != 32 || transport_plan[0].size() != 32) {
      LARCV_ERROR() << "Transport plan must be 32x32 matrix" << std::endl;
      return std::vector<float>(32, 0.0f);
    }

    // Extract column target_pmt (flows from all sources to target_pmt)
    std::vector<float> flows_to_target(32);
    for (int i = 0; i < 32; i++) {
      flows_to_target[i] = transport_plan[i][target_pmt];
    }

    return flows_to_target;
  }

  std::vector<float> SinkhornFlashDivergence::getPMTPosition(int channel) const {
    if (channel < 0 || channel >= 32) {
      LARCV_ERROR() << "Invalid PMT channel: " << channel << " (must be 0-31)" << std::endl;
      return {0.0f, 0.0f, 0.0f};
    }

    if (!_initialized) {
      LARCV_ERROR() << "PMT positions not initialized!" << std::endl;
      return {0.0f, 0.0f, 0.0f};
    }

    return _pmt_positions[channel];
  }

}
}