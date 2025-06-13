#ifndef __LARFLOW_RECO_NUVERTEX_FLASH_PREDICTION_H__
#define __LARFLOW_RECO_NUVERTEX_FLASH_PREDICTION_H__

/**
 * @brief Estimate amount of light (PE) in each optical detector based on the reconstructed prongs in a NuVertexCandidate
 *
 * We use the tools in ublarcvapp/UBPhotonLib and ublarcvapp/ParticleToPixelUtils to
 * generate a prediction of the intime flash.
 *
 */

#include <vector>
#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/opflash.h"
#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class NuVertexFlashPrediction : public larcv::larcv_base {
    
  public:
    
    /**
     * @brief Default constructor
     */
    NuVertexFlashPrediction();
    
    /**
     * @brief Destructor
     */
    virtual ~NuVertexFlashPrediction();
    
    /**
     * @brief Predict optical flash from neutrino vertex candidate
     * @param vertex_candidate The NuVertexCandidate containing tracks and showers
     * @param adc_v Vector of ADC images for each wire plane
     * @param threshold ADC threshold for pixel selection (default: 10.0)
     * @param use_trilinear Use trilinear interpolation in photon library (default: true)
     * @return Predicted larlite::opflash object
     */
    larlite::opflash predictFlash(
      const NuVertexCandidate& vertex_candidate,
      const std::vector<larcv::Image2D>& adc_v,
      const float threshold = 10.0,
      const bool use_trilinear = true
    );
    
    /**
     * @brief Structure to hold individual particle contribution information
     */
    struct ParticleContribution {
      std::string type;                     ///< "track" or "shower"
      int index;                            ///< Index in respective container
      std::map<int, float> pe_per_pmt;      ///< PE contribution per optical channel
      float total_pe;                       ///< Total PE from this particle
      float charge_collected;               ///< Total charge collected (ADC)
      float photons_emitted;                ///< Total photons emitted
      int num_spacepoints;                  ///< Number of space points processed
      
      ParticleContribution() 
        : type(""), index(-1), total_pe(0.0), charge_collected(0.0), 
          photons_emitted(0.0), num_spacepoints(0) {}
          
      ParticleContribution(const std::string& particle_type, int particle_index)
        : type(particle_type), index(particle_index), total_pe(0.0), 
          charge_collected(0.0), photons_emitted(0.0), num_spacepoints(0) {}
    };
    
    /**
     * @brief Get detailed prediction information
     * @return Map from optical channel ID to predicted photoelectrons
     */
    const std::map<int, float>& getPredictedPE() const { return _predicted_pe; }
    
    /**
     * @brief Get individual particle contributions
     * @return Vector of ParticleContribution objects
     */
    const std::vector<ParticleContribution>& getParticleContributions() const { 
      return _particle_contributions; 
    }
    
    /**
     * @brief Get total predicted photoelectrons
     * @return Total PE across all optical detectors
     */
    float getTotalPredictedPE() const;
    
    /**
     * @brief Get number of tracks processed in last prediction
     */
    int getNumTracksProcessed() const { return _num_tracks_processed; }
    
    /**
     * @brief Get number of showers processed in last prediction
     */
    int getNumShowersProcessed() const { return _num_showers_processed; }
    
    /**
     * @brief Get total charge collected from all tracks and showers
     */
    float getTotalChargeCollected() const { return _total_charge_collected; }
    
    /**
     * @brief Get total photons emitted from all sources
     */
    float getTotalPhotonsEmitted() const { return _total_photons_emitted; }
    
    /**
     * @brief Set charge-to-photon conversion parameters
     * @param adc_per_electron ADC counts per electron (default: 200.0)
     * @param mev_per_electron Ionization energy in LAr (MeV, default: 23.6e-6)
     * @param photons_per_mev Scintillation photons per MeV (default: 24000.0)
     * @param recombination_factor Fraction surviving recombination (default: 0.7)
     */
    void setChargeToPhotonParams(
      float adc_per_electron = 200.0,
      float mev_per_electron = 23.6e-6,
      float photons_per_mev = 24000.0,
      float recombination_factor = 0.7
    );
    
    /**
     * @brief Set track-to-spacepoint conversion parameters
     * @param dcol Column window around projected wire position (default: 3)
     * @param drow Row window around projected tick position (default: 3)
     * @param minstepsize Minimum step size along track in cm (default: 0.3)
     * @param maxstepsize Maximum step size along track in cm (default: 0.5)
     */
    void setTrackConversionParams(
      int dcol = 3,
      int drow = 3,
      float minstepsize = 0.3,
      float maxstepsize = 0.5
    );
    
    /**
     * @brief Set shower-to-spacepoint conversion parameters
     * @param dcol Column window around projected position (default: 3)
     * @param drow Row window around projected position (default: 3)
     */
    void setShowerConversionParams(
      int dcol = 3,
      int drow = 3
    );
    
  protected:
    
    /**
     * @brief Convert ADC charge to number of scintillation photons
     * @param adc_charge Charge in ADC counts
     * @return Number of scintillation photons
     */
    float convertChargeToPhotons(float adc_charge) const;
    
    /**
     * @brief Process tracks from vertex candidate
     * @param vertex_candidate The NuVertexCandidate
     * @param adc_v Vector of ADC images
     * @param threshold ADC threshold
     * @param use_trilinear Use trilinear interpolation
     */
    void processTracks(
      const NuVertexCandidate& vertex_candidate,
      const std::vector<larcv::Image2D>& adc_v,
      const float threshold,
      const bool use_trilinear
    );
    
    /**
     * @brief Process showers from vertex candidate
     * @param vertex_candidate The NuVertexCandidate
     * @param adc_v Vector of ADC images
     * @param threshold ADC threshold
     * @param use_trilinear Use trilinear interpolation
     */
    void processShowers(
      const NuVertexCandidate& vertex_candidate,
      const std::vector<larcv::Image2D>& adc_v,
      const float threshold,
      const bool use_trilinear
    );
    
  private:
    
    // Prediction results
    std::map<int, float> _predicted_pe;              ///< Predicted PE per optical channel
    std::vector<ParticleContribution> _particle_contributions; ///< Individual particle contributions
    int _num_tracks_processed;                       ///< Number of tracks processed
    int _num_showers_processed;                      ///< Number of showers processed
    float _total_charge_collected;                   ///< Total charge collected (ADC)
    float _total_photons_emitted;                    ///< Total photons emitted
    
    // Charge-to-photon conversion parameters
    float _adc_per_electron;                 ///< ADC counts per electron
    float _mev_per_electron;                 ///< Ionization energy in LAr (MeV)
    float _photons_per_mev;                  ///< Scintillation photons per MeV
    float _recombination_factor;             ///< Fraction surviving recombination
    
    // Track conversion parameters
    int _track_dcol;                         ///< Track column window
    int _track_drow;                         ///< Track row window
    float _track_minstepsize;                ///< Track minimum step size
    float _track_maxstepsize;                ///< Track maximum step size
    
    // Shower conversion parameters
    int _shower_dcol;                        ///< Shower column window
    int _shower_drow;                        ///< Shower row window
    
  };
  
}
}

#endif
