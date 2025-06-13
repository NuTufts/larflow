#include "NuVertexFlashPrediction.h"

#include "ublarcvapp/UBPhotonLib/PhotonVisibilityEstimator.h"
#include "ublarcvapp/ParticleToPixelUtils/TrackToSpacePoints.h"
#include "ublarcvapp/ParticleToPixelUtils/ShowerToSpacePoints.h"

namespace larflow {
namespace reco {

  NuVertexFlashPrediction::NuVertexFlashPrediction()
    : larcv::larcv_base("NuVertexFlashPrediction"),
      _num_tracks_processed(0),
      _num_showers_processed(0),
      _total_charge_collected(0.0),
      _total_photons_emitted(0.0),
      _adc_per_electron(200.0),
      _mev_per_electron(23.6e-6),
      _photons_per_mev(24000.0),
      _recombination_factor(0.7),
      _track_dcol(3),
      _track_drow(3),
      _track_minstepsize(0.3),
      _track_maxstepsize(0.5),
      _shower_dcol(3),
      _shower_drow(3)
  {
  }

  NuVertexFlashPrediction::~NuVertexFlashPrediction() {
  }

  larlite::opflash NuVertexFlashPrediction::predictFlash(
    const NuVertexCandidate& vertex_candidate,
    const std::vector<larcv::Image2D>& adc_v,
    const float threshold,
    const bool use_trilinear,
    const bool primary_prongs_only
  ) {
    
    LARCV_DEBUG() << "Starting flash prediction for vertex candidate with " 
                  << vertex_candidate.track_v.size() << " tracks and "
                  << vertex_candidate.shower_v.size() << " showers" << std::endl;
    
    // Clear previous results
    _predicted_pe.clear();
    _particle_contributions.clear();
    _num_tracks_processed = 0;
    _num_showers_processed = 0;
    _total_charge_collected = 0.0;
    _total_photons_emitted = 0.0;
    
    // Initialize predicted PE for all optical channels (32 for MicroBooNE)
    for (int ich = 0; ich < 32; ich++) {
      _predicted_pe[ich] = 0.0;
    }
    
    // Process tracks
    processTracks(vertex_candidate, adc_v, threshold, use_trilinear, primary_prongs_only);
    
    // Process showers  
    processShowers(vertex_candidate, adc_v, threshold, use_trilinear, primary_prongs_only);
    
    // Create opflash object with predicted values
    std::vector<double> pe_per_opdet;
    pe_per_opdet.reserve(32);
    for (int ich = 0; ich < 32; ich++) {
      pe_per_opdet.push_back(_predicted_pe[ich]);
    }
    
    // Create flash with time=0, timewidth=1, abstime=0, frame=0
    // These values should be set appropriately based on the vertex candidate timing
    larlite::opflash predicted_flash(
      0.0,         // time
      1.0,         // timewidth  
      0.0,         // abstime
      0,           // frame
      pe_per_opdet, // PE per optical detector
      false,       // InBeamFrame
      0,           // OnBeamTime
      1.0,         // FastToTotal
      0.0,         // yCenter (will be calculated by opflash)
      0.0,         // yWidth
      0.0,         // zCenter
      0.0          // zWidth
    );
    
    LARCV_INFO() << "Flash prediction complete: " 
                 << "Total PE = " << getTotalPredictedPE()
                 << ", Tracks processed = " << _num_tracks_processed
                 << ", Showers processed = " << _num_showers_processed
                 << ", Total charge = " << _total_charge_collected
                 << ", Total photons = " << _total_photons_emitted << std::endl;
    
    return predicted_flash;
  }

  float NuVertexFlashPrediction::getTotalPredictedPE() const {
    float total = 0.0;
    for (const auto& pe_pair : _predicted_pe) {
      total += pe_pair.second;
    }
    return total;
  }

  void NuVertexFlashPrediction::setChargeToPhotonParams(
    float adc_per_electron,
    float mev_per_electron,
    float photons_per_mev,
    float recombination_factor
  ) {
    _adc_per_electron = adc_per_electron;
    _mev_per_electron = mev_per_electron;
    _photons_per_mev = photons_per_mev;
    _recombination_factor = recombination_factor;
    
    LARCV_DEBUG() << "Charge-to-photon parameters updated: "
                  << "adc_per_electron=" << _adc_per_electron
                  << ", mev_per_electron=" << _mev_per_electron
                  << ", photons_per_mev=" << _photons_per_mev 
                  << ", recombination_factor=" << _recombination_factor << std::endl;
  }

  void NuVertexFlashPrediction::setTrackConversionParams(
    int dcol,
    int drow,
    float minstepsize,
    float maxstepsize
  ) {
    _track_dcol = dcol;
    _track_drow = drow;
    _track_minstepsize = minstepsize;
    _track_maxstepsize = maxstepsize;
    
    LARCV_DEBUG() << "Track conversion parameters updated: "
                  << "dcol=" << _track_dcol << ", drow=" << _track_drow
                  << ", minstepsize=" << _track_minstepsize
                  << ", maxstepsize=" << _track_maxstepsize << std::endl;
  }

  void NuVertexFlashPrediction::setShowerConversionParams(
    int dcol,
    int drow
  ) {
    _shower_dcol = dcol;
    _shower_drow = drow;
    
    LARCV_DEBUG() << "Shower conversion parameters updated: "
                  << "dcol=" << _shower_dcol << ", drow=" << _shower_drow << std::endl;
  }

  float NuVertexFlashPrediction::convertChargeToPhotons(float adc_charge) const {
    // Convert: ADC -> electrons -> energy -> photons
    float n_electrons = adc_charge / _adc_per_electron;
    float energy_mev = n_electrons * _mev_per_electron;
    float n_photons = energy_mev * _photons_per_mev * (1.0 - _recombination_factor);
    
    return n_photons;
  }

  void NuVertexFlashPrediction::processTracks(
    const NuVertexCandidate& vertex_candidate,
    const std::vector<larcv::Image2D>& adc_v,
    const float threshold,
    const bool use_trilinear,
    const bool primary_prongs_only
  ) {
    
    LARCV_DEBUG() << "Processing " << vertex_candidate.track_v.size() << " tracks" << std::endl;
    
    // Create track-to-spacepoint converter
    ublarcvapp::pixelutils::TrackToSpacePoints track_converter;
    track_converter.setUseChargeWeighting(false); // Use geometric positions
    
    for (size_t itrack = 0; itrack < vertex_candidate.track_v.size(); itrack++) {
      const auto& track = vertex_candidate.track_v[itrack];
      
      // Check for valid track before processing
      int num_points = track.NumberTrajectoryPoints();
      if (num_points < 2 || num_points > 100000) {
        LARCV_WARNING() << "Skipping track " << itrack 
                        << " with invalid number of trajectory points: " << num_points << std::endl;
        continue;
      }

      if ( primary_prongs_only && vertex_candidate.track_isSecondary_v.at(itrack)==1 ) {
        LARCV_DEBUG() << "Skipping seconday track (index=" << itrack << ")" << std::endl;
      }
      
      LARCV_DEBUG() << "Processing track " << itrack 
                    << " with " << num_points << " trajectory points" << std::endl;
      
      // Create individual contribution tracker
      ParticleContribution track_contrib("track", itrack);
      
      // Initialize PE per PMT for this track
      for (int ich = 0; ich < 32; ich++) {
        track_contrib.pe_per_pmt[ich] = 0.0;
      }
      
      // Convert track to space points with charge - wrap in try/catch
      std::vector<ublarcvapp::pixelutils::TrackToSpacePoints::SpacePointCharge> spacepoints;
      try {
        spacepoints = track_converter.convertTrack(
          track, 
          adc_v,
          threshold,
          _track_dcol,
          _track_drow,
          _track_minstepsize,
          _track_maxstepsize
        );
      } catch (const std::exception& e) {
        LARCV_WARNING() << "Error converting track " << itrack << " to space points: " 
                        << e.what() << ". Skipping this track." << std::endl;
        continue;
      }
      
      track_contrib.num_spacepoints = spacepoints.size();
      
      LARCV_DEBUG() << "Track " << itrack << " converted to " << spacepoints.size() << " space points" << std::endl;
      
      // Process this track separately with its own photon estimator
      if (spacepoints.size() > 0) {
        ublarcvapp::ubphotonlib::PhotonVisibilityEstimator track_photon_estimator;
        
        // Add photon sources for each space point
        for (const auto& sp : spacepoints) {
          if (sp.charge > 0.0) {
            float n_photons = convertChargeToPhotons(sp.charge);
            
            track_photon_estimator.addPhotonSource(
              sp.position.X(),
              sp.position.Y(),
              sp.position.Z(),
              n_photons
            );
            
            track_contrib.charge_collected += sp.charge;
            track_contrib.photons_emitted += n_photons;
            _total_charge_collected += sp.charge;
            _total_photons_emitted += n_photons;
          }
        }
        
        // Calculate detected photons for this track
        if (track_photon_estimator.getNumSources() > 0) {
          auto track_photons_per_pmt = track_photon_estimator.calculateDetectedPhotons(use_trilinear);
          
          // Store individual track contribution and add to total
          for (const auto& pmt_pe : track_photons_per_pmt) {
            track_contrib.pe_per_pmt[pmt_pe.first] = pmt_pe.second;
            track_contrib.total_pe += pmt_pe.second;
            _predicted_pe[pmt_pe.first] += pmt_pe.second;
          }
          
          LARCV_DEBUG() << "Track " << itrack << " contributes " << track_contrib.total_pe 
                        << " total PE from " << track_photon_estimator.getNumSources() 
                        << " photon sources" << std::endl;
        }
      }
      
      // Store this track's contribution
      _particle_contributions.push_back(track_contrib);
      _num_tracks_processed++;
    }
  }

  void NuVertexFlashPrediction::processShowers(
    const NuVertexCandidate& vertex_candidate,
    const std::vector<larcv::Image2D>& adc_v,
    const float threshold,
    const bool use_trilinear,
    const bool primary_prongs_only
  ) {
    
    LARCV_DEBUG() << "Processing " << vertex_candidate.shower_v.size() << " showers" << std::endl;
    
    // Create shower-to-spacepoint converter
    ublarcvapp::pixelutils::ShowerToSpacePoints shower_converter;
    shower_converter.setUseChargeWeighting(false); // Use geometric positions
    
    for (size_t ishower = 0; ishower < vertex_candidate.shower_v.size(); ishower++) {
      const auto& shower = vertex_candidate.shower_v[ishower];
      
      // Check for valid shower before processing
      if (shower.size() < 2) {
        LARCV_WARNING() << "Skipping shower " << ishower 
                        << " with too few hits: " << shower.size() << std::endl;
        continue;
      }

      if ( primary_prongs_only && vertex_candidate.shower_isSecondary_v.at(ishower)==1 ) {
        LARCV_DEBUG() << "Skipping seconday shower (index=" << ishower << ")" << std::endl;
      }
      
      LARCV_DEBUG() << "Processing shower " << ishower 
                    << " with " << shower.size() << " hits" << std::endl;
      
      // Create individual contribution tracker
      ParticleContribution shower_contrib("shower", ishower);
      
      // Initialize PE per PMT for this shower
      for (int ich = 0; ich < 32; ich++) {
        shower_contrib.pe_per_pmt[ich] = 0.0;
      }
      
      // Convert shower to space points with charge - wrap in try/catch
      std::vector<ublarcvapp::pixelutils::ShowerToSpacePoints::SpacePointCharge> spacepoints;
      try {
        spacepoints = shower_converter.convertShower(
          shower,
          adc_v,
          threshold,
          _shower_dcol,
          _shower_drow
        );
      } catch (const std::exception& e) {
        LARCV_WARNING() << "Error converting shower " << ishower << " to space points: " 
                        << e.what() << ". Skipping this shower." << std::endl;
        continue;
      }
      
      shower_contrib.num_spacepoints = spacepoints.size();
      
      LARCV_DEBUG() << "Shower " << ishower << " converted to " << spacepoints.size() << " space points" << std::endl;
      
      // Process this shower separately with its own photon estimator
      if (spacepoints.size() > 0) {
        ublarcvapp::ubphotonlib::PhotonVisibilityEstimator shower_photon_estimator;
        
        // Add photon sources for each space point
        for (const auto& sp : spacepoints) {
          if (sp.charge > 0.0) {
            float n_photons = convertChargeToPhotons(sp.charge);
            
            shower_photon_estimator.addPhotonSource(
              sp.position.X(),
              sp.position.Y(),
              sp.position.Z(),
              n_photons
            );
            
            shower_contrib.charge_collected += sp.charge;
            shower_contrib.photons_emitted += n_photons;
            _total_charge_collected += sp.charge;
            _total_photons_emitted += n_photons;
          }
        }
        
        // Calculate detected photons for this shower
        if (shower_photon_estimator.getNumSources() > 0) {
          auto shower_photons_per_pmt = shower_photon_estimator.calculateDetectedPhotons(use_trilinear);
          
          // Store individual shower contribution and add to total
          for (const auto& pmt_pe : shower_photons_per_pmt) {
            shower_contrib.pe_per_pmt[pmt_pe.first] = pmt_pe.second;
            shower_contrib.total_pe += pmt_pe.second;
            _predicted_pe[pmt_pe.first] += pmt_pe.second;
          }
          
          LARCV_DEBUG() << "Shower " << ishower << " contributes " << shower_contrib.total_pe 
                        << " total PE from " << shower_photon_estimator.getNumSources() 
                        << " photon sources" << std::endl;
        }
      }
      
      // Store this shower's contribution
      _particle_contributions.push_back(shower_contrib);
      _num_showers_processed++;
    }
  }

}
}
