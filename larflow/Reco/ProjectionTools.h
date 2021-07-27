#ifndef __LARFLOW_RECO_PROJECTION_TOOLS_H__
#define __LARFLOW_RECO_PROJECTION_TOOLS_H__

namespace larflow {
namespace reco {

  class ProjectionTools {
  pubic:

    ProjectionTools(){};
    virtual ~ProjectionTools(){};

    float wirecoordinate_and_grad( const std::vector<float>& pos, int plane );
    
  };
  
}
}

#endif
