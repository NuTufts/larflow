#ifndef __LARFLOW_PREP_PIXDATA_T_H__
#define __LARFLOW_PREP_PIXDATA_T_H__

namespace larflow {
namespace prep {

  /** 
   * @struct PixData_t
   *
   * @brief internal struct to represent to a pixel and provide sorting method
   *
   */
  class PixData_t {

  public:
  
  PixData_t()
    : row(0),col(0),val(0.0),idx(0)
      {};

    /** @brief constructor with row, col, value 
     *  @param[in] r row of pixel
     *  @param[in] c col of pixel
     *  @param[in] v value of pixel
     */      
  PixData_t( int r, int c, float v)
    : row(r),col(c),val(v),idx(0) 
      {};

    virtual ~PixData_t() {};
    
    int row; ///< row of pixel in image
    int col; ///< col of pixel in image
    float val; ///< value of pixel
    int idx;   ///< index in container

    /** @brief comparator based on row then col then value */
    bool operator<( const PixData_t& rhs ) const {
      if (row<rhs.row) return true;
      if ( row==rhs.row ) {
	if ( col<rhs.col ) return true;
	if ( col==rhs.col ) {
	  if ( val<rhs.val ) return true;
	}
      }
      return false;
    }
  };
  

}
}

#endif
