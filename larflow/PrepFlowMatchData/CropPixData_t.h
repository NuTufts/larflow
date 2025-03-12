#ifndef __LARFLOW_PREP_CROPPIXDATA_T_HH__
#define __LARFLOW_PREP_CROPPIXDATA_T_HH__

namespace larflow {
  namespace prep {
    
    /** 
     * @struct CropPixData_t
     *
     * @brief internal struct to represent to a pixel in cropped image and provide sorting method
     *
     */
    class CropPixData_t  {

    public:
      
    CropPixData_t()
      : row(0),col(0),rawRow(0),rawCol(0),val(0.0),idx(0)
	{};
      
      /** @brief constructor with row, col, value 
       *  @param[in] r row of pixel
       *  @param[in] c col of pixel
       *  @param[in] v value of pixel
       */      
    CropPixData_t( int r, int c, int rr, int rc, float v)
      : row(r),col(c),rawRow(rr),rawCol(rc),val(v),idx(0) {};

      virtual ~CropPixData_t() {};
      
      int row; ///< row of pixel in cropped image
      int col; ///< col of pixel in cropped image
      int rawRow; ///< row of pixel in original image
      int rawCol; ///< col of pixel in original image
      float val; ///< value of pixel
      int idx;   ///< index in container

      /** @brief equality operator based on row, col, and value */
      bool operator==( const CropPixData_t& rhs ) const {
        if(rawRow == rhs.rawRow && rawCol == rhs.rawCol && val == rhs.val) return true;
        return false;
      };
      
      /** @brief comparator based on row then col then value */
      bool operator<( const CropPixData_t& rhs ) const {
        if (row<rhs.row) return true;
        if ( row==rhs.row ) {
          if ( col<rhs.col ) return true;
          if ( col==rhs.col ) {
            if ( val<rhs.val ) return true;
          }
        }
        return false;
      };
    };

  }
}

#endif
