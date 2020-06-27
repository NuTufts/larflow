{
  // The units are cm
  // index from 1 to 10 for position dependece. See the talk
  /* https://microboone-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=26423 */

  /// YX view has Z dependence: Z sub-range from 0 to 10m per 1m
  double YX_TOP_y1_array     = 116;
  double YX_TOP_x1_array[11] = {0, 150.00, 132.56, 122.86, 119.46, 114.22, 110.90, 115.85, 113.48, 126.36, 144.21};
  double YX_TOP_y2_array[11] = {0, 110.00, 108.14, 106.77, 105.30, 103.40, 102.18, 101.76, 102.27, 102.75, 105.10};
  double YX_TOP_x2_array = 256;
    
  double YX_BOT_y1_array     = -115;
  double YX_BOT_x1_array[11] = {0, 115.71, 98.05, 92.42, 91.14, 92.25, 85.38, 78.19, 74.46, 78.86, 108.90};
  double YX_BOT_y2_array[11] = {0, -101.72, -99.46, -99.51, -100.43, -99.55, -98.56, -98.00, -98.30, -99.32, -104.20};
  double YX_BOT_x2_array = 256;

  /// ZX view has Y dependence: Y sub-range from -116 to 116cm per 24cm
  double ZX_Up_z1_array = 0;
  double ZX_Up_x1_array = 120;
  double ZX_Up_z2_array = 11;
  double ZX_Up_x2_array = 256;
    
  double ZX_Dw_z1_array     = 1037;
  double ZX_Dw_x1_array[11] = {0, 120.00, 115.24, 108.50, 110.67, 120.90, 126.43, 140.51, 157.15, 120.00, 120.00};
  double ZX_Dw_z2_array[11] = {0, 1029.00, 1029.12, 1027.21, 1026.01, 1024.91, 1025.27, 1025.32, 1027.61, 1026.00, 1026.00};
  double ZX_Dw_x2_array     = 256;
}
