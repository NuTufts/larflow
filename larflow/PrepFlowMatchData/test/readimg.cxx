//event display code (from Katie)

  #include "TH2F.h"
  #include "TCanvas.h"
  #include "TStyle.h"

int main(int nargs, char** argv) {

  std::string inputFile = argv[1]; // full path of file I want to look at
  
  auto const& wire_img = ev_img->Image2DArray()
  auto const& wireu_meta = wire_img.at(0).meta();
  
  TH2F* adc_image_u = new TH2F("adc_u","adc_u",3456,0.,3456,1008,0.,1008.);
  
  for(int row=0; row<wireu_meta.rows(); row++){
    for(int col=0; col<wireu_meta.cols(); col++){
      float adc_pix=wire_img.at(0).pixel(row,col);
      if(adc_pix >=10) adc_image_u->SetBinContent(col+1,row+1,adc_pix);
    }
  }
  gStyle->SetOptStat(0);
  TCanvas can1("can", "histograms ", 3456, 1008);
  adc_image_u->Draw("colz");
  can1.SaveAs("Images/old_u_6.png");


  return 0;

}
