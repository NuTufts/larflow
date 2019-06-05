import os,sys

def default_processor_config(input_adc_producer,input_chstatus_producer):
    """
    Make a processor that splits and crops for larflow.
    """
    processor_cfg="""ProcessDriver: {
    Verbosity: 0
    EnableFilter: false
    RandomAccess: false
    ProcessType: ["UBSplitDetector","UBCropLArFlow"]
    ProcessName: ["Split","Crop"]

    IOManager: {
      Verbosity: 0
      Name: "IOManager"
      IOMode: 2
      OutFileName: "test_crop_larflow.root"
      InputFiles: []
      InputDirs: []
      StoreOnlyType: []
      StoreOnlyName: []
      TickBackward: true
    }

    ProcessList: {
      Split: {
        Verbosity: 0
        InputProducer: "%s"
        OutputBBox2DProducer: "detsplit"
        CropInModule: true
        OutputCroppedProducer: "detsplit"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: 50
        RandomizeCrops: true
        MaxRandomAttempts: 50
        MinFracPixelsInCrop: 0.0001
      }
      Crop: {
        Verbosity:0
        InputADCProducer: "%s"
        InputBBoxProducer: "detsplit"
        InputCroppedADCProducer: "detsplit"
        InputVisiProducer: ""
        InputFlowProducer: "%s"
        InputChStatusProducer: "%s"
        OutputCroppedADCProducer:  "croppedadc"
        OutputCroppedVisiProducer: "croppedvisi"
        OutputCroppedFlowProducer: "croppedflow"
        OutputCroppedMetaProducer: "croppedmeta_defunct"
        IsMC: true
        HasVisibilityImage: false
        UseVectorizedCode: true # always true now
        MaxImages: 10
        Thresholds: [10.0,10.0,10.0]
        DoMaxPool: false # not working
        RowDownsampleFactor: 1 # not working
        ColDownsampleFactor: 1 # not working
        RequireMinGoodPixels: true # not working
        LimitOverlap: true
        MaxOverlapFraction: 0.25
        CheckFlow: true
        MakeCheckImage: false # not working
        SaveTrainingOutput: false
        OutputFilename: "croppedout.root"
      }
    }    
}"""%(input_adc_producer,input_adc_producer,"larflow",input_chstatus_producer)
    return processor_cfg

def fullsplit_processor_config(input_adc_producer,input_chstatus_producer):
    """
    Make a processor that splits and crops for larflow.
    """
    processor_cfg="""ProcessDriver: {
    Verbosity: 0
    EnableFilter: false
    RandomAccess: false
    ProcessType: ["UBSplitDetector","UBCropLArFlow"]
    ProcessName: ["Split","Crop"]

    IOManager: {
      Verbosity: 0
      Name: "IOManager"
      IOMode: 2
      OutFileName: "test_crop_larflow.root"
      InputFiles: []
      InputDirs: []
      StoreOnlyType: []
      StoreOnlyName: []
      TickBackward: true
    }

    ProcessList: {
      Split: {
        Verbosity: 0
        InputProducer: "%s"
        OutputBBox2DProducer: "detsplit"
        CropInModule: true
        OutputCroppedProducer: "detsplit"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: -1
        RandomizeCrops: false
        MaxRandomAttempts: 50
        MinFracPixelsInCrop: 0.0001
      }
      Crop: {
        Verbosity:0
        InputADCProducer: "%s"
        InputBBoxProducer: "detsplit"
        InputCroppedADCProducer: "detsplit"
        InputVisiProducer: ""
        InputFlowProducer: "%s"
        InputChStatusProducer: "%s"
        OutputCroppedADCProducer:  "croppedadc"
        OutputCroppedVisiProducer: "croppedvisi"
        OutputCroppedFlowProducer: "croppedflow"
        OutputCroppedMetaProducer: "croppedmeta_defunct"
        IsMC: true
        HasVisibilityImage: false
        UseVectorizedCode: true # always true now
        MaxImages: -1
        Thresholds: [10.0,10.0,10.0]
        DoMaxPool: false # not working
        RowDownsampleFactor: 1 # not working
        ColDownsampleFactor: 1 # not working
        RequireMinGoodPixels: true # not working
        LimitOverlap: false
        MaxOverlapFraction: 0.25
        CheckFlow: true
        MakeCheckImage: false # not working
        SaveTrainingOutput: false
        OutputFilename: "croppedout.root"
      }
    }    
}"""%(input_adc_producer,input_adc_producer,"larflow",input_chstatus_producer)
    return processor_cfg
