from __future__ import print_function
import sklearn
import xgboost
import numpy as np
import pickle
import ROOT
import array

def main():
    print("Hello World")

    infile  = ROOT.TFile("/cluster/tufts/wongjiradlab/jmills09/ubdl_gen2/gen2_checks/jobsoutput/hadd_all/full_developout_bnbfrac.root","READ")
    nue_tree         = infile.Get('Pixel_Removal_on_Intrinsic_Nue_Truth_Image')
    ext_tree         = infile.Get('Pixel_Removal_on_ExtBnB')
    bnb_tree         = infile.Get('Pixel_Removal_on_BnB_Nu_Overlay_Truth_Image')
    nue_lowe_tree    = infile.Get('Pixel_Removal_on_Intrinsic_Nue_Truth_LowE_Image')
    bnb5e19_tree     = infile.Get('Pixel_Removal_on_BnB5e19_Image')

    nue_nentries      = nue_tree.GetEntries()
    nue_lowe_nentries = nue_lowe_tree.GetEntries()
    bnb_nentries      = bnb_tree.GetEntries()
    ext_nentries      = ext_tree.GetEntries()
    bnb5e19_nentries  = bnb5e19_tree.GetEntries()


    fast_mode_entries = 1.0
    if fast_mode_entries != 1.0:
        nue_nentries      = int(round(nue_nentries*fast_mode_entries))
        nue_lowe_nentries = int(round(nue_lowe_nentries*fast_mode_entries))
        bnb_nentries      = int(round(bnb_nentries*fast_mode_entries))
        ext_nentries      = int(round(ext_nentries*fast_mode_entries))
        bnb5e19_nentries  = int(round(bnb5e19_nentries*fast_mode_entries))

    print(nue_nentries , "   Nues")
    print(nue_lowe_nentries , "   LowE")
    print(bnb_nentries , "   BnBNuMu")
    print(ext_nentries , "  ExtBnB")
    print(bnb5e19_nentries , "   BnB5e19")

    # Make Empty Numpy Arrays
    n_vars = 16

    nue_vars_np      = np.zeros((nue_nentries,n_vars))
    nue_lowe_vars_np = np.zeros((nue_lowe_nentries,n_vars))
    bnb_vars_np      = np.zeros((bnb_nentries,n_vars))
    ext_vars_np      = np.zeros((ext_nentries,n_vars))
    bnb5e19_vars_np  = np.zeros((bnb5e19_nentries,n_vars))

    nue_rse_np       = np.zeros((nue_nentries,3))
    nuelowe_rse_np   = np.zeros((nue_lowe_nentries,3))
    bnb_rse_np       = np.zeros((bnb_nentries,3))
    ext_rse_np       = np.zeros((ext_nentries,3))
    bnb5e19_rse_np   = np.zeros((bnb5e19_nentries,3))
    rse_list = [nue_rse_np, nuelowe_rse_np, bnb_rse_np, ext_rse_np, bnb5e19_rse_np]

    nue_wgts_np      = np.ones((nue_nentries))*40.0
    nue_lowe_wgts_np = np.ones((nue_lowe_nentries))*120.0
    bnb_wgts_np      = np.ones((bnb_nentries))*1.0
    ext_wgts_np      = np.ones((ext_nentries))*1.0
    # bnb5e19_wgts_np  = np.ones((bnb5e19_nentries,1))

    all_sample_arrays = [nue_vars_np, nue_lowe_vars_np, bnb_vars_np, ext_vars_np, bnb5e19_vars_np]

    tree_idx = 0
    entries_list = [nue_nentries, nue_lowe_nentries, bnb_nentries, ext_nentries, bnb5e19_nentries]
    varname = [
        "adc_pix_count           ",
        "wc_frac                 ",
        "mrcnn_frac_020          ",
        "mrcnn_frac_090          ",
        "combined_frac_020       ",
        "combined_frac_090       ",
        "hip_count_050           ",
        "hip_count_090           ",
        "mip_count_050           ",
        "mip_count_090           ",
        "shower_count_050        ",
        "shower_count_090        ",
        "michel_count_050        ",
        "michel_count_090        ",
        "delta_count_050         ",
        "delta_count_090         "
        ]
    for tree in [nue_tree, nue_lowe_tree, bnb_tree, ext_tree, bnb5e19_tree]:
        print("Doing Tree: ",tree_idx)
        for n in range(entries_list[tree_idx]):
            tree.GetEntry(n)
            all_sample_arrays[tree_idx][n][0]      = tree.adc_pix_count
            all_sample_arrays[tree_idx][n][1]      = tree.wc_frac
            all_sample_arrays[tree_idx][n][2]      = tree.mrcnn_frac_020
            all_sample_arrays[tree_idx][n][3]      = tree.mrcnn_frac_090
            all_sample_arrays[tree_idx][n][4]      = tree.combined_frac_020
            all_sample_arrays[tree_idx][n][5]      = tree.combined_frac_090
            all_sample_arrays[tree_idx][n][6]      = tree.hip_count_050
            all_sample_arrays[tree_idx][n][7]      = tree.hip_count_090
            all_sample_arrays[tree_idx][n][8]      = tree.mip_count_050
            all_sample_arrays[tree_idx][n][9]      = tree.mip_count_090
            all_sample_arrays[tree_idx][n][10]      = tree.shower_count_050
            all_sample_arrays[tree_idx][n][11]      = tree.shower_count_090
            all_sample_arrays[tree_idx][n][12]      = tree.michel_count_050
            all_sample_arrays[tree_idx][n][13]      = tree.michel_count_090
            all_sample_arrays[tree_idx][n][14]      = tree.delta_count_050
            all_sample_arrays[tree_idx][n][15]      = tree.delta_count_090
            rse_list[tree_idx][n][0] = tree.run
            rse_list[tree_idx][n][1] = tree.subrun
            rse_list[tree_idx][n][2] = tree.event



        tree_idx = tree_idx+1


    nue_evs = nue_vars_np.tolist()
    nue_lowe_evs = nue_lowe_vars_np.tolist()
    bnb_evs = bnb_vars_np.tolist()
    ext_evs = ext_vars_np.tolist()
    bnb5e19_evs = bnb5e19_vars_np.tolist()
    bnb5e19_rse = bnb5e19_rse_np.tolist()

    nue_wgts = nue_wgts_np.tolist()
    nue_lowe_wgts = nue_lowe_wgts_np.tolist()
    bnb_wgts = bnb_wgts_np.tolist()
    ext_wgts = ext_wgts_np.tolist()

    bck_len = np.zeros(len(ext_evs)).tolist()
    #DEFINE SIGNAL as bnb + nue
    # sig_len = np.ones(len(nue_evs)+len(bnb_evs)).tolist()
    # train_test_input = np.asarray(nue_evs+bnb_evs+ext_evs)
    # train_test_len   = np.asarray(sig_len+bck_len)
    # DEFINE SIGNAL as nue + nuelowe
    sig_nue_len = np.ones(len(nue_evs)+len(nue_lowe_evs)).tolist()
    train_test_nue_input = np.asarray(nue_evs+nue_lowe_evs+ext_evs)
    train_test_nue_weights = np.asarray(nue_wgts+nue_lowe_wgts+ext_wgts)
    train_test_nue_len   = np.asarray(sig_nue_len+bck_len)

    seed = 15
    test_size = 0.3
    # data_train, data_test, answers_train, answers_test = sklearn.model_selection.train_test_split(train_test_input, train_test_len, test_size=test_size, random_state=seed)
    data_nue_train, data_nue_test, wgts_nue_train, wgts_nue_test, answers_nue_train, answers_nue_test = \
         sklearn.model_selection.train_test_split(train_test_nue_input, train_test_nue_weights, train_test_nue_len, test_size=test_size, random_state=seed)

    eval_set = [(data_nue_train, answers_nue_train), (data_nue_test, answers_nue_test)]

    cosmictag_nueBDT = xgboost.XGBClassifier(silent=False,
                  scale_pos_weight=1,
                  learning_rate=0.01,
                  #colsample_bytree = .8,
                  objective='binary:logistic',
                  subsample = 0.8,
                  n_estimators=1000,
                  max_depth=5,
                  gamma=5)



    load_nue_model = True
    load_bnb_model = True
    if load_nue_model:
        print("Loading Nue Model")
        cosmictag_nueBDT = pickle.load(open("cosmictag_nueBDTweights_test.pickle", "rb"))
    else:
        print("Doing Nue Fit on size: ", data_nue_train.shape, " " , answers_nue_train.shape)
        cosmictag_nueBDT.fit(data_nue_train, answers_nue_train, sample_weight=wgts_nue_train)

    test_scores_nuebdt  = cosmictag_nueBDT.predict_proba(data_nue_test)
    train_scores_nuebdt = cosmictag_nueBDT.predict_proba(data_nue_train)
    preds_nuebdt_test = [round(test_scores_nuebdt[idx][1]) for idx in range(test_scores_nuebdt.shape[0])]
    preds_nuebdt_train = [round(train_scores_nuebdt[idx][1]) for idx in range(train_scores_nuebdt.shape[0])]

    nue_ans = np.ones(len(nue_evs)).tolist()
    bnb_ans = np.ones(len(bnb_evs)).tolist()
    nue_lowe_ans = np.ones(len(nue_lowe_evs)).tolist()
    ext_ans = np.zeros(len(ext_evs)).tolist()

    nue_scores_nuebdt       = cosmictag_nueBDT.predict_proba(np.asarray(nue_evs))
    bnb_scores_nuebdt       = cosmictag_nueBDT.predict_proba(np.asarray(bnb_evs))
    nue_lowe_scores_nuebdt  = cosmictag_nueBDT.predict_proba(np.asarray(nue_lowe_evs))
    ext_scores_nuebdt       = cosmictag_nueBDT.predict_proba(np.asarray(ext_evs))
    bnb5e19_scores_nuebdt   = cosmictag_nueBDT.predict_proba(np.asarray(bnb5e19_evs))

    preds_nuebdt_nue = [round(nue_scores_nuebdt[idx][1]) for idx in range(nue_scores_nuebdt.shape[0])]
    preds_nuebdt_bnb = [round(bnb_scores_nuebdt[idx][1]) for idx in range(bnb_scores_nuebdt.shape[0])]
    preds_nuebdt_nue_lowe = [round(nue_lowe_scores_nuebdt[idx][1]) for idx in range(nue_lowe_scores_nuebdt.shape[0])]
    preds_nuebdt_ext = [round(ext_scores_nuebdt[idx][1]) for idx in range(ext_scores_nuebdt.shape[0])]
    preds_nuebdt_bnb5e19 = [round(bnb5e19_scores_nuebdt[idx][1]) for idx in range(bnb5e19_scores_nuebdt.shape[0])]

    print("Accuracy Score NueBDT Test Set")
    print("    ",sklearn.metrics.accuracy_score(answers_nue_test,preds_nuebdt_test))
    print("Accuracy Score NueBDT Train Set")
    print("    ",sklearn.metrics.accuracy_score(answers_nue_train,preds_nuebdt_train))
    print()
    print("Feature Importances:\n")
    for idx in range(len(cosmictag_nueBDT.feature_importances_)):
        print(varname[idx] + "   ", cosmictag_nueBDT.feature_importances_[idx])
    print()
    print()
    print("NueBDT Efficiency Nue Set")
    print("    ",sklearn.metrics.accuracy_score(nue_ans,preds_nuebdt_nue))
    print("NueBDT Efficiency BnB Set")
    print("    ",sklearn.metrics.accuracy_score(bnb_ans,preds_nuebdt_bnb))
    print("NueBDT Efficiency Nue_Lowe Set")
    print("    ",sklearn.metrics.accuracy_score(nue_lowe_ans,preds_nuebdt_nue_lowe))
    print("NueBDT Efficiency Ext Set")
    print("    ",sklearn.metrics.accuracy_score(ext_ans,preds_nuebdt_ext))
    print()
    print()
    num_nue_pass_nuebdt = sklearn.metrics.accuracy_score(nue_ans,preds_nuebdt_nue,False)
    num_bnb_pass_nuebdt = sklearn.metrics.accuracy_score(bnb_ans,preds_nuebdt_bnb,False)
    num_nue_lowe_pass_nuebdt = sklearn.metrics.accuracy_score(nue_lowe_ans,preds_nuebdt_nue_lowe,False)
    num_ext_pass_nuebdt = ext_nentries - sklearn.metrics.accuracy_score(ext_ans,preds_nuebdt_ext,False)


    EXT_RUN3_POT      = (39566274.0 / 2263559.0) * 2.429524e+20 * 633.0/686.0
    BNB_RUN3_POT      = 8.98773223801e+20 * 248/312
    NUE_LOWE_RUN3_POT = 5.97440749241e+23 * 574.0/579.0
    NUE_RUN3_POT      = 4.70704675581e+22 * 2232.0/2232.0
    DATA_RUN3_POT     = 2.429524e+20  # * 850.0/3753.0

    NUE3_SCALE      = (DATA_RUN3_POT/NUE_RUN3_POT)/fast_mode_entries
    EXT3_SCALE      = (DATA_RUN3_POT/EXT_RUN3_POT)/fast_mode_entries
    BNB3_SCALE      = (DATA_RUN3_POT/BNB_RUN3_POT)/fast_mode_entries
    NUE_LOWE3_SCALE = (DATA_RUN3_POT/NUE_LOWE_RUN3_POT)/fast_mode_entries
    print()
    print("NueBDT Run3 Scaled Nues:       ",num_nue_pass_nuebdt*NUE3_SCALE)
    print("NueBDT Run3 Scaled Nue LowEs:  ",num_nue_lowe_pass_nuebdt*NUE_LOWE3_SCALE)
    print("NueBDT Run3 Scaled BnBs:       ",num_bnb_pass_nuebdt*BNB3_SCALE)
    print("NueBDT Run3 Scaled Exts:       ",num_ext_pass_nuebdt*EXT3_SCALE)
    print()
    # print("Run3 Nues:       ",num_nue_pass_nuebdt)
    # print("Run3 Nue LowEs:  ",num_nue_lowe_pass_nuebdt)
    # print("Run3 BnBs:       ",num_bnb_pass_nuebdt)
    # print("Run3 Exts:       ",num_ext_pass_nuebdt)
    # print()

    print("NueBDT Nue Purity,      Just ExtBnB:    ", (num_nue_pass_nuebdt*NUE3_SCALE) /(num_nue_pass_nuebdt*NUE3_SCALE + num_ext_pass_nuebdt*EXT3_SCALE))
    print("NueBDT Nue Lowe Purity, Just ExtBnB:    ", (num_nue_lowe_pass_nuebdt*NUE_LOWE3_SCALE) /(num_nue_lowe_pass_nuebdt*NUE_LOWE3_SCALE + num_ext_pass_nuebdt*EXT3_SCALE))
    print()
    print("NueBDT Nue Purity, All Background:         ", (num_nue_pass_nuebdt*NUE3_SCALE) /(num_nue_pass_nuebdt*NUE3_SCALE + num_bnb_pass_nuebdt*BNB3_SCALE + num_ext_pass_nuebdt*EXT3_SCALE))
    print("NueBDT BnB Purity, All Background:         ", (num_bnb_pass_nuebdt*BNB3_SCALE) /(num_nue_pass_nuebdt*NUE3_SCALE + num_bnb_pass_nuebdt*BNB3_SCALE + num_ext_pass_nuebdt*EXT3_SCALE))


    if load_nue_model != True:
        print("Dumping model")
        pickle.dump(cosmictag_nueBDT,open( "cosmictag_nueBDTweights_test.pickle", "wb" ),protocol=2)

    sig_bnb_len = np.ones(len(bnb_evs)).tolist()
    train_test_bnb_input = np.asarray(bnb_evs+ext_evs)
    train_test_bnb_weights = np.asarray(bnb_wgts+ext_wgts)
    train_test_bnb_len   = np.asarray(sig_bnb_len+bck_len)
    data_bnb_train, data_bnb_test, wgts_bnb_train, wgts_bnb_test, answers_bnb_train, answers_bnb_test = \
         sklearn.model_selection.train_test_split(train_test_bnb_input, train_test_bnb_weights, train_test_bnb_len, test_size=test_size, random_state=seed)
    eval_set = [(data_bnb_train, answers_bnb_train), (data_bnb_test, answers_bnb_test)]
    cosmictag_bnbBDT = xgboost.XGBClassifier(silent=False,
                  scale_pos_weight=1,
                  learning_rate=0.01,
                  #colsample_bytree = .8,
                  objective='binary:logistic',
                  subsample = 0.8,
                  n_estimators=1000,
                  max_depth=5,
                  gamma=5)



    if load_bnb_model:
        print("Loading BnB Model")
        cosmictag_bnbBDT = pickle.load(open("cosmictag_bnbBDTweights_test.pickle", "rb"))
    else:
        print("Doing BnB Fit on size: ", data_bnb_train.shape, " " , answers_bnb_train.shape)
        cosmictag_bnbBDT.fit(data_bnb_train, answers_bnb_train, sample_weight=wgts_bnb_train)

    test_scores_bnbbdt  = cosmictag_bnbBDT.predict_proba(data_bnb_test)
    train_scores_bnbbdt = cosmictag_bnbBDT.predict_proba(data_bnb_train)
    preds_bnbbdt_test = [round(test_scores_bnbbdt[idx][1]) for idx in range(test_scores_bnbbdt.shape[0])]
    preds_bnbbdt_train = [round(train_scores_bnbbdt[idx][1]) for idx in range(train_scores_bnbbdt.shape[0])]

    nue_scores_bnbbdt       = cosmictag_bnbBDT.predict_proba(np.asarray(nue_evs))
    bnb_scores_bnbbdt       = cosmictag_bnbBDT.predict_proba(np.asarray(bnb_evs))
    nue_lowe_scores_bnbbdt  = cosmictag_bnbBDT.predict_proba(np.asarray(nue_lowe_evs))
    ext_scores_bnbbdt       = cosmictag_bnbBDT.predict_proba(np.asarray(ext_evs))
    bnb5e19_scores_bnbbdt   = cosmictag_bnbBDT.predict_proba(np.asarray(bnb5e19_evs))

    preds_bnbbdt_nue = [round(nue_scores_bnbbdt[idx][1]) for idx in range(nue_scores_bnbbdt.shape[0])]
    preds_bnbbdt_bnb = [round(bnb_scores_bnbbdt[idx][1]) for idx in range(bnb_scores_bnbbdt.shape[0])]
    preds_bnbbdt_nue_lowe = [round(nue_lowe_scores_bnbbdt[idx][1]) for idx in range(nue_lowe_scores_bnbbdt.shape[0])]
    preds_bnbbdt_ext = [round(ext_scores_bnbbdt[idx][1]) for idx in range(ext_scores_bnbbdt.shape[0])]
    preds_bnbbdt_bnb5e19 = [round(bnb5e19_scores_bnbbdt[idx][1]) for idx in range(bnb5e19_scores_bnbbdt.shape[0])]

    print("Accuracy Score BnBBDT Test Set")
    print("    ",sklearn.metrics.accuracy_score(answers_bnb_test,preds_bnbbdt_test))
    print("Accuracy Score BnBBDT Train Set")
    print("    ",sklearn.metrics.accuracy_score(answers_bnb_train,preds_bnbbdt_train))
    print()
    print("Feature Importances:\n")
    for idx in range(len(cosmictag_bnbBDT.feature_importances_)):
        print(varname[idx] + "   ", cosmictag_bnbBDT.feature_importances_[idx])
    print()
    print()
    print("BnBBDT Efficiency Nue Set")
    print("    ",sklearn.metrics.accuracy_score(nue_ans,preds_bnbbdt_nue))
    print("BnBBDT Efficiency BnB Set")
    print("    ",sklearn.metrics.accuracy_score(bnb_ans,preds_bnbbdt_bnb))
    print("BnBBDT Efficiency Nue_Lowe Set")
    print("    ",sklearn.metrics.accuracy_score(nue_lowe_ans,preds_bnbbdt_nue_lowe))
    print("BnBBDT Efficiency Ext Set")
    print("    ",sklearn.metrics.accuracy_score(ext_ans,preds_bnbbdt_ext))
    print()
    print()
    num_nue_pass_bnbbdt = sklearn.metrics.accuracy_score(nue_ans,preds_bnbbdt_nue,False)
    num_bnb_pass_bnbbdt = sklearn.metrics.accuracy_score(bnb_ans,preds_bnbbdt_bnb,False)
    num_nue_lowe_pass_bnbbdt = sklearn.metrics.accuracy_score(nue_lowe_ans,preds_bnbbdt_nue_lowe,False)
    num_ext_pass_bnbbdt = ext_nentries - sklearn.metrics.accuracy_score(ext_ans,preds_bnbbdt_ext,False)


    EXT_RUN3_POT      = (39566274.0 / 2263559.0) * 2.429524e+20 * 633.0/686.0
    BNB_RUN3_POT      = 8.98773223801e+20 * 248/312
    NUE_LOWE_RUN3_POT = 5.97440749241e+23 * 574.0/579.0
    NUE_RUN3_POT      = 4.70704675581e+22 * 2232.0/2232.0
    DATA_RUN3_POT     = 2.429524e+20  # * 850.0/3753.0

    NUE3_SCALE      = (DATA_RUN3_POT/NUE_RUN3_POT)/fast_mode_entries
    EXT3_SCALE      = (DATA_RUN3_POT/EXT_RUN3_POT)/fast_mode_entries
    BNB3_SCALE      = (DATA_RUN3_POT/BNB_RUN3_POT)/fast_mode_entries
    NUE_LOWE3_SCALE = (DATA_RUN3_POT/NUE_LOWE_RUN3_POT)/fast_mode_entries
    print()
    print("BnBBDT Run3 Scaled Nues:       ",num_nue_pass_bnbbdt*NUE3_SCALE)
    print("BnBBDT Run3 Scaled Nue LowEs:  ",num_nue_lowe_pass_bnbbdt*NUE_LOWE3_SCALE)
    print("BnBBDT Run3 Scaled BnBs:       ",num_bnb_pass_bnbbdt*BNB3_SCALE)
    print("BnBBDT Run3 Scaled Exts:       ",num_ext_pass_bnbbdt*EXT3_SCALE)
    print()
    # print("Run3 Nues:       ",num_nue_pass_bnbbdt)
    # print("Run3 Nue LowEs:  ",num_nue_lowe_pass_bnbbdt)
    # print("Run3 BnBs:       ",num_bnb_pass_bnbbdt)
    # print("Run3 Exts:       ",num_ext_pass_bnbbdt)
    # print()

    print("BnBBDT Nue Purity,      Just ExtBnB:    ", (num_nue_pass_bnbbdt*NUE3_SCALE) /(num_nue_pass_bnbbdt*NUE3_SCALE + num_ext_pass_bnbbdt*EXT3_SCALE))
    print("BnBBDT Nue Lowe Purity, Just ExtBnB:    ", (num_nue_lowe_pass_bnbbdt*NUE_LOWE3_SCALE) /(num_nue_lowe_pass_bnbbdt*NUE_LOWE3_SCALE + num_ext_pass_bnbbdt*EXT3_SCALE))
    print()
    print("BnBBDT Nue Purity, All Background:         ", (num_nue_pass_bnbbdt*NUE3_SCALE) /(num_nue_pass_bnbbdt*NUE3_SCALE + num_bnb_pass_bnbbdt*BNB3_SCALE + num_ext_pass_bnbbdt*EXT3_SCALE))
    print("BnBBDT BnB Purity, All Background:         ", (num_bnb_pass_bnbbdt*BNB3_SCALE) /(num_nue_pass_bnbbdt*NUE3_SCALE + num_bnb_pass_bnbbdt*BNB3_SCALE + num_ext_pass_bnbbdt*EXT3_SCALE))


    if load_bnb_model != True:
        print("Dumping BnB model")
        pickle.dump(cosmictag_bnbBDT,open( "cosmictag_bnbBDTweights_test.pickle", "wb" ),protocol=2)

    nbins = 40
    ext_bdtscores_h = ROOT.TH2D("ext_bdtscores_h","ext_bdtscores_h",nbins,0,1.001,nbins,0,1.001)
    ext_bdtscores_h.SetXTitle("NueBDT Scores")
    ext_bdtscores_h.SetYTitle("BnBBDT Scores")

    nue_bdtscores_h = ROOT.TH2D("nue_bdtscores_h","nue_bdtscores_h",nbins,0,1.001,nbins,0,1.001)
    nue_bdtscores_h.SetXTitle("NueBDT Scores")
    nue_bdtscores_h.SetYTitle("BnBBDT Scores")

    nuelowe_bdtscores_h = ROOT.TH2D("nuelowe_bdtscores_h","nuelowe_bdtscores_h",nbins,0,1.001,nbins,0,1.001)
    nuelowe_bdtscores_h.SetXTitle("NueBDT Scores")
    nuelowe_bdtscores_h.SetYTitle("BnBBDT Scores")

    bnb_bdtscores_h = ROOT.TH2D("bnb_bdtscores_h","bnb_bdtscores_h",nbins,0,1.001,nbins,0,1.001)
    bnb_bdtscores_h.SetXTitle("NueBDT Scores")
    bnb_bdtscores_h.SetYTitle("BnBBDT Scores")

    bnb5e19_bdtscores_h = ROOT.TH2D("bnb5e19_bdtscores_h","bnb5e19_bdtscores_h",nbins,0,1.001,nbins,0,1.001)
    bnb5e19_bdtscores_h.SetXTitle("NueBDT Scores")
    bnb5e19_bdtscores_h.SetYTitle("BnBBDT Scores")

    outfile     = ROOT.TFile("BDTScores_and_Distributions.root","RECREATE")
    nuetree     = ROOT.TTree("NueIntrinsicTree","NueIntrinsicTree")
    nuelowetree = ROOT.TTree("NueLowEIntrinsicTree","NueLowEIntrinsicTree")
    bnbtree     = ROOT.TTree("BnBNuMuIntrinsicTree","BnBNuMuIntrinsicTree")
    exttree     = ROOT.TTree("ExtBnBTree","ExtBnBTree")
    bnb5e19tree = ROOT.TTree("BnB5e19Tree","BnB5e19Tree")

    nue_run = array.array("d",[-1.0])
    nue_subrun = array.array("d",[-1.0])
    nue_event = array.array("d",[-1.0])
    nue_nueBDTScore = array.array("d",[-1.0])
    nue_bnbBDTScore = array.array("d",[-1.0])
    nuetree.Branch("run",nue_run, "nue_run/D")
    nuetree.Branch("subrun",nue_subrun, "nue_subrun/D")
    nuetree.Branch("event",nue_event, "nue_event/D")
    nuetree.Branch("nueBDTScore",nue_nueBDTScore, "nue_nueBDTScore/D")
    nuetree.Branch("bnbBDTScore",nue_bnbBDTScore, "nue_bnbBDTScore/D")

    nuelowe_run = array.array("d",[-1.0])
    nuelowe_subrun = array.array("d",[-1.0])
    nuelowe_event = array.array("d",[-1.0])
    nuelowe_nueBDTScore = array.array("d",[-1.0])
    nuelowe_bnbBDTScore = array.array("d",[-1.0])
    nuelowetree.Branch("run",nuelowe_run, "nuelowe_run/D")
    nuelowetree.Branch("subrun",nuelowe_subrun, "nuelowe_subrun/D")
    nuelowetree.Branch("event",nuelowe_event, "nuelowe_event/D")
    nuelowetree.Branch("nueBDTScore",nuelowe_nueBDTScore, "nuelowe_nueBDTScore/D")
    nuelowetree.Branch("bnbBDTScore",nuelowe_bnbBDTScore, "nuelowe_bnbBDTScore/D")

    bnb_run = array.array("d",[-1.0])
    bnb_subrun = array.array("d",[-1.0])
    bnb_event = array.array("d",[-1.0])
    bnb_nueBDTScore = array.array("d",[-1.0])
    bnb_bnbBDTScore = array.array("d",[-1.0])
    bnbtree.Branch("run",bnb_run, "bnb_run/D")
    bnbtree.Branch("subrun",bnb_subrun, "bnb_subrun/D")
    bnbtree.Branch("event",bnb_event, "bnb_event/D")
    bnbtree.Branch("nueBDTScore",bnb_nueBDTScore, "bnb_nueBDTScore/D")
    bnbtree.Branch("bnbBDTScore",bnb_bnbBDTScore, "bnb_bnbBDTScore/D")

    ext_run = array.array("d",[-1.0])
    ext_subrun = array.array("d",[-1.0])
    ext_event = array.array("d",[-1.0])
    ext_nueBDTScore = array.array("d",[-1.0])
    ext_bnbBDTScore = array.array("d",[-1.0])
    exttree.Branch("run",ext_run, "ext_run/D")
    exttree.Branch("subrun",ext_subrun, "ext_subrun/D")
    exttree.Branch("event",ext_event, "ext_event/D")
    exttree.Branch("nueBDTScore",ext_nueBDTScore, "ext_nueBDTScore/D")
    exttree.Branch("bnbBDTScore",ext_bnbBDTScore, "ext_bnbBDTScore/D")

    bnb5e19_run = array.array("d",[-1.0])
    bnb5e19_subrun = array.array("d",[-1.0])
    bnb5e19_event = array.array("d",[-1.0])
    bnb5e19_nueBDTScore = array.array("d",[-1.0])
    bnb5e19_bnbBDTScore = array.array("d",[-1.0])
    bnb5e19tree.Branch("run",bnb5e19_run, "bnb5e19_run/D")
    bnb5e19tree.Branch("subrun",bnb5e19_subrun, "bnb5e19_subrun/D")
    bnb5e19tree.Branch("event",bnb5e19_event, "bnb5e19_event/D")
    bnb5e19tree.Branch("nueBDTScore",bnb5e19_nueBDTScore, "bnb5e19_nueBDTScore/D")
    bnb5e19tree.Branch("bnbBDTScore",bnb5e19_bnbBDTScore, "bnb5e19_bnbBDTScore/D")

    for i in range(nue_scores_bnbbdt.shape[0]):
        nue_run[0] = nue_rse_np[i][0]
        nue_subrun[0] = nue_rse_np[i][1]
        nue_event[0] = nue_rse_np[i][2]
        nue_nueBDTScore[0] = nue_scores_nuebdt[i][1]
        nue_bnbBDTScore[0] = nue_scores_bnbbdt[i][1]
        nue_bdtscores_h.Fill(nue_scores_nuebdt[i][1],nue_scores_bnbbdt[i][1],NUE3_SCALE)
        nuetree.Fill()

    for i in range(nue_lowe_scores_bnbbdt.shape[0]):
        nuelowe_run[0] = nuelowe_rse_np[i][0]
        nuelowe_subrun[0] = nuelowe_rse_np[i][1]
        nuelowe_event[0] = nuelowe_rse_np[i][2]
        nuelowe_nueBDTScore[0] = nue_lowe_scores_nuebdt[i][1]
        nuelowe_bnbBDTScore[0] = nue_lowe_scores_bnbbdt[i][1]
        nuelowe_bdtscores_h.Fill(nue_lowe_scores_nuebdt[i][1],nue_lowe_scores_bnbbdt[i][1],NUE_LOWE3_SCALE)
        nuelowetree.Fill()

    for i in range(bnb_scores_bnbbdt.shape[0]):
        bnb_run[0] = bnb_rse_np[i][0]
        bnb_subrun[0] = bnb_rse_np[i][1]
        bnb_event[0] = bnb_rse_np[i][2]
        bnb_nueBDTScore[0] = bnb_scores_nuebdt[i][1]
        bnb_bnbBDTScore[0] = bnb_scores_bnbbdt[i][1]
        bnb_bdtscores_h.Fill(bnb_scores_nuebdt[i][1],bnb_scores_bnbbdt[i][1],BNB3_SCALE)
        bnbtree.Fill()
    for i in range(ext_scores_bnbbdt.shape[0]):
        ext_run[0] = ext_rse_np[i][0]
        ext_subrun[0] = ext_rse_np[i][1]
        ext_event[0] = ext_rse_np[i][2]
        ext_nueBDTScore[0] = ext_scores_nuebdt[i][1]
        ext_bnbBDTScore[0] = ext_scores_bnbbdt[i][1]
        ext_bdtscores_h.Fill(ext_scores_nuebdt[i][1],ext_scores_bnbbdt[i][1],EXT3_SCALE)
        exttree.Fill()

    for i in range(bnb5e19_scores_bnbbdt.shape[0]):
        bnb5e19_run[0] = bnb5e19_rse_np[i][0]
        bnb5e19_subrun[0] = bnb5e19_rse_np[i][1]
        bnb5e19_event[0] = bnb5e19_rse_np[i][2]
        bnb5e19_nueBDTScore[0] = bnb5e19_scores_nuebdt[i][1]
        bnb5e19_bnbBDTScore[0] = bnb5e19_scores_bnbbdt[i][1]
        bnb5e19_bdtscores_h.Fill(bnb5e19_scores_nuebdt[i][1],bnb5e19_scores_bnbbdt[i][1])
        bnb5e19tree.Fill()

#
    c1 = ROOT.TCanvas("canoe","canoe",1400,1000)
    ROOT.gStyle.SetOptStat(0);
    nue_bdtscores_h.Draw("COLZ")
    c1.SaveAs("test/twobdtscores_nuesample.png")
    nuelowe_bdtscores_h.Draw("COLZ")
    c1.SaveAs("test/twobdtscores_nuelowesample.png")
    bnb_bdtscores_h.Draw("COLZ")
    c1.SaveAs("test/twobdtscores_bnbsample.png")
    ext_bdtscores_h.Draw("COLZ")
    c1.SaveAs("test/twobdtscores_extsample.png")
    bnb5e19_bdtscores_h.Draw("COLZ")
    c1.SaveAs("test/twobdtscores_bnb5e19sample.png")

    nuetree.Write()
    nuelowetree.Write()
    bnbtree.Write()
    exttree.Write()
    bnb5e19tree.Write()

    nue_bdtscores_h.Write()
    nuelowe_bdtscores_h.Write()
    bnb_bdtscores_h.Write()
    ext_bdtscores_h.Write()
    bnb5e19_bdtscores_h.Write()
    # outfile.Write()
    outfile.Close()
    return 0




if __name__ == "__main__":
    main()
