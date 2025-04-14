import os,sys

outdir = 'output/outdir_mcc9_v13_bnb_nu_corsika_training'
#outdir = 'output/outdir_mcc9_v13_bnb_nu_corsika_validation'
files = os.listdir(outdir)
for fh5 in files:
    fh5 = fh5.strip()
    if fh5[-3:]==".h5" and "larcv_mctruth" in fh5:
        frename = fh5.replace("larcv_mctruth","dlshowermodel_trainingdata")
        cmd = f'mv {outdir}/{fh5} {outdir}/{frename}'
        print(cmd)
        #os.system(cmd)
