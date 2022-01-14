import ROOT

# We use ROOT to load our class+python bindings
ROOT.gSystem.Load('./libSmallClusterRemoval.so')

# Load the namespace
from ROOT import larvoxelprepdata as larvoxelprepdata

# make an instance of our class
remover = larvoxelprepdata.SmallClusterRemoval()
print(remover)
