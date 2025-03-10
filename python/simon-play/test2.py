import os
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams
import pandas as pd


# This is supposed to cause a crash, but doesn't...
df_orig = pd.read_csv('original_5dim.csv')
syn = Synthesizer(df_orig,
                  anonymization_params=AnonymizationParams(layer_noise_sd=0,
                  low_count_params=SuppressionParams(low_threshold=0, layer_sd=0)))
syn.sample()
print(syn.clusters)

df_orig = pd.read_csv('tx2019.csv')
syn = Synthesizer(df_orig,
                  anonymization_params=AnonymizationParams(layer_noise_sd=0,
                  low_count_params=SuppressionParams(low_threshold=0, layer_sd=0)))
syn.sample()
print(syn.clusters)
