import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

from syndiffix import Synthesizer
from dataclasses import dataclass

test_data = pd.read_csv("original_5dim.csv")

os.makedirs("plots", exist_ok=True)

####################PARAMETERS###################
@dataclass(frozen=False)
class FlatteningInterval:
    lower: int = 2
    upper: int = 5
    
@dataclass(frozen=False)
class SuppressionParams:
    low_threshold: int = 0
    layer_sd: float = 0.1
    low_mean_gap: float = 2.0

@dataclass(frozen=False)
class BucketizationParams:
    singularity_low_threshold: int = 5
    range_low_threshold: int = 15
    precision_limit_row_fraction: int = 10000
    precision_limit_depth_threshold: int = 15
    
@dataclass(frozen=False)
class AnonymizationParams:
  # Each noise layer seed is salted before being hashed with a cryptographically-strong algorithm.
  # The salt value needs to have at least 64 bits of entropy (equal or higher than that of the seed).
  # If the provided salt is empty, a per-system salt will be generated and used.
  salt: bytes = b""
  low_count_params: SuppressionParams = SuppressionParams()
  outlier_count: FlatteningInterval = FlatteningInterval()
  top_count: FlatteningInterval = FlatteningInterval()
  layer_noise_sd: float = 0


def run_synth(target = None, anon = "With_Anon"):
    print(f"Do synthesis for target {target}")
    ####################SEQUENCE 1 ###########################################

    var = ['num1', 'num2', 'cat1', 'cat2','geo' ]
    subset_target = test_data.iloc[:,test_data.columns.get_indexer(var) ]

    #syntheizer
    if anon == "No_Anon":
        synthesizer_ = Synthesizer(raw_data = subset_target,
                               target_column = target,
                               bucketization_params = BucketizationParams(),
                               anonymization_params = AnonymizationParams())
    else:
        synthesizer_ = Synthesizer(raw_data = subset_target,
                               target_column = target)
    sd_i_ = synthesizer_.sample()
    print(var)
    print(sd_i_.columns)
    pp.pprint(synthesizer_.clusters)


    ####################SEQUENCE 2 ###########################################

    var = ['geo', 'num1', 'num2', 'cat1', 'cat2' ]
    subset_target_ = test_data.iloc[:,test_data.columns.get_indexer(var) ]

    #syntheizer
    if anon == "No_Anon":
        synthesizer_ = Synthesizer(raw_data = subset_target,
                               target_column = target,
                               bucketization_params = BucketizationParams(),
                               anonymization_params = AnonymizationParams())
    else:
        synthesizer_ = Synthesizer(raw_data = subset_target,
                               target_column = target)
    sd_i_r = synthesizer_.sample()
    print(var)
    print(sd_i_r.columns)
    pp.pprint(synthesizer_.clusters)


    ######################## COMPARe ######################


    f, axes = plt.subplots(1, 3,figsize= (10,5))
    sns.scatterplot(data = subset_target, x = "num1", y ="num2", ax=axes[0], alpha = 0.5)
    sns.scatterplot(data = sd_i_, x = "num1", y ="num2", ax=axes[1], alpha = 0.5)
    sns.scatterplot(data = sd_i_r, x = "num1", y ="num2", ax=axes[2], alpha = 0.5)
    axes[0].set_xlim(-10000,300000)
    axes[1].set_xlim(-10000,300000)
    axes[2].set_xlim(-10000,300000)
    axes[0].set_title(f"original ({anon})")
    axes[1].set_title(f"Geo last ({anon})")
    axes[2].set_title(f"Geo first ({anon})")
    plt.savefig(os.path.join("plots", f"order1_{target}_{anon}.png"))
    plt.close()

    # Create a figure with two subplots side by side
    f, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Left subplot
    sns.scatterplot(data=subset_target, x="num1", y="num2", ax=axes[0], alpha=0.5, color='blue', label='subset_target')
    sns.scatterplot(data=sd_i_, x="num1", y="num2", ax=axes[0], alpha=0.5, color='red', label='sd_i_')
    axes[0].set_xlim(-10000, 300000)
    axes[0].set_title(f"Original vs Geo last ({anon})")
    axes[0].legend()

    # Right subplot
    sns.scatterplot(data=subset_target, x="num1", y="num2", ax=axes[1], alpha=0.5, color='blue', label='subset_target')
    sns.scatterplot(data=sd_i_r, x="num1", y="num2", ax=axes[1], alpha=0.5, color='red', label='sd_i_r')
    axes[1].set_xlim(-10000, 300000)
    axes[1].set_title(f"Original vs Geo first ({anon})")
    axes[1].legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"order2_{target}_{anon}.png"))
    plt.close()

run_synth(target="num2", anon = 'With_Anon')
run_synth(target=None, anon = 'With_Anon')
run_synth(target="num2", anon = 'No_Anon')
run_synth(target=None, anon = 'No_Anon')