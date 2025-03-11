import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams, BucketizationParams

latest_syndiffix = True

test_data = pd.read_csv("original_5dim.csv")

os.makedirs("plots", exist_ok=True)

def run_synth(df_orig, filename,
              target = '', 
              layer_noise_sd = 1.0,
              low_threshold = 3,
              low_mean_gap = 2.0,
              layer_sd = 1.0,
              singularity_low_threshold = 5,
              range_low_threshold = 15,
              precision_limit_row_fraction = 10000,
              precision_limit_depth_threshold = 15):

    file_path = os.path.join("plots", f"{filename}.png")
    if os.path.exists(file_path):
        return

    defaults = {
        "target": {"def": "", "p": "target"},
        "layer_noise_sd": {"def": 1.0, "p": "noise"},
        "low_threshold": {"def": 3, "p": "sp_low"},
        "low_mean_gap": {"def": 2.0, "p": "sp_gap"},
        "layer_sd": {"def": 1.0, "p": "sp_sd"},
        "singularity_low_threshold": {"def": 5, "p": "sing_low"},
        "range_low_threshold": {"def": 15, "p": "range_low"},
        "precision_limit_row_fraction": {"def": 10000, "p": "prec_frac"},
        "precision_limit_depth_threshold": {"def": 15, "p": "prec_depth"},
    }

    params = '          \n'

    if defaults["target"]["def"] != target:
        params += f"{defaults['target']['p']}={target}\n"
    if defaults["layer_noise_sd"]["def"] != layer_noise_sd:
        params += f"{defaults['layer_noise_sd']['p']}={layer_noise_sd}\n"
    if defaults["low_threshold"]["def"] != low_threshold:
        params += f"{defaults['low_threshold']['p']}={low_threshold}\n"
    if defaults["low_mean_gap"]["def"] != low_mean_gap:
        params += f"{defaults['low_mean_gap']['p']}={low_mean_gap}\n"
    if defaults["layer_sd"]["def"] != layer_sd:
        params += f"{defaults['layer_sd']['p']}={layer_sd}\n"
    if defaults["singularity_low_threshold"]["def"] != singularity_low_threshold:
        params += f"{defaults['singularity_low_threshold']['p']}={singularity_low_threshold}\n"
    if defaults["range_low_threshold"]["def"] != range_low_threshold:
        params += f"{defaults['range_low_threshold']['p']}={range_low_threshold}\n"
    if defaults["precision_limit_row_fraction"]["def"] != precision_limit_row_fraction:
        params += f"{defaults['precision_limit_row_fraction']['p']}={precision_limit_row_fraction}\n"
    if defaults["precision_limit_depth_threshold"]["def"] != precision_limit_depth_threshold:
        params += f"{defaults['precision_limit_depth_threshold']['p']}={precision_limit_depth_threshold}\n"

    if target == '':
        target = None
    syn = Synthesizer(df_orig, target_column = target,
                      anonymization_params=AnonymizationParams(layer_noise_sd=layer_noise_sd,
                         low_count_params=SuppressionParams(low_threshold=low_threshold,
                                                            layer_sd=layer_sd,
                                                            low_mean_gap=low_mean_gap)),
                      bucketization_params=BucketizationParams(
                           singularity_low_threshold=singularity_low_threshold,
                           range_low_threshold=range_low_threshold,
                           precision_limit_row_fraction=precision_limit_row_fraction,
                           precision_limit_depth_threshold=precision_limit_depth_threshold))
    df_syn = syn.sample()
    if latest_syndiffix:
        from syndiffix.stitcher import get_cluster
        cluster = get_cluster(syn)
    else:
        cluster = syn.clusters
    params += '\nInitial:\n'
    for col in cluster['initial']:
        params += f"{col}\n"

    print("------------------------------------------")
    print(params)
    pp.pprint(cluster)

    # Create the scatterplot with extra space on the right
    fig, ax = plt.subplots(figsize=(8, 4))  # Set the figure size

    # Plot for df_orig
    sns.scatterplot(data=df_orig, x='num1', y='num2', s=10, color='blue', label='df_orig', ax=ax)

    # Plot for df_syn
    sns.scatterplot(data=df_syn, x='num1', y='num2', s=10, color='red', label='df_syn', ax=ax)

    # Customize the plot
    ax.set_xlabel('num1')
    ax.set_ylabel('num2')
    ax.set_title(f"{filename}")
    ax.set_xlim(-5000, 200000)  # Set x limits
    ax.legend()  # Place the legend in the default location
    ax.grid(True)

    # Add the text from the params variable to the right of the plot
    text = fig.text(1.05, 0.5, params, fontsize=10, verticalalignment='center', transform=ax.transAxes)

    # Adjust the layout to provide more space on the right
    plt.tight_layout()

    # Save the plot with extra space for the text
    plt.savefig(file_path, bbox_inches='tight', bbox_extra_artists=[text])
    plt.close()

# for each column in test_data, get the distinct number of values, and the description
for col in test_data.columns:
    print(f"Column: {col}")
    print(test_data[col].describe())
    print(test_data[col].nunique())

run_synth(test_data, "cmp_no_bucketization",
        singularity_low_threshold = 0,
        range_low_threshold = 0,
)
run_synth(test_data, "cmp_no_noise_supp_bucket",
        singularity_low_threshold = 0,
        range_low_threshold = 0,
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data, "cmp_defaults")
run_synth(test_data, "cmp_no_noise_supp",
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data, "cmp_min_supp",
        low_threshold = 2,
        low_mean_gap = 1,
        layer_sd = 1,
)

run_synth(test_data, "cmp_target_no_bucketization",
        target = 'num2',
        singularity_low_threshold = 0,
        range_low_threshold = 0,
)
run_synth(test_data, "cmp_target_defaults",
        target = 'num2',
)
run_synth(test_data, "cmp_target_no_noise_supp",
        target = 'num2',
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data, "cmp_target_min_supp",
        target = 'num2',
        low_threshold = 2,
        low_mean_gap = 1,
        layer_sd = 1,
)

test_data_no_cat1 = test_data.drop(columns=['cat1'])
run_synth(test_data_no_cat1, "cmp_no_cat1_no_noise_supp",
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)

# remove the geo column from test_data
test_data_no_geo = test_data.drop(columns=['geo'])

run_synth(test_data_no_geo, "cmp_no_geo_no_bucketization",
        singularity_low_threshold = 0,
        range_low_threshold = 0,
)
run_synth(test_data_no_geo, "cmp_no_geo_no_noise_supp_bucket",
        singularity_low_threshold = 0,
        range_low_threshold = 0,
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data_no_geo, "cmp_no_geo_defaults")
run_synth(test_data_no_geo, "cmp_no_geo_no_noise_supp",
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data_no_geo, "cmp_no_geo_min_supp",
        low_threshold = 2,
        low_mean_gap = 1,
        layer_sd = 1,
)

run_synth(test_data_no_geo, "cmp_no_geo_target_no_bucketization",
        target = 'num2',
        singularity_low_threshold = 0,
        range_low_threshold = 0,
)
run_synth(test_data_no_geo, "cmp_no_geo_target_defaults",
        target = 'num2',
)
run_synth(test_data_no_geo, "cmp_no_geo_target_no_noise_supp",
        target = 'num2',
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)
run_synth(test_data_no_geo, "cmp_no_geo_target_min_supp",
        target = 'num2',
        low_threshold = 2,
        low_mean_gap = 1,
        layer_sd = 1,
)

run_synth(test_data_no_geo[['num1', 'num2']], "cmp_2col_defaults")
run_synth(test_data_no_geo[['num1', 'num2']], "cmp_2col_target_min_supp",
        low_threshold = 2,
        low_mean_gap = 1,
        layer_sd = 1,
)
run_synth(test_data_no_geo[['num1', 'num2']], "cmp_2col_target_no_noise_supp",
        layer_noise_sd = 0,
        low_threshold = 0,
        low_mean_gap = 0,
        layer_sd = 0,
)