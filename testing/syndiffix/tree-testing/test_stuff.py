import itertools
import sys
sys.path.append(r"C:\paul\GitHub\my-code\testing\syndiffix\df-builders")
import matplotlib.pyplot as plt
import os
from df_builder import df_build, df_describe, name_from_params
from plotter import plot_pdf, plot_scatter, plot_heat
from syndiffix import Synthesizer
from syndiffix.tools import tree_to_df, df_to_tree, dump_placeholder_tree, row_to_node, TestNodeForest, plot_2d_nodes_boxes, plot_1d_nodes_bars, plot_1d_orig_anon_cdf, ks_measure
from leaf_stuff import *
import pprint

def run_test(nrows, cors, cols, range_extend_fraction=0.25, dump_nodes=False, leafs_mode='none'):
    print(f"Running test with nrows={nrows}, cors={cors}, cols={cols}, range_extend_fraction={range_extend_fraction}, dump_nodes={dump_nodes}, leafs_mode={leafs_mode}")
    display_counts = False
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    results = {}
    results['run_params'] = {'range_extend_fraction': range_extend_fraction,
                             'leafs_mode': leafs_mode}
    df, params = df_build(nrows=nrows, cors=cors, cols=cols)
    results['dataset_params'] = params
    name = name_from_params(params)
    lmc = {'none': 'no', 'simple': 'sim', 'leaf_only': 'lfo'}
    name += f"_ref{int(range_extend_fraction*100)}_lm{lmc[leafs_mode]}"
    results_path = f"results/{name}.json"
    # Check to see if results file already exists, and return if it does
    if os.path.exists(results_path):
        print(f"Results file {results_path} already exists. Skipping test.")
        return None, None
    results['dataset_params']['name'] = name
    print(f"Dataset name: {name}")
    describe_str, describe_dict = df_describe(df)
    print(describe_str)
    results['dataset_describe'] = describe_dict

    plot_dir = f'plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)
    # for each column in df that is a float, run plot_pdf
    for col in df.columns:
        plt = plot_pdf(df[[col]], file_name=name)
        plt.savefig(f"{plot_dir}/pdf_col{col}.png")
        plt.close()
    # for each pair of columns in df, run plot_scatter and plot_heat
    if len(df.columns) >= 2:
        for col1 in range(len(df.columns)):
            for col2 in range(col1+1, len(df.columns)):
                df_pair = df[[df.columns[col1], df.columns[col2]]]
                plt = plot_scatter(df_pair, file_name=name)
                plt.savefig(f"{plot_dir}/scat_cols{col1}_{col2}.png")
                plt.close()
                plt = plot_heat(df_pair, file_name=name)
                plt.savefig(f"{plot_dir}/heat_cols{col1}_{col2}.png")
                plt.close()

    syn = Synthesizer(df)
    df_sdx = syn.sample()
    some_problem_found = False
    tnf = TestNodeForest(syn, leafs_mode=leafs_mode, range_extend_fraction=range_extend_fraction)
    # use itertools to iterate through every combination of columns
    for r in range(1, len(df.columns) + 1):
        for comb in itertools.combinations(range(len(df.columns)), r):
            print("==========================================================")
            print(f"Combination: {comb}:")
            nodes = tnf.test_nodes[comb]
            #print("Original tree:")
            #nodes.dump_tree_from_root()
            print(nodes.df_tree_in.columns)
            print(nodes.df_tree_in.dtypes)
            #print(df_tree.to_string())
            tree = df_to_tree(nodes.df_tree_in)
            #print("Reconstructed placeholder tree:")
            #dump_placeholder_tree(tree)
            # Loop through df_tree, convert each row to a node, and print the node
            #print("Rows as nodes:")
            for index, row in nodes.df_tree_in.iterrows():
                d = row_to_node(row)
                #print(f"Row {index} as dict:")
                #pp.pprint(d)
            df_test = nodes.assigned_values_ex
            results['nodes_stats'] = nodes.nodes_in_stats
            results['nodes_supp_stats'] = nodes.nodes_supp_in_stats
            results['leafs_stats'] = nodes.leafs_in_stats
            print(f"All nodes: {len(nodes.nodes_in)}")
            pp.pprint(nodes.nodes_in_stats)
            if dump_nodes:
                pp.pprint(nodes.nodes_in)
            print(f"All non-suppressed nodes from nodes_supp: {len(nodes.nodes_supp_in)}")
            pp.pprint(nodes.nodes_supp_in_stats)
            if dump_nodes:
                pp.pprint(nodes.nodes_supp_in)
            print(f"All leafs: {len(nodes.leafs_in)}")
            pp.pprint(nodes.leafs_in_stats)
            if dump_nodes:
                pp.pprint(nodes.leafs_in)
            problems_found, integrity_results = nodes.integrity_checks()
            results['integrity_check'] = {'problems_found': problems_found, 'results': integrity_results}
            if problems_found:
                some_problem_found = True
                if dump_nodes is False:
                    print("Nodes:")
                    pp.pprint(nodes.nodes_in)
                    print("Nodes_supp:")
                    pp.pprint(nodes.nodes_supp_in)
                    print("Leafs:")
                    pp.pprint(nodes.leafs_in)
                print("Integrity problems found:")
                pp.pprint(integrity_results)
            if len(comb) == 2:
                file_name = f"2d_nodes_{'_'.join(map(str, comb))}"
                plt = plot_2d_nodes_boxes(nodes.nodes_in, [0,1], file_name=name, display_counts=display_counts)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()
                file_name = f"2d_nodes_supp_{'_'.join(map(str, comb))}"
                plt = plot_2d_nodes_boxes(nodes.nodes_supp_in, [0,1], file_name=name, display_counts=display_counts)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()
                file_name = f"2d_leafs_{'_'.join(map(str, comb))}"
                plt = plot_2d_nodes_boxes(nodes.leafs_in, [0,1], file_name=name, display_counts=display_counts)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()
                for i, col_index in enumerate(comb):
                    file_name = f"1d_2_nodes_col{col_index}_{'_'.join(map(str, comb))}"
                    plt = plot_1d_nodes_bars(nodes.nodes_in, i, file_name=name)
                    plt.savefig(f"{plot_dir}/{file_name}.png")
                    plt.close()
                    file_name = f"1d_2_nodes_col{col_index}_supp_{'_'.join(map(str, comb))}"
                    plt = plot_1d_nodes_bars(nodes.nodes_supp_in, i, file_name=name)
                    plt.savefig(f"{plot_dir}/{file_name}.png")
                    plt.close()
                    file_name = f"1d_2_leafs_col{col_index}_{'_'.join(map(str, comb))}"
                    plt = plot_1d_nodes_bars(nodes.leafs_in, i, file_name=name)
                    plt.savefig(f"{plot_dir}/{file_name}.png")
                    plt.close()
            if len(comb) == 1:
                col_name = df.columns[comb[0]]
                col_index = comb[0]
                ks_sdx, _ = ks_measure(df[[col_name]], df_sdx[[col_name]])
                results['ks_sdx'] = ks_sdx
                ks_test = None
                file_name = f"1d_cdf_col{col_index}_{'_'.join(map(str, comb))}"
                if df_test is not None:
                    ks_test, _ = ks_measure(df[[col_name]], df_test[[col_name]])
                    results['ks_test'] = ks_test
                    title = f"\nKS Synthesizer: {ks_sdx:.4f}, KS TestNodes: {ks_test:.4f}"
                    plt = plot_1d_orig_anon_cdf(df_orig=df[[col_name]], df_sdx=df_sdx[[col_name]], df_test=df_test[[col_name]], file_name=name+title)
                    plt.savefig(f"{plot_dir}/{file_name}.png")
                    plt.close()
                else:
                    title = f"\nKS Synthesizer: {ks_sdx:.4f}"
                    plt = plot_1d_orig_anon_cdf(df_orig=df[[col_name]], df_sdx=df_sdx[[col_name]], file_name=name+title)
                    plt.savefig(f"{plot_dir}/{file_name}.png")
                    plt.close()
                file_name = f"1d_1_nodes_col{col_index}_{'_'.join(map(str, comb))}"
                plt = plot_1d_nodes_bars(nodes.nodes_in, 0, file_name=name)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()
                file_name = f"1d_1_nodes_col{col_index}_supp_{'_'.join(map(str, comb))}"
                plt = plot_1d_nodes_bars(nodes.nodes_supp_in, 0, file_name=name)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()
                file_name = f"1d_1_leafs_col{col_index}_{'_'.join(map(str, comb))}"
                plt = plot_1d_nodes_bars(nodes.leafs_in, 0, file_name=name)
                plt.savefig(f"{plot_dir}/{file_name}.png")
                plt.close()

    # Save results as json file
    with open(f"results/{name}.json", "w") as f:
        import json
        json.dump(results, f, indent=4)
    return results, some_problem_found