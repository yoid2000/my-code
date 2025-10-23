from df_builder import df_build, name_from_params, df_describe
from plotter import plot_pdf, plot_scatter, plot_heat, plot_bins
import matplotlib.pyplot as plt
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

do_quit = True

# make directory plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

nrows = 100
cols = [
    {'type': 'con', 'skew': 'none'},
    {'type': 'cat', 'nuniq': 5, 'skew': 'weak'},
    {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': 'strong', 'bumps': 0}
]
cors = [
    [[0, 1], 'strong'],
    [[1, 2], 'weak']
]


# Create several 2-column dataframes, both hybrid, for each correlation strength and each skew
for cors in [[[[0, 1], strength]] for strength in ['none', 'weak', 'strong', 'perfect']]:
    if cors[0][1] == 'none':
        cors = None
    for skews in ['none', 'weak', 'strong']:
        for bumps in range(4):
            cols = [
                {'type': 'con', 'skew': skews, 'bumps': bumps},
                {'type': 'con', 'skew': skews, 'bumps': bumps},
            ]
            df, params = df_build(nrows=nrows, cors=cors, cols=cols)
            file_name = name_from_params(params)
            print(f"Name: {file_name}")
            pp.pprint(params)
            print(df.describe())
            print(df_describe(df))
            file_name_col = f"{file_name}_both"
            plt = plot_scatter(df, file_name=file_name)
            plt.savefig(f"plots/{file_name_col}_scat.png")
            plt.close()
            plt = plot_heat(df, file_name=file_name)
            plt.savefig(f"plots/{file_name_col}_heat.png")
            plt.close()
            for col_index in [0,1]:
                file_name_col = f"{file_name}_col{col_index}"
                plt = plot_pdf(df[[df.columns[col_index]]], file_name=file_name_col, point_thresh=10)
                plt.savefig(f"plots/{file_name_col}.png")
                plt.close()

if do_quit: quit()

# Create a bunch of 1-column 'cat' dataframes, each with different skew
for skew in ['none', 'weak', 'mid', 'strong']:
    for bumps in range(4):
        cols = [
            {'type': 'cat', 'nuniq': 16, 'skew': skew, 'bumps': bumps}
        ]
        df, params = df_build(nrows=nrows, cors=None, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        plt = plot_bins(df, file_name=file_name)
        plt.savefig(f"plots/{file_name}.png")
        plt.close()

if do_quit: quit()


# Create a bunch of 1-column 'hybrid' dataframes, each with different skew and number of bumps (0 to 3)
for skew in ['none', 'weak', 'mid', 'strong']:
    for bumps in range(4):
        cols = [
            {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skew, 'bumps': bumps}
        ]
        df, params = df_build(nrows=nrows, cors=None, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        plt = plot_pdf(df, file_name=file_name, point_thresh=10)
        plt.savefig(f"plots/{file_name}.png")
        plt.close()

if do_quit: quit()

# Create a bunch of 1-column 'con' dataframes, each with different skew and number of bumps (0 to 3)
for skew in ['none', 'weak', 'mid', 'strong']:
    for bumps in range(4):
        cols = [
            {'type': 'con', 'skew': skew, 'bumps': bumps}
        ]
        df, params = df_build(nrows=nrows, cors=None, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        plt = plot_pdf(df, file_name=file_name, point_thresh=10)
        plt.savefig(f"plots/{file_name}.png")
        plt.close()

if do_quit: quit()

# Create several 2-column dataframes, both hybrid, for each correlation strength and each skew
for cors in [[[[0, 1], strength]] for strength in ['none', 'weak', 'strong', 'perfect']]:
    if cors[0][1] == 'none':
        cors = None
    for skews in ['none', 'weak', 'strong']:
        cols = [
            {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skews},
            {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skews},
        ]
        df, params = df_build(nrows=nrows, cors=cors, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        file_name_col = f"{file_name}_both"
        plt = plot_scatter(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_scat.png")
        plt.close()
        plt = plot_heat(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_heat.png")
        plt.close()
        for col_index in [0,1]:
            file_name_col = f"{file_name}_col{col_index}"
            plt = plot_pdf(df[[df.columns[col_index]]], file_name=file_name_col, point_thresh=10)
            plt.savefig(f"plots/{file_name_col}.png")
            plt.close()

if do_quit: quit()

# Create several 2-column dataframes, one continuous, one hybrid, for each correlation strength and each skew
for cors in [[[[0, 1], strength]] for strength in ['none', 'weak', 'strong', 'perfect']]:
    if cors[0][1] == 'none':
        cors = None
    for skews in ['none', 'weak', 'strong']:
        cols = [
            {'type': 'con', 'skew': skews},
            {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skews},
        ]
        df, params = df_build(nrows=nrows, cors=cors, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        file_name_col = f"{file_name}_both"
        plt = plot_scatter(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_scat.png")
        plt.close()
        plt = plot_heat(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_heat.png")
        plt.close()
        for col_index in [0,1]:
            file_name_col = f"{file_name}_col{col_index}"
            plt = plot_pdf(df[[df.columns[col_index]]], file_name=file_name_col, point_thresh=10)
            plt.savefig(f"plots/{file_name_col}.png")
            plt.close()

if do_quit: quit()

# Create several 1-column hybrid dataframes with different distributions
cors = None
for skews in ['none', 'weak', 'strong']:
    cols = [
        {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skews},
    ]
    df, params = df_build(nrows=nrows, cors=cors, cols=cols)
    file_name = name_from_params(params)
    print(f"Name: {file_name}")
    pp.pprint(params)
    print(df.describe())
    print(df_describe(df))
    plt = plot_pdf(df, file_name=file_name, point_thresh=10)
    plt.savefig(f"plots/{file_name}.png")
    plt.close()



# Create several 2-column dataframes, both continuous, for each correlation strength and each skew
for cors in [[[[0, 1], strength]] for strength in ['none', 'weak', 'strong', 'perfect']]:
    if cors[0][1] == 'none':
        cors = None
    for skews in ['none', 'weak', 'strong']:
        cols = [
            {'type': 'con', 'skew': skews},
            {'type': 'con', 'skew': skews}
        ]
        df, params = df_build(nrows=nrows, cors=cors, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        file_name_col = f"{file_name}_both"
        plt = plot_scatter(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_scat.png")
        plt.close()
        plt = plot_heat(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_heat.png")
        plt.close()
        for col_index in [0,1]:
            file_name_col = f"{file_name}_col{col_index}"
            plt = plot_pdf(df[[df.columns[col_index]]], file_name=file_name_col, point_thresh=10)
            plt.savefig(f"plots/{file_name_col}.png")
            plt.close()


# Create several 2-column dataframes, both categorical, for each correlation strength and each skew
for cors in [[[[0, 1], strength]] for strength in ['none', 'weak', 'strong', 'perfect']]:
    if cors[0][1] == 'none':
        cors = None
    for skews in ['none', 'weak', 'strong']:
        cols = [
            {'type': 'cat', 'nuniq': 10, 'skew': skews},
            {'type': 'cat', 'nuniq': 20, 'skew': skews},
        ]
        df, params = df_build(nrows=nrows, cors=cors, cols=cols)
        file_name = name_from_params(params)
        print(f"Name: {file_name}")
        pp.pprint(params)
        print(df.describe())
        print(df_describe(df))
        file_name_col = f"{file_name}_both"
        plt = plot_scatter(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_scat.png")
        plt.close()
        plt = plot_heat(df, file_name=file_name)
        plt.savefig(f"plots/{file_name_col}_heat.png")
        plt.close()
        for col_index in [0,1]:
            file_name_col = f"{file_name}_col{col_index}"
            plt = plot_pdf(df[[df.columns[col_index]]], file_name=file_name_col, point_thresh=10)
            plt.savefig(f"plots/{file_name_col}.png")
            plt.close()



# Create several 1-column continuous dataframes with different distributions
cors = None
for skews in ['none', 'weak', 'strong']:
    cols = [
        {'type': 'con', 'skew': skews},
    ]
    df, params = df_build(nrows=nrows, cors=cors, cols=cols)
    file_name = name_from_params(params)
    print(f"Name: {file_name}")
    pp.pprint(params)
    print(df.describe())
    print(df_describe(df))
    plt = plot_pdf(df, file_name=file_name, point_thresh=10)
    plt.savefig(f"plots/{file_name}.png")
    plt.close()

# Create several 1-column categorical dataframes with different distributions
cors = None
for skews in ['none', 'weak', 'strong']:
    cols = [
        {'type': 'cat', 'nuniq': 10, 'skew': skews},
    ]
    df, params = df_build(nrows=nrows, cors=cors, cols=cols)
    file_name = name_from_params(params)
    print(f"Name: {file_name}")
    pp.pprint(params)
    print(df.describe())
    print(df_describe(df))
    plt = plot_pdf(df, file_name=file_name, point_thresh=10)
    plt.savefig(f"plots/{file_name}.png")
    plt.close()

