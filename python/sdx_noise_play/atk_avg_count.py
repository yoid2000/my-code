import itertools
import os
import json
from buckets import Bucketizer, count_rows_by_col_vals
from table_generators import TableMaker
from attacker import Attacker

outFile = 'simple_avg_med_count.json'

def do_bucketize(bucketizer, col_vals, ndeep):
    target_columns = list(col_vals.keys())
    all_columns = list(bucketizer.df.columns)
    for i in range(ndeep):
        for comb in itertools.combinations(all_columns, len(target_columns)+i):
            if all(item in comb for item in target_columns):
                # This combination includes the target_columns
                bucketizer.bucketizer(columns=list(comb))

def run_attack(df, alg, col_vals, ndeep):
    true_count = count_rows_by_col_vals(df, col_vals)
    if true_count < 10:
        print(f"Table isn't big enough.")
        print(true_count, col_vals)
        quit()
    # Let's compare no noise with noise
    res = {'true': true_count}
    for sd in [0,1.0]:
        bucketizer = Bucketizer(df=df, std_dev = sd)
        do_bucketize(bucketizer, col_vals, ndeep)
        attacker = Attacker()
        attacker.add_buckets_by_combination(bucketizer.buckets_by_combination)
        est_count_avg, est_count_med = attacker.avg_med_count(col_vals)
        if sd == 0 and (est_count_avg != true_count or est_count_med != true_count):
            print(f"Bad no_noise {true_count}, {est_count_avg}, {est_count_med}")
            print(col_vals)
            quit()
        if sd == 0:
            continue
        if alg == 'simple_avg':
            res['est'] = est_count_avg
        elif alg == 'simple_med':
            res['est'] = est_count_med
    return res

def already_done(allResults, alg, C, V, N, ntar, ndeep):
    for res in allResults:
        if res['alg'] == alg and res['C'] == C and res['V'] == V and res['N'] == N and res['num_targets'] == ntar and res['explore_depth'] == ndeep:
            return True
    return False

def run_attacks(alg, C, V, ntar, ndeep):
    if os.path.exists(outFile):
        with open(outFile, 'r') as f:
            allResults = json.load(f)
    else:
        allResults = []
    # Every value needs a reasonable number of appearances with every other value
    # so that we don't suffer LCF
    N = 15
    for _ in range(ntar + ndeep):
        N *= V
    if already_done(allResults, alg, C, V, N, ntar, ndeep):
        print("already_done", alg, C, V, N, ntar, ndeep)
        return
    outcomes = []
    tests = []
    print(f"Try simple averaging attack with num_targets {ntar} and explore_depth {ndeep} ({N} rows, {C} columns, {V} values)")
    while True:
        tm = TableMaker()
        df = tm.simple_uniform(N=N, C=C, V=V)  # Example dataframe
        if len(outcomes) > max_samples:
            break
        for col_comb in itertools.combinations(list(df.columns), ntar):
            distinct_df = df[list(col_comb)].drop_duplicates()
            for _, row in distinct_df.iterrows():
                col_vals = {}
                for col in list(col_comb):
                    col_vals[col] = row[col]
                res = run_attack(df, alg, col_vals, ndeep)
                outcomes.append(1 if res['est'] == res['true'] else 0)
                tests.append(col_vals)
                print(f"{len(outcomes)},",flush=True,end='')
                if len(outcomes) > max_samples:
                    break
            if len(outcomes) > max_samples:
                break
    num_success = sum(outcomes)
    success_rate = int((num_success*100)/len(outcomes))
    print(f"{num_success} successes out of {len(outcomes)} for {success_rate}%")
    allResults.append({'N':N,'C':C,'V':V,'num_targets':ntar,'explore_depth':ndeep, 'alg':alg, 'samples':len(outcomes),'success_rate':success_rate})
    allResults = sorted(allResults, key=lambda x:x['success_rate'], reverse=True)
    with open(outFile, 'w') as f:
        json.dump(allResults, f, indent=4)
                
max_samples = 100
for V in [2, 5, 10]:
    for C in [10,20,40,80]:
        for alg in ['simple_avg', 'simple_med']:
            # ntar is the number of target columns (the number of columns for which we are
            # trying to learn the exact count)
            # ndeep is the number of additional levels we are going to sample
            for ndeep in [1,2,3]:
                for ntar in [1,2]:
                    run_attacks(alg, C, V, ntar, ndeep)
quit()
bucketizer = Bucketizer(df=df, std_dev = 0)
bucketizer.bucketizer(columns=['C1'])
bucketizer.bucketizer(columns=['C1', 'C2'])
bucketizer.bucketizer(columns=['C1', 'C3'])
bucketizer.bucketizer(columns=['C1', 'C4'])
bucketizer.bucketizer(columns=['C1', 'C5'])

attacker = Attacker()