import itertools
import os
import json
from buckets import Bucketizer, count_rows_by_col_vals
from table_generators import TableMaker
from attacker import Attacker

def do_bucketize(bucketizer, col_vals, btype):
    # bucketize the single column
    target_columns = list(col_vals.keys())
    bucketizer.bucketizer(columns=target_columns)
    if btype == '2dim':
        # Get all combinations with target_columns and one other
        all_columns = list(bucketizer.df.columns)
        for comb in itertools.combinations(all_columns, len(target_columns)+1):
            if all(item in comb for item in target_columns):
                # This combination includes the target_columns
                bucketizer.bucketizer(columns=list(comb))

def run_attack(col_vals, btype):
    true_count = count_rows_by_col_vals(df, col_vals)
    # Let's compare no noise with noise
    res = {'true': true_count}
    for sd, label in [[0,'no_noise'],[1.0,'with_noise']]:
        bucketizer = Bucketizer(df=df, std_dev = sd)
        do_bucketize(bucketizer, col_vals, btype)
        attacker = Attacker()
        attacker.add_buckets_by_combination(bucketizer.buckets_by_combination)
        est_count_avg, est_count_med = attacker.avg_med_count(col_vals)
        res[label+'_avg'] = est_count_avg
        res[label+'_med'] = est_count_med
    return res

def already_done(allResults, C, V, N, btype):
    for res in allResults:
        if res['C'] == C and res['V'] == V and res['N'] == N and res['type'] == btype:
            return True
    return False

outFile = 'simple_avg_med_count.json'
if os.path.exists(outFile):
    with open(outFile, 'r') as f:
        allResults = json.load(f)
else:
    allResults = []

for C in [10,20,40,80]:
    for V in [2, 5, 10, 20]:
        # Every value needs a reasonable number of appearances with every other value
        # so that we don't suffer LCF
        N = V * V * 20
        for btype in ['2dim', '1dim']:
            if already_done(allResults, C, V, N, btype):
                print("already_done", C, V, N, btype)
                continue
            outcomes_avg = []
            outcomes_med = []
            tests = []
            print(f"Try simple averaging attack with {btype} ({N} rows, {C} columns, {V} values)")
            for _ in range(5):
                tm = TableMaker()
                df = tm.simple_uniform(N=N, C=C, V=V)  # Example dataframe

                for column in list(df.columns):
                    for value in df[column].unique():
                        col_vals = {column:value}
                        res = run_attack(col_vals, btype)
                        if (res['no_noise_avg'] != res['true'] or 
                            res['no_noise_med'] != res['true']):
                            print(f"Bad no_noise {res}")
                            quit()
                        outcomes_avg.append(1 if res['with_noise_avg'] == res['true'] else 0)
                        outcomes_med.append(1 if res['with_noise_med'] == res['true'] else 0)
                        tests.append(col_vals)
            num_success_avg = sum(outcomes_avg)
            success_rate_avg = int((num_success_avg*100)/len(outcomes_avg))
            print(f"{num_success_avg} successes out of {len(outcomes_avg)} for {success_rate_avg}%")
            allResults.append({'N':N,'C':C,'V':V,'type':btype,'alg':'simple avg', 'samples':len(outcomes_avg),'success_rate':success_rate_avg})
            num_success_med = sum(outcomes_med)
            success_rate_med = int((num_success_med*100)/len(outcomes_med))
            print(f"{num_success_med} successes out of {len(outcomes_med)} for {success_rate_med}%")
            allResults.append({'N':N,'C':C,'V':V,'type':btype,'alg':'simple med', 'samples':len(outcomes_med),'success_rate':success_rate_med})
            allResults = sorted(allResults, key=lambda x:x['success_rate'], reverse=True)
            with open(outFile, 'w') as f:
                json.dump(allResults, f, indent=4)
                
quit()
bucketizer = Bucketizer(df=df, std_dev = 0)
bucketizer.bucketizer(columns=['C1'])
bucketizer.bucketizer(columns=['C1', 'C2'])
bucketizer.bucketizer(columns=['C1', 'C3'])
bucketizer.bucketizer(columns=['C1', 'C4'])
bucketizer.bucketizer(columns=['C1', 'C5'])

attacker = Attacker()