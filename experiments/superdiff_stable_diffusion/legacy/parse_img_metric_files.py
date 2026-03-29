import os
import re
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import json
import scipy.stats as st

# BASEDIR = "/projects/superdiff/saved_sd_results"
BASEDIR = "/m/cs/work/blohmp1/comp_div1000/saved_sd_results"


with open("taskinfo.json") as f:
    TASKINFO = json.load(f)
print(TASKINFO)
TASKS = set([elem['dir_'] for elem in TASKINFO])
print("TASKS", TASKS)

with open("neg_taskinfo.json") as f:
    TASKINFO = json.load(f)
print(TASKINFO)
NEGTASKS = set([elem['dir_'] for elem in TASKINFO])
print("NEGTASKS", NEGTASKS)




def parse_csv(path_, metric=None, op='and'):
    df = pd.read_csv(path_)
    if op == "and":
        col = f"min_{metric}"
        vals = df[col].values
    elif "or" in op:
        col_1 = f"{metric}_raw_score_1"
        vals_1 = np.array(df[col_1].values)

        col_2 = f"{metric}_raw_score_2"
        vals_2 = np.array(df[col_2].values)
        if op == "or_diff":
            vals = np.abs(vals_1 - vals_2)
        elif op == "or_max":
            vals = [max(a, b) for a, b in zip(vals_1, vals_2)]
    elif op == "contrast":
        col_1 = f"{metric}_raw_score_1"
        vals_1 = np.array(df[col_1].values)

        col_2 = f"{metric}_raw_score_2"
        vals_2 = np.array(df[col_2].values)
        
        vals = vals_1 - vals_2

    return vals

def get_csvs(rootdir):
    rootdir = Path(rootdir)
    csv_list = sorted(rootdir.glob('*.csv'))
    return csv_list

def parse_clip_or_ir(metric, methods, op="and"):
    taskrank = {}
    for method in methods:
        print("="*100)
        results = []
        assert len(TASKS) == 20
        np.random.seed(42)
        for task in (sorted(TASKS) if op != "contrast" else sorted(NEGTASKS)):
            mod = "_T0.1"
            if method == "joint":
                
                if "or" in op:
                    mod = "_or"
                dir_ = f"{BASEDIR}/metrics_sd_ab{mod}"
                csv_ab = Path(dir_) / f"metrics_sd_ab{mod}_{task}.csv"
                task_results_ab = parse_csv(csv_ab, metric, op)

                dir_ = f"{BASEDIR}/metrics_sd_ba{mod}"
                csv_ba = Path(dir_) / f"metrics_sd_ba{mod}_{task}.csv"
                task_results_ba = parse_csv(csv_ba, metric, op)
                if np.mean(task_results_ab) >= np.mean(task_results_ba):
                    # print("AB CHOSEN")
                    task_results = task_results_ab
                else: 
                    # print("BA CHOSEN")
                    task_results = task_results_ba
            elif method == "coin_flip":
                dir_ = f"{BASEDIR}/metrics_sd_a"
                csv_a = Path(dir_) / f"metrics_sd_a_{task}.csv"
                task_results_a = parse_csv(csv_a, metric, op)

                dir_ = f"{BASEDIR}/metrics_sd_b"
                csv_b = Path(dir_) / f"metrics_sd_b_{task}.csv"
                task_results_b = parse_csv(csv_b, metric, op)
                
                coins = np.random.choice([0, 1], size=len(task_results_b))
                task_results = [task_results_a[i] if coins[i] == 0 else task_results_b[i] for i in range(len(coins))]

            else:
                dir_ = f"{BASEDIR}/metrics_{method}"
                csv = Path(dir_) / f"metrics_{method}_{task}.csv"
                # print("CURRENT TASK:", csv.name)
                task_results = parse_csv(csv, metric, op)
            # print("METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(task_results), np.std(task_results)))
            if task not in taskrank:
                taskrank[task] = {}
            taskrank[task][method] = np.mean(task_results)
            results.extend(task_results)
        # assert len(results) == 400
        print("GLOBAL RESULTS, METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(results), np.std(results)))


def parse_improvement_over_baseline(metric, methods, baseline_method, op="and"):
    taskrank = {}
    for method in methods:
        print("="*100)
        results = []
        assert len(TASKS) == 20
        np.random.seed(42)
        for task in (sorted(TASKS) if op != "contrast" else sorted(NEGTASKS)):
            mod = "_T0.1"
            if method == "joint":
                dir_ = f"{BASEDIR}/metrics_sd_ab{mod}"
                csv_ab = Path(dir_) / f"metrics_sd_ab{mod}_{task}.csv"
                task_results_ab = parse_csv(csv_ab, metric, op)

                dir_ = f"{BASEDIR}/metrics_sd_ba{mod}"
                csv_ba = Path(dir_) / f"metrics_sd_ba{mod}_{task}.csv"
                task_results_ba = parse_csv(csv_ba, metric, op)
                if np.mean(task_results_ab) >= np.mean(task_results_ba):
                    # print("AB CHOSEN")
                    task_results = task_results_ab
                else: 
                    # print("BA CHOSEN")
                    task_results = task_results_ba
            elif method == "coin_flip":
                dir_ = f"{BASEDIR}/metrics_sd_a"
                csv_a = Path(dir_) / f"metrics_sd_a_{task}.csv"
                task_results_a = parse_csv(csv_a, metric, op)

                dir_ = f"{BASEDIR}/metrics_sd_b"
                csv_b = Path(dir_) / f"metrics_sd_b_{task}.csv"
                task_results_b = parse_csv(csv_b, metric, op)
                
                coins = np.random.choice([0, 1], size=len(task_results_b))
                task_results = [task_results_a[i] if coins[i] == 0 else task_results_b[i] for i in range(len(coins))]

            else:
                dir_ = f"{BASEDIR}/metrics_{method}"
                csv = Path(dir_) / f"metrics_{method}_{task}.csv"
                # print("CURRENT TASK:", csv.name)
                task_results = parse_csv(csv, metric, op)

                bdir_ = f"{BASEDIR}/metrics_{baseline_method}"
                bcsv = Path(bdir_) / f"metrics_{baseline_method}_{task}.csv"
                # print("CURRENT TASK:", csv.name)
                btask_results = parse_csv(bcsv, metric, op)

            # print("METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(task_results-btask_results), np.std(task_results-btask_results)))
            if task not in taskrank:
                taskrank[task] = {}
            taskrank[task][method] = np.mean(task_results-btask_results)
            # print(task_results-btask_results)
            results.extend(task_results-btask_results)
        # assert len(results) == 400
        t_val = st.t.ppf(0.995, df=len(results)-1)
        se = np.std(results, ddof=1)/np.sqrt(len(results)) #N=400
        print("GLOBAL RESULTS, METHOD:: {}-{} | mean ± std:: {:.4f} ± {:.4f}, 99% CI: {}".format(method,baseline_method, np.mean(results), np.std(results), se*t_val))





op="and"
methods_and = ['and_T0.1', 'avg_T0.1', "joint", "dombi_and_T0.1", "dombi_and", "dombi_and_T10.0"]

# methods_and = ['and_T0.1', "dombi_and_T0.1", "dombi_and", "dombi_and_T10.0"]

op_c="contrast"
methods_contrast = ['icn', "and_not_superdiff", 
                    "dombi_contrast","dombi_contrast_gamma3.0","dombi_contrast_gamma10.0",
                    "dombi_contrast_T0.1","dombi_contrast_T0.1_gamma3.0","dombi_contrast_T0.1_gamma10.0",#]
                    "dombi_contrast_T10.0","dombi_contrast_T10.0_gamma3.0","dombi_contrast_T10.0_gamma10.0"]

# print("clip")
# parse_clip_or_ir("clip", methods_and, op=op)
# print("image rewards")
# parse_clip_or_ir("ir", methods_and, op=op)

print("clip")
parse_clip_or_ir("clip", methods_contrast, op=op_c)
print("image rewards")
parse_clip_or_ir("ir", methods_contrast, op=op_c)



print("improvements over and_T0.1 baseline")
# print("clip")
# parse_improvement_over_baseline("clip",methods_and,"and_T0.1",op)
# print("image reward")
# parse_improvement_over_baseline("ir",methods_and,"and_T0.1",op)

print("clip")
parse_improvement_over_baseline("clip",methods_contrast,"icn",op_c)
print("image reward")
parse_improvement_over_baseline("ir",methods_contrast,"icn",op_c)

# parse_tifa(methods_and, op)
