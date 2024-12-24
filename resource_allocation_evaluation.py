from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 10)

plt.rcParams['font.family'] = 'serif'

# --- JOB EXECUTION DATA LOADING --- #

CLOUD_EXECUTION_RUNTIME_DATA = './gcp_runtime_data/gcp_dataproc.csv'
JOB_PROFILING_DATA_DIR = './profiling_data/'

"""
    GCP on demand VM costs (USD/h) in their Frankfurt data center as of December 1st 2024
    https://cloud.google.com/compute/vm-instance-pricing

    - $0.04073  USD per vCPU per hour
    - $0.005458 USD per GB memory per hour

    Examples:

    GCP VM name   CPU RAM  USD/h
    n2-highcpu-8   8  8GB $0.369504
    n2-highcpu-32 32 32GB $1.478016

    n2-standard-4  4 16GB $0.250248
    n2-standard-8  8 32GB $0.500496

    n2-highmem-4   4 32GB $0.337576
    n2-highmem-8   8 64GB $0.675152

"""

HOURLY_VCPU_COST = 0.040730
HOURLY_GB_MEM_COST = 0.005458
GiB = 2**30
NODE_MEMORY_OVERHEAD = 1.5*GiB # OS + Spark

MEMORY_DEMANDING_ALGORITHMS = {'sort', 'join', 'kmeans', 'logisticregression', 'linearregression'}
MEMORY_YIELDING_ALGORITHMS = {'grep', 'wordcount', 'groupbycount', 'selectwhereorderby'}

config_names = {  # (scale-out, total vCPU cores in cluster, total GB RAM in cluster)
    (8, 64, 64):    '#01',
    (8, 64, 256):   '#02',
    (8, 64, 512):   '#03',
    (4, 16, 128):   '#04',
    (4, 32, 128):   '#05',
    (4, 128, 128):  '#06',
    (2, 16, 128):   '#07',
    (8, 32, 128):   '#08',
    (16, 64, 256):  '#09',
    (16, 128, 128): '#10',
}

cloud_df = pd.read_csv(CLOUD_EXECUTION_RUNTIME_DATA)

# Derive additional values of interest
cloud_df['total_cluster_memory'] = cloud_df['memory_gb_per_node'] * cloud_df['scaleout']
cloud_df['total_cluster_vcpu'] = cloud_df['vcpu_per_node'] * cloud_df['scaleout']
cloud_df['memory_gb_per_vcpu'] = cloud_df['memory_gb_per_node'] // cloud_df['vcpu_per_node']
cloud_df['gb_mem_seconds'] = cloud_df['total_cluster_memory'] * cloud_df['runtime']
cloud_df['vcpu_seconds'] = cloud_df['total_cluster_vcpu'] * cloud_df['runtime']
cloud_df['dollar_cost'] = \
    cloud_df['vcpu_seconds'] * HOURLY_VCPU_COST/3600 \
    + cloud_df['gb_mem_seconds'] * HOURLY_GB_MEM_COST/3600

for _, job_df in cloud_df.groupby(['algorithm', 'dataset_size_bytes']):
    cloud_df.loc[job_df.index, 'normalized_runtime'] = \
        job_df['runtime'] / job_df['runtime'].min()
    cloud_df.loc[job_df.index, 'normalized_dollar_cost'] = \
        job_df['dollar_cost'] / job_df['dollar_cost'].min()
    cloud_df.loc[job_df.index, 'normalized_gb_mem_seconds'] = \
        job_df['gb_mem_seconds'] / job_df['gb_mem_seconds'].min()
    cloud_df.loc[job_df.index, 'normalized_vcpu_seconds'] =  \
        job_df['vcpu_seconds'] / job_df['vcpu_seconds'].min()


algorithms = set(cloud_df['algorithm'])
config_keys = ['scaleout', 'total_cluster_vcpu', 'total_cluster_memory']
cluster_configs = cloud_df[config_keys].drop_duplicates()

# --- JOB EXECUTION DATA ANALYSIS & VISUALIZATION --- #


# print(f"\nColumns of the Dataframe:\n{', '.join(cloud_df.keys())}\n")
# print(f"\nData analytics algorithms:\n{', '.join(algorithms)}")
# print(f"\nCluster configuration options:\n{cluster_configs.sort_values(config_keys)}")
# print(f"\nResulting Costs:\n{cloud_df['normalized_dollar_cost'].describe()}")
# print(f"\nResulting Runtimes:\n{cloud_df['normalized_runtime'].describe()}")


# --- CLUSTER RESOURCE SELECTION EVALUATION --- #


def get_row(df, **kwargs):
    rows = df.query(" and ".join(f"{k} == {repr(v)}" for k, v in kwargs.items()))
    assert len(rows) == 1, f"{rows}\n{kwargs}"
    return rows.iloc[0]


def get_rows(df, **kwargs):
    rows = df.query(" and ".join(f"{k} == {repr(v)}" for k, v in kwargs.items()))
    return rows


def usable_memory_bytes(config):
    scaleout, _, total_cluster_memory_GiB = config
    return total_cluster_memory_GiB*GiB - scaleout*NODE_MEMORY_OVERHEAD


# -- Selection strategies -- #

def minimize_cpu(algorithm, dataset_size, min_usable_memory_bytes=0):

    total_vcpu_rankings = sorted([
        (usable_memory_bytes(config) < min_usable_memory_bytes,
         df['total_cluster_vcpu'].mean(),
         config)
        for config, df in cloud_df.groupby(config_keys)
    ])  # [(x,y,z), (...), ...]

    _, __, lowest_vcpu_config = total_vcpu_rankings[0]

    return pd.Series(lowest_vcpu_config, index=config_keys)


def maximize_cpu(algorithm, dataset_size, min_usable_memory_bytes=0):

    total_vcpu_rankings = sorted([
        (usable_memory_bytes(config) < min_usable_memory_bytes,
         df['total_cluster_vcpu'].mean(),
         config)
        for config, df in cloud_df.groupby(config_keys)
    ])  # [(x,y,z), (...), ...]

    _, __, highest_vcpu_config = total_vcpu_rankings[-1]
    return pd.Series(highest_vcpu_config, index=config_keys)


def minimize_memory(algorithm, dataset_size, min_usable_memory_bytes=0):
    # unique config with 64GB RAM when min_usable_memory_bytes==0

    total_memory_rankings = sorted([
        (usable_memory_bytes(config) < min_usable_memory_bytes,
         df['total_cluster_memory'].mean(),
         df['total_cluster_vcpu'].mean(),
         df['scaleout'].mean(),
         config)
        for config, df in cloud_df.groupby(config_keys)
    ])
    lowest_memory_config = total_memory_rankings[0][-1]
    return pd.Series(lowest_memory_config, index=config_keys)


def maximize_memory(algorithm, dataset_size):  # unique config with 256GB RAM
    return cluster_configs.sort_values('total_cluster_memory').iloc[-1]


def random_selection():
    pass


def Juggler(algorithm, dataset_size):

    profiling_df = pd.read_csv(JOB_PROFILING_DATA_DIR + 'cache_usage_monitoring.csv')
    profiling_df = profiling_df.query(f"algorithm == '{algorithm}'")

    x, y = profiling_df['dataset_size_bytes'], profiling_df['max_cached_bytes']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    is_linear = abs(r_value) > .9
    if not is_linear:
        return None

    predicted_cache_use = slope * dataset_size + intercept
    return minimize_memory(algorithm, dataset_size, min_usable_memory_bytes=predicted_cache_use)


def Crispy(algorithm, dataset_size):

    def BFA(algorithm, dataset_size):
        all_other_jobs = cloud_df.query(f"algorithm in {list(algorithms - {algorithm})}")

        config_cheapness_rankings = sorted([ ( df['normalized_dollar_cost'].mean(), config)
            for config, df in all_other_jobs.groupby(config_keys) ])

        _, bfa = config_cheapness_rankings[0]
        return pd.Series(bfa, index=config_keys)

    profiling_df = pd.read_csv(JOB_PROFILING_DATA_DIR + 'psrecord_monitoring.csv') \
        .query(f"algorithm == '{algorithm}'")

    x, y = profiling_df['dataset_size_bytes'], profiling_df['max_allocated_memory_bytes']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    is_linear = abs(r_value) > .9
    if not is_linear:
        return BFA(algorithm, dataset_size)

    predicted_cache_use = slope * dataset_size + intercept

    return minimize_cpu(algorithm, dataset_size, min_usable_memory_bytes=predicted_cache_use)


def Flora_one_class(algorithm, dataset_size, min_usable_memory_bytes=0):

    all_other_algorithms = algorithms - {algorithm}
    assert len(all_other_algorithms) == 8
    all_other_jobs = cloud_df.query(f"algorithm in {list(all_other_algorithms)}")

    config_cheapness_rankings = sorted([
        (usable_memory_bytes(config) < min_usable_memory_bytes,
         df['normalized_dollar_cost'].mean(),
         config)
        for config, df in all_other_jobs.groupby(config_keys)
    ])  # [(...), (...), ...]

    has_too_little_memory, _, cheapest_config_for_other_jobs = config_cheapness_rankings[0]
    if has_too_little_memory:
        print(f"Flora_one_class could not find a config with enough memory."
              f"for {algorithm} with input dataset size {dataset_size//GiB}GiB.")
    return pd.Series(cheapest_config_for_other_jobs, index=config_keys)


def Flora(algorithm, dataset_size):

    if algorithm in MEMORY_DEMANDING_ALGORITHMS:
        related_algorithms = MEMORY_DEMANDING_ALGORITHMS - {algorithm}
    else:
        related_algorithms = MEMORY_YIELDING_ALGORITHMS - {algorithm}

    related_jobs = cloud_df.query(f"algorithm in {list(related_algorithms)}")

    config_cheapness_rankings = sorted([
        (df['normalized_dollar_cost'].mean(), config) for config, df in related_jobs.groupby(config_keys)
    ])

    _, cheapest_config_for_related_jobs = config_cheapness_rankings[0]
    return pd.Series(cheapest_config_for_related_jobs, index=config_keys)


def min_cost(algorithm, dataset_size):
    job = cloud_df.query("algorithm == @algorithm and dataset_size_bytes == @dataset_size")
    cheapest_config_for_given_job = job.sort_values('normalized_dollar_cost').iloc()[0]
    return pd.Series(cheapest_config_for_given_job, index=config_keys)


def min_runtime(algorithm, dataset_size):
    job = cloud_df.query("algorithm == @algorithm and dataset_size_bytes == @dataset_size")
    fastest_config_for_given_job = job.sort_values('normalized_runtime').iloc()[0]
    return pd.Series(fastest_config_for_given_job, index=config_keys)


# -- Evaluation -- #

mean = lambda iterable: np.array(iterable).mean()

approaches = [
    random_selection,
    minimize_cpu,
    maximize_cpu,
    minimize_memory,
    maximize_memory,
    Juggler,
    Flora_one_class,
    Crispy,
    Flora,
    min_cost,
    min_runtime,
]


def evaluate(resource_selection_strategy, algorithms=algorithms, verbose=False):
    if resource_selection_strategy == random_selection:
        return (cloud_df['normalized_dollar_cost'].mean(), cloud_df['normalized_runtime'].mean())

    normalized_dollar_costs = []
    normalized_runtimes = []
    if verbose:
        print(f"{'Algorithm':23s}  dataset  cost  rt  config")
    for algorithm in sorted(algorithms):
        dataset_sizes = sorted(set(get_rows(cloud_df, algorithm=algorithm)['dataset_size_bytes']))
        for dataset_size in dataset_sizes:
            config = resource_selection_strategy(algorithm, dataset_size)
            if config is None:
                continue

            # print(f"{algorithm:20s} {dataset_size//(GiB):7d}GiB {tuple(config)}")

            execution = get_row(cloud_df, algorithm=algorithm,
                                dataset_size_bytes=dataset_size, **config)

            normalized_dollar_costs.append(execution['normalized_dollar_cost'])
            normalized_runtimes.append(execution['normalized_runtime'])
            if verbose:
                print(f"{algorithm:18s} {dataset_size/(GiB):7.0f} GiB  "
                      f"{execution['normalized_dollar_cost']:.3f} "
                      f"{execution['normalized_runtime']:.3f} "
                      f"{str(tuple(config)):14s}"
                      f"{config_names[tuple(config)]:14s}")

    res = (mean(normalized_dollar_costs), mean(normalized_runtimes))
    return res


def run_main_experiment(results_file_name):

    res = []
    for approach in approaches:
        normalized_dollar_cost, normalized_runtime = evaluate(approach)
        res.append((approach.__name__, normalized_dollar_cost, normalized_runtime))

    columns = ["approach", "normalized_dollar_cost", "normalized_runtime"]
    pd.DataFrame(res, columns=columns).to_csv(results_file_name, index=False)


def present_main_experiment(results_file_name):

    df = pd.read_csv(results_file_name)
    print(df.sort_values(['normalized_dollar_cost']).iloc[::-1])


def run_mislabeling_experiment(results_file_name):
    # How do misclassifications impact overall selection quality?

    def evaluate_Flora_misclassification(algorithm):

        global MEMORY_DEMANDING_ALGORITHMS
        global MEMORY_YIELDING_ALGORITHMS
        MEMORY_DEMANDING_ALGORITHMS_ORIGINAL = {'sort', 'join', 'kmeans', 'logisticregression', 'linearregression'}
        MEMORY_YIELDING_ALGORITHMS_ORIGINAL = {'grep', 'wordcount', 'groupbycount', 'selectwhereorderby'}

        if algorithm in MEMORY_DEMANDING_ALGORITHMS:
            MEMORY_DEMANDING_ALGORITHMS -= {algorithm}
            MEMORY_YIELDING_ALGORITHMS |= {algorithm}
        else:
            MEMORY_YIELDING_ALGORITHMS -= {algorithm}
            MEMORY_DEMANDING_ALGORITHMS |= {algorithm}

        res = evaluate(Flora, {algorithm})
        MEMORY_DEMANDING_ALGORITHMS = MEMORY_DEMANDING_ALGORITHMS_ORIGINAL
        MEMORY_YIELDING_ALGORITHMS = MEMORY_YIELDING_ALGORITHMS_ORIGINAL

        return res

    res = []
    for algorithm in algorithms:
        normalized_dollar_cost, normalized_runtime = evaluate(Flora, {algorithm})
        res.append(('Flora', algorithm, True, normalized_dollar_cost, normalized_runtime))
        normalized_dollar_cost, normalized_runtime = evaluate_Flora_misclassification(algorithm)
        res.append(('Flora', algorithm, False, normalized_dollar_cost, normalized_runtime))

    columns = ["approach", "algorithm", "correct_classification",
               "normalized_dollar_cost", "normalized_runtime"]
    pd.DataFrame(res, columns=columns).to_csv(results_file_name, index=False)


def present_mislabeling_experiment(results_file_name):

    df = pd.read_csv(results_file_name)
    print(df[["algorithm", "correct_classification", "normalized_dollar_cost"]].sort_values(['algorithm', 'correct_classification']))


def run_labeling_accuracy_experiment(results_file_name):

    run_mislabeling_experiment(results_file_name='results/flora_mislabeling.csv')
    df = pd.read_csv('results/flora_mislabeling.csv')

    res = []
    for num_misclassifications in range(0, len(algorithms)+1):
        accumulate_cost = []
        accumulate_runtime = []
        for misclassified_algos in combinations(algorithms, num_misclassifications):
            wellclassified_algos = algorithms - set(misclassified_algos)
            x = df.query('algorithm in @wellclassified_algos and correct_classification == True '
                     'or algorithm in @misclassified_algos and correct_classification == False')
            accumulate_cost.append(x['normalized_dollar_cost'].mean())
            accumulate_runtime.append(x['normalized_runtime'].mean())
        # print(('Flora', num_misclassifications, np.array(accumulate_cost).mean(), np.array(accumulate_runtime).mean()))
        res.append(('Flora', num_misclassifications, np.array(accumulate_cost).mean(), np.array(accumulate_runtime).mean()))

    columns = ["approach", "num_misclassifications", "mean_normalized_dollar_cost", "mean_normalized_runtime"]
    pd.DataFrame(res, columns=columns).to_csv(results_file_name, index=False)


def present_labeling_accuracy_experiment(results_file_name):
    df = pd.read_csv(results_file_name)
    print(df[['approach','num_misclassifications','mean_normalized_dollar_cost','mean_normalized_runtime']])

    plt.plot([-1, 12], [1.940660, 1.940660], '-', linewidth=2.0, label='random selection', color='darkgray')
    plt.plot([-1, 12], [1.336259, 1.336259], '-', linewidth=2.0, label='Flora with one class', color='cornflowerblue')
    estimated_cost_when_guessing = (1.484+1.592)/2
    plt.plot([9/2, 9/2], [0, estimated_cost_when_guessing], '-', linewidth=1.5, color='purple')
    plt.plot([-1, 9/2], [estimated_cost_when_guessing, estimated_cost_when_guessing], linestyle=(1,(3.45,7.0)), linewidth=1.5, color='purple')
    plt.text(9/2+0.05, 1.2, r'$\leftarrow$ Guessing the class', verticalalignment='top', horizontalalignment='left', color='purple', fontsize=11)
    plt.text(9/2+0.05, 1.145, r'     with a coin flip', verticalalignment='top', horizontalalignment='left', color='purple', fontsize=11)

    plt.plot(
        df['num_misclassifications'],
        df['mean_normalized_dollar_cost'],
        linestyle=(1,(3, 6.0)),
        linewidth=1.5,
        color='blue',
    )
    plt.scatter(
        df['num_misclassifications'],
        df['mean_normalized_dollar_cost'],
        marker='x',
        color='blue',
        s=80,
        linewidths=2.5,
        label="Flora"
    )

    plt.grid(color='gray', linestyle='-', linewidth=.2, zorder=3, alpha=.4, axis='y')
    plt.legend(loc='upper left')
    plt.title("Flora's Classification Accuracy and Execution Cost", fontsize=14)
    plt.ylim((0.9999, 2.2))
    plt.xlim((-.2, 9.2))
    # plt.xscale('log')
    plt.xticks(range(0,10), [r'$\frac{'+str(n*2)+'}{18}$' for n in range(0,10)], fontsize=15)
    plt.yticks(np.arange(1.0, 2.3, 0.1))
    plt.xlabel('Ratio of misclassified jobs when using Flora', fontsize=13)
    plt.ylabel('Monetary cost of execution [normalized]', fontsize=13)
    plt.savefig('results/labeling_accuracy_experiment.pdf', bbox_inches='tight')
    plt.savefig('results/labeling_accuracy_experiment.svg', bbox_inches='tight')
    #plt.show()
    plt.clf()


def run_mem_cpu_cost_ratio_experiment(results_file_name):

    res = []
    for approach in approaches:

        HOURLY_VCPU_COST = 1
        HOURLY_GB_MEM_COST = 0.13400441934691873

        for mem_cost in np.arange(0.01, 11.01, 0.01):
            HOURLY_GB_MEM_COST = round(mem_cost, 2)
            cloud_df['dollar_cost'] = \
                cloud_df['vcpu_seconds'] * HOURLY_VCPU_COST/3600 \
                + cloud_df['gb_mem_seconds'] * HOURLY_GB_MEM_COST/3600

            for _, job_df in cloud_df.groupby(['algorithm', 'dataset_size_bytes']):
                cloud_df.loc[job_df.index, 'normalized_runtime'] = \
                    job_df['runtime'] / job_df['runtime'].min()
                cloud_df.loc[job_df.index, 'normalized_dollar_cost'] = \
                    job_df['dollar_cost'] / job_df['dollar_cost'].min()


            normalized_dollar_cost, normalized_runtime = evaluate(approach)
            res.append((approach.__name__, f'{mem_cost:.02f}', normalized_dollar_cost))

    columns = ["approach", "mem_cpu_cost_ratio", "normalized_dollar_cost"]
    pd.DataFrame(res, columns=columns).to_csv(results_file_name, index=False)


def present_mem_cpu_cost_ratio_experiment(results_file_name):
    df = pd.read_csv(results_file_name)

    fat, thin, very_thin = 2.5, 1.5, 1.0
    line_properties = {
        'minimize_cpu': ('minimize CPU', 'darkgray', thin, (0, (1, 5.0))),
        'maximize_cpu': ('maximize CPU', 'darkgray', thin, (0, (1, 1.5))),
        'minimize_memory': ('minimize memory', 'darkgray', thin, (0, (3, 6.0))),
        'maximize_memory': ('maximize memory', 'darkgray', thin, (0, (3, 1.5))),
        'random_selection': ('random selection', 'darkgray', very_thin, 'solid'),
        'Crispy': ('Crispy', 'red', fat, 'solid'),
        'Juggler': ('Juggler', 'orange', fat, 'solid'),
        'Flora_one_class': ('Flora with one class', 'cornflowerblue', thin, 'solid'),
        'Flora': ('Flora', 'blue', fat, 'solid'),
    }

    for approach_name, properties in line_properties.items():
        approach_df = df.query(f'approach == "{approach_name}"')
        label, color, linewidth, linestyle = properties
        plt.plot(
            approach_df['mem_cpu_cost_ratio'],
            approach_df['normalized_dollar_cost'],
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        plt.legend(loc='upper right')
    plt.plot([0.134004, 0.134004], [0, 10], '-', linewidth=.5, color='green')
    plt.text(0.126, 0.985, '^GCP 2024-12-01', verticalalignment='top', horizontalalignment='left', color='green', fontsize=9.5)
    plt.title('Resource Selection Approaches Minimizing Cost', fontsize=14)
    plt.ylim((0.9999, 2.6))
    plt.xlim((1e-2, 1.1e1))
    plt.xscale('log')
    plt.xlabel('hourly_cost(1 GB RAM) ÷ hourly_cost(1 vCPU core)', fontsize=13)
    plt.ylabel('Monetary cost of execution [normalized]', fontsize=13)
    plt.savefig('results/prices_experiment.pdf', bbox_inches='tight')
    plt.savefig('results/prices_experiment.svg', bbox_inches='tight')
    # plt.show()
    plt.clf()


# --- To run / present each evaluation experiment, uncomment the respective line ---

# run_main_experiment(results_file_name='results/main_experiment.csv')
present_main_experiment(results_file_name='results/main_experiment.csv')

# -- Main experiment details --
# cost, runtime = evaluate(Flora, verbose=True)
# cost, runtime = evaluate(Flora_one_class, verbose=True)
# cost, runtime = evaluate(Juggler, verbose=True)
# cost, runtime = evaluate(Crispy, verbose=True)

# run_mislabeling_experiment('results/flora_mislabeling.csv')
# present_mislabeling_experiment('results/flora_mislabeling.csv')

# run_labeling_accuracy_experiment('results/flora_labeling_accuracy.csv')
# present_labeling_accuracy_experiment('results/flora_labeling_accuracy.csv')

# NOTE:
# - This experiment can't be run before the others, since it modifies the cloud_df's pricing data
# - It takes a couple of minutes to complete
# run_mem_cpu_cost_ratio_experiment(results_file_name='results/mem_cpu_cost_ratio.csv')
present_mem_cpu_cost_ratio_experiment(results_file_name='results/mem_cpu_cost_ratio.csv')

