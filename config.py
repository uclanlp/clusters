import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="pang04", help='dataset?')
parser.add_argument('--outdir', type=str, default=None, help='output dir?')
parser.add_argument(
    '--utility', type=str, default="submod", help='utility model?')
parser.add_argument('--k', type=int, default=None, help='num clusters?')
parser.add_argument('--seed', type=int, default=1, help='seed?')
parser.add_argument('--budget', type=int, default=500, help='budget?')
parser.add_argument(
    '--cric_cls', type=int, default=1, help='Index of critical class')
parser.add_argument('--num_runs', type=int, default=10, help='num runs?')
parser.add_argument('--start_seed', type=int, default=0, help='starting seed?')
parser.add_argument(
    '--regret',
    action="store_true",
    default=False,
    help='Plot regret instead?')
parser.add_argument(
    '--tao', type=float, default=0.65, help='threshold for cric class?')
parser.add_argument(
    '--var', type=float, default=None, help='gaussian variance?')
parser.add_argument(
    '--num_processes', type=int, default=10, help='num processes?')
parser.add_argument(
    '--clip', type=int, default=50, help='Clip plot at budget?')
parser.add_argument(
    '--cluster_algo', type=str, default="kmeans_both", help="clustering algo?")
parser.add_argument('--clf', type=str, default="logistic", help="classifier?")
parser.add_argument('--rep', type=str, default="svd", help="representation?")
parser.add_argument(
    "--elbow",
    action="store_true",
    default=False,
    help="run elbow method and exit")

# args = parser.parse_args()
# if args.outdir is None:
#     args.outdir = "out_" + args.dataset
# if args.clip is None:
#     args.clip = args.budget

# if args.dataset == "pang04":
#     if args.k is None:
#         args.k = 6
#     if args.var is None:
#         args.var = 0.001
# elif args.dataset == "pang05":
#     if args.k is None:
#         args.k = 6
#     if args.var is None:
#         args.var = 0.001
# elif args.dataset == "mcauley15":
#     if args.k is None:
#         args.k = 6
#     if args.var is None:
#         args.var = 1e-4
# elif args.dataset == "kaggle13":
#     if args.k is None:
#         args.k = 6
#     if args.var is None:
#         args.var = 1e-3
#     args.clf = "kagglecnn"
#     args.rep = "pixelsvd"

# print ("clusters", args.k)

# clf_config_str = "%s_%s" % (args.dataset, args.clf)
# clf_pred_file = os.path.join(args.outdir, clf_config_str + ".npz")
# clf_ckpt = os.path.join(args.outdir, "checkpoint_" + clf_config_str + ".ckpt")

# config_str = "%s_%s_%s_%s_cls_%d_%s_k_%d_b_%d_%.3f" % (args.dataset, args.clf,
#                                                        args.rep, args.utility,
#                                                        args.cric_cls,
#                                                        args.cluster_algo,
#                                                        args.k, args.budget,
#                                                        args.var)
# util_config_str = "%s_%s_%s_%s_cls_%d_%.3f" % (args.dataset, args.clf,
#                                                args.rep, args.utility,
#                                                args.cric_cls, args.var)
# algo_config_str = util_config_str + "_b_%d" % args.budget

# prior_types = []
# if args.utility == "submod":
#     prior_types = [
#         "conf",
#         # "uniform",
#     ]

# name_map = {
#     "greedy_optimal_noupdate": "Upper bound (tractable)",
#     "greedy_conf_noupdate": "Greedy-conf-fix",
#     "greedy_uniform_noupdate": "Greedy-uniform",
#     "uub_bandit": "UUB (Lakkaraju et al. 2017)",
#     "greedy_Most uncertain": "Most-uncertain",
#     "greedy_conf_cluster": "Greedy-conf",
#     "greedy_uniform_cluster": "Greedy-uniform"
# }
# linestyle_map = {
#     "greedy_optimal_noupdate": "--",
#     "greedy_conf_noupdate": "-.",
#     "greedy_uniform_noupdate": "--",
#     "uub_bandit": "-.",
#     "greedy_Most uncertain": ":",
#     "greedy_conf_cluster": "-",
#     "greedy_uniform_cluster": ":"
# }
# color_map = {
#     "greedy_optimal_noupdate": "orange",
#     "greedy_conf_noupdate": "black",
#     "greedy_uniform_noupdate": "blue",
#     "uub_bandit": "green",
#     "greedy_Most uncertain": "red",
#     "greedy_conf_cluster": "black",
#     "greedy_uniform_cluster": "black"
# }


# cluster_algos = [
#     # "kmeans"
# ]

# bandit_algos = [
#     # "optimal",
#     # "submodgreedy",
#     # "ucb",
#     # "random",
#     # "epsilongreedy",
#     "uub",
# ]

# utilityfile = os.path.join(args.outdir, "utility_%s.npz" % util_config_str)
# unkunk_histfile = os.path.join(args.outdir,
#                                "unkunkhist_%s.pdf" % util_config_str)
# unkunk_2dfile = os.path.join(args.outdir, "unkunk2d_%s.pdf" % util_config_str)
# coverage_scatterfile = os.path.join(args.outdir,
#                                     "coverage_%s.pdf" % util_config_str)
# coverage_scatterfile2 = os.path.join(
#     args.outdir, "coveragetimesconf_%s.pdf" % util_config_str)
# distance_histfile = os.path.join(
#     args.outdir, "utility_%s_%s_%s_%s_cls_%d_%.3f_hist.pdf" %
#     (args.dataset, args.clf, args.rep, args.utility, args.cric_cls, args.var))
# most_uncertainfile = os.path.join(args.outdir,
#                                   algo_config_str + "_most_uncertain.npy")
# clusterfile = os.path.join(args.outdir, "cluster_%s_%s_%s_%d.npy" %
#                            (args.dataset, args.rep, args.cluster_algo, args.k))
# repfile = os.path.join(args.outdir, "rep_%s_%s.npy" % (args.dataset, args.rep))


# def get_algodir(algo):
#     algodir = os.path.join(args.outdir, config_str + "_" + algo)
#     return algodir


# def get_adap_greedy_file(prior_type, update_type, cluster_algo=""):
#     algofile = os.path.join(args.outdir,
#                             algo_config_str + "greedy_%s_%s%s.npy" %
#                             (prior_type, update_type, cluster_algo))
#     return algofile
