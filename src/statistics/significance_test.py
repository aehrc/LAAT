from src.util.util import *
import pickle as pkl
from tqdm import tqdm
import random


def calculate_rand_test(y_true, y_1, y_2, n_shuffle=9999, rng_seed=1234, is_multilabel=True, one_side=False):
    random.seed(rng_seed)
    diffs = {}
    pos_count = {}
    scores_1 = calculate_eval_metrics(ids=None, true_labels=y_true, pred_probs=y_1, is_multilabel=is_multilabel)
    scores_2 = calculate_eval_metrics(ids=None, true_labels=y_true, pred_probs=y_2, is_multilabel=is_multilabel)

    for key in scores_1:
        if "micro" in key or "macro" in key:
            # print(key, scores_1[key], scores_2[key])
            diffs[key] = abs(scores_1[key] - scores_2[key])
            # print(key, scores_1[key], scores_2[key], diffs[key])
            if one_side:
                diffs[key] = (scores_1[key] - scores_2[key])
            pos_count[key] = 0
            # print(diffs[key])

    for i in tqdm(range(n_shuffle)):
        shuffled_y_1, shuffled_y_2 = shuffle_wrap(y_1, y_2)
        shuffled_scores_1 = calculate_eval_metrics(ids=None, true_labels=y_true, pred_probs=shuffled_y_1, is_multilabel=is_multilabel)
        shuffled_scores_2 = calculate_eval_metrics(ids=None, true_labels=y_true, pred_probs=shuffled_y_2, is_multilabel=is_multilabel)
        for key in scores_1:
            if "micro" in key or "macro" in key:

                diff = abs(shuffled_scores_1[key] - shuffled_scores_2[key])
                if one_side:
                    diff = (shuffled_scores_1[key] - shuffled_scores_2[key])
                # print(key, shuffled_scores_1[key], shuffled_scores_2[key], diff)

                if diff >= diffs[key]:
                    pos_count[key] += 1

    for key in pos_count:
        print("{}: {} - {}".format(key, pos_count[key], (pos_count[key] + 1) * 1.0 / (n_shuffle + 1)))


def shuffle_wrap(y1, y2):
    assert len(y1) == len(y2)
    shuffled_y1 = []
    shuffled_y2 = []
    for i in range(len(y1)):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            shuffled_y1.append(y2[i])
            shuffled_y2.append(y1[i])
        else:
            shuffled_y1.append(y1[i])
            shuffled_y2.append(y2[i])
    return shuffled_y1, shuffled_y2


def read_results_from_file(pkl_file_path, level=-1, dataset="test"):
    with open(pkl_file_path, "rb") as f:
        data = pkl.load(f)
        results = data[dataset]
        max_level = -1
        for key in results:
            if "level" in key:
                if int(key.split("_")[1]) > max_level:
                    max_level = int(key.split("_")[1])
        if level < 0:
            level = max_level

        results = results["level_{}".format(level)]

        for key in results:
            if "micro_f1" in key or "macro_f1" in key:
                print("{}: {}".format(key, results[key]))
        return results["true_labels"], results["pred_probs"]


if __name__ == "__main__":
    true_labels, y_1 = read_results_from_file("/Users/vu028/Desktop/ICDcoding/MIMIC-II-Full/LAAT/result.pkl", dataset="test")
    true_labels, y_2 = read_results_from_file("/Users/vu028/Desktop/ICDcoding/MIMIC-II-Full/JointLAAT/result.pkl", dataset="test")

    calculate_rand_test(true_labels, y_2, y_1, n_shuffle=30, one_side=False)