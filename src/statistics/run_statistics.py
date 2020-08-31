import json
import re
from os import listdir
from os.path import isfile, join
import pprint
import csv
from copy import deepcopy


def process_all_files(folder_path, problem_names=None, metric="micro_f1", level=2, max_epoch=-1, attention_mode="label"):
    slurm_files = [join(folder_path, f) if "slurm" in f else "" for f in listdir(folder_path)
                   if isfile(join(folder_path, f))]
    best_result_dict = {}
    for slurm_file in slurm_files:
        if slurm_file == "":
            continue
        # print(slurm_file)
        settings, best_epoch, max_val, best_results = process_a_file(slurm_file, metric, level, max_epoch=max_epoch)
        if settings is None:
            continue

        if problem_names is not None and settings["problem_name"] not in problem_names:
            continue
        if "cnn_model" in settings:
            print("CNN: {}".format(slurm_file))
        if "rnn_model" not in settings or settings["attention_mode"] != attention_mode:
            continue
        if settings["hidden_size"] != 512:
            continue
        #
        if settings["level_projection_size"] != 128:
            continue
        key = "{}_{}_{}_{}_{}".format(attention_mode, settings["problem_name"],
                                settings["joint_mode"],
                                settings["rnn_model"],
                                settings["embedding_file"].split("/")[-1])
        if key not in best_result_dict:
            best_result_dict[key] = {"best_val": max_val, "settings": settings, "epoch": best_epoch,
                                     "best_results": best_results, "file": slurm_file}
        else:
            if best_result_dict[key]["best_val"] < max_val:
                best_result_dict[key] = {"best_val": max_val, "settings": settings, "epoch": best_epoch,
                                         "best_results": best_results, "file": slurm_file}
    with open("results.csv", "w") as csvfile:
        fieldnames = ["Setting", "MACRO_auc", "MICRO_auc", "MACRO_f1", "MICRO_f1", "MACRO_P@5", "MACRO_P@8", "MACRO_P@15",
                      "filename", "hidden_size", "level_projection_size"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key in best_result_dict:
            print("==================={}===================".format(key))
            print(key)
            print("file: {}".format(best_result_dict[key]["file"]))
            print("hidden_size: {}".format(best_result_dict[key]["settings"]["hidden_size"]))
            print("level_projection_size: {}".format(best_result_dict[key]["settings"]["level_projection_size"]))
            print("epoch: {}".format(best_result_dict[key]["epoch"]))
            print("best_val: {}".format(best_result_dict[key]["best_val"]))
            results = best_result_dict[key]["best_results"]
            if results is None:
                continue
            n_level = len(results["valid"])

            print("RESULT ON VALID")
            i = level
            if i < 0:
                i = n_level - 1
            print("===LEVEL {}===".format(i))
            valid_results = print_scores_per_level(results["valid"][i]["micro"],
                                   results["valid"][i]["macro"],
                                   best_result_dict[key]["epoch"], "valid")
            valid_results["Setting"] = "VALID_" + key
            valid_results["filename"] = best_result_dict[key]["file"]
            valid_results["hidden_size"] = best_result_dict[key]["settings"]["hidden_size"]
            valid_results["level_projection_size"] = best_result_dict[key]["settings"]["level_projection_size"]

            keys = list(valid_results.keys())
            for fn in keys:
                if fn not in fieldnames:
                    del valid_results[fn]

            writer.writerow(valid_results)
            print("RESULT ON TEST")
            print("===LEVEL {}===".format(i))
            test_results = print_scores_per_level(results["test"][i]["micro"],
                                   results["test"][i]["macro"],
                                   best_result_dict[key]["epoch"], "test")

            test_results["Setting"] = "TEST_" + key
            test_results["filename"] = best_result_dict[key]["file"]
            test_results["hidden_size"] = best_result_dict[key]["settings"]["hidden_size"]
            test_results["level_projection_size"] = best_result_dict[key]["settings"]["level_projection_size"]
            keys = list(test_results.keys())
            for fn in keys:
                if fn not in fieldnames:
                    del test_results[fn]

            writer.writerow(test_results)

            # print(pprint.pformat(best_epoch[key]["settings"], indent=4))


def process_a_file(file_path, metric="micro_f1", level=2, max_epoch=-1):

    with open(file_path) as f:
        content = f.read()
        settings = get_settings(content)
        results = get_results(content)
        best_epoch, max_val = get_best_by_metric(results, metric, level, max_epoch=max_epoch)
        if settings is None or results is None or best_epoch is None:
            print("FAIL: {}".format(file_path))
            return settings, best_epoch, max_val, None
        return settings, best_epoch, max_val, results[best_epoch]


from collections import OrderedDict


def print_scores_per_level(micro_scores, macro_scores, epoch, prefix_text, digits=5):
    metrics = ["accuracy", "auc", "precision", "recall", "f1", "P@1", "P@5", "P@8", "P@10", "P@15"]
    macros = []
    micros = []
    results = dict()
    for metric in metrics:
        macro_name = metric
        if macro_name in macro_scores:
            score = round(macro_scores[macro_name], digits)
            if score > 0:
                macros.append("{}: {}".format(metric, score))
                results["MACRO_{}".format(metric)] = score
        micro_name = metric
        if micro_name in micro_scores:
            score = round(micro_scores[micro_name], digits)
            if score > 0:
                micros.append("{}: {}".format(metric, score))
                results["MICRO_{}".format(metric)] = score
    if epoch is not None:
        print("Results on {} at epoch #{}: \n[MICRO]\t{}\n[MACRO]\t{}\n"
                    .format(prefix_text, epoch, "\t".join(micros), "\t".join(macros)))

    else:
        print("Results on {}: \n[MICRO]\t{}\n[MACRO]\t{}\n"
                    .format(prefix_text, "\t".join(micros), "\t".join(macros)))

    return results


def get_best_by_metric(results, metric="micro_f1", level=-1, max_epoch=-1):
    max_val = -1
    best_epoch = None
    if results is None:
        return best_epoch, max_val

    for epoch in results:
        try:
            if int(epoch.replace("#", "")) > max_epoch > 0:
                continue
            if level == -1:
               level = len(results[epoch]["valid"]) - 1
            valid_score = results[epoch]["valid"][level][metric.split("_")[0]][metric.split("_")[1]]
            if valid_score > max_val:
                max_val = valid_score
                best_epoch = epoch
        except:
            pass

    return best_epoch, max_val


def get_results(content):
    result_dict = dict()
    results, idx = get_content_by_flags(content, "Learning rate at epoch", "\n\n\n")
    if results is None:
        return None
    epoch, valid_results = extract_results(results.split("with Averaged Loss")[1])
    epoch, test_results = extract_results(results.split("with Averaged Loss")[2])
    result_dict[epoch] = {"valid": valid_results, "test": test_results}

    while idx > 0:
        results, idx = get_content_by_flags(content, "Learning rate at epoch", "\n\n\n", idx)
        if results is not None:
            epoch, valid_results = extract_results(results.split("with Averaged Loss")[1])
            epoch, test_results = extract_results(results.split("with Averaged Loss")[2])
            result_dict[epoch] = {"valid": valid_results, "test": test_results}
    return result_dict


def extract_results(results):

    result_levels = results.split("\n\n")
    scores = {}
    epoch = get_content_by_flags(results, "set at epoch ", " ")[0]
    level = 0
    loss = float(results.strip().split("\n")[0])
    # print(loss)
    for result_level in result_levels:
        if "[MICRO]" in result_level and "[MACRO]"in result_level:
            result_level += "\n"
            # print(result_level)
            micro_scores = convert_str_2_dict(get_content_by_flags(result_level, "[MICRO]", "\n")[0])
            macro_scores = convert_str_2_dict(get_content_by_flags(result_level, "[MACRO]", "\n")[0])
            micro_scores["loss"] = -loss
            macro_scores["loss"] = -loss
            scores[level] = {"micro": micro_scores, "macro": macro_scores}
            level += 1
    return epoch, scores


def convert_str_2_dict(scores):
    scores = scores.strip().split("\t")
    score_dict = {}
    for score in scores:
        score_dict[score.split(": ")[0]] = float(score.split(": ")[1])

    return score_dict


def get_settings(content):
    settings = get_content_by_flags(content, "INFO Training with", "}")[0] 
    if settings is None:
        return None
    settings += "}"
    settings = re.sub("\"model.*?,\n", "", settings.strip().replace("'", "\"")
                      .replace("False", "false")
                      .replace("True", "true")
                      .replace("None", "null")
                      )
    settings = json.loads(settings)
    return settings


def get_content_by_flags(content: str, start_flag, end_flag, start_idx=0):
    if content.find(start_flag, start_idx) < 0:
        return None, -1

    start_idx = content.index(start_flag, start_idx)
    if start_idx > 0:
        if content.find(end_flag, start_idx + len(start_flag)) < 0:
            return None, -1
        end_idx = content.index(end_flag, start_idx + len(start_flag))
        if end_idx > start_idx:
            return content[start_idx + len(start_flag):end_idx], end_idx
    return None, -1


import argparse
def create_args_parser():
    args = argparse.ArgumentParser(description="Processing results")
    args.add_argument("--problem_names", type=str, default=["mimiciii_2_full", "mimiciii_single_full"],
                      nargs='+', help="The list of problems")
    args.add_argument("--attention_mode", type=str, default="label", choices=["label", "caml"])
    args.add_argument("--metric", type=str, default="micro_f1")
    args.add_argument("--level", type=int, default=-1)
    args.add_argument("--max_epoch", type=int, default=-1)
    args.add_argument("--folder_path", type=str, required=True)
    return args


if __name__ == "__main__":
    args = create_args_parser().parse_args()
    process_all_files(args.folder_path,
                      problem_names=args.problem_names,
                      metric=args.metric, level=args.level, max_epoch=args.max_epoch, attention_mode=args.attention_mode)
