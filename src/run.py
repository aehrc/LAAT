import sys
sys.path.append("c:\\Users\\micha\\Desktop\\LAAT")

from src.training import *
import os
from src.util.util import set_random_seed
import copy

# set the random seed if needed, disable by default
# set_random_seed(random_seed=42)


def generate_checkpoint_dir_path(args):
    model_setting = get_model_setting(args)
    cp_args = copy.deepcopy(args)
    del cp_args.n_epoch, cp_args.patience, cp_args.save_results, cp_args.save_best_model, cp_args.save_results_on_train
    del cp_args.resume_training, cp_args.problem_name, cp_args.metric_level, cp_args.main_metric

    checkpoint_dir_path = "{}/{}/{}".format(args.checkpoint_dir, args.problem_name,
                                                     "{}_{}".format(model_setting, to_md5("{}".format(cp_args))))
    return checkpoint_dir_path


def main():
    training_data, valid_data, test_data, vocab, args, logger, saved_vocab_path = prepare_data()
    logger.info("{}.{}.{}".format(len(training_data), len(valid_data), len(test_data)))
    saved_data_file_path = "{}.data.pkl".format(saved_vocab_path.split(".pkl")[0])

    checkpoint_dir_path = generate_checkpoint_dir_path(args)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    checkpoint_path = "{}/checkpoint.pkl".format(checkpoint_dir_path)
    args.best_model_path = "{}/best_model.pkl".format(checkpoint_dir_path)
    args.result_path = "{}/result.pkl".format(checkpoint_dir_path)
    run_with_validation(training_data, valid_data, test_data, vocab, args,
                        logger=logger, saved_data_file_path=saved_data_file_path,
                        checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()

