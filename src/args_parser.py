import argparse
from email.policy import default
from src.models.rnn import *
from src.models.cnn import *


def create_args_parser():
    parser = argparse.ArgumentParser(description="DNN for Text Classifications")
    parser.add_argument("--problem_name", type=str, default="mimic-iii_single_full", required=False,
                        help="The problem name is used to load the configuration from config.json")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--n_epoch", type=str, default="2,3,5,10,50")
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping")

    parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1,
                        help="Use lr scheduler to reduce the learning rate during training")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.9,
                        help="Reduce the learning rate by the scheduler factor")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5,
                        help="The lr scheduler patience")

    parser.add_argument("--joint_mode", type=str, choices=["flat", "hierarchical", "hicu"], default="hierarchical")
    parser.add_argument("--level_projection_size", type=int, default=128)

    parser.add_argument("--main_metric", default="micro_f1",
                        help="the metric to be used for validation",
                        choices=["macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
                                 "micro_accuracy", "micro_precision", "micro_recall", "micro_f1", "micro_auc", "loss",
                                 "macro_P@1", "macro_P@5", "macro_P@8", "macro_P@10", "macro_P@15"])
    parser.add_argument("--metric_level", type=int, default=1,
                        help="The label level to be used for validation:"
                             "\n\tn: The n-th level if n >= 0 (started with 0)"
                             "\n\tif n > max_level, n is set to max_level"
                             "\n\tif n < 0, use the average of all label levels"
                             )

    parser.add_argument("--multilabel", default=1, type=int, choices=[0, 1])

    parser.add_argument("--shuffle_data", type=int, choices=[0, 1], default=1)

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")

    parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_results", type=int, choices=[0, 1], default=1)
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--save_results_on_train", action='store_true', default=False)
    parser.add_argument("--resume_training", action='store_true', default=False)

    parser.add_argument("--max_seq_length", type=int, default=4000)
    parser.add_argument("--min_seq_length", type=int, default=-1)
    parser.add_argument("--min_word_frequency", type=int, default=-1)

    # Embedding
    parser.add_argument("--mode", type=str, default="static",
                        choices=["rand", "static", "non_static", "multichannel"],
                        help="The mode to init embeddings:"
                             "\n\t1. rand: initialise the embedding randomly"
                             "\n\t2. static: using pretrained embeddings"
                             "\n\t2. non_static: using pretrained embeddings with fine tuning"
                             "\n\t2. multichannel: using both static and non-static modes")

    parser.add_argument("--embedding_mode", type=str, default="fasttext",
                        help="Choose the embedding mode which can be fasttext, word2vec")
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument("--embedding_file", type=str, default=None)

    # Attention
    parser.add_argument("--attention_mode", type=str, choices=["hard", "self", "label", "caml"], default=None)
    parser.add_argument("--d_a", type=int, help="The dimension of the first dense layer for self attention", default=-1)
    parser.add_argument("--r", type=int, help="The number of hops for self attention", default=-1)
    parser.add_argument("--use_regularisation", action='store_true', default=False)
    parser.add_argument("--penalisation_coeff", type=float, default=0.01)

    # HiCu
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--decoder", type=str, choices=['HierarchicalHyperbolic', 'CodeTitleHierarchicalHyperbolic', 'Hierarchical',
                                                   'CodeTitle', 'RandomlyInitialized'], default='HierarchicalHyperbolic')
    parser.add_argument("--hyperbolic_dim", type=int, default=50)
    parser.add_argument('--cat_hyperbolic', action="store_const", const=True, default=False)
    parser.add_argument("--loss", type=str, choices=['BCE', 'ASL', 'ASLO'], default='BCE')
    parser.add_argument("--asl_config", type=str, default='1,0,0.05')
    parser.add_argument("--asl_reduction", type=str, choices=['mean', 'sum'], default='sum')
    parser.add_argument('--disable_attention_linear', action="store_const", const=True, default=False)

    sub_parsers = parser.add_subparsers()

    _add_sub_parser_for_cnn(sub_parsers)
    _add_sub_parser_for_rnn(sub_parsers)

    return parser


def _add_sub_parser_for_rnn(subparsers):
    args = subparsers.add_parser("RNN")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")


def _add_sub_parser_for_cnn(subparsers):
    args = subparsers.add_parser("CNN")
    args.add_argument("--cnn_model", type=str, choices=["CONV1D", "TCN"], default="CONV1D")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers for TCN")
    args.add_argument("--out_channels", type=int, default=100, help="The number of out channels")
    args.add_argument("--kernel_size", type=int, default=5, help="The kernel sizes.")
    args.set_defaults(model=WordCNN)

