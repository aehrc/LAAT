from torch import optim

from src.util.preprocessing import *
from src.util.util import to_md5
from src.util.util import get_n_training_labels

from src.data_helpers.dataloaders import TextDataset, TextDataLoader, HiCuTextDataset
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.util.icd_hierarchy import generate_code_hierarchy
from src.args_parser import *
from src.models.loss import AsymmetricLoss
import pickle
import pprint

from transformers import AdamW

from gensim.models.poincare import PoincareModel


def _load_and_cache_data(train_data, valid_data, test_data, vocab, args, logger, saved_data_file_path):
    """
    Load the data from cached file or cache the data if there is no cached file

    :param train_data:
    :param valid_data:
    :param test_data:
    :param vocab:
    :param args:
    :param logger:
    :param saved_data_file_path:
    :return:
    """
    import logging
    if len(logging.getLogger().handlers) > 0:
        logging.getLogger().handlers.pop()  # This is to remove the transformer logger

    logger.info("Saved dataset path: {}".format(saved_data_file_path))
    if os.path.exists(saved_data_file_path):
        try:
            with open(saved_data_file_path, 'rb') as f:
                data = pickle.load(f)
                vocab.logger = logger

                data["train"].multilabel = bool(args.multilabel)
                data["valid"].multilabel = bool(args.multilabel)
                data["test"].multilabel = bool(args.multilabel)

                return data["train"], data["valid"], data["test"]
        except:
            pass

    # Build train/valid/test data loaders
    if args.joint_mode == "hicu":
        train_dataset = HiCuTextDataset(train_data, vocab,
                                    max_seq_length=args.max_seq_length,
                                    min_seq_length=args.min_seq_length,
                                    sort=True,
                                    multilabel=args.multilabel)

        valid_dataset = HiCuTextDataset(valid_data, vocab,
                                    max_seq_length=args.max_seq_length,
                                    min_seq_length=args.min_seq_length,
                                    sort=True,
                                    multilabel=args.multilabel)

        test_dataset = HiCuTextDataset(test_data, vocab,
                                max_seq_length=args.max_seq_length,
                                min_seq_length=args.min_seq_length,
                                sort=True, multilabel=args.multilabel)
    else:
        train_dataset = TextDataset(train_data, vocab,
                                    max_seq_length=args.max_seq_length,
                                    min_seq_length=args.min_seq_length,
                                    sort=True,
                                    multilabel=args.multilabel)

        valid_dataset = TextDataset(valid_data, vocab,
                                    max_seq_length=args.max_seq_length,
                                    min_seq_length=args.min_seq_length,
                                    sort=True,
                                    multilabel=args.multilabel)

        test_dataset = TextDataset(test_data, vocab,
                                max_seq_length=args.max_seq_length,
                                min_seq_length=args.min_seq_length,
                                sort=True, multilabel=args.multilabel)
    # try:
    with open(saved_data_file_path, 'wb') as f:
        data = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # except:
    #     logger.error("Cannot cache the datasets!")
    vocab.logger = logger
    return train_dataset, valid_dataset, test_dataset


def _train_model(train_data, valid_data, test_data,
                 vocab, args,
                 init_state_dict=None,
                 logger=None,
                 saved_data_file_path=None,
                 checkpoint_path=None
                 ):
    """
    Train the data
    :param train_data:
    :param valid_data:
    :param test_data:
    :param vocab:
    :param args:
    :param init_state_dict:
    :param logger:
    :param saved_data_file_path:
    :param checkpoint_path:
    :return:
    """
    args.multilabel = bool(args.multilabel)
    model = args.model(vocab, args)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    model.to(vocab.device)

    train_dataset, valid_dataset, test_dataset = _load_and_cache_data(train_data, valid_data, test_data,
                                                                      vocab, args, logger, saved_data_file_path)

    logger.info("{} instances with {} tokens, {} in the train dataset"
                .format(train_dataset.size, train_dataset.n_total_tokens,
                        ", ".join(["Level_{} with {} labels".format(level, len(train_dataset.labels[level]))
                                   for level in range(vocab.n_level())])))

    logger.info("{} instances with {} tokens, {} in the valid dataset"
                .format(valid_dataset.size, valid_dataset.n_total_tokens,
                        ", ".join(["Level_{} with {} labels".format(level, len(valid_dataset.labels[level]))
                                   for level in range(vocab.n_level())])))

    logger.info("{} instances with {} tokens, {} in the test dataset"
                .format(test_dataset.size, test_dataset.n_total_tokens,
                        ", ".join(["Level_{} with {} labels".format(level, len(test_dataset.labels[level]))
                                   for level in range(vocab.n_level())])))

    train_dataloader = TextDataLoader(dataset=train_dataset, vocab=vocab, batch_size=args.batch_size)

    valid_dataloader = TextDataLoader(dataset=valid_dataset, vocab=vocab, batch_size=args.batch_size)

    test_dataloader = TextDataLoader(dataset=test_dataset, vocab=vocab, batch_size=args.batch_size)

    # Train the model
    if args.optimiser.lower() == "adagrad":
        optimiser = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimiser.lower() == "adam":
        betas = (0.9, 0.999)
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    elif args.optimiser.lower() == "adamw":
        betas = (0.9, 0.999)
        optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    elif args.optimiser.lower() == "sgd":
        optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimiser.lower() == "adadelta":
        optimiser = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    lr_plateau = None
    if args.use_lr_scheduler:
        lr_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                          mode="max",
                                                          factor=args.lr_scheduler_factor,
                                                          patience=args.lr_scheduler_patience,
                                                          min_lr=0.0001)
    if args.multilabel:
        if args.loss == "BCE":
            criterions = [nn.BCEWithLogitsLoss() for _ in range(vocab.n_level())]
        elif args.loss == "ASL":
            asl_config = [float(c) for c in args.asl_config.split(',')]
            criterions = [AsymmetricLoss(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                         clip=asl_config[2], reduction=args.asl_reduction) for _ in range(vocab.n_level())]
    else:
        criterions = [nn.CrossEntropyLoss() for _ in range(vocab.n_level())]

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      test_dataloader=test_dataloader,
                      criterions=criterions,
                      optimiser=optimiser,
                      lr_scheduler=lr_plateau,
                      vocab=vocab,
                      logger=logger,
                      args=args,
                      checkpoint_path=checkpoint_path)
    best_model, scores = trainer.train(n_epoch=args.n_epoch, patience=args.patience)

    evaluator = Evaluator(model=best_model,
                          vocab=vocab,
                          criterions=criterions,
                          n_training_labels=get_n_training_labels(train_dataloader)[-1])

    del model, lr_plateau, optimiser, evaluator, trainer, criterions
    return best_model, scores  # either on valid or test


def _get_labels(data, args):
    n_label_level = len(data[0][1])
    out_labels = [[] for _ in range(n_label_level)]
    for (feature, labels, _) in data:
        for label_lvl in range(len(labels)):
            out_labels[label_lvl].extend(labels[label_lvl])
    unique_labels = []
    args.n_labels = []
    for lvl in range(len(out_labels)):
        unique_labels.append(list(set(out_labels[lvl])))
        args.n_labels.append(len(unique_labels[lvl]))

    out_labels = unique_labels

    return out_labels


def run_with_validation(training_data, valid_data, test_data,
                        vocab, args, logger=None,
                        saved_data_file_path=None,
                        checkpoint_path=None):
    best_model, scores = _train_model(
        train_data=training_data, valid_data=valid_data, test_data=test_data,
        vocab=vocab, args=args, logger=logger,
        saved_data_file_path=saved_data_file_path, checkpoint_path=checkpoint_path)

    return best_model, scores


def load_cached_data(file_path: str) -> tuple:
    """
    Load cached data from the pickle file
    :param file_path: str
    :return: tuple
        a list of features and a list of corresponding labels
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data["vocab"], data["training_data"], data["valid_data"], data["test_data"]


def save_data(data: dict,
              file_path: str) -> None:
    """
    Save data to a pickle file
    :param data: dict
    :param file_path: str
    :return: None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    return None


def _create_log_file_folder(args):
    strtime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_setting = get_model_setting(args)
    return "{}/{}.{}.{}".format(args.problem_name, strtime, model_setting, to_md5("{}".format(args)))


def get_model_setting(args):
    if "attention_mode" not in args:
        args.attention_mode = None
    additive_information = ""

    if "WordCNN" in args.model.__name__:
        additive_information = "_{}_{}_{}".format(args.cnn_model, args.kernel_size, args.out_channels)
    elif "RNN" == args.model.__name__:
        additive_information = "_{}_{}_{}".format(args.rnn_model, args.n_layers, args.hidden_size)
    model_setting = "{}.{}.{}.{}.{}" \
        .format(args.model.__name__ + additive_information, args.mode, args.attention_mode, args.lr, args.dropout)
    return model_setting


def prepare_data():
    import logging
    if len(logging.getLogger().handlers) > 0:
        logging.getLogger().handlers.pop()  # This is to remove the transformer logger

    parser = create_args_parser()
    args = parser.parse_args()
    logger = create_logger(logger_name=_create_log_file_folder(args))
    logger.info("Training with \n{}\n".format(pprint.pformat(args.__dict__, indent=4)))

    # Read configuration from the problem name specified in the config.json file
    if "problem_name" in args:
        problem_name = args.problem_name
    args.save_best_model = bool(args.save_best_model)

    if "penalisation_coeff" not in args:
        args.penalisation_coeff = 0.0
    configuration = read_config(problem_name)

    embedding_file = None
    if "embedding_file" in args:
        if args.embedding_file:
            embedding_file = args.embedding_file

    embedding_mode = None
    if "embedding_mode" in args:
        if args.embedding_mode:
            embedding_mode = args.embedding_mode

    cache_folder = "{}/{}".format(configuration["cache_dir"], problem_name)
    create_folder_if_not_exist(cache_folder)
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_vocab_file_name = "{}.pkl".format(
        to_md5("{}{}{}{}{}{}{}".format(json.dumps(configuration),
                                       embedding_file,
                                       embedding_mode,
                                       device_name,
                                       args.min_word_frequency,
                                       args.max_seq_length,
                                       args.min_seq_length
                                       )))
    cached_file_name = os.path.join(cache_folder, save_vocab_file_name)

    if "use_regularisation" not in args:
        args.use_regularisation = False

    if os.path.exists(cached_file_name):
        vocab, training_data, valid_data, test_data = load_cached_data(cached_file_name)
        data = training_data + valid_data + test_data
        labels = _get_labels(data, args)

        # generate hierarchy for ICD codes
        if args.joint_mode == "hicu":
            labels, hierarchy = generate_code_hierarchy(labels[0])
            
            poincare_embeddings = None
            if args.decoder.find("Hyperbolic") != -1:
                # train poincare (hyperbolic) embeddings
                relations = set()
                for k, v in hierarchy[4].items():
                    relations.add(('root', v[0]))
                    for i in range(4):
                        relations.add(tuple(v[i:i+2]))
                relations = list(relations)
                poincare = PoincareModel(relations, args.hyperbolic_dim, negative=10)
                poincare.train(epochs=50)
                poincare_embeddings = poincare.kv

            vocab.update_hierarchy(hierarchy, poincare_embeddings)

        vocab.update_labels(labels)
        logger.info("Loaded vocab and data from file")

    else:
        # Prepare the data for training DNN
        # use all the labels
        data_dir = configuration["data_dir"]
        training_data = read_data(configuration, data_dir + "/train.csv")
        valid_data = read_data(configuration, data_dir + "/valid.csv")
        test_data = read_data(configuration, data_dir + "/test.csv")
        data = training_data + valid_data + test_data

        labels = _get_labels(data, args)

        # generate hierarchy for ICD codes
        if args.joint_mode == "hicu":
            labels, hierarchy = generate_code_hierarchy(labels[0])

        # Prepare the vocabulary
        vocab = Vocab(data, labels,
                      min_word_frequency=args.min_word_frequency,
                      word_embedding_mode=embedding_mode,
                      word_embedding_file=embedding_file)

        if args.joint_mode == "hicu":
            poincare_embeddings = None
            if args.decoder.find("Hyperbolic") != -1:
                # train poincare (hyperbolic) embeddings
                relations = set()
                for k, v in hierarchy[4].items():
                    relations.add(('root', v[0]))
                    for i in range(4):
                        relations.add(tuple(v[i:i+2]))
                relations = list(relations)
                poincare = PoincareModel(relations, args.hyperbolic_dim, negative=10)
                poincare.train(epochs=50)
                poincare_embeddings = poincare.kv

            vocab.update_hierarchy(hierarchy, poincare_embeddings)

        logger.info("Preparing the vocab")
        vocab.prepare_vocab()
        saved_objects = {"vocab": vocab,
                         "training_data": training_data,
                         "valid_data": valid_data,
                         "test_data": test_data}

        save_data(saved_objects, cached_file_name)
        logger.info("Saved vocab and data to files")
    logger.info("Using {}".format(vocab.device))

    logger.info("# levels: {}".format(len(labels)))
    for label_lvl in range(len(labels)):
        logger.info("# labels at level {}: {}".format(label_lvl, len(labels[label_lvl])))

    if args.metric_level >= vocab.n_level():
        args.metric_level = vocab.n_level() - 1

    return training_data, valid_data, test_data, vocab, args, logger, cached_file_name


