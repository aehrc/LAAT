from src.evaluator import *
import numpy as np
import os
from torch.autograd import Variable
from src.models.attentions.attention_layer import *
np.set_printoptions(precision=5)
from tqdm import tqdm
from collections import OrderedDict
import shutil


class Trainer:
    def __init__(self, model: nn.Module,
                 train_dataloader: TextDataLoader,
                 valid_dataloader: TextDataLoader,
                 test_dataloader: TextDataLoader,
                 criterions,
                 optimiser,
                 lr_scheduler,
                 vocab,
                 logger,
                 args,
                 checkpoint_path,
                 ):
        """
        The initialisation model
        :param model: The machine learning model
        :param train_dataloader: Training dataloader
        :param valid_dataloader: Validation dataloader
        :param test_dataloader: Test dataloader
        :param criterions: Criterion to generate loss
        :param optimiser: Adam/AdamW ...
        :param lr_scheduler: Reduce 10% of learning rate every 5 epochs
        :param vocab: Vocabulary
        :param logger:
        :param args:
        :param checkpoint_path: Path where the model is saved
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.criterions = criterions
        self.optimiser = optimiser
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.vocab = vocab
        self.args = args
        self.save_best_model = args.save_best_model

        # for self attetion
        self.use_regularisation = args.use_regularisation
        self.penalisation_coeff = args.penalisation_coeff

        self.multilabel = args.multilabel
        self.save_results = args.save_results
        self.checkpoint_path = checkpoint_path
        self.n_training_labels = get_n_training_labels(train_dataloader)

        if self.save_results:
            self.saved_result_path = args.result_path
        self.saved_last_model_path = None
        if self.save_best_model:
            self.best_model_path = args.best_model_path

            if self.best_model_path is None:
                self.best_model_path = "{}/best_model.pt".format(os.path.dirname(logger.handlers[0].baseFilename))

            saved_model_dir = os.path.dirname(self.best_model_path)
            if not os.path.exists(saved_model_dir):
                logger.info("Create {} for saving the model".format(saved_model_dir))
                os.makedirs(saved_model_dir)

        self.main_metric = args.main_metric
        self.metric_level = "level_{}".format(args.metric_level)
        if args.metric_level < 0:
            self.metric_level = "average"

        self.start_epoch = 0
        self.best_val = None
        self.saved_test_scores = None
        self.best_epoch_num = 1

    def train_single_epoch(self, index):
        """
        This is to train a single epoch
        :param index: epoch index
        :return: scores
        """
        self.model.train()

        if bool(self.args.shuffle_data):
            self.train_dataloader.dataset.shuffle_data()
        losses = []
        true_labels = [[] for _ in range(self.vocab.n_level())] if self.args.joint_mode != "hicu" else [[]]
        pred_probs = [[] for _ in range(self.vocab.n_level())] if self.args.joint_mode != "hicu" else [[]]
        ids = []
        all_loss_list = []
        progress_bar = tqdm(self.train_dataloader, unit="batches", desc="Training at epoch #{}".format(index))
        progress_bar.clear()
        self.optimiser.zero_grad()
        batch_id = 0

        for text_batch, label_batch, length_batch, id_batch in progress_bar:
            batch_id += 1
            text_batch = text_batch.to(device)
            for idx in range(len(label_batch)):
                label_batch[idx] = label_batch[idx].to(device)

            if type(length_batch) == list:
                for i in range(len(length_batch)):
                    length_batch[i] = length_batch[i].to(device)
            else:
                length_batch = length_batch.to(device)

            output, attn_weights = self.model(text_batch, length_batch)
            loss_list = []

            if self.args.joint_mode != "hicu":
                for level in range(len(output)):
                    level_labels = label_batch[level]
                    true_labels[level].extend(level_labels.cpu().numpy())
                    loss_list.append(self.criterions[level](output[level], level_labels))

                for level in range(len(loss_list)):
                    if len(all_loss_list) < len(loss_list):
                        all_loss_list.append([loss_list[level].item()])
                    else:
                        all_loss_list[level].append(loss_list[level].item())

                ids.extend(id_batch)
                for level in range(len(output)):
                    if self.multilabel:
                        output[level] = torch.sigmoid(output[level])
                        output[level] = output[level].detach().cpu().numpy()
                        pred_probs[level].extend(output[level])
                    else:
                        output[level] = torch.softmax(output[level], 1)
                        output[level] = output[level].detach().cpu().numpy()
                        pred_probs[level].extend(output[level].tolist())
                loss = get_loss(loss_list, self.n_training_labels)
                loss.backward()
                losses.append(loss.item())

                self.optimiser.step()
                self.optimiser.zero_grad()
            else:
                level_labels = label_batch[self.cur_depth]
                true_labels[0].extend(level_labels.cpu().numpy())
                loss_list.append(self.criterions[self.cur_depth](output[0], level_labels))

                for level in range(len(loss_list)):
                    if len(all_loss_list) < len(loss_list):
                        all_loss_list.append([loss_list[level].item()])
                    else:
                        all_loss_list[level].append(loss_list[level].item())

                ids.extend(id_batch)
                for level in range(len(output)):
                    if self.multilabel:
                        output[level] = torch.sigmoid(output[level])
                        output[level] = output[level].detach().cpu().numpy()
                        pred_probs[level].extend(output[level])
                    else:
                        output[level] = torch.softmax(output[level], 1)
                        output[level] = output[level].detach().cpu().numpy()
                        pred_probs[level].extend(output[level].tolist())
                loss = get_loss(loss_list, [self.n_training_labels[self.cur_depth]])
                loss.backward()
                losses.append(loss.item())

                self.optimiser.step()
                self.optimiser.zero_grad()


        scores = OrderedDict()
        if self.args.joint_mode != "hicu":
            for level in range(len(output)):
                if self.args.save_results_on_train:
                    scores["level_{}".format(level)] = calculate_eval_metrics(ids, true_labels[level],
                                                                            pred_probs[level], self.multilabel)
                else:
                    scores["level_{}".format(level)] = {}
                scores["level_{}".format(level)]["loss"] = -np.mean(all_loss_list[level]).item()

            scores["average"] = average_scores(scores)
            scores["average"]["loss"] = np.mean(losses).item()
            progress_bar.refresh(True)
            progress_bar.clear(True)
            progress_bar.close()

        else:
            for level in range(len(output)):
                if self.args.save_results_on_train:
                    scores["level_{}".format(self.cur_depth)] = calculate_eval_metrics(ids, true_labels[level],
                                                                            pred_probs[level], self.multilabel)
                else:
                    scores["level_{}".format(self.cur_depth)] = {}
                scores["level_{}".format(self.cur_depth)]["loss"] = -np.mean(all_loss_list[level]).item()

            scores["average"] = average_scores(scores)
            scores["average"]["loss"] = np.mean(losses).item()
            progress_bar.refresh(True)
            progress_bar.clear(True)
            progress_bar.close()

        return scores

    def calculate_penalisation(self, attn_weights, batch_size):
        """
        This is the penalisation for the self attention
        :param attn_weights:
        :param batch_size:
        :return:
        """
        transposed_attn_weights = attn_weights.transpose(1, 2)
        identity = torch.eye(attn_weights.size(1))
        identity = Variable(identity.unsqueeze(0).expand(batch_size, attn_weights.size(1), attn_weights.size(1))).to(
            device)
        penal = AttentionLayer.l2_matrix_norm(attn_weights @ transposed_attn_weights - identity)
        return penal

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint_path)
        if is_best:
            shutil.copyfile(self.checkpoint_path, self.best_model_path)

    @staticmethod
    def format_number(number):
        return abs(round(number, ndigits=ndigits))

    def train(self,
              n_epoch,
              patience: int = 5):
        

        if self.args.resume_training:
            if os.path.isfile(self.checkpoint_path):
                self.logger.info("=> loading checkpoint '{}'".format(self.checkpoint_path))
                checkpoint = torch.load(self.checkpoint_path)
                self.start_epoch = checkpoint['epoch']
                self.best_val = checkpoint['best_val']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimiser.load_state_dict(checkpoint['optimiser'])
                self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                                 .format(self.checkpoint_path, checkpoint['epoch']))
                self.saved_test_scores = checkpoint['test_scores']
                self.best_epoch_num = checkpoint['best_epoch_num']
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.checkpoint_path))

        self.cur_depth = 5 - self.args.depth

        n_epoch = [int(epoch) for epoch in n_epoch.split(",")]

        while self.cur_depth < 5:

            print("Training model at depth {}:".format(self.cur_depth))
            if self.cur_depth != 0:
                self.model.attention.change_depth(self.cur_depth)

            best_valid_scores = self.best_val
            saved_test_scores = self.saved_test_scores
            saved_train_scores = None
            check_to_stop = 0
            best_epoch_num = self.best_epoch_num
            # best_state_dict = None
            evaluator = Evaluator(self.model, self.vocab, [self.criterions[self.cur_depth]], [self.n_training_labels[self.cur_depth]])
            for e in range(self.start_epoch + 1, n_epoch[self.cur_depth] + 1):
                self.logger.info("Training epoch #{}".format(e))
                train_scores = self.train_single_epoch(e)
                epoch_loss = train_scores["average"]["loss"]

                if self.model.args.joint_mode != "hicu":
                    valid_scores = evaluator.evaluate(self.valid_dataloader)
                    test_scores = evaluator.evaluate(self.test_dataloader)
                else:
                    valid_scores = evaluator.evaluate(self.valid_dataloader, self.cur_depth)
                    test_scores = evaluator.evaluate(self.test_dataloader, self.cur_depth)

                if self.lr_scheduler is not None and self.cur_depth == 4:
                    self.lr_scheduler.step(valid_scores[self.metric_level][self.main_metric])
                    for param_group in self.optimiser.param_groups:
                        self.logger.info("Learning rate at epoch #{}: {}".format(e, param_group["lr"]))

                if self.args.save_results_on_train:
                    self.logger.info("Loss on Train at epoch #{}: {}, {} on Train: {}, {} on Valid: {}".
                                    format(e, self.format_number(epoch_loss),
                                            self.main_metric,
                                            self.format_number(train_scores[self.metric_level][self.main_metric]),
                                            self.main_metric,
                                            self.format_number(valid_scores[self.metric_level][self.main_metric])))
                else:
                    self.logger.info("Loss on Train at epoch #{}: {}, {} on Valid: {}".
                                    format(e, abs(round(epoch_loss, ndigits=ndigits)),
                                            self.main_metric,
                                            abs(round(valid_scores[self.metric_level][self.main_metric], ndigits=ndigits))))

                is_best = False
                if best_valid_scores is None or best_valid_scores[self.metric_level][self.main_metric] < \
                        valid_scores[self.metric_level][self.main_metric]:
                    check_to_stop = 0
                    best_valid_scores = valid_scores
                    saved_test_scores = test_scores
                    saved_train_scores = train_scores
                    best_epoch_num = e
                    if self.save_best_model:
                        is_best = True

                else:
                    check_to_stop += 1
                    self.logger.info("[CURRENT BEST] ({}) {} on Valid set: {}"
                                    .format(self.metric_level,
                                            self.main_metric,
                                            self.format_number(best_valid_scores[self.metric_level][self.main_metric]),
                                            ))
                    self.logger.info("Early stopping: {}/{}".format(check_to_stop, patience + 1))

                if check_to_stop == 0:
                    self.logger.info("[NEW BEST] ({}) {} on Valid set: {}"
                                    .format(self.metric_level,
                                            self.main_metric,
                                            self.format_number(best_valid_scores[self.metric_level][self.main_metric]),
                                            ))
                    log_scores(valid_scores, self.logger, e, "Valid set")

                # log_scores(test_scores, self.logger, e, "Test set")

                # Checkpoint save
                lr_scheduler_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()

                self.save_checkpoint({
                    'epoch': e,
                    'state_dict': self.model.state_dict(),
                    'best_val': best_valid_scores,
                    'test_scores': saved_test_scores,
                    'optimiser': self.optimiser.state_dict(),
                    'best_epoch_num': best_epoch_num,
                    'lr_scheduler': lr_scheduler_state_dict
                }, is_best)

                if check_to_stop > patience > 0:
                    self.logger.warn("Early stopped on Valid set!")
                    break

            self.cur_depth += 1

        self.logger.info("=================== BEST ===================")
        # log_scores(saved_train_scores, self.logger, best_epoch_num, "Training set")
        log_scores(best_valid_scores, self.logger, best_epoch_num, "Valid set")
        log_scores(saved_test_scores, self.logger, best_epoch_num, "Test set")

        if self.save_results:
            import pickle
            results = {"train": saved_train_scores, "valid": best_valid_scores, "test": saved_test_scores,
                       "params": self.args, "index2label": self.vocab.index2label}

            with open(self.saved_result_path, 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        if self.save_best_model and os.path.isfile(self.best_model_path):
            self.logger.info("=> loading best model '{}'".format(self.best_model_path))
            best_model = torch.load(self.best_model_path)
            self.model.load_state_dict(best_model['state_dict'])
        return self.model, best_valid_scores

