from src.data_helpers.dataloaders import *
import warnings
from src.data_helpers.vocab import device

from src.util.util import *
from tqdm import tqdm
from collections import OrderedDict

warnings.filterwarnings('ignore')  # want to remove some warning from sklearn


class Evaluator:
    def __init__(self,
                 model: torch.nn.Module,
                 vocab,
                 criterions,
                 n_training_labels,
                 ):
        self.model = model

        self.vocab = vocab
        self.index_to_label = vocab.index2label
        self.multilabel = model.args.multilabel
        self.criterions = criterions
        self.n_training_labels = n_training_labels

    def evaluate(self,
                 dataloader: TextDataLoader,
                 cur_depth=0) -> dict:
        """
        Evaluate the model on the dataset
        :param dataloader: TextDataLoader
            The input data set
        :return: dict
            The evaluation scores
        """

        self.model.eval()
        true_labels = [[] for _ in range(self.vocab.n_level())] if self.model.args.joint_mode != "hicu" else [[]]
        pred_probs = [[] for _ in range(self.vocab.n_level())] if self.model.args.joint_mode != "hicu" else [[]]
        ids = []

        losses = []
        all_loss_list = []

        for text_batch, label_batch, length_batch, id_batch in \
                tqdm(dataloader, unit="batches", desc="Evaluating"):

            text_batch = text_batch.to(device)
            for idx in range(len(label_batch)):
                label_batch[idx] = label_batch[idx].to(device)

            if type(length_batch) == list:
                for i in range(len(length_batch)):
                    length_batch[i] = length_batch[i].to(device)
            else:
                length_batch = length_batch.to(device)

            if self.model.args.joint_mode != "hicu":
                true_label_batch = []
                for idx in range(len(label_batch)):
                    true_label_batch.append(label_batch[idx].cpu().numpy())
                true_labels.extend(true_label_batch)
            else:
                true_label_batch = []
                true_label_batch.append(label_batch[cur_depth].cpu().numpy())
                true_labels.extend(true_label_batch)
                
            ids.extend(id_batch)
            with torch.no_grad():
                output, attn_weights = self.model(text_batch, length_batch)

            loss_list = []

            if self.model.args.joint_mode != "hicu":
                for label_lvl in range(len(output)):
                    level_labels = label_batch[label_lvl]
                    true_labels[label_lvl].extend(level_labels.cpu().numpy())
                    loss_list.append(self.criterions[label_lvl](output[label_lvl], level_labels))

                for level in range(len(loss_list)):
                    if len(all_loss_list) < len(loss_list):
                        all_loss_list.append([loss_list[level].item()])
                    else:
                        all_loss_list[level].append(loss_list[level].item())

                ids.extend(id_batch)
                probs = [None] * len(output)
                for label_lvl in range(len(output)):
                    if self.multilabel:

                        output[label_lvl] = torch.sigmoid(output[label_lvl])
                        output[label_lvl] = output[label_lvl].detach().cpu().numpy()
                        probs[label_lvl] = output[label_lvl].tolist()
                        pred_probs[label_lvl].extend(output[label_lvl])
                    else:
                        output[label_lvl] = torch.softmax(output[label_lvl], 1)
                        output[label_lvl] = output[label_lvl].detach().cpu().numpy()
                        probs[label_lvl] = output[label_lvl].tolist()
                        pred_probs[label_lvl].extend(output[label_lvl].tolist())
                loss = get_loss(loss_list, self.n_training_labels)
                losses.append(loss.item())
            else:
                level_labels = label_batch[cur_depth]
                true_labels[0].extend(level_labels.cpu().numpy())
                loss_list.append(self.criterions[0](output[0], level_labels))

                for level in range(len(loss_list)):
                    if len(all_loss_list) < len(loss_list):
                        all_loss_list.append([loss_list[level].item()])
                    else:
                        all_loss_list[level].append(loss_list[level].item())

                ids.extend(id_batch)
                probs = [None] * len(output)
                for label_lvl in range(len(output)):
                    if self.multilabel:

                        output[label_lvl] = torch.sigmoid(output[label_lvl])
                        output[label_lvl] = output[label_lvl].detach().cpu().numpy()
                        probs[label_lvl] = output[label_lvl].tolist()
                        pred_probs[label_lvl].extend(output[label_lvl])
                    else:
                        output[label_lvl] = torch.softmax(output[label_lvl], 1)
                        output[label_lvl] = output[label_lvl].detach().cpu().numpy()
                        probs[label_lvl] = output[label_lvl].tolist()
                        pred_probs[label_lvl].extend(output[label_lvl].tolist())

                loss = get_loss(loss_list, self.n_training_labels)
                losses.append(loss.item())

        scores = OrderedDict()
        if self.model.args.joint_mode != "hicu":
            for label_lvl in range(len(output)):
                scores["level_{}".format(label_lvl)] = calculate_eval_metrics(ids, true_labels[label_lvl],
                                                                            pred_probs[label_lvl], self.multilabel)
                scores["level_{}".format(label_lvl)]["loss"] = -np.mean(all_loss_list[label_lvl]).item()
            scores["average"] = average_scores(scores)
            scores["average"]["loss"] = np.mean(losses).item()
        else:
            for label_lvl in range(len(output)):
                scores["level_{}".format(cur_depth)] = calculate_eval_metrics(ids, true_labels[label_lvl],
                                                                            pred_probs[label_lvl], self.multilabel)
                scores["level_{}".format(cur_depth)]["loss"] = -np.mean(all_loss_list[label_lvl]).item()
            scores["average"] = average_scores(scores)
            scores["average"]["loss"] = np.mean(losses).item()

        return scores
