import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import *
import random
import torch
import hashlib


# This is for shuffling the data and initialise the "UNKNOWN" word embedding
random.seed(0)
np.random.seed(0)


# This is to set the random seed if needed
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ndigits = 5  # rounding


def average_scores(scores):
    avg_scores = dict()
    for key in scores:
        if key.startswith("level"):
            for metric in scores[key]:
                if metric.startswith("macro") or metric.startswith("micro") or metric == "loss":
                    if metric not in avg_scores:
                        avg_scores[metric] = []
                    avg_scores[metric].append(scores[key][metric])

    for metric in avg_scores:
        avg_scores[metric] = sum(avg_scores[metric]) / len(avg_scores[metric])
    return avg_scores


def normalise_labels(labels, n_label):
    norm_labels = []
    for label in labels:
        one_hot_vector_label = [0] * n_label
        one_hot_vector_label[label] = 1
        norm_labels.append(one_hot_vector_label)
    return np.asarray(norm_labels)


def calculate_scores(true_labels, pred_labels, pred_probs, average="macro", is_multilabel=True):
    max_size = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[: max_size]
    pred_labels = pred_labels[: max_size]
    pred_probs = pred_probs[: max_size]
    p_1 = 0
    p_5 = 0
    p_8 = 0
    p_10 = 0
    p_15 = 0
    if pred_probs is not None:
        if not is_multilabel:
            normalised_labels = normalise_labels(true_labels, len(pred_probs[0]))
            auc_score = roc_auc(normalised_labels, pred_probs, average=average)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average=average)
            recall = recall_score(true_labels, pred_labels, average=average)
            f1 = f1_score(true_labels, pred_labels, average=average)
        else:
            if average == "macro":
                accuracy = macro_accuracy(true_labels, pred_labels)  # categorical accuracy
                precision, recall, f1 = macro_f1(true_labels, pred_labels)
                p_ks = precision_at_k(true_labels, pred_probs, [1, 5, 8, 10, 15])
                p_1 = p_ks[0]
                p_5 = p_ks[1]
                p_8 = p_ks[2]
                p_10 = p_ks[3]
                p_15 = p_ks[4]

            else:
                accuracy = micro_accuracy(true_labels, pred_labels)
                precision, recall, f1 = micro_f1(true_labels, pred_labels)
            auc_score = roc_auc(true_labels, pred_probs, average)

            # For significance test
            # accuracy = -1
            # if average == "macro":
            #
            #     precision, recall, f1 = macro_f1(true_labels, pred_labels)
            #     p_ks = precision_at_k(true_labels, pred_probs, [5])
            #     p_5 = p_ks[0]
            #     auc_score = roc_auc(true_labels, pred_probs, average)
            # else:  # micro
            #     precision, recall, f1 = -1, -1, -1
            #     auc_score = -1

    else:
        auc_score = -1

    output = {"{}_precision".format(average): precision, "{}_recall".format(average): recall,
              "{}_f1".format(average): f1, "{}_accuracy".format(average): accuracy,
              "{}_auc".format(average): auc_score, "{}_P@1".format(average): p_1, "{}_P@5".format(average): p_5,
              "{}_P@8".format(average): p_8, "{}_P@10".format(average): p_10, "{}_P@15".format(average): p_15}

    return output


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def macro_accuracy(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (union_size(true_labels, pred_labels, 0) + 1e-10)
    return np.mean(num)


def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_accuracy(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / union_size(flat_true, flat_pred, 0)


def recall_at_k(true_labels, pred_probs, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(pred_probs)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = true_labels[i, tk].sum()
        denom = true_labels[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(true_labels, pred_probs, ks=[1, 5, 8, 10, 15]):
    # num true labels in top k predictions / k
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
    return output


def roc_auc(true_labels, pred_probs, average="macro"):
    if pred_probs.shape[0] <= 1:
        return

    fpr = {}
    tpr = {}
    if average == "macro":
        # get AUC for each label individually
        relevant_labels = []
        auc_labels = {}
        for i in range(true_labels.shape[1]):
            # only if there are true positives for this label
            if true_labels[:, i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
                if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                    auc_score = auc(fpr[i], tpr[i])
                    if not np.isnan(auc_score):
                        auc_labels["auc_%d" % i] = auc_score
                        relevant_labels.append(i)

        # macro-AUC: just average the auc scores
        aucs = []
        for i in relevant_labels:
            aucs.append(auc_labels['auc_%d' % i])
        score = np.mean(aucs)
    else:
        # micro-AUC: just look at each individual prediction
        flat_pred = pred_probs.ravel()
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), flat_pred)
        score = auc(fpr["micro"], tpr["micro"])

    return score


def multiclass_roc_auc(true_labels, pred_labels, average="macro"):
    lb = LabelBinarizer()
    lb.fit(true_labels)
    true_labels = lb.transform(true_labels)
    pred_labels = lb.transform(pred_labels)
    return roc_auc_score(true_labels, pred_labels, average=average)


def union_size(x, y, axis):
    return np.logical_or(x, y).sum(axis=axis).astype(float)


def intersect_size(x, y, axis):
    return np.logical_and(x, y).sum(axis=axis).astype(float)


def log_scores(scores, logger, epoch, prefix_text, digits=ndigits):
    if scores is None:
        return

    logger.info("Results on {} at epoch #{} with Averaged Loss {}".
                format(prefix_text, epoch, abs(round(scores["average"]["loss"], digits))))
    for level in scores:
        if "level" in level:
            logger.info("======== Results at {} ========".format(level))
            log_scores_per_level(scores[level], logger, epoch, prefix_text, digits)


def log_scores_per_level(scores, logger, epoch, prefix_text, digits=ndigits):
    metrics = ["accuracy", "auc", "precision", "recall", "f1", "P@1", "P@5", "P@8", "P@10", "P@15"]
    macros = []
    micros = []
    for metric in metrics:
        macro_name = "{}_{}".format("macro", metric)
        if macro_name in scores:
            macros.append("{}: {}".format(metric, round(scores[macro_name], digits)))

        micro_name = "{}_{}".format("micro", metric)
        if micro_name in scores:
            micros.append("{}: {}".format(metric, round(scores[micro_name], digits)))
    if epoch is not None:
        logger.info("Results on {} at epoch #{} with Loss {}: \n[MICRO]\t{}\n[MACRO]\t{}\n"
                    .format(prefix_text, epoch, abs(round(scores["loss"], digits)), "\t".join(micros), "\t".join(macros)))
    else:
        logger.info("Results on {}: \n[MICRO]\t{}\n[MACRO]\t{}\n"
                    .format(prefix_text, "\t".join(micros), "\t".join(macros)))


def calculate_eval_metrics(ids, true_labels, pred_probs, is_multilabel):
    true_labels = np.asarray(true_labels)
    pred_probs = np.asarray(pred_probs)
    if is_multilabel:
        pred_labels = np.rint(pred_probs)
    else:
        if type(true_labels[0]) == list:
            true_labels = np.argmax(true_labels, 1)
        pred_labels = np.argmax(pred_probs, 1)

    macro_scores = calculate_scores(true_labels, pred_labels, pred_probs, "macro", is_multilabel)
    micro_scores = calculate_scores(true_labels, pred_labels, pred_probs, "micro", is_multilabel)
    scores = macro_scores
    scores.update(micro_scores)
    scores["ids"] = ids
    scores["true_labels"] = true_labels
    scores["pred_probs"] = pred_probs
    return scores


def convert_to_text_for_evaluation(vocab, batch_data):
    size = len(batch_data.size())

    output = []
    if size == 2:
        for each_doc in batch_data:
            a_doc_text = []
            for each_word in each_doc:
                word_id = each_word.item()
                if word_id != vocab.index_of_word(vocab.PAD_TOKEN):
                    a_doc_text.append(vocab.index2word[word_id])
            output.append(a_doc_text)
    elif size == 3:
        for each_doc in batch_data:
            a_doc_text = []
            for each_sent in each_doc:
                a_sent_text = []
                for each_word in each_sent:
                    word_id = each_word.item()
                    if word_id != vocab.index_of_word(vocab.PAD_TOKEN):
                        a_sent_text.append(vocab.index2word[word_id])
                if len(a_sent_text) > 0:
                    a_doc_text.append(a_sent_text)
            output.append(a_doc_text)
    return output


def convert_to_text(vocab, batch_data):
    size = len(batch_data.size())

    output = []
    if size == 2:
        for each_doc in batch_data:
            a_doc_text = []
            for each_word in each_doc:
                word_id = each_word.item()
                if word_id != vocab.index_of_word(vocab.PAD_TOKEN):
                    a_doc_text.append(vocab.index2word[word_id])
            output.append(" ".join(a_doc_text))
    elif size == 3:
        for each_doc in batch_data:
            a_doc_text = []
            for each_sent in each_doc:
                a_sent_text = []
                for each_word in each_sent:
                    word_id = each_word.item()
                    if word_id != vocab.index_of_word(vocab.PAD_TOKEN):
                        a_sent_text.append(vocab.index2word[word_id])
                if len(a_sent_text) > 0:
                    a_doc_text.append(" ".join(a_sent_text))
            output.append(a_doc_text)
    return output


def get_loss(loss_list, n_training_labels):
    loss = None
    n_total_label = 0
    for i in range(len(loss_list)):
        n_label = n_training_labels[i]
        n_total_label += n_label
        if loss is None:
            loss = n_label * loss_list[i]
        else:
            loss += n_label * loss_list[i]
    return loss/n_total_label


def get_n_training_labels(train_dataloader):
    n_training_labels = []
    for i in range(len(train_dataloader.dataset.labels)):
        n_training_labels.append(len(train_dataloader.dataset.labels[i]))
    return n_training_labels


def to_md5(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()