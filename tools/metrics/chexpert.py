from tools.chexbert import CheXbert
from tools.metrics.natural_language import NaturalLanguage
from tools.utils import enumerated_save_path
import os
import pandas as pd
import time
import torch
import evaluate


def calculate_nlg_metrics(preds, gts):
    # BLEU, METEOR, ROUGE-L, Perplexity
    print("Calculating NLG metrics")
    results = {}
    bleu = evaluate.load("bleu")
    gts_bleu = [[gt] for gt in gts]
    for n in range(1, 5):
        result = bleu.compute(predictions=preds, references=gts_bleu, max_order=n)
        results[f"BLEU@{n}"] = result["bleu"]
    meteor = evaluate.load("meteor")
    result = meteor.compute(predictions=preds, references=gts)
    results["METEOR"] = result["meteor"]
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=preds, references=gts)
    results["ROUGE_L"] = result["rougeL"]
    perplexity = evaluate.load("perplexity", module_type="metric")
    # result = perplexity.compute(predictions=preds, model_id="gpt2")
    # results["Perplexity"] = result["mean_perplexity"]
    return results

def calculate_ce_metrics(preds, gts):
    # P,R, F1
    print("Calculating CE metrics")
    labeler = ChexpertLabeler()
    preds_labels, gts_labels = [], []
    for pred, gt in tqdm(zip(preds, gts), total=len(preds)):
        pred_label = labeler.get_label(pred)
        gt_label = labeler.get_label(gt)
        preds_labels.append([1 if pred_label[k] == 1 else 0 for k in CATEGORIES])
        gts_labels.append([1 if gt_label[k] == 1 else 0 for k in CATEGORIES])
    precision, recall, f1_score, _ = precision_recall_fscore_support(gts_labels, preds_labels, average="micro")
    result = {"PRECISION": precision, "RECALL": recall, "F1_SCORE": f1_score}
    return result


def get_label(report_list):
    labeler = ChexpertLabeler()
    label_list = []
    for report in tqdm(report_list):
        label = labeler.get_label(report)
        label = [1 if label[k] == 1 else 0 for k in CATEGORIES]
        label_list.append(label)
    del labeler
    return np.array(label_list)


def calculate_ce_metrics_mp(preds, gts, num_processes=os.cpu_count()-1):
    # P,R, F1
    print("Calculating CE metrics")
    # preds = [clean_report(pred) for pred in preds]
    # gts = [clean_report(gt) for gt in gts]
    preds_gts = preds + gts
    num_processes = min(num_processes, len(preds_gts))
    preds_gts_chunks = list(batched(preds_gts, math.ceil(len(preds_gts)/num_processes)))
    if num_processes != len(preds_gts_chunks): num_processes = len(preds_gts_chunks)
    pool = Pool(num_processes)
    preds_gts_labels = pool.imap(get_label, preds_gts_chunks)
    pool.close()
    pool.join()
    preds_gts_labels = np.concatenate(list(preds_gts_labels))
    preds_labels, gts_labels = preds_gts_labels[:len(preds)], preds_gts_labels[len(preds):]
    precision, recall, f1_score, _ = precision_recall_fscore_support(gts_labels, preds_labels, average="micro")
    result = {"PRECISION": precision, "RECALL": recall, "F1_SCORE": f1_score}
    return result


class CheXpertMetrics(NaturalLanguage):

    is_differentiable = False
    full_state_update = False

    def __init__(
        self,
        ckpt_dir,
        bert_path,
        checkpoint_path,
        mbatch_size=16,
        save_class_scores=False,
        save_outputs=False,
        exp_dir=None,
    ):
        super().__init__(dist_sync_on_step=False)

        self.ckpt_dir = ckpt_dir
        self.bert_path = bert_path
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.save_class_scores = save_class_scores
        self.save_outputs = save_outputs
        self.exp_dir = exp_dir

    def mini_batch(self, iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def compute(self):
        preds, gts = [], []
        for pred, gt, _ in self.pairs:
            preds.append(pred)
            gts.append(gt)
        result = calculate_nlg_metrics(preds, gts)
        result.update(calculate_ce_metrics_mp(preds, gts))
        scores = {
            # 'ce_precision_macro': precision_class.mean(),
            # 'ce_recall_macro': recall_class.mean(),
            # 'ce_f1_macro': f1_class.mean(),
            'chexpert_precision_micro': result["PRECISION"],
            'chexpert_recall_micro': result["RECALL"],
            'chexpert_f1_micro': result["F1_SCORE"],
            'custom_bleu@1': result["BLEU@1"],
            'custom_bleu@2': result["BLEU@2"],
            'custom_bleu@3': result["BLEU@3"],
            'custom_bleu@4': result["BLEU@4"],
            'custom_rougel': result["ROUGE_L"],
            'custom_meteor': result["METEOR"],
            # 'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
            # 'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
            # 'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
            # 'ce_num_examples': len(preds),
        }
        print(scores)
        return scores