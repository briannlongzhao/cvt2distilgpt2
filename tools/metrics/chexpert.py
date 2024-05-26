from tools.chexbert import CheXbert
from tools.metrics.natural_language import NaturalLanguage
from tools.utils import enumerated_save_path
import os
import pandas as pd
import time
import torch

from .eval_metrics import calculate_ce_metrics_mp, calculate_nlg_metrics



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