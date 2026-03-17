

# DAG Search algorithm (?)





# from https://github.com/kmccleary3301/nested_learning/blob/main/src/nested_learning/levels.py







from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a nested-learning level."""

    name: str
    update_period: int
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            msg = f"update_period for level {self.name} must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps for level {self.name} must be non-negative"
            raise ValueError(msg)
        if self.jitter < 0:
            msg = f"jitter for level {self.name} must be non-negative"
            raise ValueError(msg)


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]):
        self._specs: Dict[str, LevelSpec] = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {name: LevelState() for name in self._specs}
        self._step: int = 0
        self._timeline: List[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1
        self._timeline.append({"step": self._step, "level": name})

    def levels_in_frequency_order(self) -> List[LevelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.update_period)

    def stats(self) -> Dict[str, LevelState]:
        return {
            name: LevelState(state.last_step, state.updates) for name, state in self._state.items()
        }

    def timeline(self) -> List[dict]:
        return list(self._timeline)


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Ensure deterministic ordering and validate duplicates."""

    specs = list(entries)
    seen = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            msg = f"Duplicate level spec {spec.name}"
            raise ValueError(msg)
        seen.add(spec.name)
        ordered.append(spec)
    return ordered
















# ----------------------------------------------------




# from https://github.com/dhruvramani/Neural-Architecture-Search-with-RL/blob/master/model_lenet/src/eval_performance.py



# Useless File
import sys
import numpy as np
from collections import Counter
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error, label_ranking_loss, hamming_loss, accuracy_score

def patk(predictions, labels):
    pak = np.zeros(3)
    K = np.array([1, 3, 5])
    for i in range(predictions.shape[0]):
        pos = np.argsort(-predictions[i, :])
        y = labels[i, :]
    y = y[pos]
    for j in range(3):
        k = K[j]
        pak[j] += (np.sum(y[:k]) / k)
    pak = pak / predictions.shape[0]
    return pak * 100.

'''
def precision_at_k(predictions, labels, k):
    act_set = 
'''
def cm_precision_recall(prediction,truth):
  """Evaluate confusion matrix, precision and recall for given set of labels and predictions
     Args
       prediction: a vector with predictions
       truth: a vector with class labels
     Returns:
       cm: confusion matrix
       precision: precision score
       recall: recall score"""
  confusion_matrix = Counter()

  positives = [1]

  binary_truth = [x in positives for x in truth]
  binary_prediction = [x in positives for x in prediction]

  for t, p in zip(binary_truth, binary_prediction):
    confusion_matrix[t,p] += 1

  cm = np.array([confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False]])
  #print cm
  precision = (cm[0]/(cm[0]+cm[2]+0.000001))
  recall = (cm[0]/(cm[0]+cm[3]+0.000001))
  return cm, precision, recall

def bipartition_scores(labels,predictions):
    """ Computes bipartitation metrics for a given multilabel predictions and labels
      Args:
        logits: Logits tensor, float - [batch_size, NUM_LABELS].
        labels: Labels tensor, int32 - [batch_size, NUM_LABELS].
      Returns:
        bipartiation: an array with micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1"""
    sum_cm=np.zeros((4))
    macro_precision=0
    macro_recall=0
    for i in range(labels.shape[1]):
        truth=labels[:,i]
        prediction=predictions[:,i]
        cm,precision,recall=cm_precision_recall(prediction, truth)
        sum_cm+=cm
        macro_precision+=precision
        macro_recall+=recall
    
    macro_precision=macro_precision/labels.shape[1]
    macro_recall=macro_recall/labels.shape[1]
    macro_f1 = 2*(macro_precision)*(macro_recall)/(macro_precision+macro_recall+0.000001)
    
    micro_precision = sum_cm[0]/(sum_cm[0]+sum_cm[2]+0.000001)
    micro_recall=sum_cm[0]/(sum_cm[0]+sum_cm[3]+0.000001)
    micro_f1 = 2*(micro_precision)*(micro_recall)/(micro_precision+micro_recall+0.000001)
    bipartiation = np.asarray([micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1])
    return bipartiation


def evaluate(predictions, labels, threshold=0.4, multi_label=True):
    '''
        True Positive  :  Label : 1, Prediction : 1
        False Positive :  Label : 0, Prediction : 1
        False Negative :  Label : 0, Prediction : 0
        True Negative  :  Label : 1, Prediction : 0
        Precision      :  TP/(TP + FP)
        Recall         :  TP/(TP + FN)
        F Score        :  2.P.R/(P + R)
        Ranking Loss   :  The average number of label pairs that are incorrectly ordered given predictions
        Hammming Loss  :  The fraction of labels that are incorrectly predicted. (Hamming Distance between predictions and labels)
    '''
    assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
    metrics = dict()
    if not multi_label:
        metrics['bae'] = BAE(labels, predictions)
        labels, predictions = np.argmax(labels, axis=1), np.argmax(predictions, axis=1)

        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = \
            precision_recall_fscore_support(labels, predictions, average='micro')
        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['coverage'], \
            metrics['average_precision'], metrics['ranking_loss'], metrics['pak'], metrics['hamming_loss'] \
            = 0, 0, 0, 0, 0, 0, 0, 0

    else:
        metrics['coverage'] = coverage_error(labels, predictions)
        metrics['average_precision'] = label_ranking_average_precision_score(labels, predictions)
        metrics['ranking_loss'] = label_ranking_loss(labels, predictions)
        
        for i in range(predictions.shape[0]):
            predictions[i, :][predictions[i, :] >= threshold] = 1
            predictions[i, :][predictions[i, :] < threshold] = 0

        metrics['bae'] = 0
        metrics['patk'] = patk(predictions, labels)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], metrics['macro_precision'], \
            metrics['macro_recall'], metrics['macro_f1'] = bipartition_scores(labels, predictions)
    return metrics