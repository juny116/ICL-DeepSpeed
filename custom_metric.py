# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets
from numpy import indices
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

_DESCRIPTION = """
"""


_KWARGS_DESCRIPTION = """
"""


_CITATION = """
"""

class CUSTOM_METRIC(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": {
                        "idx": datasets.Sequence(datasets.Value("int32")),
                        "label": datasets.Sequence(datasets.Value("int32")),
                        "probs": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    }
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": {
                        "idx": datasets.Value("int32"),
                        "label": datasets.Value("int32"),
                        "probs": datasets.Sequence(datasets.Value("float32")),
                    }
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None):
        idx_list = []
        preds = []
        refs = []
        probs = []
        for p, r in zip(predictions, references):
            if r['idx'] not in idx_list:
                preds.append(p)
                refs.append(r['label'])
                probs.append(r['probs'])
                idx_list.append(r['idx'])

        preds = [x for _, x in sorted(zip(idx_list, preds))]
        refs = [x for _, x in sorted(zip(idx_list, refs))]
        probs = [x for _, x in sorted(zip(idx_list, probs))]

        accuracy = accuracy_score(refs, preds, normalize=normalize, sample_weight=sample_weight)   
        f1 = f1_score(
            refs, preds, labels=None, pos_label=1, average="macro", sample_weight=sample_weight
        )
        return {
            "accuracy": accuracy * 100, 
            "f1": float(f1) * 100 if f1.size == 1 else f1,
            "labels": refs,
            "predictions": preds,
            "probs": probs
        }
