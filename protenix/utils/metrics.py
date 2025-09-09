# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from protenix.utils.distributed import gather_and_merge
from protenix.utils.profiler import profile_function

common_aggregator = {
    "avg": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "pct90": lambda x: np.percentile(x, 90),
    "pct99": lambda x: np.percentile(x, 99),
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "nonzero_avg": lambda x: np.mean(x[x != 0]) if np.any(x != 0) else np.nan,
}


class SimpleMetricAggregator(object):
    """A quite simple metrics calculator that only do simple metrics aggregation."""

    def __init__(
        self, aggregator_names=None, gather_before_calc=True, need_gather=True
    ):
        super(SimpleMetricAggregator, self).__init__()
        self.gather_before_calc = gather_before_calc
        self.need_gather = need_gather
        self._metric_data = {}

        self.aggregators = {name: common_aggregator[name] for name in aggregator_names}

    @profile_function("metric_aggregator_add")
    def add(self, key, value, namespace="default"):
        value_dict = self._metric_data.setdefault(namespace, {})
        value_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = torch.tensor([value])
        elif isinstance(value, torch.Tensor):
            # Keep as tensor, just detach to avoid gradient computation
            try:
                value = value.detach()
                if value.dim() == 0:
                    value = value.unsqueeze(0)  # Make scalar into 1D tensor
            except RuntimeError as e:
                raise RuntimeError(f"Error processing tensor for key {key}: {str(e)}")
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        value_dict[key].append(value)

    def calc(self):
        metric_data, self._metric_data = self._metric_data, {}
        if self.need_gather and self.gather_before_calc:
            metric_data = gather_and_merge(
                metric_data, aggregation_func=lambda x: sum(x, [])
            )
        results = {}
        for agg_name, agg_func in self.aggregators.items():
            for namespace, value_dict in metric_data.items():
                for key, data in value_dict.items():
                    plain_key = f"{namespace}/{key}" if namespace != "default" else key
                    plain_key = f"{plain_key}.{agg_name}"
                    # Convert tensors to CPU numpy arrays for aggregation
                    concatenated_data = torch.cat(data, dim=0).cpu().numpy()
                    results[plain_key] = agg_func(concatenated_data)
        if self.need_gather and not self.gather_before_calc:  # need gather after calc
            results = gather_and_merge(results, aggregation_func=np.mean)
        return results
