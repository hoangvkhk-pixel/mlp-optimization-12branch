"""Population-wide cooperative MLP batching for the legacy 12-branch runner.

This keeps the original candidate evaluator and SHADE logic unchanged.  The
only difference is how MLP queries are scheduled: candidates yield at MLP
queries, requests are grouped into one normal/duck TensorFlow batch, then the
candidates resume in their original order.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from greenlet import greenlet, getcurrent

from new20_sizing_eval import (
    AERO_INPUT_COLS,
    AERO_OUTPUT_COLS,
    INPUT_COLS_V2,
    MlpBackend,
    evaluate_candidate,
    failed_candidate_result,
)


class BatchProxy(MlpBackend):
    def __init__(self):
        self.cache = {}

    def predict_outputs(self, case, alpha, delta):
        values = tuple(float(case[k]) for k in AERO_INPUT_COLS[:-2]) + (
            float(alpha),
            float(delta),
        )
        if values not in self.cache:
            family = "duck" if float(case["a_S_rel"]) > 0.5 else "normal"
            self.cache[values] = getcurrent().parent.switch((family, values))
        return self.cache[values]

    def predict_outputs_batch(self, case, alpha, delta):
        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        delta = np.asarray(delta, dtype=float).reshape(-1)
        if alpha.shape != delta.shape:
            raise ValueError("alpha and delta batch shapes differ")
        return [self.predict_outputs(case, a, d) for a, d in zip(alpha, delta)]


class PopulationBatchEvaluator:
    def __init__(self, normal_dir, duck_dir):
        self.models = {
            "normal": MlpBackend(normal_dir),
            "duck": MlpBackend(duck_dir),
        }
        for model in self.models.values():
            if list(model.output_cols) != AERO_OUTPUT_COLS:
                raise ValueError("Unexpected MLP output order")
            if list(getattr(model.sx, "feature_names_in_", AERO_INPUT_COLS)) != AERO_INPUT_COLS:
                raise ValueError("Unexpected MLP input order")
        self.stats = {}
        self.errors = []

    def evaluate(
        self,
        population,
        *,
        mpay=600.0,
        w_rpm=5800.0,
        t=25.0,
        type_power="DBC",
        gamma=0.87,
        ce1=0.285,
        ce2=0.27,
    ):
        rows = [dict(zip(INPUT_COLS_V2, row.tolist())) for row in np.asarray(population, dtype=float)]
        results = [None] * len(rows)
        self.errors = []
        counts = Counter()

        def evaluate_one(i):
            try:
                result = evaluate_candidate(
                    rows[i],
                    BatchProxy(),
                    i,
                    mpay=mpay,
                    w_rpm=w_rpm,
                    t=t,
                    type_power=type_power,
                    gamma=gamma,
                    ce1=ce1,
                    ce2=ce2,
                )
                if not np.isfinite(result).all():
                    raise ValueError("Non-finite sizing output")
                return result
            except Exception as exc:
                self.errors.append({"candidate": i, "error": repr(exc)})
                return failed_candidate_result(rows[i])

        tasks = [greenlet(lambda i=i: evaluate_one(i)) for i in range(len(rows))]
        pending = {}
        for i, task in enumerate(tasks):
            value = task.switch()
            if task.dead:
                results[i] = value
            else:
                pending[i] = value

        while pending:
            predictions = {}
            for family in ("normal", "duck"):
                indices = [i for i, req in pending.items() if req[0] == family]
                if not indices:
                    continue
                model = self.models[family]
                inputs = np.asarray([pending[i][1] for i in indices], dtype=np.float32)
                if not np.isfinite(inputs).all():
                    raise ValueError("Non-finite input to inference batch")
                transformed = model.sx.transform(inputs)
                scaled = model.model(transformed, training=False).numpy()
                outputs = model.sy.inverse_transform(scaled)
                if not np.isfinite(outputs).all():
                    raise ValueError("Non-finite model batch output")
                counts[family + "_calls"] += 1
                counts[family + "_rows"] += len(indices)
                counts[family + "_max_batch"] = max(
                    counts[family + "_max_batch"], len(indices)
                )
                for i, output in zip(indices, outputs):
                    predictions[i] = dict(zip(model.output_cols, output.astype(float)))

            next_pending = {}
            for i in pending:
                value = tasks[i].switch(predictions[i])
                if tasks[i].dead:
                    results[i] = value
                else:
                    next_pending[i] = value
            pending = next_pending

        self.stats = dict(counts)
        self.stats["failed_candidates"] = len(self.errors)
        return np.asarray(results, dtype=float)

