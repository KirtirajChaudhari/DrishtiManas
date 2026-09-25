import clsx from "clsx";
import { useMemo, useState } from "react";

import { ConfusionMatrix, RunsAccuracyBars, RunsLossChart, TrainingCurve } from "../components/charts";
import { ErrorBox, PageHeader, Section, Spinner, Stat, Table } from "../components/ui";
import { useReport } from "../hooks/useReport";
import type { EvalReport, MLPConfig, ModelReport, SelectionRun, TuningExperiment, TuningRun } from "../lib/api";
import { ACTIVATION_LABEL, LOSS_LABEL, OPTIMIZER_LABEL, fixed, num, pct } from "../lib/format";

const TOC = [
  ["results", "Results"],
  ["architecture", "Architecture"],
  ["training", "Training & convergence"],
  ["ensemble", "Ensemble & ablation"],
  ["evaluation", "Evaluation"],
  ["baselines", "Baselines"],
  ["tuning", "Hyperparameter tuning"],
  ["data", "Dataset & features"]
];

function Architecture({ report }: { report: ModelReport }) {
  const layers = report.final.layers;
  const cfg = report.final.config;
  const names = layers.map((_, i) => (i === 0 ? "Input features" : i === layers.length - 1 ? "Output (softmax)" : `Hidden ${i}`));
  return (
    <div className="flex flex-wrap items-stretch gap-2">
      {layers.map((n, i) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className={clsx(
              "min-w-[112px] rounded-xl px-4 py-3 ring-1",
              i === 0 && "bg-stone-100 ring-stone-200 dark:bg-stone-800 dark:ring-stone-700",
              i > 0 && i < layers.length - 1 && "bg-brand-50 ring-brand-200 dark:bg-brand-900/30 dark:ring-brand-800",
              i === layers.length - 1 && "bg-amber-50 ring-amber-200 dark:bg-amber-900/20 dark:ring-amber-800"
            )}
          >
            <p className="text-xs muted">{names[i]}</p>
            <p className="font-mono text-lg font-semibold tabular-nums">{num(n)}</p>
            <p className="text-[11px] muted">
              {i === 0 ? report.features.selected : i === layers.length - 1 ? "4 classes" : ACTIVATION_LABEL[cfg.activation]}
            </p>
          </div>
          {i < layers.length - 1 && <span className="text-stone-400" aria-hidden="true">→</span>}
        </div>
      ))}
    </div>
  );
}

function ConfigTable({ cfg, extra }: { cfg: MLPConfig; extra?: [string, string][] }) {
  const rows: [string, string][] = [
    ["Hidden layers", cfg.hidden_layers.length ? cfg.hidden_layers.join(" – ") : "none"],
    ["Activation (hidden)", ACTIVATION_LABEL[cfg.activation] ?? cfg.activation],
    ["Output activation", "Softmax"],
    ["Cost function", LOSS_LABEL[cfg.loss] ?? cfg.loss],
    ["Optimizer", OPTIMIZER_LABEL[cfg.optimizer] ?? cfg.optimizer],
    ["Learning rate", `${cfg.learning_rate}${cfg.lr_decay !== 1 ? ` (×${cfg.lr_decay}/epoch)` : ""}`],
    ["Batch size", String(cfg.batch_size)],
    ["Dropout (hidden layers)", cfg.dropout > 0 ? pct(cfg.dropout, 0) : "none"],
    ["L2 regularisation λ", String(cfg.l2)],
    ["Max epochs / early stopping", `${cfg.epochs} / patience ${cfg.early_stopping_patience ?? "off"}`],
    ["Class weights", cfg.class_weights ? cfg.class_weights.map((w) => w.toFixed(2)).join(", ") : "none (balanced data)"],
    ...(extra ?? [])
  ];
  return (
    <dl className="grid grid-cols-1 gap-x-6 text-sm sm:grid-cols-2">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 border-b border-stone-100 py-2 dark:border-stone-800">
          <dt className="muted">{k}</dt>
          <dd className="text-right font-mono text-[13px]">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function PerClassTable({ ev }: { ev: EvalReport }) {
  return (
    <Table
      head={["Class", "Precision", "Recall", "F1-score", "Support"]}
      align={["left", "right", "right", "right", "right"]}
      rows={[
        ...ev.per_class.map((c) => [<span className="font-medium">{c.class}</span>, pct(c.precision), pct(c.recall), pct(c.f1), num(c.support)]),
        [<span className="font-semibold">Macro average</span>, pct(ev.precision_macro), pct(ev.recall_macro), pct(ev.f1_macro), num(ev.per_class.reduce((a, c) => a + c.support, 0))],
        [<span className="font-semibold">Weighted average</span>, pct(ev.precision_weighted), pct(ev.recall_weighted), pct(ev.f1_weighted), ""]
      ]}
    />
  );
}

function TuningPanel({ exp, baseValue }: { exp: TuningExperiment; baseValue?: string }) {
  if (exp.key === "epochs") {
    const runs = exp.runs as TuningRun[];
    return (
      <div className="space-y-6">
        <p className="text-sm muted">
          One run for {exp.history?.length} epochs with early stopping switched off. Validation loss is lowest at epoch{" "}
          <strong>{exp.best}</strong>; after that the training loss keeps falling while validation loss stops improving,
          which is overfitting. That is why the final model uses early stopping on the validation loss.
        </p>
        {exp.history && <TrainingCurve history={exp.history} metric="loss" />}
        <Table
          head={["Epochs", "Training loss", "Validation loss", "Training acc.", "Validation acc."]}
          align={["left", "right", "right", "right", "right"]}
          rows={runs.map((r) => [r.label, fixed(r.train_loss ?? 0), fixed(r.val_loss ?? 0), pct(r.train_acc), pct(r.val_acc)])}
        />
      </div>
    );
  }
  const runs = exp.runs as TuningRun[];
  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h4 className="mb-2 text-sm font-medium">Accuracy (at the best epoch)</h4>
          <RunsAccuracyBars runs={runs} />
        </div>
        <div>
          <h4 className="mb-2 text-sm font-medium">Validation loss vs epochs</h4>
          <RunsLossChart runs={runs} />
        </div>
      </div>
      <Table
        head={["Value", "Val. accuracy", "Val. macro-F1", "Train accuracy", "Epochs (best)", "Parameters", "Time"]}
        align={["left", "right", "right", "right", "right", "right", "right"]}
        rows={runs.map((r) => [
          <span className={clsx("font-mono", String(r.value) === String(exp.best) && "font-semibold text-brand-700 dark:text-brand-400")}>
            {r.label}
            {String(r.value) === String(exp.best) ? " ★" : ""}
            {baseValue !== undefined && String(r.value) === baseValue ? " (start)" : ""}
          </span>,
          pct(r.val_acc),
          pct(r.val_f1 ?? 0),
          pct(r.train_acc),
          `${r.epochs_run} (${r.best_epoch})`,
          num(r.params ?? 0),
          `${(r.seconds ?? 0).toFixed(1)} s`
        ])}
      />
    </div>
  );
}

function Tuning({ report }: { report: ModelReport }) {
  const exps = report.tuning.experiments.filter((e) => e.key !== "selection");
  const selection = report.tuning.experiments.find((e) => e.key === "selection");
  const [active, setActive] = useState(exps[0]?.key);
  const exp = exps.find((e) => e.key === active) ?? exps[0];

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="-mx-1 mb-6 flex gap-1 overflow-x-auto pb-1" role="tablist" aria-label="Hyperparameter">
          {exps.map((e) => (
            <button
              key={e.key}
              role="tab"
              aria-selected={e.key === exp.key}
              onClick={() => setActive(e.key)}
              className={clsx(
                "whitespace-nowrap rounded-full px-3.5 py-1.5 text-sm font-medium",
                e.key === exp.key
                  ? "bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900"
                  : "text-stone-600 ring-1 ring-stone-200 hover:bg-stone-100 dark:text-stone-400 dark:ring-stone-700 dark:hover:bg-stone-800"
              )}
            >
              {e.title}
            </button>
          ))}
        </div>
        <TuningPanel exp={exp} baseValue={exp.incumbent !== undefined ? String(exp.incumbent) : undefined} />
      </div>

      {selection && (
        <div className="card">
          <h3 className="font-semibold">Best model selected</h3>
          <p className="mt-1 text-sm muted">
            The configuration found by the greedy search is compared with the original base configuration on the
            validation set. The winner, ranked by macro-F1, goes on to final training.
          </p>
          <div className="mt-4">
            <Table
              head={["Candidate", "Hidden layers", "Activation", "Optimizer / LR", "Batch", "L2", "Val. accuracy", "Val. macro-F1"]}
              align={["left", "left", "left", "left", "right", "right", "right", "right"]}
              rows={(selection.runs as SelectionRun[]).map((r, i) => [
                <span className={clsx("font-medium", i === 0 && "text-brand-700 dark:text-brand-400")}>
                  {r.name}
                  {i === 0 ? " ★" : ""}
                </span>,
                r.config.hidden_layers.join("-") || "none",
                ACTIVATION_LABEL[r.config.activation],
                `${OPTIMIZER_LABEL[r.config.optimizer]} / ${r.config.learning_rate}`,
                String(r.config.batch_size),
                String(r.config.l2),
                r.val_acc === null ? "–" : pct(r.val_acc),
                pct(r.val_f1)
              ])}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function ReportPage() {
  const { report, error } = useReport();
  const classCodes = useMemo(() => report?.dataset.classes.map((c) => c.code) ?? [], [report]);

  if (error) return <ErrorBox message={`Could not load the model report: ${error}`} />;
  if (!report) return <Spinner label="Loading model report…" />;

  const test = report.final.test;
  const final = report.final;
  const counts = report.dataset.counts;
  const bestBaseline = report.baselines.find((b) => b.name.startsWith("Baseline MLP"));

  return (
    <div>
      <PageHeader eyebrow="Model report" title="How well does the network classify OCT scans?">
        Every number on this page comes from <code className="font-mono text-[13px]">python -m ml.train</code>, run on{" "}
        {new Date(report.generated_at).toLocaleString()}. Hyperparameters were chosen on the validation split. The test
        split ({num(counts.test.reduce((a, b) => a + b, 0))} images) was used once, for the final evaluation.
        {report.quick && (
          <span className="mt-2 block font-medium text-amber-700 dark:text-amber-400">
            Smoke-test report: produced with --quick on a tiny subset, so these numbers are not meaningful.
          </span>
        )}
      </PageHeader>

      <nav className="mb-10 flex flex-wrap gap-x-4 gap-y-2 text-sm" aria-label="On this page">
        {TOC.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="text-brand-700 underline-offset-2 hover:underline dark:text-brand-400">
            {label}
          </a>
        ))}
      </nav>

      <div className="space-y-14">
        <Section id="results" title="Results on the held-out test set">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Stat label="Accuracy" value={pct(test.accuracy)} hint={bestBaseline ? `Baseline MLP: ${pct(bestBaseline.test_acc)}` : undefined} />
            <Stat label="Precision (macro)" value={pct(test.precision_macro)} />
            <Stat label="Recall (macro)" value={pct(test.recall_macro)} />
            <Stat label="F1-score (macro)" value={pct(test.f1_macro)} />
          </div>
        </Section>

        <Section id="architecture" title="Selected architecture" subtitle="A fully connected feed-forward network (multilayer perceptron) implemented from scratch in NumPy: forward propagation, backpropagation and the optimizers are all hand-written.">
          <div className="card space-y-6">
            <Architecture report={report} />
            <ConfigTable
              cfg={final.config}
              extra={[
                ["Trainable parameters", num(final.parameters)],
                ["Training images", `${num(report.dataset.final_training_size)} (${report.dataset.final_training_regime})`]
              ]}
            />
          </div>
        </Section>

        <Section
          id="training"
          title="Training and convergence"
          subtitle={`Cost (cross-entropy + L2 penalty) and accuracy per epoch. Training stopped after ${final.epochs_trained} epochs; the weights from epoch ${final.best_epoch}, which had the lowest validation loss, were restored.`}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h3 className="mb-3 text-sm font-semibold">Loss vs epochs</h3>
              <TrainingCurve history={final.history} metric="loss" />
            </div>
            <div className="card">
              <h3 className="mb-3 text-sm font-semibold">Accuracy vs epochs</h3>
              <TrainingCurve history={final.history} metric="acc" />
            </div>
          </div>
          {final.regimes.length > 1 && (
            <div className="card mt-6">
              <h3 className="text-sm font-semibold">Final training candidates</h3>
              <p className="mt-1 text-xs muted">
                The selected hyperparameters retrained on the full-size data: class-balanced undersampling vs all 97k
                images with a class-weighted loss, each at the tuned learning rate and at one third of it. Selected by
                validation macro-F1.
              </p>
              <div className="mt-3">
                <Table
                  head={["Candidate", "Epochs", "Val. accuracy", "Val. macro-F1"]}
                  align={["left", "right", "right", "right"]}
                  rows={final.regimes.map((r, i) => [
                    <span className={clsx("font-mono text-[13px]", i === 0 && "font-semibold text-brand-700 dark:text-brand-400")}>
                      {r.regime}
                      {i === 0 ? " ★" : ""}
                    </span>,
                    String(r.epochs ?? "–"),
                    r.val_acc === undefined ? "–" : pct(r.val_acc),
                    pct(r.val_f1)
                  ])}
                />
              </div>
            </div>
          )}
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <Stat
              label="Gradient check"
              value={report.gradient_check.max_relative_error.toExponential(1)}
              hint={`Max relative error between backprop and numerical gradients (${report.gradient_check.passed ? "passed" : "failed"}; < 1e-4 is correct)`}
            />
            <Stat
              label="Loss reduction"
              value={`${fixed(final.history[0].train_loss, 2)} → ${fixed(final.history[final.best_epoch - 1].train_loss, 2)}`}
              hint="Training cost, first epoch → best epoch"
            />
            <Stat
              label="Gradient norm"
              value={`${fixed(final.history[0].grad_norm, 2)} → ${fixed(final.history[final.history.length - 1].grad_norm, 2)}`}
              hint="Mean mini-batch ‖∇J‖, first → last epoch"
            />
          </div>
        </Section>

        <Section
          id="ensemble"
          title="Ensemble, calibration and what each step contributed"
          subtitle="Two post-hoc steps applied after hyperparameter tuning. Both were chosen on validation data; the table at the bottom shows, for honesty, whether each one also helped on the test set."
        >
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-sm font-semibold">Ensemble</h3>
              <p className="mt-1 text-sm muted">
                {final.ensemble.size} models with the selected hyperparameters, differing only in random seed (weight
                initialisation and mini-batch shuffling), are trained independently, and every prediction averages their
                softmax outputs. Averaging reduces variance when the members&apos; mistakes are only partly correlated.
                Here it raised validation macro-F1 from {pct(final.ensemble.members[0]?.val_f1 ?? 0)} to{" "}
                {pct(final.ensemble.uncalibrated_validation.f1_macro)}, but did not improve the test score (see the
                table below).
              </p>
              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Ensemble size" value={String(final.ensemble.size)} />
                <Stat label="Total parameters" value={num(final.ensemble.total_parameters)} hint={`${num(final.parameters)} per model`} />
                <Stat label="Single model (seed 42)" value={pct(final.ensemble.members[0]?.val_f1 ?? 0)} hint="Validation macro-F1" />
                <Stat
                  label="Ensemble average"
                  value={pct(final.ensemble.uncalibrated_validation.f1_macro)}
                  hint="Validation macro-F1, uncalibrated"
                />
              </div>
              <div className="mt-4">
                <Table
                  head={["Model", "Seed", "Val. accuracy", "Val. macro-F1"]}
                  align={["left", "right", "right", "right"]}
                  rows={final.ensemble.members.map((m, i) => [`Member ${i + 1}`, String(m.seed), pct(m.val_acc), pct(m.val_f1)])}
                />
              </div>
            </div>

            <div className="card">
              <h3 className="text-sm font-semibold">Class-prior calibration (logit adjustment)</h3>
              <p className="mt-1 text-sm muted">
                The network over-predicts CNV, the class that dominates training and is hardest to separate from Drusen.
                A per-class additive bias <code className="font-mono text-xs">b</code> changes the decision rule from{" "}
                <code className="font-mono text-xs">argmax(p)</code> to <code className="font-mono text-xs">argmax(log p + b)</code>.
                It never touches the trained weights; it only shifts each class&apos;s effective threshold. The bias is
                found by coordinate ascent on a class-balanced subsample of the validation set
                {final.calibration.calibration_set_size ? ` (${num(final.calibration.calibration_set_size)} images)` : ""},
                because the test set, like any real deployment, is balanced while the full validation split is not.
              </p>
              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                {report.dataset.classes.map((c, i) => (
                  <Stat key={c.code} label={c.code} value={final.calibration.bias[i]?.toFixed(2) ?? "0.00"} hint="log-bias" />
                ))}
              </div>
              <p className="mt-4 text-sm">
                Macro-F1 on the balanced calibration subsample:{" "}
                <span className="font-mono">{pct(final.calibration.val_f1_before)}</span> →{" "}
                <span className="font-mono font-semibold text-brand-700 dark:text-brand-400">{pct(final.calibration.val_f1_after)}</span>.
                {final.calibration.full_validation_f1_before !== undefined && final.calibration.full_validation_f1_after !== undefined && (
                  <>
                    {" "}
                    On the full (imbalanced) validation split it moves {pct(final.calibration.full_validation_f1_before)} →{" "}
                    {pct(final.calibration.full_validation_f1_after)}: it deliberately trades a little performance on the
                    majority classes for recall on Drusen.
                  </>
                )}
              </p>
            </div>

            {final.ablation && (
              <div className="card">
                <h3 className="text-sm font-semibold">What each step contributed</h3>
                <p className="mt-1 text-sm muted">
                  Reported after all choices were made on validation. The test set has only 1,000 images, so differences of
                  one or two points are within noise
                  {final.ablation.significance_vs_previous
                    ? ` (standard error of accuracy ≈ ±${(final.ablation.significance_vs_previous.accuracy_standard_error * 100).toFixed(1)} points)`
                    : ""}
                  .
                </p>
                <div className="mt-4">
                  <Table
                    head={["Step", "Val. macro-F1", "Test accuracy", "Test macro-F1", "Test Drusen F1"]}
                    align={["left", "right", "right", "right", "right"]}
                    rows={[
                      ...(final.ablation.previous_version ? [final.ablation.previous_version] : []),
                      ...final.ablation.rows
                    ].map((r, i, all) => [
                      <span className={clsx(i === all.length - 1 && "font-semibold text-brand-700 dark:text-brand-400")}>
                        {r.step}
                        {i === all.length - 1 ? " ★" : ""}
                      </span>,
                      pct(r.val_f1),
                      pct(r.test_acc),
                      pct(r.test_f1),
                      pct(r.test_per_class_f1.DRUSEN ?? 0)
                    ])}
                  />
                </div>
                {final.ablation.significance_vs_previous && (
                  <p className="mt-4 text-sm leading-relaxed">
                    <strong>Is the improvement real?</strong> A paired McNemar test on the same test images compares the
                    served model with the previous version. The overall accuracy gain is <strong>not</strong> statistically
                    significant (p = {final.ablation.significance_vs_previous.overall.p_value.toFixed(2)}). The gain on
                    Drusen is: recall rose from {pct(final.ablation.significance_vs_previous.drusen.recall_before)} to{" "}
                    {pct(final.ablation.significance_vs_previous.drusen.recall_after)} (
                    {final.ablation.significance_vs_previous.drusen.prev_wrong_new_right} Drusen scans newly correct vs{" "}
                    {final.ablation.significance_vs_previous.drusen.prev_right_new_wrong} newly wrong, p &lt; 0.0001).
                  </p>
                )}
                {final.ablation.latency_ms_median !== undefined && (
                  <p className="mt-2 text-xs muted">
                    Measured latency of the full served path (decode, preprocess, features, {final.ensemble.size} models,
                    calibration): median {final.ablation.latency_ms_median.toFixed(1)} ms, 95th percentile{" "}
                    {(final.ablation.latency_ms_p95 ?? 0).toFixed(1)} ms on one CPU core.
                  </p>
                )}
              </div>
            )}
          </div>
        </Section>

        <Section id="evaluation" title="Confusion matrix and per-class metrics" subtitle="Test set. CNV, DME and Drusen are all abnormal; mistaking one for Normal is the most costly error.">
          <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
            <div className="card">
              <ConfusionMatrix matrix={test.confusion_matrix} labels={classCodes} />
            </div>
            <div className="card">
              <PerClassTable ev={test} />
              <p className="mt-4 text-xs muted">
                Validation set ({num(report.final.validation.per_class.reduce((a, c) => a + c.support, 0))} images): accuracy{" "}
                {pct(final.validation.accuracy)}, macro-F1 {pct(final.validation.f1_macro)}.
              </p>
            </div>
          </div>
        </Section>

        <Section id="baselines" title="Baseline accuracy vs our network" subtitle="Reference models trained on the same features and tuning subset, compared with the final network.">
          <div className="card">
            <Table
              head={["Model", "Val. accuracy", "Val. macro-F1", "Test accuracy", "Test macro-F1"]}
              align={["left", "right", "right", "right", "right"]}
              rows={[
                ...report.baselines.map((b) => [
                  <div>
                    <p className="font-medium">{b.name}</p>
                    <p className="text-xs muted">{b.description}</p>
                  </div>,
                  pct(b.val_acc),
                  pct(b.val_f1),
                  pct(b.test_acc),
                  pct(b.test_f1)
                ]),
                [
                  <div>
                    <p className="font-semibold text-brand-700 dark:text-brand-400">DrishtiManas MLP (tuned, ours) ★</p>
                    <p className="text-xs muted">From-scratch NumPy network with the selected hyperparameters.</p>
                  </div>,
                  pct(final.validation.accuracy),
                  pct(final.validation.f1_macro),
                  pct(test.accuracy),
                  pct(test.f1_macro)
                ]
              ]}
            />
          </div>
        </Section>

        <Section
          id="tuning"
          title="Hyperparameter tuning"
          subtitle={`Greedy coordinate search. The hyperparameters are tuned one at a time, in the order of the tabs. Each experiment starts from the best configuration found so far ("start"), and a new value is adopted (★) only if it raises validation macro-F1 by at least 0.5 points. Every run trains on a class-balanced subset of ${num(report.dataset.tuning_subset_per_class * 4)} images and is scored on the validation split.`}
        >
          <Tuning report={report} />
        </Section>

        <Section id="data" title="Dataset, preprocessing and feature extraction">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h3 className="text-sm font-semibold">{report.dataset.name}</h3>
              <p className="mt-1 text-xs muted">
                {report.dataset.source} · {report.dataset.license}
              </p>
              <div className="mt-4">
                <Table
                  head={["Class", "Train", "Validation", "Test"]}
                  align={["left", "right", "right", "right"]}
                  rows={report.dataset.classes.map((c, i) => [
                    <span>
                      <span className="font-medium">{c.code}</span> <span className="text-xs muted">{c.name}</span>
                    </span>,
                    num(counts.train[i]),
                    num(counts.val[i]),
                    num(counts.test[i])
                  ])}
                />
              </div>
              <p className="mt-3 text-xs muted">
                The training split is imbalanced (Normal and CNV dominate). Tuning used a class-balanced subset, and the
                final model was trained with {report.dataset.final_training_regime === "balanced" ? "class-balanced undersampling" : "class-weighted cross-entropy"}.
              </p>
            </div>
            <div className="card">
              <h3 className="text-sm font-semibold">Preprocessing pipeline</h3>
              <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-sm">
                {report.preprocessing.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ol>
              <h3 className="mt-6 text-sm font-semibold">Feature extraction comparison</h3>
              <div className="mt-2">
                <Table
                  head={["Features", "Dimensions", "Val. accuracy", "Val. macro-F1"]}
                  align={["left", "right", "right", "right"]}
                  rows={report.features.comparison.map((f) => [
                    <span className={clsx(f.label === report.features.selected && "font-semibold text-brand-700 dark:text-brand-400")}>
                      {f.label}
                      {f.label === report.features.selected ? " ★" : ""}
                    </span>,
                    num(f.dim),
                    pct(f.val_acc),
                    pct(f.val_f1 ?? 0)
                  ])}
                />
              </div>
              <p className="mt-3 text-xs muted">
                Pixels = raw grayscale intensities. HOG = histograms of oriented gradients (9 orientation bins per cell,
                2×2-cell blocks with L2-Hys normalisation), which capture the edges and contours of the retinal layers.
              </p>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
