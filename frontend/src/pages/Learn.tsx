import type { ReactNode } from "react";

import { FunctionPlot } from "../components/charts";
import { PageHeader, Section } from "../components/ui";
import { useReport } from "../hooks/useReport";
import { pct } from "../lib/format";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

const ACTIVATIONS = [
  {
    name: "ReLU",
    formula: "g(z) = max(0, z)        g′(z) = 1 if z > 0 else 0",
    fn: (z: number) => Math.max(0, z),
    d: (z: number) => (z > 0 ? 1 : 0),
    note: "Cheap to compute, and the gradient does not shrink for z > 0, so deep networks train quickly. Units can 'die' if z stays negative."
  },
  {
    name: "Leaky ReLU",
    formula: "g(z) = z if z > 0 else 0.01·z",
    fn: (z: number) => (z > 0 ? z : 0.01 * z),
    d: (z: number) => (z > 0 ? 1 : 0.01),
    note: "Keeps a small slope for negative inputs so that no neuron stops learning."
  },
  {
    name: "Sigmoid",
    formula: "g(z) = 1 / (1 + e^(−z))   g′(z) = g(z)·(1 − g(z))",
    fn: sigmoid,
    d: (z: number) => sigmoid(z) * (1 - sigmoid(z)),
    note: "Squashes to (0, 1). Its derivative is at most 0.25, so gradients vanish as they flow back through many layers."
  },
  {
    name: "Tanh",
    formula: "g(z) = tanh(z)          g′(z) = 1 − tanh²(z)",
    fn: Math.tanh,
    d: (z: number) => 1 - Math.tanh(z) ** 2,
    note: "A zero-centred sigmoid with range (−1, 1). Stronger gradients than the sigmoid, but it still saturates."
  }
];

function Prose({ children }: { children: ReactNode }) {
  return <div className="max-w-3xl space-y-3 text-[15px] leading-relaxed text-stone-700 dark:text-stone-300">{children}</div>;
}

export default function LearnPage() {
  const { report } = useReport();
  const perceptron = report?.baselines.find((b) => b.name.startsWith("Single-layer"));
  const final = report?.final;
  const size = report?.dataset.image_shape[0] ?? 64;
  const res = `${size} × ${size}`;

  return (
    <div>
      <PageHeader eyebrow="How it works" title="From an OCT scan to a diagnosis, one layer at a time">
        This page covers the theory the web app demonstrates: what a multilayer perceptron is, how forward propagation,
        activation functions, the cost function and backpropagation work together, and where the approach reaches its limits.
      </PageHeader>

      <div className="space-y-14">
        <Section title="1. The problem and the data">
          <Prose>
            <p>
              Optical coherence tomography (OCT) produces cross-sectional images of the retina. Ophthalmologists use it to
              spot <strong>choroidal neovascularization (CNV)</strong>, <strong>diabetic macular edema (DME)</strong> and{" "}
              <strong>drusen</strong>, which are leading causes of vision loss, and to tell them apart from a{" "}
              <strong>normal</strong> retina.
            </p>
            <p>
              We use OCTMNIST, the MedMNIST v2 version of the Kermany et al. (2018) dataset: 109,309 labelled B-scans,
              available as {res} grayscale images and split into training (97,477), validation (10,832) and test
              (1,000) sets.
              Every image, whether from the dataset or uploaded in the browser, goes through the same preprocessing:
              grayscale, centre square crop, resize to {res}, and scaling to [0, 1]. We then extract{" "}
              <em>features</em>: the raw pixels and/or a HOG descriptor, standardised with the training-set mean and
              standard deviation.
            </p>
          </Prose>
        </Section>

        <Section title="2. The multilayer perceptron (MLP)">
          <Prose>
            <p>
              A perceptron computes a weighted sum of its inputs plus a bias and passes it through an activation function.
              An MLP stacks these units into <strong>layers</strong>. Every neuron in one layer connects to every neuron in
              the next. The input layer holds the feature vector, one or more <strong>hidden layers</strong> learn
              intermediate representations, and the output layer has one neuron per class.
            </p>
            <p>
              A single-layer perceptron can only draw straight decision boundaries (hyperplanes), so it fails on problems
              that are not linearly separable, such as XOR. A hidden layer with a non-linear activation lets the network
              bend those boundaries. By the universal approximation theorem, one wide enough hidden layer can approximate
              any continuous function.
              {perceptron && final && (
                <>
                  {" "}
                  On OCTMNIST, the single-layer perceptron reaches {pct(perceptron.test_acc)} test accuracy; the tuned MLP
                  reaches {pct(final.test.accuracy)}.
                </>
              )}
            </p>
          </Prose>
        </Section>

        <Section title="3. Forward propagation">
          <Prose>
            <p>
              For each layer <em>l</em>, the network takes the previous layer&apos;s activations, applies an affine
              transformation, then applies the activation function. The output layer uses softmax, which turns the raw
              scores (logits) into probabilities that sum to 1:
            </p>
          </Prose>
          <pre className="formula mt-4">{`A[0] = X                                   (N × d input features)
Z[l] = A[l−1] · W[l] + b[l]                 (affine transform)
A[l] = g(Z[l])                              (hidden layers: ReLU / tanh / …)
P    = softmax(Z[L]),   P_k = e^{Z_k} / Σ_j e^{Z_j}
ŷ    = argmax_k P_k                         (predicted class)`}</pre>
          <Prose>
            <p className="mt-4">
              Weights are initialised randomly with He initialisation (std = √(2 / fan_in)) for ReLU-family activations
              and Xavier initialisation (√(1 / fan_in)) for sigmoid and tanh. This keeps the signal from exploding or
              vanishing as it passes through the layers. Biases start at zero.
            </p>
          </Prose>
        </Section>

        <Section title="4. Activation functions" subtitle="Solid line: g(z). Dashed line: its derivative g′(z), which backpropagation multiplies into the gradient at every layer.">
          <div className="grid gap-4 sm:grid-cols-2">
            {ACTIVATIONS.map((a) => (
              <div key={a.name} className="card">
                <h3 className="font-semibold">{a.name}</h3>
                <p className="mt-1 font-mono text-xs muted">{a.formula}</p>
                <div className="mt-3">
                  <FunctionPlot fn={a.fn} deriv={a.d} />
                </div>
                <p className="mt-2 text-sm muted">{a.note}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section title="5. Cost function: categorical cross-entropy">
          <Prose>
            <p>
              The cost J measures how far the predicted probabilities are from the true one-hot labels. For multi-class
              classification we minimise cross-entropy, plus an L2 penalty that discourages large weights and reduces
              overfitting:
            </p>
          </Prose>
          <pre className="formula mt-4">{`J = −(1/N) Σ_i Σ_k  w_{y_i} · y_ik · log(p_ik)   +   (λ/2) Σ_l ‖W[l]‖²

y_ik ∈ {0,1}  one-hot label         p_ik  softmax probability
w_c           optional class weight  λ     L2 regularisation strength`}</pre>
          <Prose>
            <p className="mt-4">
              Cross-entropy heavily penalises confident wrong answers: the loss for a true class predicted at 1% is
              −log 0.01 ≈ 4.6. Combined with softmax, its gradient with respect to the logits is simply{" "}
              <code className="font-mono text-sm">(P − Y) / N</code>. We also trained with mean squared error for
              comparison (see the tuning results): its gradient passes through the softmax Jacobian and becomes very
              small when the network is confidently wrong, so it learns more slowly.
            </p>
          </Prose>
        </Section>

        <Section title="6. Backpropagation and gradient descent">
          <Prose>
            <p>
              Backpropagation applies the chain rule from the output layer back to the input and computes ∂J/∂W and ∂J/∂b
              for every layer in a single backward pass:
            </p>
          </Prose>
          <pre className="formula mt-4">{`dZ[L]   = w ⊙ (P − Y) / N                          (softmax + cross-entropy)
dW[l]   = A[l−1]ᵀ · dZ[l]  +  λ · W[l]
db[l]   = Σ_rows dZ[l]
dA[l−1] = dZ[l] · W[l]ᵀ
dZ[l−1] = dA[l−1] ⊙ g′(Z[l−1])                       (repeat down to l = 1)

SGD + momentum:  v ← μ·v − η·dW,   W ← W + v          (η = learning rate, μ = 0.9)`}</pre>
          <Prose>
            <p className="mt-4">
              Each epoch shuffles the training set and processes it in mini-batches. Each batch runs forward
              propagation, computes the loss, runs backpropagation and applies an optimizer step. To confirm the
              hand-derived gradients, we compare them with centred finite differences (J(θ+ε) − J(θ−ε)) / 2ε.
              {report && (
                <>
                  {" "}
                  The largest relative difference was{" "}
                  <strong>{report.gradient_check.max_relative_error.toExponential(1)}</strong>, which shows the
                  backpropagation is implemented correctly.
                </>
              )}{" "}
              Convergence shows in the loss-vs-epoch curves on the model report: the training cost falls steadily, the
              gradient norm shrinks, and early stopping keeps the epoch with the lowest validation loss.
            </p>
          </Prose>
        </Section>

        <Section title="7. Hyperparameters we tuned">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              ["Learning rate η", "Step size of each update. Too small converges slowly; too large oscillates or diverges."],
              ["Hidden neurons", "Width of a layer. More neurons give more capacity, but also more parameters and more risk of overfitting."],
              ["Hidden layers", "Depth. 0 layers is a linear classifier; more layers learn hierarchical features but are harder to optimise."],
              ["Batch size", "Samples per gradient estimate. Small batches give noisy updates that regularise; large batches give smoother, faster epochs."],
              ["Activation", "The non-linearity. It controls gradient flow (ReLU vs saturating sigmoid/tanh)."],
              ["Epochs", "Passes over the data. Too few underfits and too many overfits, which early stopping prevents."],
              ["Optimizer", "Plain SGD, SGD with momentum (smooths the updates), or Adam (adaptive per-weight step sizes)."],
              ["Cost function", "Cross-entropy vs mean squared error on the softmax output."],
              ["L2 penalty λ", "Weight decay. It trades training fit for generalisation."],
              ["Dropout", "Randomly zeroes a fraction of hidden units on every training batch, so no unit can rely on any other single unit; disabled at inference."]
            ].map(([t, d]) => (
              <div key={t} className="card !p-5">
                <h3 className="text-sm font-semibold">{t}</h3>
                <p className="mt-1 text-sm muted">{d}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section title="8. Beyond hyperparameter tuning: ensembling and calibration">
          <Prose>
            <p>
              Two more techniques run after the hyperparameter search, on top of the single best configuration it
              finds. Neither one touches the network&apos;s architecture or how it is trained &mdash; they only change
              how its predictions are combined and read.
            </p>
            <p>
              <strong>Ensembling.</strong> Several networks with the identical selected hyperparameters are trained
              from different random seeds, so each ends up in a slightly different region of weight space (different
              initial weights, different mini-batch order). Averaging their softmax outputs cancels out some of each
              individual model&apos;s idiosyncratic mistakes &mdash; a classic bias-variance argument: the members
              share the same bias, but their errors are only partially correlated, so the variance of the average is
              lower than the variance of any one member.
              {report && (
                <>
                  {" "}
                  Here, {report.final.ensemble.size} models are averaged. That raised validation macro-F1 from{" "}
                  <strong>{pct(report.final.ensemble.members[0]?.val_f1 ?? 0)}</strong> (a single model) to{" "}
                  <strong>{pct(report.final.ensemble.uncalibrated_validation.f1_macro)}</strong>, but it did not raise
                  the test score: a useful reminder that a gain on validation is a hypothesis, not a guarantee.
                </>
              )}{" "}
              The cost is proportional: N models means N forward passes per prediction, but each pass is a few matrix
              multiplications, so the whole ensemble still answers in about a millisecond.
            </p>
            <p>
              <strong>Class-prior calibration.</strong> A softmax classifier predicts{" "}
              <code className="font-mono text-sm">argmax P</code>, the class with the highest probability. The
              training set is dominated by Normal and CNV, and CNV is visually hard to separate from Drusen, so the
              network learns a boundary that over-predicts CNV. A small post-hoc fix from the class-imbalance
              literature, logit adjustment, searches for a per-class additive bias{" "}
              <code className="font-mono text-sm">b</code> so the rule becomes{" "}
              <code className="font-mono text-sm">argmax(log P + b)</code>: an over-predicted class gets a negative
              bias, an under-predicted class a positive one. It never touches a trained weight. The bias must be tuned
              on data whose class mix matches where the model will be used. Here that means a class-balanced sample of
              the validation set, because the test set and a real screening deployment are balanced while the full
              validation split is not. Tuning it on the imbalanced split instead improved validation but made the test
              score worse.
              {report && (
                <>
                  {" "}
                  On the balanced calibration sample, macro-F1 moves from{" "}
                  <strong>{pct(report.final.calibration.val_f1_before)}</strong> to{" "}
                  <strong>{pct(report.final.calibration.val_f1_after)}</strong>.
                  {report.final.ablation?.significance_vs_previous && (
                    <>
                      {" "}
                      Together with dropout, it lifts Drusen recall on the test set from{" "}
                      {pct(report.final.ablation.significance_vs_previous.drusen.recall_before)} (previous version) to{" "}
                      {pct(report.final.ablation.significance_vs_previous.drusen.recall_after)}.
                    </>
                  )}
                </>
              )}
            </p>
          </Prose>
        </Section>

        <Section title="9. Limitations of the MLP">
          <Prose>
            <ul className="list-disc space-y-2 pl-5">
              <li>
                <strong>No spatial awareness.</strong> The image is flattened into a vector, so the network does not know
                which pixels are neighbours, and a small shift of the retina changes every input. Convolutional networks
                (CNNs) build in translation equivariance and weight sharing, and reach higher accuracy on OCT.
              </li>
              <li>
                <strong>Many parameters.</strong> Every input is wired to every hidden neuron: the first layer alone has
                (input dimensions × hidden width) weights. That makes overfitting likely and limits how large an input
                resolution is practical, which is one reason we work at {res}.
              </li>
              <li>
                <strong>Relies on feature engineering.</strong> Handcrafted HOG features help, but they are fixed; a CNN
                learns its own filters.
              </li>
              <li>
                <strong>Low resolution.</strong> At {res} pixels, fine retinal detail such as small drusen deposits is
                lost. This is why Drusen is the hardest class.
              </li>
              <li>
                <strong>Non-convex optimisation.</strong> Results depend on initialisation and hyperparameters, and
                training can stall on plateaus or in poor local minima.
              </li>
              <li>
                <strong>Overconfident outside its domain.</strong> Softmax always picks a class, even for a photo of a
                cat. The app warns about colour and low-confidence inputs, but the model cannot say &quot;I don&apos;t
                know&quot;.
              </li>
              <li>
                <strong>Black box.</strong> The network gives no clinical explanation for its decision. It is a
                screening aid, not a diagnosis.
              </li>
            </ul>
          </Prose>
        </Section>
      </div>
    </div>
  );
}
