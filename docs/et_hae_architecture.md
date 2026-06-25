# ET-HAE Architecture

## Scope

This document describes the implemented ET-HAE component only. RECCON integration is the next layer.

## Frozen ET Predictor

The frozen predictor is `skboy/emotion_et_2nd_model`. It maps text to word-level predicted eye-tracking features.

\[
\tilde{\mathbf{t}} = f_{\mathrm{ET}}(x_{1:n})
\]

Only the predicted `TRT` channel is used by ET-HAE v1.

\[
\tilde{t}_i = \tilde{\mathbf{y}}_{i,\mathrm{TRT}}
\]

The predictor is not updated during ET-HAE training.

## Noisy Heatmap

Predicted TRT is converted into a probability distribution over valid word positions.

\[
g_i^0 =
\frac{\log(1 + \max(\tilde{t}_i, 0)) + \epsilon}
{\sum_j \left(\log(1 + \max(\tilde{t}_j, 0)) + \epsilon\right)}
\]

This is the noisy input heatmap.

## Target Heatmap

Training targets come from scaled human ET data. Raw unscaled TRT is not used.

\[
g_i =
\frac{\log(1 + \max(t_i^{scaled}, 0)) + \epsilon}
{\sum_j \left(\log(1 + \max(t_j^{scaled}, 0)) + \epsilon\right)}
\]

## Trainable ET-HAE

ET-HAE is a deterministic denoising autoencoder over word sequences.

\[
\hat{\mathbf{g}} = h_\phi(x_{1:n}, \mathbf{g}^0)
\]

Implemented v1 structure:

\[
\mathbf{e}_i = E[x_i]
\]

\[
\mathbf{u}_i = \mathrm{MLP}(g_i^0)
\]

\[
\mathbf{z}_i^{(0)} = \mathrm{LayerNorm}(\mathbf{e}_i + \mathbf{u}_i)
\]

\[
\mathbf{z}^{(\ell+1)} =
\mathrm{ResidualConvBlock}_{\ell}(\mathbf{z}^{(\ell)})
\]

\[
\hat{g}_i =
\mathrm{softmax}_i(W_o\mathbf{z}_i^{(L)})
\]

The convolution stack is 1D over sequence positions. It is part of ET-HAE, not part of the RECCON baseline model.

## Loss

\[
\mathcal{L}_{\mathrm{HAE}}
=
D_{\mathrm{KL}}(\mathbf{g}\parallel\hat{\mathbf{g}})
+ \lambda \mathcal{L}_{\mathrm{rank}}
\]

The KL term trains distribution reconstruction. The rank term encourages the refined heatmap to preserve high-salience versus low-salience ordering.

## Current Verified Smoke Result

The smoke run uses `target_noise` to verify the ET-HAE machinery without requiring Hugging Face downloads.

Verified artifacts:

- `artifacts/et_hae_data/smoke.jsonl`
- `artifacts/et_hae_checkpoints/smoke/best_model.pt`
- `artifacts/et_hae_checkpoints/smoke/last_model.pt`
- `artifacts/et_hae_checkpoints/smoke/train_summary.json`
- `artifacts/predicted_heatmaps/smoke_sentence.json`

