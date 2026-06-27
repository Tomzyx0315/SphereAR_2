# SphereAR with Conditional Flow Heads

This fork adds one-step conditional normalizing flow heads for each continuous
latent token. The autoregressive backbone and hyperspherical VAE stay unchanged;
the token prediction head can be switched from diffusion/flow matching to a
conditional neural spline flow, or to a sphere-native projected flow.

## Method

For each token, the AR transformer produces a condition vector `h_i` and the
head models `p(z_i | z_<i, y)`. The Euclidean `FlowHead` uses:

- conditional diagonal Gaussian base distribution;
- rational-quadratic spline coupling layers;
- learned invertible linear mixing between coupling layers;
- exact negative log-likelihood training;
- single-pass sampling from the conditional base distribution.

The generated token is normalized by the existing VAE projection path before it
is fed back into the AR cache and before VAE decoding. This preserves the fixed
latent radius used by the rest of the model.

This design follows the normalizing-flow line of work behind
[RealNVP](https://arxiv.org/abs/1605.08803),
[Glow](https://arxiv.org/abs/1807.03039), and
[Neural Spline Flows](https://arxiv.org/abs/1906.04032). The main reason to try
it here is that each head only models a 16-dimensional token, where expressive
spline flows are cheap and avoid the multi-step diffusion sampler.

Recent adjacent work points in a similar direction:
[TarFlow](https://arxiv.org/abs/2412.06329) and
[STARFlow](https://arxiv.org/abs/2506.06276) scale transformer autoregressive
flows for images; [Jet](https://arxiv.org/abs/2412.15129) revisits
transformer-based flows; [Show-o2](https://arxiv.org/abs/2506.15564) and
[TMD](https://arxiv.org/abs/2601.09881) attach flow heads to larger
multimodal or diffusion backbones. I did not find a public paper that matches
this repo's exact setup: one conditional spline-flow head per 16-D SphereAR
token.

The flow paths do not use classifier-free guidance. Use sampling temperature to
trade diversity for sharpness, and optionally anneal it across token positions:

```text
temperature < 1.0  lower diversity, often cleaner samples
temperature = 1.0  base sampling
temperature > 1.0  higher diversity, often noisier samples
temperature_schedule = constant | linear
```

This repo keeps the original diffusion CFG path as the baseline; the linear CFG
schedule is still the default there.

## Environment

- PyTorch with CUDA
- FlashAttention
- TensorFlow for metric evaluation

The original training and sampling scripts still require DDP/NCCL and GPUs.

## Data

ImageNet can be used either as an extracted `ImageFolder` tree or directly from
the original tar file:

```shell
data_path=/path/to/ILSVRC2012_img_train.tar
```

When a tar file is used, the first run builds:

```text
ILSVRC2012_img_train.tar.index
```

The index stores image offsets and labels so the dataset does not need to be
decompressed.

## Train AR with Flow Head

Use `--head-type flow` or `--head-type flow-sphere` and pass an existing VAE
checkpoint:

```shell
data_path=/path/to/ILSVRC2012_img_train.tar
result_path=/path/to/flow_ar_run
vae_ckpt=/path/to/vae.pt

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
train.py --results-dir $result_path --data-path $data_path \
--image-size 256 --model SphereAR-B --epochs 400 \
--patch-size 16 --latent-dim 16 --trained-vae $vae_ckpt \
--head-type flow-sphere --flow-layers 8 --flow-bins 16 \
--lr 3e-4 --global-batch-size 512 --ema 0.9999
```

Useful flow options:

```text
--flow-layers              number of spline coupling blocks
--flow-bins                rational-quadratic spline bins
--flow-hidden-mul          flow hidden dim multiplier over model diff_dim
--flow-conditioner-depth   residual MLP depth inside each conditioner
--flow-tail-bound          spline interval is [-bound, bound]
--flow-noise-std           training dequantization noise on target latents
--flow-base-scale-bound    clamp range for conditional base log-scale
```

Sampling options:

```text
--temperature
--temperature-schedule
```

Checkpoints are written to `last.pt`, with `prev.pt` and periodic
`epoch_*.pt` snapshots.

## Sample

Flow sampling ignores `--cfg-scale` and `--sample-steps`. Set
`--temperature` instead:

```shell
ckpt=/path/to/flow_ar_run/last.pt
sample_dir=/path/to/samples

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
sample_ddp.py --model SphereAR-B --head-type flow-sphere --ckpt $ckpt \
--sample-dir $sample_dir --per-proc-batch-size 256 \
--temperature 0.9 --temperature-schedule constant --to-npz
```

The script writes PNGs first and, with `--to-npz`, converts the first 50,000
images into:

```text
$sample_dir/<run-name>.npz
```

Try temperatures such as `0.7`, `0.8`, `0.9`, and `1.0`.

## Evaluate

Download the ImageNet 256 reference batch used by OpenAI's guided-diffusion
evaluation protocol:

```text
VIRTUAL_imagenet256_labeled.npz
```

Then run:

```shell
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/generated.npz
```

The evaluator reports Inception Score, FID, sFID, Precision, and Recall, and
writes the metrics next to the generated `.npz`.

## Train Preview

Training does not run FID. It periodically saves a small preview batch to:

```text
$result_path/train_samples/step_xxxxxxx/
```

Useful preview options:

```text
--preview-every-steps
--preview-num-samples
--preview-sample-steps
--preview-temperature
--preview-temperature-schedule
```

Preview sampling follows the same head-specific rules as full sampling: flow
uses temperature, diffusion uses CFG.

## Diffusion Head Baseline

The original head is still available:

```shell
--head-type diff
```

Diffusion sampling continues to use `--cfg-scale` and `--sample-steps`, with
the original linear CFG schedule kept in code.

## Notes

- Flow checkpoints are not shape-compatible with diffusion-head checkpoints.
- `flow` and `flow-sphere` are not checkpoint-compatible with each other.
- EMA sampling is enabled by default when the checkpoint contains `ema`; use
  `--no-ema` to sample raw weights.
- `flow-sphere` models the token on the sphere `S^{15}_{sqrt(16)}` with an
  exact stereographic chart, so its likelihood lives on spherical support
  rather than in ambient Euclidean space.
