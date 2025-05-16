# Representation superposition is an underlying mechanism of neural scaling laws

This is the github repo for the paper ["Superposition Yields Robust Neural Scaling"](https://arxiv.org/abs/2505.10465).

Superposition means that models represent more features than dimensions they have, which is true for LLMs since there are too many things to represent in language. We find that superposition leads to a power-law loss with width, leading to the observed neural scaling law.

The code of the following figure is ['./exp/exp-17.py'](./exp/exp-17.py)

<p align="center" width="100%">
<img src="./figures/Fig-1-2.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## The toy model

We use Anthropic's toy model of superposition, adding weight decay or growth to control the degree of superposition.

<p align="center" width="100%">
<img src="./figures/Fig-2-2.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## Weight decay

Weight decay (or growth when the value is negative) can control superposition reflected by the fraction of represented features.

The code of the following figure is ['./exp/exp-10.py'](./exp/exp-10.py) and ['./exp/exp-10-3.py'](./exp/exp-10-3.py)

<p align="center" width="100%">
<img src="./figures/Fig-3-2.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## Rich phenomena

We need to answer when the loss is a power law with model dimension, and what control the power law exponent (we call it model exponent here).

We analyze the data from ['./exp/exp-17.py'](./exp/exp-17.py), ['./exp/exp-10.py'](./exp/exp-10.py) and ['./exp/exp-10-3.py'](./exp/exp-10-3.py) in the following figure.

<p align="center" width="100%">
<img src="./figures/Fig-4-3.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## Weak superposition regime

In the weak superposition regime (weight decay is large), the loss is well described by the expected number of activated but unlearned features, which is a power law once the feature distribution is.

The data are from ['./exp/exp-10.py'](./exp/exp-10.py) and ['./exp/exp-10-3.py'](./exp/exp-10-3.py).

<p align="center" width="100%">
<img src="./figures/Fig-5-1.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## Strong superposition

Scaling behavior in the strong superposition regime is robust due to generic geometric fact that when many more vectors are squeezed into a lower dimensional space, their overlaps scale inversely proportional to square root of dimension.

The data are from ['./exp/exp-10.py'](./exp/exp-10.py) and ['./exp/exp-10-3.py'](./exp/exp-10-3.py).

<p align="center" width="100%">
<img src="./figures/Fig-6-3.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## Activation density

The scaling exponents are robust to the number of expected activated features in one data point.

The data are from ['./exp/exp-15.py'](./exp/exp-15.py).

<p align="center" width="100%">
<img src="./figures/Fig-7-2.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

## LLMs

LLMs agree with the toy model results in the strong superposition regime from underlying overlaps between representations to loss scaling with model dimension.

Analysis of overlaps is in ['./LLMs/overlap-0.py'](./LLMs/overlap-0.py). We also analyzed norm distribution in ['./LLMs/norm-0.py'](./LLMs/norm-0.py) and token frequencies in ['./LLMs/token-freq-0.py'](./LLMs/token-freq-0.py) (see Appendix in the paper). Loss evaluation can be found in ['./LLMs/cali-1.py'](./LLMs/cali-1.py).

<p align="center" width="100%">
<img src="./figures/Fig-8-2.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>