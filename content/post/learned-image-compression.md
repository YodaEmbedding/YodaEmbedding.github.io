+++
date = "2024-01-01T00:00:00-08:00"
title = "Learned image compression: Introduction"
slug = "learned-image-compression"
categories = [ "Compression" ]
tags = [ "Compression", "Deep learning" ]
# headline = "Headline goes here"
# readingtime = true
# aliases = ["2024/01/some-previous-name/"]
# draft = true
katex = true
+++


Learned image compression (LIC) applies deep learning models to the task of image compression.
It is also sometimes known as deep image compression or neural image compression.
This article introduces learned image compression, and in the future, will be updated to provide a brief survey of the current landscape and state-of-the-art (SOTA).



## What is data compression?

The storage and transmission of data is a fundamental aspect of computing.
However, every bit stored or transmitted incurs a cost.
To mitigate this cost, we turn to data compression.
Data compression is the process of reducing the amount of bits required to represent, store, or transmit data.

The modern age of the Internet would look very different without data compression: web pages would take much longer to load, and images and video would be much lower in resolution.
The transmission of an uncompressed video stream would require thousands of times more bits than its compressed counterpart.
Video streaming services such as Netflix and YouTube would suffer from a much higher operating cost, and would be much less accessible to the average consumer.

Data compression algorithms, known as *codecs*, are often specialized for encoding a particular type of data.
Common types of data include text, images, video, audio, and point clouds.
Data from such sources often contains redundancy or patterns which can be identified and eliminated by a compression algorithm to represent the data more compactly.
For instance, pixels that are near each other in an image are often similar in color, which is a phenomenon known as *spatial redundancy*.




## Data compression: an example

<!--
- \\( P(X = \text{Sunny ☀️}) = \frac{1}{2} \\)
- \\( P(X = \text{Cloudy ⛅}) = \frac{1}{4} \\)
- \\( P(X = \text{Rainy 🌧️}) = \frac{1}{4} \\)
 -->

As an example of how data compression works, consider the random variable \\( X \\) representing the summer weather in the city of Vancouver, Canada.
Let us say that the possible weather conditions
\\( (\texttt{Sunny}, \texttt{Rainy}, \texttt{Cloudy}) \\) abbreviated as
\\( (\texttt{S}, \texttt{R}, \texttt{C}) \\) are predicted to occur with the probabilities
\\( (\frac{1}{2}, \frac{1}{4}, \frac{1}{4}) \\), respectively.
To compress a sequence of weather observations \\( X\_1, X\_2, \ldots, X\_n \\), we can use a codebook that maps each weather condition to a binary string:
\\[
\begin{align*}
  \texttt{S} &\rightarrow \texttt{0}, \\\
  \texttt{R} &\rightarrow \texttt{10}, \\\
  \texttt{C} &\rightarrow \texttt{11}.
\end{align*}
\\]
Then, a sequence of weather observations such as the 64-bit ASCII string "\\( \texttt{SRCSSRCS} \\)" can be represented more compactly as the *encoded* 12-bit binary string "\\( \texttt{010110010110} \\)".

<!--
Technically, the ASCII string is actually 01010011 01010010 01000011 01010011 01010011 01010010 01000011 01010011.

TODO(review): This could be written more clearly, or perhaps even with a figure...
 -->

Notably, for any given input \\( x \in \\{\texttt{S}, \texttt{R}, \texttt{C}\\} \\), the length in bits of its encoded representation is equal to \\( -\log\_2 P(X = x) \\).
That is,
- \\( -\log\_2 \frac{1}{2} = 1 \text{ bit} \\) for \\( \texttt{S} \\),
- \\( -\log\_2 \frac{1}{4} = 2 \text{ bits} \\) for \\( \texttt{R} \\), and
- \\( -\log\_2 \frac{1}{4} = 2 \text{ bits} \\) for \\( \texttt{C} \\).

This codebook is optimal if \\( X\_1, X\_2, \ldots, X\_n \\) are independently and identically distributed (i.i.d.).

However, many raw sources studied in compression are not i.i.d.
In fact, there are often patterns or correlations between consecutive elements.
For instance, in the weather example, it is more likely that the weather will be \\( \texttt{R} \\) on a given day if it was \\( \texttt{C} \\) on the previous day.
After a \\( \texttt{C} \\) is encountered, the probability distribution used for encoding should be reevaluated to a more realistic prediction, e.g. \\( (\frac{1}{4}, \frac{1}{2}, \frac{1}{4}) \\).
The codebook must then also be updated to dynamically match this new encoding distribution:
\\[
\begin{align*}
  \texttt{S} &\rightarrow \texttt{10}, \\\
  \texttt{R} &\rightarrow \texttt{0}, \\\
  \texttt{C} &\rightarrow \texttt{11}.
\end{align*}
\\]
This is an optimal codebook for the new encoding distribution.
It should be used instead of the general codebook whenever the previous weather observation is \\( \texttt{C} \\).

We can compress even further by determining a more accurate encoding distribution that predicts the next observation more accurately.
More sophisticated probability modeling might also take into account the weather from the past several days, or from the same day in previous years.
A good model might blend in other related sources of information such as past and current humidity, temperature, and wind speed.
It might also analyze such information on multiple scales: locally within the city, within the province, or within the continent.
Such probability modeling is the work of a meteorologist... and also a data compression researcher!
In data compression, this process of determining the encoding distribution on-the-fly based on previous information is known as *context modeling*; and more generally, for any way of determining the encoding distribution, as [*entropy modeling*](#entropy-modeling).

<!--
If the ... does not match ..., then the "compressed" result will be inefficient.
Good compression requires accurate modeling of the data source.
More generally, any encoding probability distribution can be perfectly coded using an arithmetic coder.

Reader should now appreciate probability distribution-based modeling.
 -->



## Learning-based compression: the current landscape

Deep learning based approaches have recently been applied to data compression.
Learning-based approaches have demonstrated compression performance that is competitive with traditional standard codecs.
For instance, {{< cref "fig:intro/rd-curves" >}} compares the Rate-Distortion (RD) performance curves for popular and state-of-the-art (SOTA) codecs in image compression, evaluated on generic non-specialized datasets.

{{< figure
  src="assets/img/learned-compression/rd-curves-image-kodak-psnr-rgb.png"
  label="fig:intro/rd-curves"
>}}
Rate-Distortion (RD) curves for various image compression codecs evaluated on the Kodak test dataset.
{{< /figure >}}

<!--
| ![RD curves of various image compression codecs on the Kodak dataset](/~mulhaq/assets/img/learned-compression/rd-curves-image-kodak-psnr-rgb.png) |
|:--:|
| Rate-Distortion (RD) curves for various image compression codecs on the Kodak dataset. |

Non-learned codecs are marked by dashed lines.
Learning-based codecs offer further advantages by being easier to tune for targeted data sources, e.g. faces or screencasts.  % NEEDSCITATION

These techniques can be applied to many types of data sources, including images, video, audio, and point clouds.
-->

Learned compression has been applied to various types of data including images, video, and point clouds.
For learned image compression, most prominent are approaches based on Ballé *et al.* [^ref-balle2018variational]'s compressive variational autoencoder (VAE), including [^ref-minnen2018joint] [^ref-cheng2020learned] [^ref-he2022elic].
Other approaches based on RNNs and GANs have also been applied, including [^ref-toderici2017rnn] [^ref-mentzer2020highfidelity].
Works in learned point cloud compression include [^ref-yan2019deep] [^ref-he2022density] [^ref-pang2022graspnet] [^ref-fu2022octattention] [^ref-you2022ipdae], and works in learned video compression include [^ref-rippel2019learned] [^ref-agustsson2020scalespaceflow] [^ref-hu2021fvc] [^ref-ho2022canf].

Currently, one factor inhibiting industry adoption of learning-based codecs is that they are much more computationally expensive than traditional codecs like JPEG and WebP.
In fact, learned compression codecs exceed reasonable computational budgets by a factor of 100--10000x.
To remedy this, there is work being done towards designing low-complexity codecs for image compression, including [^ref-galpin2023entropy] [^ref-ladune2023coolchic] [^ref-leguay2023lowcomplexity] [^ref-kamisli2023lowcomplexity].

Learned compression has also shown benefits when applied to learned machine or computer vision tasks.
In Coding for Machines (CfM) --- also referred to as Video Coding for Machines (VCM) [^ref-duan2020vcm] --- compression is used for machine tasks such as classification, object detection, and semantic segmentation.
In this paradigm, the encoder-side device compresses the input into a compact task-specialized bitstream that is transmitted to the decoder-side device or server for further inference.
This idea of partially processing the input allows for significantly lower bitrates in comparison to transmitting the entire unspecialized input for inference.
Extending this technique, scalable multi-task codecs such as [^ref-choi2021latentspace] [^ref-choi2022sichm] allocate a small base bitstream for machine tasks, and a larger enhancement bitstream for a higher-quality input reconstruction intended for human viewing.




## Compression architecture overview

A simple compression architecture used by both traditional and learned compression methods alike is shown in {{< cref "fig:intro/arch-comparison/factorized" >}}.


{{< figure label="fig:intro/arch-comparison"
  mode="container"
  caption="High-level comparison of codec architectures."
>}}
{{< figure mode="subfigure"
  src="assets/img/learned-compression/arch-overview-factorized.png"
  label="fig:intro/arch-comparison/factorized"
>}}
Simple compression architecture.
{{< /figure >}}
{{< figure mode="subfigure"
  src="assets/img/learned-compression/arch-overview-hyperprior.png"
  label="fig:intro/arch-comparison/hyperprior"
>}}
Hyperprior compression architecture.
{{< /figure >}}
{{< /figure >}}


This architecture consists of several components.
The table below lists some common choices for the components in this architecture.

<!-- {{< cref "tbl:intro/codec_components" >}} -->


{{< table
  label="tbl:intro/codec_components"
  caption="Components of a compression architecture for various image compression codecs."
>}}
| Model | Quantizer | Entropy coding | Analysis transform (\\( g\_a \\)) | Synthesis transform (\\( g\_s \\)) |
|:--:|:--:|:--:|:--:|:--:|
| JPEG | non-uniform | zigzag + RLE, Huffman | \\( 8 \times 8 \\) block DCT | \\( 8 \times 8 \\) block DCT<sup>-1</sup> |
| JPEG 2000 | uniform dead-zone or TCQ | arithmetic | multilevel DWT | multilevel DWT<sup>-1</sup> |
| bmshj2018-factorized | uniform | arithmetic | (Conv, GDN) \\( \times 4 \\) | (ConvT, IGDN) \\( \times 4 \\) |
{{< /table >}}

<!--
  \caption[Compression architecture overview for image codecs]{%
    Components of a compression architecture for various image compression codecs.%
  }
  \label{tbl:intro/codec_components}
    \toprule
    \thead{Model}
      & \thead{Quantizer}
      & \thead{Entropy coding}
      & \thead{Analysis transform \\ (\\( g\_a \\))}
      & \thead{Synthesis transform \\ (\\( g\_s \\))}
      \\
    \midrule
    JPEG
      & non-uniform
      & \makecell{zigzag + RLE, \\ Huffman}
      & \\( 8 \times 8 \\) block DCT
      & \\( 8 \times 8 \\) block DCT\\( ^{-1} \\)
      \\
    JPEG 2000
      & \makecell{uniform dead-zone \\ or trellis coded (TCQ)}
      & arithmetic
      & multilevel DWT
      & multilevel DWT\\( ^{-1} \\)
      \\
    bmshj2018-factorized
      & uniform
      & arithmetic
      & (Conv, GDN) \\( \times 4 \\)
      & (ConvT, IGDN) \\( \times 4 \\)
      \\
    \bottomrule
 -->

In this architecture, the input \\( \boldsymbol{x} \\) goes through a transform \\( g\_a \\) to generate an intermediate representation \\( y \\), which is quantized to \\( \boldsymbol{\hat{y}} \\).
Then, \\( \boldsymbol{\hat{y}} \\) is losslessly entropy coded to generate a transmittable bitstream from which \\( \boldsymbol{\hat{y}} \\) can be perfectly reconstructed.
(at the decoder side.)
Finally, \\( \boldsymbol{\hat{y}} \\) is fed into a synthesis transform \\( g\_s \\) which reconstructs an approximation of \\( \boldsymbol{x} \\), which is labeled \\( \boldsymbol{\hat{x}} \\).

Each of the components of the standard compression architecture are described in further detail below:

-   **Analysis transform** (\\( g\_a \\)):
    The input first is transformed by the \\( g\_a \\) transform into a transformed representation \\( \boldsymbol{y} \\).
    This transform often outputs a signal that contains less redundancy than within the input signal, and has its energy compacted into a smaller dimension.
    For instance, the JPEG codec transforms \\( 8 \times 8 \\) blocks from the input image using a discrete cosine transform (DCT).
    This concentrates most of the signal energy into the low-frequency components that are often the dominating frequency component within natural images.
    In the case of learned compression, the analysis transform is often a nonlinear transform comprised of multiple deep layers and many parameters.
    For instance, the bmshj2018-factorized model's \\( g\_a \\) transform contains 4 downsampling convolutional layers interleaved with GDN [^ref-balle2016gdn] nonlinear activations, totaling 1.5M to 3.5M parameters.

-   **Quantization**:
    The analysis transform outputs coefficients contained in a rather large (potentially even continuous) support.
    However, much of this precision is not necessary for a reasonably accurate reconstruction.
    Thus, we drop most of this unneeded information by binning the transformed coefficients into a much smaller discretized support.
    There are many choices for the reconstructed quantization bin values and boundaries.
    Popular quantizers include the uniform, dead-zone, Lloyd-Max, and Trellis-coded quantizers.
    Ballé *et al.* [^ref-balle2018variational] use a uniform quantizer during inference, which is replaced with additive unit-width uniform noise during training.
    More recently, the STE quantizer has also been used during training.

-   **Entropy coding**:
    The resulting \\( \boldsymbol{\hat{y}} \\) is losslessly compressed using an entropy coding method.
    The entropy coder is targeted to match a specific encoding distribution.
    Whenever the encoding distribution correctly predicts an encoded symbol with high probability, the relative bit cost for encoding that symbol is reduced.
    Thus, some entropy models are context-adaptive, and change the encoding distribution on-the-fly in order to accurately probabilistically predict the next encoded symbol value.
    Huffman coding is used in JPEG, though it has trouble replicating a given target encoding probability distribution and also at adapting to dynamically changing encoding distributions.
    Thus, more recent codecs prefer to use arithmetic coders, which can much better approximate rapidly changing target encoding distributions.
    The CompressAI [^ref-begaint2020compressai] implementation uses rANS [^ref-duda2013asymmetric] [^ref-giesen2014ryg_rans], a popular recent innovation that is quite fast under certain conditions.

-   **Synthesis transform** (\\( g\_s \\)):
    Finally, the reconstructed quantized \\( \boldsymbol{\hat{y}} \\) is fed into a synthesis transform \\( g\_s \\), which produces \\( \boldsymbol{\hat{x}} \\).
    In JPEG, this is simply the inverse DCT.
    Similar to the analysis transform, in learned compression, the synthesis transform consists of several layers and many parameters.
    For instance, the bmshj2018-factorized model's \\( g\_s \\) transform contains 4 upsampling transposed convolutional layers interleaved with IGDN nonlinear activations, totaling 1.5M to 3.5M parameters.

The length of the bitstream is known as the rate cost \\( R\_{\boldsymbol{\hat{y}}} \\), which we seek to minimize.
We also seek to minimize the distortion \\( D(\boldsymbol{x}, \boldsymbol{\hat{x}}) \\), which is typically the mean squared error (MSE) between \\( \boldsymbol{x} \\) and \\( \boldsymbol{\hat{x}} \\).
To balance these two competing goals, it is common to introduce a Lagrangian trade-off hyperparameter \\( \lambda \\), so that the quantity sought to be minimized is \\( L = R\_{\boldsymbol{\hat{y}}} + \lambda \\, D(\boldsymbol{x}, \boldsymbol{\hat{x}}) \\).


<!--
bmshj2018-factorized parameters and MACs/pixel:

>>> N, M = 128, 192
>>> sum([3 * N, N * N, N * N, N * M]) * 5**2 + sum([N**2, N**2, N**2])
1492352
>>> sum([3 * N / 2**2, N * N / 4**2, N * N / 8**2, N * M / 16**2]) * 5**2
36800.0

>>> N, M = 192, 320
>>> sum([3 * N, N * N, N * N, N * M]) * 5**2 + sum([N**2, N**2, N**2])
3504192
>>> sum([3 * N / 2**2, N * N / 4**2, N * N / 8**2, N * M / 16**2]) * 5**2
81600.0
 -->



## Entropy modeling

A given element \\( \hat{y}\_i \in \mathbb{Z} \\) of the latent tensor \\( \boldsymbol{\hat{y}} \\) is compressed using its encoding distribution \\( p\_{{\hat{y}}\_i} : \mathbb{Z} \to [0, 1] \\), as visualized in {{< cref "fig:intro/encoding-distribution" >}}.
The rate cost for encoding \\( \hat{y}\_i \\) is the negative log-likelihood, \\( R\_{{\hat{y}}\_i} = -\log\_2 p\_{{\hat{y}}\_i}({\hat{y}}\_i) \\), measured in bits.
Afterward, the exact same encoding distribution is used by the decoder to reconstruct the encoded symbol.

{{< figure
  src="assets/img/learned-compression/encoding-distribution.png"
  label="fig:intro/encoding-distribution"
>}}
Visualization of an encoding distribution used for compressing a single element \\( \hat{y}\_i \\).
{{< /figure >}}


The encoding distributions are determined using an *entropy model*;
{{< cref "fig:intro/encoding-distributions" >}} visualizes the encoding distributions generated by well-known entropy models.
These are used to compress a latent tensor \\( \boldsymbol{\hat{y}} \\) with dimensions \\( M\_y \times H\_y \times W\_y \\).
The exact total rate cost for encoding \\( \boldsymbol{\hat{y}} \\) using \\( p\_{\boldsymbol{\hat{y}}} \\) is simply the sum of the negative log-likelihoods of each element, \\( R\_{\boldsymbol{\hat{y}}} = \sum\_i -\log\_2 p\_{{\hat{y}}\_i}({\hat{y}}\_i) \\).

{{< figure
  label="fig:intro/encoding-distributions"
  mode="container"
  caption=`
Visualization of encoding distributions used for compressing a latent tensor \\( \boldsymbol{\hat{y}} \\) with dimensions \\( M\_y \times H\_y \times W\_y \\).
In (a), the encoding distributions within a given channel are all the same since the elements within a channel are assumed to be i.i.d. w.r.t. each other.
Furthermore, in the case of the fully factorized entropy bottleneck used by Ballé *et al.*, each encoding distribution is a static non-parametric distribution.
In (b), the encoding distributions for each element are uniquely determined, often by [conditioning](https://en.wikipedia.org/wiki/Conditional_probability) on side information or on previously decoded elements (known as autoregression).
Furthermore, in the case of the scale hyperprior used by Ballé *et al.*, the encoding distributions are Gaussian distributions parameterized by a mean and variance.
`
>}}

{{< figure
  src="assets/img/learned-compression/encoding-distributions-factorized.png"
  label="fig:intro/encoding-distributions/factorized"
  mode="subfigure"
>}}
fully factorized
{{< /figure >}}

{{< figure
  src="assets/img/learned-compression/encoding-distributions-conditional.png"
  label="fig:intro/encoding-distributions/conditional"
  mode="subfigure"
>}}
conditional
{{< /figure >}}

{{< /figure >}}



Some popular choices for entropy models are discussed below.



### "Fully factorized" entropy bottleneck

One entropy model is the "fully factorized" *entropy bottleneck*, as introduced by Ballé *et al.* [^ref-balle2018variational].
Let \\( p\_{{\hat{y}}\_{c,i}} : \mathbb{Z} \to [0, 1] \\) denote the probability mass distribution used to encode the \\( i \\)-th element \\( \hat{y}\_{c,i} \\) from the \\( c \\)-th channel of \\( \boldsymbol{\hat{y}} \\).
The same encoding distribution \\( p\_{{\hat{y}}\_c} \\) is used for all elements within the \\( c \\)-th channel, i.e., \\( p\_{{\hat{y}}\_{c,i}} = p\_{{\hat{y}}\_c}, \forall i \\).
This entropy model works best when all such elements \\( \hat{y}\_{c,1}, \hat{y}\_{c,2}, \ldots, \hat{y}\_{c,N} \\) are independently and identically distributed (i.i.d.).

Ballé *et al.* [^ref-balle2018variational] model the encoding distribution as a static non-parametric distribution that is computed as the binned area under a probability density function \\( f\_{c} : \mathbb{R} \to \mathbb{R} \\), with a corresponding cumulative distribution function \\( F\_{c} : \mathbb{R} \to [0, 1] \\).
Then,
\\[
\begin{equation}
  % \label{eqn:p_y_c_integral}
  p\_{\boldsymbol{\hat{y}}\_c}(\hat{y}\_{c,i})
  = \int\_{-\frac{1}{2}}^{\frac{1}{2}} f(\hat{y}\_{c,i} + \tau) \\, d\tau
  = F\_{c}(\hat{y}\_{c,i} + 1/2) - F\_{c}(\hat{y}\_{c,i} - 1/2).
\end{equation}
\\]
\\( F\_{c} \\) is modelled using a small fully-connected network composed of five linear layers with channels of sizes \\( [1, 3, 3, 3, 3, 1] \\), whose parameters are tuned during training.
Note that \\( F\_{c} \\) is not conditioned on any other information, and is thus static.



### Mean-scale hyperprior

<!-- Another popular entropy model is the mean-scale variant of the "hyperprior" model introduced by Ballé *et al.* [^ref-balle2018variational]. -->
Let \\( f\_i(y) = \mathcal{N}(y; {\mu\_i}, {\sigma\_i}^2) \\) be a Gaussian distribution with mean \\( {\mu\_i} \\) and variance \\( {\sigma\_i}^2 \\).
Then, like in (1), <!-- {{< cref "eqn:p_y_c_integral" >}}, --> the encoding distribution \\( p\_{\boldsymbol{\hat{y}}\_i} \\) is defined as the binned area under \\( f\_i \\):
\\[
\begin{equation}
  % \label{eqn:p_y_i_integral}
  p\_{\boldsymbol{\hat{y}}\_i}(\hat{y}\_i)
  = \int\_{-\frac{1}{2}}^{\frac{1}{2}} f\_i(\hat{y}\_i + \tau) \\, d\tau.
  % = F\_i(\hat{y}\_i + 1/2) - F\_i(\hat{y}\_i - 1/2).
\end{equation}
\\]
In the mean-scale variant of the "hyperprior" model introduced by Ballé *et al.* [^ref-balle2018variational],
the parameters \\( {\mu\_i} \\) and \\( {\sigma\_i}^2 \\) are computed by
\\( [{\mu\_i}, {\sigma\_i}] = (h\_s(\boldsymbol{\hat{z}}))\_i \\).
Here, the latent representation \\( \boldsymbol{\hat{z}} = \operatorname{Quantize}[h\_a(\boldsymbol{y})] \\) is computed by the analysis transform \\( h\_a \\), and then encoded using an entropy bottleneck and transmitted as *side information*;
and \\( h\_s \\) is a synthesis transform.
This architecture is visualized in {{< cref "fig:intro/arch-comparison/hyperprior" >}}.
Cheng *et al.* [^ref-cheng2020learned] define \\( f\_i \\) as a mixture of \\( K \\) Gaussians --- known as a Gaussian mixture model (GMM) --- with parameters \\( {\mu}\_{i}^{(k)} \\) and \\( {\sigma}\_{i}^{(k)} \\) for each Gaussian, alongside an affine combination of weights \\( {w}\_{i}^{(1)}, \ldots, {w}\_{i}^{(K)} \\) that satisfy the constraint \\( \sum\_k {w}\_{i}^{(k)} = 1 \\).
A GMM encoding distribution is thus defined as
\\( f\_i(y) = \sum\_{k=1}^{K} {w}\_{i}^{(k)} \\, \mathcal{N}(y; {\mu}\_{i}^{(k)}, [{\sigma}\_{i}^{(k)}]^2) \\).

Note that *scale* refers to the width \\( \sigma\_i \\) of the Gaussian distribution, not to the fact that the latent \\( \boldsymbol{\hat{y}} \\) is spatially downsampled and then upsampled.
Ballé *et al.* [^ref-balle2018variational] originally introduced a "scale hyperprior" model, which assumes a fixed mean \\( \mu\_i = 0 \\), though it was later shown that allowing the mean \\( \mu\_i \\) to also vary can improve performance.
In the CompressAI implementation, the scale hyperprior model constructs the encoding distributions \\( p\_{\boldsymbol{\hat{y}}\_i} \\) from the parameters \\( \mu\_i \\) and \\( \sigma\_i \\) using a [*Gaussian conditional*](https://interdigitalinc.github.io/CompressAI/_modules/compressai/entropy_models/entropy_models.html#GaussianConditional) component.
The Gaussian conditional is not an entropy model on its own, by our definition.
(This article defines an entropy model as a function that computes the encoding distributions \\( p\_{\boldsymbol{\hat{y}}} \\) for a given latent tensor \\( \boldsymbol{\hat{y}} \\).)







## Conclusion

This article reviewed basic concepts in learned image compression, which is an ever-improving field that shows promise.

Later in this series, we will look at various popular papers, and other interesting works.





<!--

KaTeX notes:

_  -> \_
\, -> \\,





Follow Wikipedia aliases style...

Learned image compression (LIC),
also known as
deep image compression,
neural image compression (NIC),
...

CLIC (Challenge on Learned Image Compression)

Include references to these names?




TODO:

BibTeX / BibLaTeX.
mbt/Cheng2020.
Then publish. Update later with more recent works.

Introduction and survey of learned image compression.

Also introduce more advanced learned entropy models (autoregressive):
mbt, Cheng, Checkerboard, ELIC, ...
Other popular or interesting works: HiFiC, MLIC++/etc, Sandwiched, Cool CHIC, low entropy, ...
At the very least, mention a list of interesting papers. But figures from papers and brief overview of methods and architecture are nice too.
Other model types: GAN, RNN, ...

For our "paper club" presentation, too?

Floating table of contents.

 -->



<div id="refs"></div>

[^ref-balle2018variational]: J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in *Proc. ICLR*, 2018. Available: <https://arxiv.org/abs/1802.01436>

[^ref-minnen2018joint]: D. Minnen, J. Ballé, and G. D. Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” *Advances in neural information processing systems*, vol. 31, 2018, Available: <https://arxiv.org/abs/1809.02736>

[^ref-cheng2020learned]: Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in *Proc. IEEE/CVF CVPR*, 2020, pp. 7939–7948. Available: <https://arxiv.org/abs/2001.01568>

[^ref-he2022elic]: D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding,” in *Proc. IEEE/CVF CVPR*, 2022, pp. 5718–5727. Available: <https://arxiv.org/abs/2203.10886>

[^ref-toderici2017rnn]: G. Toderici *et al.*, “Full resolution image compression with recurrent neural networks,” in *Proc. IEEE/CVF CVPR*, IEEE, 2017, pp. 5306–5314. doi: [10.1109/cvpr.2017.577](https://doi.org/10.1109/cvpr.2017.577).

[^ref-mentzer2020highfidelity]: F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson, “High-fidelity generative image compression,” *Advances in Neural Information Processing Systems*, vol. 33, pp. 11913–11924, 2020, Available: <https://proceedings.neurips.cc/paper_files/paper/2020/file/8a50bae297807da9e97722a0b3fd8f27-Paper.pdf>

[^ref-yan2019deep]: W. Yan, Y. shao, S. Liu, T. H. Li, Z. Li, and G. Li, “Deep AutoEncoder-based lossy geometry compression for point clouds.” 2019. Available: <https://arxiv.org/abs/1905.03691>

[^ref-he2022density]: Y. He, X. Ren, D. Tang, Y. Zhang, X. Xue, and Y. Fu, “Density-preserving deep point cloud compression,” in *Proc. IEEE/CVF CVPR*, 2022, pp. 2323–2332. doi: [10.1109/CVPR52688.2022.00237](https://doi.org/10.1109/CVPR52688.2022.00237).

[^ref-pang2022graspnet]: J. Pang, M. A. Lodhi, and D. Tian, “GRASP-Net: Geometric residual analysis and synthesis for point cloud compression,” in *Proc. 1st int. Workshop on advances in point cloud compression, processing and analysis*, 2022. Available: <https://arxiv.org/abs/2209.04401>

[^ref-fu2022octattention]: C. Fu, G. Li, R. Song, W. Gao, and S. Liu, “OctAttention: Octree-based large-scale contexts model for point cloud compression,” in *Proc. AAAI*, Jun. 2022, pp. 625–633. doi: [10.1609/aaai.v36i1.19942](https://doi.org/10.1609/aaai.v36i1.19942).

[^ref-you2022ipdae]: K.-S. You, P. Gao, and Q. T. Li, “IPDAE: Improved patch-based deep autoencoder for lossy point cloud geometry compression,” in *Proc. 1st int. Workshop on advances in point cloud compression, processing and analysis*, 2022. Available: <https://arxiv.org/abs/2208.02519>

[^ref-rippel2019learned]: O. Rippel, S. Nair, C. Lew, S. Branson, A. G. Anderson, and L. Bourdev, “Learned video compression,” in *Proc. IEEE/CVF ICCV*, 2019, pp. 3454–3463. Available: <https://arxiv.org/abs/1811.06981>

[^ref-agustsson2020scalespaceflow]: E. Agustsson, D. Minnen, N. Johnston, J. Ballé, S. J. Hwang, and G. Toderici, “Scale-Space Flow for end-to-end optimized video compression,” in *Proc. IEEE/CVF CVPR*, 2020, pp. 8500–8509. doi: [10.1109/CVPR42600.2020.00853](https://doi.org/10.1109/CVPR42600.2020.00853).

[^ref-hu2021fvc]: Z. Hu, G. Lu, and D. Xu, “FVC: A new framework towards deep video compression in feature space,” in *Proc. IEEE/CVF CVPR*, 2021, pp. 1502–1511. doi: [10.1109/cvpr46437.2021.00155](https://doi.org/10.1109/cvpr46437.2021.00155).

[^ref-ho2022canf]: Y.-H. Ho, C.-P. Chang, P.-Y. Chen, A. Gnutti, and W.-H. Peng, “CANF-VC: Conditional augmented normalizing flows for video compression,” in *Proc. ECCV*, Springer, 2022, pp. 207–223. Available: <https://arxiv.org/abs/2207.05315>

[^ref-galpin2023entropy]: F. Galpin, M. Balcilar, F. Lefebvre, F. Racapé, and P. Hellier, “Entropy coding improvement for low-complexity compressive auto-encoders,” in *Proc. IEEE DCC*, 2023, pp. 338–338. doi: [10.1109/DCC55655.2023.00080](https://doi.org/10.1109/DCC55655.2023.00080).

[^ref-ladune2023coolchic]: T. Ladune, P. Philippe, F. Henry, G. Clare, and T. Leguay, “COOL-CHIC: Coordinate-based low complexity hierarchical image codec,” in *Proc. IEEE/CVF ICCV*, 2023, pp. 13515–13522. Available: <https://arxiv.org/abs/2212.05458>

[^ref-leguay2023lowcomplexity]: T. Leguay, T. Ladune, P. Philippe, G. Clare, and F. Henry, “Low-complexity overfitted neural image codec.” 2023. Available: <https://arxiv.org/abs/2307.12706>

[^ref-kamisli2023lowcomplexity]: F. Kamisli, “Learned lossless image compression through interpolation with low complexity,” *IEEE Trans. Circuits Syst. Video Technol.*, pp. 1–1, 2023, doi: [10.1109/TCSVT.2023.3273578](https://doi.org/10.1109/TCSVT.2023.3273578).

[^ref-duan2020vcm]: L.-Y. Duan, J. Liu, W. Yang, T. Huang, and W. Gao, “Video Coding for Machines: A paradigm of collaborative compression and intelligent analytics,” *IEEE Trans. Image Process.*, vol. 29, pp. 8680–8695, 2020, Available: <https://arxiv.org/abs/2001.03569>

[^ref-choi2021latentspace]: H. Choi and I. V. Bajić, “Latent-space scalability for multi-task collaborative intelligence,” in *Proc. IEEE ICIP*, 2021, pp. 3562–3566. doi: [10.1109/ICIP42928.2021.9506712](https://doi.org/10.1109/ICIP42928.2021.9506712).

[^ref-choi2022sichm]: H. Choi and I. V. Bajić, “Scalable image coding for humans and machines,” *IEEE Trans. Image Process.*, vol. 31, pp. 2739–2754, Mar. 2022, doi: [10.1109/tip.2022.3160602](https://doi.org/10.1109/tip.2022.3160602).

[^ref-balle2016gdn]: J. Ballé, V. Laparra, and E. P. Simoncelli, “Density modeling of images using a generalized normalization transformation,” in *Proc. ICLR*, 2016. Available: <https://arxiv.org/abs/1511.06281>

[^ref-begaint2020compressai]: J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: A PyTorch library and evaluation platform for end-to-end compression research.” 2020. Available: <https://arxiv.org/abs/2011.03029>

[^ref-duda2013asymmetric]: J. Duda, “Asymmetric numeral systems: Entropy coding combining speed of huffman coding with compression rate of arithmetic coding.” 2013. Available: <https://arxiv.org/abs/1311.2540>

[^ref-giesen2014ryg_rans]: F. Giesen, “Ryg_rans.” GitHub, 2014. Available: <https://github.com/rygorous/ryg_rans>

