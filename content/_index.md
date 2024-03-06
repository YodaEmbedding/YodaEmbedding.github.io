---
title: "Home"

name: "Mateen Ulhaq"

# Home page picture is controlled via .site.params.params.imgname
# imgname:
#   name: "assets/img/expanse.jpg"
#   alt: ""
#   type: image/jpeg

# More sources can be added (optional) using
# imgOther:
#   - name: $IMAGE_PATH
#     type: $IMAGE_TYPE
#   - name: $IMAGE_PATH
#     type: $IMAGE_TYPE
# ...

# personal_title: "MASc. Engineering"
personal_title: "Software Engineer"

# address:
#   -
#     name: Everywhere
#     street: Nowhere
#     postal_code: "000000"
#     locality: Earth

# Add an email with a mailto: hyperlink
# email: aaaa@example.com
# Add an email "image" for spam protection. With light and dark mode
# emailImg:
#   dark: /assets/img/dark_email.png
#   light: /assets/img/light_email.png

publications:

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "Learned Compression of Encoding Distributions"
    date: 2024
    journal: "IEEE ICIP"
    image: "assets/img/thumbnails/2024-icip-learned-compression-of-encoding-distributions.png"
    citation: |
      @inproceedings{ulhaq2024encodingdistributions,
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        booktitle         = {Proc. IEEE ICIP},
        date              = {2024},
        pubstate          = {submitted},
        title             = {Learned Compression of Encoding Distributions},
      }
    pdf: "assets/pdf/2024-icip-learned-compression-of-encoding-distributions.pdf"  # NOTE: Preprint.
    links:
      - name: Document
        url: "assets/pdf/2024-icip-learned-compression-of-encoding-distributions.pdf"  # NOTE: Preprint.
    description: |
      The entropy bottleneck introduced by Ballé et al. is a common component used in many learned compression models.
      It encodes a transformed latent representation using a static distribution whose parameters are learned during training.
      However, the actual distribution of the latent data may vary wildly across different inputs.
      The static distribution attempts to encompass all possible input distributions, thus fitting none of them particularly well.
      This unfortunate phenomenon, sometimes known as the amortization gap, results in suboptimal compression.
      To address this issue, we propose a method that dynamically adapts the encoding distribution to match the latent data distribution for a specific input.
      First, our model estimates a better encoding distribution for a given input.
      This distribution is then compressed and transmitted as an additional side-information bitstream.
      Finally, the decoder reconstructs the encoding distribution and uses it to decompress the corresponding latent data.
      Our method achieves a Bjøntegaard-Delta (BD)-rate gain of -7.10% on the Kodak test dataset when applied to the standard fully-factorized architecture.
      Furthermore, considering computational complexity, the transform used by our method is an order of magnitude cheaper in terms of Multiply-Accumulate (MAC) operations compared to related side-information methods such as the scale hyperprior.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "Scalable Human-Machine Point Cloud Compression"
    date: 2024
    journal: "Picture Coding Symposium (PCS)"
    image: "assets/img/thumbnails/2024-pcs-scalable-human-machine-point-cloud-compression.png"
    citation: |
      @inproceedings{ulhaq2024scalablepointcloud,
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        booktitle         = {Proc. PCS},
        date              = {2024},
        pubstate          = {accepted},
        title             = {Scalable Human-Machine Point Cloud Compression},
      }
    pdf: "assets/pdf/2024-pcs-scalable-human-machine-point-cloud-compression.pdf"  # NOTE: Preprint.
    links:
      - name: Document
        url: "assets/pdf/2024-pcs-scalable-human-machine-point-cloud-compression.pdf"  # NOTE: Preprint.
    description: |
      Due to the limited computational capabilities of edge devices, deep learning inference can be quite expensive.
      One remedy is to compress and transmit point cloud data over the network for server-side processing.
      Unfortunately, this approach can be sensitive to network factors, including available bitrate.
      Luckily, the bitrate requirements can be reduced without sacrificing inference accuracy by using a machine task-specialized codec.
      In this paper, we present a scalable codec for point-cloud data that is specialized for the machine task of classification, while also providing a mechanism for human viewing.
      In the proposed scalable codec, the "base" bitstream supports the machine task, and an "enhancement" bitstream may be used for better input reconstruction performance for human viewing.
      We base our architecture on PointNet++, and test its efficacy on the ModelNet40 dataset.
      We show significant improvements over prior non-specialized codecs.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
    title: "Master's thesis: Learned Compression for Images and Point Clouds"
    date: 2023
    journal: "Simon Fraser University"
    image: "assets/img/thumbnails/2023-learned-compression-for-images-and-point-clouds-masc-thesis.png"
    citation: |
      @thesis{ulhaq2023thesismasc,
        author            = {Ulhaq, Mateen},
        author+an:default = {1=me},
        institution       = {Simon Fraser University},
        date              = {2023},
        title             = {Learned Compression for Images and Point Clouds},
        type              = {MASc thesis},
      }
    pdf: "assets/pdf/2023-learned-compression-for-images-and-point-clouds-masc-thesis.pdf"
    links:
      - name: Document
        url: "assets/pdf/2023-learned-compression-for-images-and-point-clouds-masc-thesis.pdf"
        # url: "https://summit.sfu.ca/_flysystem/fedora/2024-02/etd22847.pdf"
        # url: "https://summit.sfu.ca/item/37955"
      - name: Slides
        url: "assets/pdf/2023-learned-compression-for-images-and-point-clouds-masc-thesis-defense-slides.pdf"
    description: |
      Over the last decade, deep learning has shown great success at performing computer vision tasks, including classification, super-resolution, and style transfer.
      Now, we apply it to data compression to help build the next generation of multimedia codecs.
      This thesis provides three primary contributions to this new field of learned compression.
      First, we present an efficient low-complexity entropy model that dynamically adapts the encoding distribution to a specific input by compressing and transmitting the encoding distribution itself as side information.
      Secondly, we propose a novel lightweight low-complexity point cloud codec that is highly specialized for classification, attaining significant reductions in bitrate compared to non-specialized codecs.
      Lastly, we explore how motion within the input domain between consecutive video frames is manifested in the corresponding convolutionally-derived latent space.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "Learned Point Cloud Compression for Classification"
    date: 2023
    journal: "IEEE MMSP"
    image: "assets/img/thumbnails/2023-mmsp-learned-point-cloud-compression-for-classification.png"
    citation: |
      @inproceedings{ulhaq2023pointcloud,
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        booktitle         = {Proc. IEEE MMSP},
        date              = {2023},
        eprint            = {2308.05959},
        eprintclass       = {eess.IV},
        eprinttype        = {arXiv},
        title             = {Learned Point Cloud Compression for Classification},
      }
    pdf: "https://arxiv.org/pdf/2308.05959.pdf"
    links:
      - name: Document
        url: "https://arxiv.org/pdf/2308.05959.pdf"
      - name: Slides
        url: "assets/pdf/2023-mmsp-learned-point-cloud-compression-for-classification-slides.pdf"
        # url: "https://raw.githubusercontent.com/multimedialabsfu/learned-point-cloud-compression-for-classification/assets/main/assets/slides.pdf"
      - name: Code
        url: "https://github.com/multimedialabsfu/learned-point-cloud-compression-for-classification"
    description: |
      Deep learning is increasingly being used to perform machine vision tasks such as classification, object detection, and segmentation on 3D point cloud data.
      However, deep learning inference is computationally expensive.
      The limited computational capabilities of end devices thus necessitate a codec for transmitting point cloud data over the network for server-side processing.
      Such a codec must be lightweight and capable of achieving high compression ratios without sacrificing accuracy.
      Motivated by this, we present a novel point cloud codec that is highly specialized for the machine task of classification.
      Our codec, based on PointNet, achieves a significantly better rate-accuracy trade-off in comparison to alternative methods.
      In particular, it achieves a 94% reduction in BD-bitrate over non-specialized codecs on the ModelNet40 dataset.
      For low-resource end devices, we also propose two lightweight configurations of our encoder that achieve similar BD-bitrate reductions of 93% and 92% with 3% and 5% drops in top-1 accuracy, while consuming only 0.470 and 0.048 encoder-side kMACs/point, respectively.
      Our codec demonstrates the potential of specialized codecs for machine analysis of point clouds, and provides a basis for extension to more complex tasks and datasets in the future.

  -
    authors:
      - name: Ezgi Özyılkan
      - name: Mateen Ulhaq
        me: true
      - name: Hyomin Choi
      - name: Fabien Racapé
    title: "Learned disentangled latent representations for scalable image coding for humans and machines"
    date: 2023
    journal: "IEEE DCC"
    image: "assets/img/thumbnails/2023-dcc-learned-disentangled-latent-representations-for-scalable-image-coding-for-humans-and-machines.png"
    citation: |
      @inproceedings{ozyilkan2023learned,
        author            = {Özyılkan, Ezgi and Ulhaq, Mateen and Choi, Hyomin and Racapé, Fabien},
        author+an:default = {1=first;2=me,first},
        booktitle         = {Proc. IEEE DCC},
        date              = {2023},
        eprint            = {2301.04183},
        eprintclass       = {eess.IV},
        eprinttype        = {arXiv},
        pages             = {42--51},
        title             = {Learned disentangled latent representations for scalable image coding for humans and machines},
      }
    pdf: "https://arxiv.org/pdf/2301.04183.pdf"
    links:
      - name: Document
        url: "https://arxiv.org/pdf/2301.04183.pdf"
      - name: Slides
        url: "assets/pdf/2023-dcc-learned-disentangled-latent-representations-for-scalable-image-coding-for-humans-and-machines-slides.pdf"
    description: |
      As an increasing amount of image and video content will be analyzed by machines, there is demand for a new codec paradigm that is capable of compressing visual input primarily for the purpose of computer vision inference, while secondarily supporting input reconstruction.
      In this work, we propose a learned compression architecture that can be used to build such a codec.
      We introduce a novel variational formulation that explicitly takes feature data relevant to the desired inference task as input at the encoder side.
      As such, our learned scalable image codec encodes and transmits two disentangled latent representations for object detection and input reconstruction.
      We note that compared to relevant benchmarks, our proposed scheme yields a more compact latent representation that is specialized for the inference task.
      Our experiments show that our proposed system achieves a bit rate savings of 40.6% on the primary object detection task compared to the current state-of-the-art, albeit with some degradation in performance for the secondary input reconstruction task.

  -
    authors:
      - name: Hyomin Choi
      - name: Fabien Racapé
      - name: Shahab Hamidi-Rad
      - name: Mateen Ulhaq
        me: true
      - name: Simon Feltman
    title: "Frequency-aware Learned Image Compression for Quality Scalability"
    date: 2022
    journal: "IEEE VCIP"
    image: "assets/img/thumbnails/2022-vcip-frequency-aware-learned-image-compression-for-quality-scalability.png"
    citation: |
      @inproceedings{choi2022frequencyaware,
        author            = {Choi, Hyomin and Racapé, Fabien and Hamidi-Rad, Shahab and Ulhaq, Mateen and Feltman, Simon},
        author+an:default = {4=me},
        booktitle         = {Proc. IEEE VCIP},
        date              = {2022},
        doi               = {10.1109/VCIP56404.2022.10008818},
        eprint            = {2301.01290},
        eprintclass       = {eess.IV},
        eprinttype        = {arXiv},
        pages             = {1--5},
        title             = {Frequency-aware Learned Image Compression for Quality Scalability},
      }
    pdf: "https://arxiv.org/pdf/2301.01290.pdf"
    links:
      - name: Document
        url: "https://arxiv.org/pdf/2301.01290.pdf"
    description: |
      Spatial frequency analysis and transforms serve a central role in most engineered image and video lossy codecs, but are rarely employed in neural network (NN)-based approaches.
      We propose a novel NN-based image coding framework that utilizes forward wavelet transforms to decompose the input signal by spatial frequency.
      Our encoder generates separate bitstreams for each latent representation of low and high frequencies.
      This enables our decoder to selectively decode bitstreams in a quality-scalable manner.
      Hence, the decoder can produce an enhanced image by using an enhancement bitstream in addition to the base bitstream.
      Furthermore, our method is able to enhance only a specific region of interest (ROI) by using a corresponding part of the enhancement latent representation.
      Our experiments demonstrate that the proposed method shows competitive rate-distortion performance compared to several non-scalable image codecs.
      We also showcase the effectiveness of our two-level quality scalability, as well as its practicality in ROI quality enhancement.

  -
    authors:
      - name: Saeed Ranjbar Alvar
      - name: Mateen Ulhaq
        me: true
      - name: Hyomin Choi
      - name: Ivan V. Bajić
    title: "Joint Image Compression and Denoising via Latent-Space Scalability"
    date: 2022
    journal: "Frontiers in Signal Processing"
    image: "assets/img/thumbnails/2022-frontiers-joint-image-compression-and-denoising-via-latent-space-scalability.png"
    citation: |
      @article{alvar2022joint,
        author            = {Alvar, Saeed Ranjbar and Ulhaq, Mateen and Choi, Hyomin and Bajić, Ivan V.},
        author+an:default = {2=me},
        publisher         = {Frontiers Media {SA}},
        date              = {2022},
        doi               = {10.3389/frsip.2022.932873},
        eprint            = {2205.01874},
        eprintclass       = {eess.IV},
        eprinttype        = {arXiv},
        journaltitle      = {Frontiers in Signal Processing},
        title             = {Joint Image Compression and Denoising via Latent-Space Scalability},
        volume            = {2},
      }
    pdf: "https://arxiv.org/pdf/2205.01874.pdf"
    links:
      - name: Document
        url: "https://arxiv.org/pdf/2205.01874.pdf"
    description: |
      When it comes to image compression in digital cameras, denoising is traditionally performed prior to compression.
      However, there are applications where image noise may be necessary to demonstrate the trustworthiness of the image, such as court evidence and image forensics.
      This means that noise itself needs to be coded, in addition to the clean image itself.
      In this paper, we present a learning-based image compression framework where image denoising and compression are performed jointly.
      The latent space of the image codec is organized in a scalable manner such that the clean image can be decoded from a subset of the latent space (the base layer), while the noisy image is decoded from the full latent space at a higher rate.
      Using a subset of the latent space for the denoised image allows denoising to be carried out at a lower rate.
      Besides providing a scalable representation of the noisy input image, performing denoising jointly with compression makes intuitive sense because noise is hard to compress; hence, compressibility is one of the criteria that may help distinguish noise from the signal.
      The proposed codec is compared against established compression and denoising benchmarks, and the experiments reveal considerable bitrate savings compared to a cascade combination of a state-of-the-art codec and a state-of-the-art denoiser.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "Latent Space Motion Analysis for Collaborative Intelligence"
    date: 2021
    journal: "IEEE ICASSP"
    image: "assets/img/thumbnails/2021-icassp-latent-space-motion-analysis-for-collaborative-intelligence.png"
    citation: |
      @inproceedings{ulhaq2021analysis,
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        booktitle         = {Proc. IEEE ICASSP},
        date              = {2021},
        doi               = {10.1109/ICASSP39728.2021.9413603},
        eprint            = {2102.04018},
        eprintclass       = {cs.CV},
        eprinttype        = {arXiv},
        pages             = {8498--8502},
        title             = {Latent Space Motion Analysis for Collaborative Intelligence},
      }
    pdf: "https://arxiv.org/pdf/2102.04018.pdf"
    links:
      - name: Document
        url: "https://arxiv.org/pdf/2102.04018.pdf"
      - name: Talk
        url: "https://www.youtube.com/watch?v=_no6R1sNpHo"
    description: |
      When the input to a deep neural network (DNN) is a video signal, a sequence of feature tensors is produced at the intermediate layers of the model.
      If neighboring frames of the input video are related through motion, a natural question is, "what is the relationship between the corresponding feature tensors?"
      By analyzing the effect of common DNN operations on optical flow, we show that the motion present in each channel of a feature tensor is approximately equal to the scaled version of the input motion.
      The analysis is validated through experiments utilizing common motion models.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "ColliFlow: A Library for Executing Collaborative Intelligence Graphs"
    date: 2020
    journal: "NeurIPS (demo)"
    image: "assets/img/thumbnails/2020-neurips-demo-colliflow.png"
    citation: |
      @misc{ulhaq2020colliflow,
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        url               = {https://yodaembedding.github.io/neurips-2020-demo/},
        date              = {2020},
        note              = {demoed at NeurIPS},
        title             = {{ColliFlow}: A Library for Executing Collaborative Intelligence Graphs},
      }
    pdf: "https://yodaembedding.github.io/neurips-2020-demo/"
    links:
      - name: Code
        url: "https://github.com/YodaEmbedding/colliflow"
      - name: Slides
        url: "assets/pdf/2020-neurips-demo-colliflow-slides.pdf"
      - name: Talk
        url: "https://www.youtube.com/watch?v=ApvKOTlmflo"
        # - name: Project website
        #   url: "https://yodaembedding.github.io/neurips-2020-demo/"
    description: |
      Collaborative intelligence has emerged as a promising strategy to bring "AI to the edge".
      In a typical setting, a learning model is distributed between an edge device and the cloud, each part performing its own share of inference.
      We present ColliFlow — a library for executing collaborative intelligence graphs — which makes it relatively easy for researchers and developers to construct and deploy collaborative intelligence systems.
    #
    # Collaborative intelligence is a technique for using more than one computing device to perform a computational task.
    # A possible application of this technique is to assist mobile client edge devices in performing inference of deep learning models by sharing the workload with a server.
    # In one typical setup, the mobile device performs a partial inference of the model, up to an intermediate layer.
    # The output tensor of this intermediate layer is then transmitted over a network (e.g. WiFi, LTE, 3G) to a server, which completes the remaining inference, and then transmits the result back to the client.
    # Such a strategy can reduce network usage, resulting in reduced bandwidth costs, lower energy consumption, faster inference, and provide better privacy guarantees.
    # A working implementation of this was shown in our demo at NeurIPS 2019.
    # This year, we present a library that will enable researchers and developers to create collaborative intelligence systems themselves quickly and easily.
    #
    # This demo presents a new library for developing and deploying collaborative intelligence systems. Computational and communication subprocesses are expressed as a directed acyclic graph. Expressing the entire process as a computational graph provides several advantages including modularity, graph serializability and transmission, and easier scheduling and optimization.
    #
    # Library features include:
    #
    #     Graph definition via a functional API inspired by Keras and PyTorch
    #     Over-the-network execution of graphs that span across multiple devices
    #     API for Android (Kotlin/Java) edge clients and servers (Python)
    #     Integration with Reactive Extensions (Rx)
    #     Asynchronous execution and multi-threading support
    #     Backpressure handling
    #     Modules for network transmission of compressed feature tensor data

  -
    authors:
      - name: Mateen Ulhaq
        me: true
    title: "Bachelor's thesis: Mobile-Cloud Inference for Collaborative Intelligence"
    date: 2020
    journal: "Simon Fraser University"
    image: "assets/img/thumbnails/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis.png"
    citation: |
      @thesis{ulhaq2020thesisbasc,
        author            = {Ulhaq, Mateen},
        author+an:default = {1=me},
        institution       = {Simon Fraser University},
        date              = {2020},
        eprint            = {2306.13982},
        eprintclass       = {cs.LG},
        eprinttype        = {arXiv},
        title             = {Mobile-Cloud Inference for Collaborative Intelligence},
        type              = {BASc honors thesis},
      }
    # pdf: "https://arxiv.org/pdf/2306.13982.pdf"
    pdf: "assets/pdf/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis.pdf"
    links:
      - name: Document
        # url: "https://arxiv.org/pdf/2306.13982.pdf"
        url: "assets/pdf/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis.pdf"
      - name: Slides
        url: "assets/pdf/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis-defence-slides.pdf"
    description: |
      As AI applications for mobile devices become more prevalent, there is an increasing need for faster execution and lower energy consumption for deep learning model inference.
      Historically, the models run on mobile devices have been smaller and simpler in comparison to large state-of-the-art research models, which can only run on the cloud.
      However, cloud-only inference has drawbacks such as increased network bandwidth consumption and higher latency.
      In addition, cloud-only inference requires the input data (images, audio) to be fully transferred to the cloud, creating concerns about potential privacy breaches.

      There is an alternative approach: shared mobile-cloud inference.
      Partial inference is performed on the mobile in order to reduce the dimensionality of the input data and arrive at a compact feature tensor, which is a latent space representation of the input signal.
      The feature tensor is then transmitted to the server for further inference.
      This strategy can reduce inference latency, energy consumption, and network bandwidth usage, as well as provide privacy protection, because the original signal never leaves the mobile.
      Further performance gain can be achieved by compressing the feature tensor before its transmission.

  -
    authors:
      - name: Mateen Ulhaq
        me: true
      - name: Ivan V. Bajić
    title: "Shared Mobile-Cloud Inference for Collaborative Intelligence"
    date: 2019
    journal: "NeurIPS (demo)"
    image: "assets/img/thumbnails/2019-neurips-demo-shared-mobile-cloud-inference-for-collaborative-intelligence.png"
    citation: |
      @misc{ulhaq2019shared,
        abstract          = {Interactive mobile Android app demo},
        author            = {Ulhaq, Mateen and Bajić, Ivan V.},
        author+an:default = {1=me},
        url               = {https://youtu.be/sHySFCUzh6s},
        date              = {2019},
        eprint            = {2002.00157},
        eprintclass       = {cs.AI},
        eprinttype        = {arXiv},
        note              = {demoed at NeurIPS},
        title             = {Shared Mobile-Cloud Inference for Collaborative Intelligence},
        venue             = {Vancouver, Canada},
      }
    pdf: "https://www.youtube.com/watch?v=sHySFCUzh6s"
    links:
      - name: Code
        url: "https://github.com/YodaEmbedding/collaborative-intelligence"
      - name: Video
        url: "https://www.youtube.com/watch?v=sHySFCUzh6s"
      # - name: Report
      #   url: "https://arxiv.org/pdf/2002.00157.pdf"
    description: |
      As AI applications for mobile devices become more prevalent, there is an increasing need for faster execution and lower energy consumption for neural model inference.
      Historically, the models run on mobile devices have been smaller and simpler in comparison to large state-of-the-art research models, which can only run on the cloud.
      However, cloud-only inference has drawbacks such as increased network bandwidth consumption and higher latency.
      In addition, cloud-only inference requires the input data (images, audio) to be fully transferred to the cloud, creating concerns about potential privacy breaches.
      We demonstrate an alternative approach: shared mobile-cloud inference.
      Partial inference is performed on the mobile in order to reduce the dimensionality of the input data and arrive at a compact feature tensor, which is a latent space representation of the input signal.
      The feature tensor is then transmitted to the server for further inference.
      This strategy can improve inference latency, energy consumption, and network bandwidth usage, as well as provide privacy protection, because the original signal never leaves the mobile.
      Further performance gain can be achieved by compressing the feature tensor before its transmission.

---

<!-- TODO Move this to the top of the page. -->
<!-- Social media links -->
<div style="text-align: center">
<a href="https://stackoverflow.com/users/365102/mateen-ulhaq"><img src="assets/img/icons/stackoverflow-32x32.png" alt="Stack Overflow" title="Stack Overflow" style="display: inline-block; height: 16px; width: auto" /></a>&nbsp;
<a href="https://github.com/YodaEmbedding"><img src="assets/img/icons/github-32x32.png" alt="GitHub" title="GitHub" style="display: inline-block; height: 16px; width: auto" /></a>&nbsp;
<a href="https://www.linkedin.com/in/mulhaq"><img src="assets/img/icons/linkedin-32x32.png" alt="LinkedIn" title="LinkedIn" style="display: inline-block; height: 16px; width: auto" /></a>&nbsp;
<a href="https://muntoo.deviantart.com/gallery"><img src="assets/img/icons/photography-32x32.png" alt="Photography/Art" title="Photography/Art" style="display: inline-block; height: 16px; width: auto" /></a>
</div>


<!-- # Bio -->

I am a former Master's student from Simon Fraser University.
I am interested in deep learning, artificial intelligence, programming languages (e.g. Python, C++, Rust, Haskell, etc.), compilers, compression, open source, Linux, terminal code editors (Neovim), mathematics, image processing, photography, and physics.
<!-- , and since I did my undergrad in Engineering Physics, I guess one might also say physics. :) -->

<!--
[![Stack Overflow](assets/img/icons/stackoverflow-32x32.png)](https://stackoverflow.com/users/365102/mateen-ulhaq)
[![GitHub](assets/img/icons/github-32x32.png)](https://github.com/YodaEmbedding)
[![LinkedIn](assets/img/icons/linkedin-32x32.png)](https://www.linkedin.com/in/mulhaq)
[![DeviantArt](assets/img/icons/photography-32x32.png)](https://muntoo.deviantart.com/gallery)
 -->

<!--
Some of my recent publications include:

- Learned compression of encoding distributions (ICIP 2024) \[Submitted\]
- Scalable human-machine point cloud compression (PCS 2024) \[Accepted\]
- Learned compression for images and point clouds (Master's thesis) [[PDF](assets/pdf/2023-learned-compression-for-images-and-point-clouds-masc-thesis.pdf)] [[Slides](assets/pdf/2023-learned-compression-for-images-and-point-clouds-masc-thesis-defense-slides.pdf)]
- Learned point cloud compression for classification (MMSP 2023) [[Paper](https://arxiv.org/abs/2308.05959)] [[Slides](https://raw.githubusercontent.com/multimedialabsfu/learned-point-cloud-compression-for-classification/assets/main/assets/slides.pdf)] [[Code](https://github.com/multimedialabsfu/learned-point-cloud-compression-for-classification)]
- Latent space motion analysis for collaborative intelligence (ICASSP 2021) [[Paper](https://arxiv.org/abs/2102.04018)] [[Talk](https://www.youtube.com/watch?v=_no6R1sNpHo)]
- Mobile-cloud inference for collaborative intelligence (Bachelor's honors thesis) [[PDF](assets/pdf/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis.pdf)] [[Slides](assets/pdf/2020-mobile-cloud-inference-for-collaborative-intelligence-basc-thesis-defence-slides.pdf)]
- ColliFlow: a library for executing collaborative intelligence graphs (NeurIPS 2020 demo) [[Talk](https://www.youtube.com/watch?v=ApvKOTlmflo)]
- Shared mobile-cloud inference for collaborative intelligence (NeurIPS 2019 demo) [[Talk](https://www.youtube.com/watch?v=sHySFCUzh6s)] [[Code](https://github.com/YodaEmbedding/collaborative-intelligence)]
-->

Some fun projects I've worked on:

- [Building a mini autograd engine (Python) [Slides]](assets/pdf/2024-building-a-mini-autograd-autodiff-engine-slides.pdf)
- [CompressAI Trainer (Python)](https://github.com/InterDigitalInc/CompressAI-Trainer)
- [Particle filter likelihood kernel on FPGA and GPU (C++, HLS, CUDA) [Slides]](assets/pdf/2021-particle-filter-likelihood-kernel-fpga-hls-slides.pdf)
- [Chess engine (Rust)](https://github.com/YodaEmbedding/rs-chess)
- [Frece: frecency indexed database (Rust)](https://github.com/YodaEmbedding/frece)
- [Dotfiles](https://github.com/YodaEmbedding/dotfiles)
- [Easy slurm (Python)](https://github.com/YodaEmbedding/easy-slurm)
- [Scrobblez: customizable last.fm scrobbler (Python)](https://github.com/YodaEmbedding/scrobblez)
- [Improving the tactical awareness of deep neural network chess engines (Python) [Report]](assets/pdf/2021-ensc-413-improving-the-tactical-awareness-of-deep-neural-network-chess-engines.pdf)
- [Chess "play all moves" challenge webapp (JavaScript)](https://github.com/YodaEmbedding/chess-speedrun)
- [Fruit tetris game (C++, OpenGL)](https://www.youtube.com/watch?v=pfS8h6n60_M)
<!-- - [ARM assembly racing game](https://github.com/YodaEmbedding/ARMRacingGame) -->
<!-- - [Raytracer]() -->
<!-- flow-rate-app -->
<!-- colliflow: https://www.youtube.com/watch?v=ApvKOTlmflo -->

<!-- TODO "Blog post" about LIC... for traffic. :) Or just a Wikipedia article... -->

<!-- Maybe interactive JS game... :) -->

