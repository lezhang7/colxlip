# FLAIR: VLM with Fine-grained Language-informed Image Representations

**Authors:** Rui Xiao, Sanghwan Kim, Mariana-Iuliana Georgescu, Zeynep Akata, Stephan Alaniz

## Abstract
CLIP has shown impressive results in aligning images and
texts at scale. However, its ability to capture detailed visual features remains limited because CLIP matches images and texts at a global level. To address this issue, we propose **FLAIR**, **F**ine-grained **La**nguage-informed **I**mage
**R**epresentations, an approach that utilizes long and detailed image descriptions to learn localized image embed
dings. By sampling diverse sub-captions that describe fine-grained details about an image, we train our vision-
language model to produce not only global embeddings but also text-specific image representations. Our model
introduces text-conditioned attention pooling on top of local image tokens to produce fine-grained image representations that excel at retrieving detailed image content. We achieve state-of-the-art performance on both, existing mul-
timodal retrieval benchmarks, as well as, our newly introduced fine-grained retrieval task which evaluates vision-
language models’ ability to retrieve partial image content. Furthermore, our experiments demonstrate the effectiveness of FLAIR trained on 30M image-text pairs in capturing fine-grained visual information, including zero-shot semantic segmentation, outperforming models trained on billions of pairs.

## Main Difference from Previous Methods
![](assets/method_compare_github.png "FLAIR compared with previous CLIP-based methods")


## Methodology
![](assets/methodology_github.png "A general workflow for FLAIR")

