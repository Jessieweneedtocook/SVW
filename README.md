> **Provenance & License**
> This project (“SVW”) is a derivative of [VINE](https://github.com/Shilin-LU/VINE), © NTUITIVE (Nanyang Technological University), distributed under the **NTUITIVE Non-Commercial License** (see `LICENSE`).
> **Modifications for semantic embedding by Jessie Smith (jessieweneedtocook), 2025.**
> All code in this repository is for **non-commercial use only**.


## Overview

As generative AI models (e.g., Stable Diffusion, DALL-E) become capable of producing photorealistic images and edits, watermarking has emerged as a defense for **content provenance**. The original [VINE](https://github.com/Shilin-LU/VINE) framework introduced **VINE-B** (baseline) and **VINE-R** (robust) as strong post-processing watermarking models, embedding 100-bit watermarks via SDXL-Turbo with impressive robustness to global edits.

However, **VINE-R embeds watermarks uniformly across image features**, making it vulnerable to **local, semantic manipulations** (e.g., object removal or replacement).  

**VINE-S** addresses this limitation by adding **semantic embedding** through a **stability predictor** that guides watermark placement into semantically stable regions of an image.

---

## Key Contributions of VINE-S

1. **Stability Predictor**
   - A CNN trained on HumanEdit + PIPE datasets to predict per-pixel likelihood of being edited.
   - Produces a **stability mask** highlighting robust embedding regions.

2. **Semantic Integration**
   - The stability mask is passed to an updated **ConditionAdaptor**:
     - Input channels increased from 6 → 9 (image + secret + mask).
     - Embeds watermarks preferentially in stable regions.
   - New **embed mask loss** penalizes embedding into unstable regions.

3. **Modified Training Pipeline**
   - Retains the original VINE 3-stage training:
     - **Stage 1:** Basic robustness (common noise, blur, JPEG, etc.).
     - **Stage 2:** Perceptual + pixel fidelity.
     - **Stage 3:** Fine-tuning with InstructPix2Pix surrogate edits.
   - Extended with stability-aware objectives.
  
## Dissertation

This repository is based on my undergraduate dissertation:

**Jessie E S Smith (2025).**  
*Robust Invisible Watermarking in the Age of Generative AI.*  
BSc Computer Science, Newcastle University.  

📄 [Read the full dissertation (PDF)](docs/Diss_FINAL.pdf)

4. **Improved Local Robustness**
   - Demonstrated **state-of-the-art robustness under local, object-based generative attacks** (e.g., remove-and-replace pipelines).
   - Outperforms VINE-R when parts of an image are selectively manipulated.
