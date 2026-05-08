# Automated Side Branch Detection in OCT Imaging Using Deep Learning

This project develops a deep learning pipeline to automatically detect blood vessel side branches in optical coherence tomography (OCT) video. The goal is to improve detection robustness, reduce manual annotation burden, and support scalable clinical workflows for intravascular imaging analysis.
  
## Introduction

Coronary heart disease (CHD) is a leading global cause of mortality (Shi et al., 2024), often requiring intravascular imaging to assess arterial and guide treatment. Optical coherence tomography (OCT) provides high-resolution imaging of coronary vessels and is widely used to evaluate plaque morphology and support interventional decision-making (Nagaraja, Kalra and Puri, 2020).

A key challenge in OCT analysis is the identification of vascular side branches, which are commonly used as anatomical landmarks for co-registering OCT with other modalities such as intravascular ultrasound (IVUS) (Kubo et al., 2021). This process is currently performed manually and is time-consuming, restricting implementation and scalability in clinical workflows.

Existing deep learning approaches, including Faster R-CNN-based methods, operate on single OCT frames and are limited by false detections caused by catheter shadows, overlapping structures, and ambiguous vascular morphology. Importantly, these approaches do not leverage temporal information across adjacent frames, despite clinicians routinely using sequential context to resolve uncertainty.

This project investigates multi-frame deep learning strategies for automated side branch detection in OCT imaging. We extend Faster R-CNN by incorporating temporal feature aggregation and explore attention-based architectures to model spatiotemporal relationships across frames. The goal is to improve robustness in ambiguous regions and reduce false positives and false negatives.

The models are trained and evaluated using OCT data from a cohort of 300 patients from a previously published clinical trial (Räber et al., 2022)., with clinician-provided annotations used as ground truth. Performance is compared against a baseline Faster R-CNN model.

## Dataset

## Method

This project explores two main modelling approaches:

#### Baseline Model
- Faster R-CNN applied to individual OCT frames  
- Frame-by-frame detection without temporal context  

#### Proposal
- Multi-frame feature aggregation across consecutive OCT frames  
- CNN backbone for spatial feature extraction  
- Attention-based mechanisms to capture spatiotemporal relationships  
- Improved discrimination of side branches using temporal consistency

This design is reflective of clinical practice, where clinicians interpret adjacent frames to resolve ambiguous vascular structures.

## Dependencies

## Project Structure

## References

Kubo, T., Terada, K., Ino, Y., Shiono, Y., Tu, S., Tsao, T. P., Chen, Y., Park, D. W. (2021) ‘Combined use of multiple intravascular imaging techniques in acute coronary syndrome.’ Frontiers in cardiovascular medicine, 8, p.824128.

Nagaraja, V., Kalra, A. and Puri, R. (2020). ‘When to use intravascular ultrasound or optical coherence tomography during percutaneous coronary intervention?’ Cardiovascular diagnosis and therapy, 10 (5), pp.1429–1444.

Räber, L., Yasushi, E., Tatsuhiko, O., Losdat, S., Häner, D. J., Lonborg, J., Fahrni, G., Iglesias, F. J., van Geuns, R., Ondracek, S. A., Juul Jensen, R. D. M., Zanchin, C., Stortecky, S., Spirk, D., Siontis, M. C. G., Saleh, L., Matter, M. C., Daemen, J., Mach, F., Heg, D., Windecker, S., Engstrom, T., Lang, M. I., Koskinas, C. K. (2022). ‘Effect of alirocumab added to high-intensity statin therapy on coronary atherosclerosis in patientswith acute myocardial infarction: The PACMAN-AMI randomized clinical trial: The PACMAN-AMI randomized clinical trial.’ JAMA: the journal of the American Medical Association, 327 (18), pp.1771–1781

Shi, H., Xia, Y., Cheng, Y., Liang, P., Cheng, M., Zhang, B., Liang, Z., Wang, Y., Xie, W. (2024) ‘Global Burden of ischemic heart disease from 2022 to 2050: Projections of incidence, prevalence, deaths, and disability-adjusted life years.’ European Heart Journal - Quality of care & clinical outcomes, 11(4), 355–366

