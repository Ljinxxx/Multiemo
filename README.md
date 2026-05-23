# Identity-Aware Cross-Task Graph Framework for Multimodal Emotion Recognition in Conversations

This repository contains the implementation of our identity-aware cross-task graph framework for multimodal emotion recognition in conversations.

The code is built upon the MultiEMO framework and extends it with graph-based multi-task learning, speaker identity modeling, adversarial identity disentanglement, and identity leakage probing.

## 1. Overview

Multimodal Emotion Recognition in Conversations (MERC) aims to identify the emotion of each utterance in a dialogue using text, audio, and visual information.

Existing MERC methods usually focus on context modeling, multimodal fusion, contrastive learning, and graph-based utterance relation modeling. However, speaker identity-related cues may be implicitly encoded into emotion-oriented representations. Such identity cues may help model speaker-specific expression patterns, but may also become shortcuts for emotion prediction.

To address this issue, we propose an identity-aware cross-task graph framework. The model explicitly connects emotion recognition and speaker identity modeling through a graph-based multi-task module, and uses adversarial identity disentanglement to reduce speaker identity leakage in emotion-oriented features.

## 2. Main Components

The repository contains the following main components:

```text
Dataset/
  IEMOCAPDataset.py
  MELDDataset.py

Loss/
  SampleWeightedFocalContrastiveLoss.py
  SoftHGRLoss.py

Model/
  MultiEMO_Model.py
  MultiAttn.py
  CrossTaskGNN.py
  DialogueRNN.py
  MLP.py
  Resnet101.py
  VisExtNet.py

Train/
  TrainMultiEMO_clean.py
  ProbeIdentityLeakage_clean.py
  Run_Disentangle_Probe_24G_clean.sh