# Synthetic-Voice-Detection-Vocoder-Artifacts

# LibriSeVoc Dataset
1. We are the first to identify neural vocoders as a source of features to expose synthetic human voices.
   Here are the differences shown by the six vocoders compared to the original audio:
   ![image](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/6c3381c4-af7e-4ce2-a446-b3c76bf52aee)

2. We provide LibriSeVoC as a dataset of self-vocoding samples created with six state-of-the-art vocoders to highlight and exploit the vocoder artifacts.
   The distribution of data set and audio length is shown in the following tables:
   <img width="503" alt="image" src="https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/c74fdb20-a5b7-4109-b833-821dd8dd6230">
   ![image](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/3dd129af-7785-4223-9b50-e41bc772317a)


# Deepfake Detection
We propose a new approach to detecting synthetic human voices by exposing signal artifacts left by neural vocoders by modifying and improved the RawNet2 baseline by adding multi-loss, lowering the error rate from 6.10% to 4.54% on ASVspoof Dataset.
This is the framework of the proposed synthesized voice detection method:
   <img width="517" alt="image" src="https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts/assets/90001788/c46df06b-6d62-4b0f-a9d2-f5ffc4e378b9">

# Paper
For more details please read our paper: **https://arxiv.org/abs/2304.13085**

