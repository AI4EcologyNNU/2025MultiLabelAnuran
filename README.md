# Dual-branch Mamba based multi-label Anuran species classification

**Authors:** Yuji Wang, Mingying Zhu, Juan Gabriel Colonna, Jie Xie* 

In this paper, we propose a novel multi-label audio classification framework for anuran call recognition. Combining CNN-based feature extraction with a dual-branch Mamba module, our proposed model can effectively capture the complex and multi-dimensional characteristics of anuran vocalizations. The use of MM, CBAM, and ZLPRLoss can further improve the classification performance.

## Repository Structure
**config.yaml:** Configuration file defining parameters such as data paths, batch size, learning rate, model settings, and more.

**anuraset.py:** Dataset class for handling data operations.

**step0_audio_pre_process.py:** Enhance and expand the original dataset.

**step1_split_train_and_validation_dataset.py:** Divide the training set and the validation set.

**step2_extract_feature.py:** Extract the features of the spectrogram.

**step3_CNN.py:** The CNN model implementation.

**step4_evaluate.py:** Utility functions for model evaluation.

**train_and_validate.py:** contains training and validation functions.

**tools.py:** contains various utility functions used in the code.
