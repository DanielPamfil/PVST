# PVST: Pyramid Vision Transformer for Saliency Detection

## Overview
The Pyramid Vision Transformer (PVST) project is an advanced deep learning model designed for saliency detection tasks. Utilizing a hierarchical representation of images, PVST employs transformer architectures to efficiently process visual data and predict areas of an image that attract human attention.

## Features
- **Transformer-Based Architecture**: Utilizes the Pyramid Vision Transformer, incorporating a multi-scale approach for enhanced performance.
- **Saliency Detection**: Specialized in identifying and highlighting salient regions in images.
- **Hierarchical Image Processing**: Processes images in a hierarchical manner for better feature extraction.
- **Evaluation Metrics**: Includes various metrics like MAE, F-measure, AUC, E-measure, S-measure, and AP for comprehensive performance analysis.
- **CUDA Support**: Optimized for CUDA-enabled devices for faster computation.

## Installation

To set up the PVST project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DanielPamfil/PVST.git
   cd PVST
   ```

2. **Install Dependencies**:
   Ensure that Python (version 3.x) is installed, along with PyTorch. Other dependencies can be installed via:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**:
   Organize your dataset in the following format:
   ````
   -- Data
   |-- DUTS
   |   |-- DUTS-TR
   |   |-- | DUTS-TR-Image
   |   |-- | DUTS-TR-Mask
   |   |-- DUTS-TE
   |   |-- | DUTS-TE-Image
   |   |-- | DUTS-TE-Mask
   |-- ECSSD
   |   |--images
   |   |--GT
   ...
   ````

4. **Model Weights**:
   If pre-trained weights are available, place them in the [pretrained_model](pretrained_model)  directory.

## Usage

To train or test the model, follow these instructions:

1. **Training**:
   Run the model by command line by specifying the required arguments. For example, to train the model, r:
   ```bash
   python train_test_eval.py --Training True--test_paths DUTS/DUTS-TE --pretrained_model ./pretrained_model/pvt_v2_b3.pth --save_model_dir checkpoint/pvt_v2_b3/ --arch pvt_v2_b3
   ```

2. **Testing**:
   After training, or with pre-trained weights, evaluate the model using:
   ```bash
   python train_test_eval.py --Testing True --test_paths DUTS/DUTS-TE --pretrained_model ./pretrained_model/pvt_v2_b3.pth --save_model_dir checkpoint/pvt_v2_b3/ 

   ```

3. **Evaluation**:
   Run the evaluation script to calculate metrics like MAE, F-measure, etc.:
     ```bash
     python train_test_eval.py --Evaluation True --test_paths DUTS/DUTS-TE --pretrained_model ./pretrained_model/pvt_v2_b3.pth --save_model_dir checkpoint/pvt_v2_b3/ 
   
     ```

3. **Training, Testing and Evaluation**:
   To train, test, and evaluate the model in one go, run:
   ```bash
   python train_test_eval.py --Training True --Testing True --Evaluation True --test_paths DUTS/DUTS-TE --pretrained_model ./pretrained_model/pvt_v2_b3.pth --save_model_dir checkpoint/pvt_v2_b3/ 
   ```

## Results Showcase

Here are some GIFs showcasing the results obtained using PVST for saliency detection:

1. **Example 1 - Bike:**

   ![Saliency Detection GIF 1](gifs/bike.gif)

2. **Example 2 - Boat:**

   ![Saliency Detection GIF 2](gifs/boat.gif)

3. **Example 3 - Persons:**

   ![Saliency Detection GIF 3](gifs/person.gif)

## Contribution

Contributions to the PVST project are welcome. Please adhere to the project's coding style and submit pull requests for any new features or bug fixes.

## License

Specify the project's license here.

---

For more detailed information, refer to the specific documentation files provided for each component of the project.