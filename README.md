# FraudFusion

FraudFusion is a Data Science Capstone project focused on improving fraud detection by generating high-quality synthetic fraud samples. The project uses a diffusion-based generative model—based on the FraudDiffuse paper—to augment highly imbalanced fraud datasets. By leveraging techniques such as an adaptive non-fraud prior, probability-based loss, triplet (contrastive) loss, and engineered feature range loss, FraudFusion aims to produce synthetic fraud data that closely resembles real fraud cases while being diverse enough to enhance training of fraud detection models.


## Project Overview

- **Motivation:**  
  Modern financial systems face severe challenges with imbalanced datasets in fraud detection. Real-world fraud samples are extremely scarce, causing traditional models to underperform. FraudFusion exploits state-of-the-art diffusion models to generate synthetic fraud cases, supplementing limited real samples and improving overall detection performance.

- **Approach:**  
  The model progressively transforms noise into realistic synthetic fraud examples by:
  - **Adaptive Prior:** Learning from non-fraud data to guide generative sampling.
  - **Probability-Based Loss:** Encouraging outputs that adhere to the non-fraud distribution.
  - **Triplet Loss:** Enforcing contrast between fraudulent and non-fraudulent samples.
  - **Engineered Range Loss:** Clipping engineered (e.g., time-derived) feature outputs within observed realistic ranges.

- **Dataset:**  
  The current implementation uses the Sparkov dataset. Data preprocessing includes log transformations for monetary and skewed features, datetime feature extraction, and standard scaling of numeric data. ([Sparkov Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection))


## File Structure

- **Data_Preprocessing.ipynb**  
  Performs data cleaning, merging, missing value handling, feature engineering (e.g., converting date-of-birth into age, datetime decompositions), log transformations, and scaling.  
- **Model_Development_v4.ipynb**  
  A paper-faithful implementation that trains the diffusion model using fraud samples only and employs an adaptive non-fraud prior.
- **Model_Development_Baseline.ipynb**  
  Builds a complete pipeline with additional loss components including probability-based, triplet, and engineered range losses.
- **Model_Development_Baseline_improved.ipynb**  
  An enhanced version with GPU support, detailed training logs, comprehensive statistical evaluations, and refined data generation.
- **FraudDiffuse_Model.ipynb**  
  (Optional) Contains an alternative implementation incorporating time and label embeddings with a slightly different architectural approach.
- **README.md**  
  This file, which provides an overview and instructions for the project.


## Requirements

Ensure you have the following packages installed:
- Python 3.8 or above
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- PyTorch
- tqdm

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib torch tqdm
```

## Setup and Execution

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/FraudFusion.git
   cd FraudFusion
   ```

2. **Prepare the Data:**

   - Place the Sparkov dataset CSV files (e.g., `fraudTrain.csv` and `fraudTest.csv`) in the `Data/Sparkov` folder.
   - Run the `Data_Preprocessing.ipynb` notebook to preprocess and generate required artifacts such as the standardized scaler, categorical vocabulary, and mapping.

3. **Model Training and Evaluation:**

   - Start with `Model_Development_v4.ipynb` to validate the basic diffusion model approach.
   - Explore `Model_Development_Baseline.ipynb` for a full baseline implementation with multiple loss terms.
   - For further refinements and evaluation, use `Model_Development_Baseline_improved.ipynb`.
   - (Optionally) Check out `FraudDiffuse_Model.ipynb` for an alternate implementation with additional feature embeddings.

4. **Generating Synthetic Fraud Samples:**

   After training the model (preferably using the Baseline Improved implementation), use the synthesis section to generate synthetic fraud samples. The notebook will generate outputs (both numeric and categorical parts) and provide inverse-transformed results for evaluation.



## Mathematical Formulation

The core of the synthetic fraud generation process is based on diffusion models. Key equations include:

- **Forward Diffusion Process:**

  \[
  x_t = \sqrt{\hat{\alpha}_t} \, x_0 + \sqrt{1 - \hat{\alpha}_t} \, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
  \]

  where:
  - \(x_0\) is the original (clean) input.
  - \(x_t\) is the noisy input at timestep \(t\).
  - \(\hat{\alpha}_t\) is the cumulative product of \(\alpha\) up to timestep \(t\).

- **Reverse Diffusion Process and Loss Function:**

  The model is trained to predict the noise added, and the overall loss is defined as:

  \[
  L_{\text{total}} = L_{\text{norm}} + w_1 \, L_{\text{prior}} + w_2 \, L_{\text{triplet}} + \lambda_{\text{eng}} \, L_{\text{eng}}
  \]

  where:
  - \(L_{\text{norm}}\) is the MSE loss between the predicted and true noise.
  - \(L_{\text{prior}}\) is a probability-based loss ensuring adherence to the non-fraud distribution.
  - \(L_{\text{triplet}}\) is a contrastive loss enforcing separation between fraudulent and non-fraudulent samples.
  - \(L_{\text{eng}}\) penalizes values of engineered features falling outside the observed range.
  - \(w_1\), \(w_2\), and \(\lambda_{\text{eng}}\) are weighting parameters.

## Model Architecture Overview

The network architecture for the noise predictor consists of the following components:

- **Input Handling:**  
  - **Numeric Features:** Directly used after standard scaling.
  - **Categorical Features:** Passed through embedding layers.
  - **Time Embedding:** A normalized timestep is concatenated to the feature vector.

- **MLP Structure:**  
  - The MLP comprises several fully connected layers with LeakyReLU activations, which predict the noise vector.


## Training Details

- **Hyperparameters:**  
  - Number of diffusion timesteps: 800  
  - Learning rate: 0.001  
  - Batch size: 40  
  - Number of epochs: 200  

- **Training Monitoring:**  
  Loss values are reported every 10 epochs to track convergence.

## Evaluation

The quality of synthetic fraud samples is evaluated by:
- Comparing statistical distributions (mean, standard deviation, percentiles) of real and synthetic fraud data.
- Visualizations such as histograms and density plots for numeric features.
- Using statistical tests (e.g., KS test and Anderson–Darling test) to ensure distributional similarity.
- Finally, augmenting fraud detection training data to observe improvements in classification metrics.

## Citation

> [FraudDiffuse: Diffusion-aided Synthetic Fraud Augmentation for Improved Fraud Detection](https://dl.acm.org/doi/pdf/10.1145/3677052.3698658)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback regarding FraudFusion, please contact:
- Raghu Ram Sattanapalle – [sattanapalle.r@northeastern.edu](mailto:sattanapalle.r@northeastern.edu)
- Anish Rao - [rao.anish@northeastern.edu](mailto:rao.anish@northeastern.edu)

---