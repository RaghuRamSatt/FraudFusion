
# FraudFusion

FraudFusion is a Data Science Capstone project focused on improving fraud detection by generating high-quality synthetic fraud samples. The project uses a diffusion-based generative model—based on the FraudDiffuse paper—to augment highly imbalanced fraud datasets. By leveraging techniques such as an adaptive non-fraud prior, probability-based loss, triplet (contrastive) loss, engineered feature range loss, and amount distribution loss, FraudFusion aims to produce synthetic fraud data that closely resembles real fraud cases while being diverse enough to enhance training of fraud detection models.


## Project Overview

- **Motivation:**  
  Modern financial systems face severe challenges with imbalanced datasets in fraud detection. Real-world fraud samples are extremely scarce, causing traditional models to underperform. FraudFusion exploits state-of-the-art diffusion models to generate synthetic fraud cases, supplementing limited real samples and improving overall detection performance.

- **Approach:**  
  The model progressively transforms noise into realistic synthetic fraud examples by:
  - **Adaptive Prior:** Learning from non-fraud data to guide generative sampling.
  - **Probability-Based Loss:** Encouraging outputs that adhere to the non-fraud distribution.
  - **Triplet Loss:** Enforcing contrast between fraudulent and non-fraudulent samples.
  - **Engineered Range Loss:** Clipping engineered (e.g., time-derived) feature outputs within observed realistic ranges.
  - **Amount Distribution Loss:** Matching the bimodal distribution of transaction amounts with emphasis on higher-value fraud.

- **Dataset:**  
  The current implementation uses the Sparkov dataset. Data preprocessing includes log transformations for monetary and skewed features, datetime feature extraction, and standard scaling of numeric data. ([Sparkov Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection))


## Project Structure

FraudFusion is organized into the following directories:

- **Diffusion Models/**  
  Contains all diffusion model development notebooks showing the evolution from baseline to Version 7, along with trained model weights (.pth files):
  - `Baseline_improved_v7.ipynb`: Final enhanced model with distribution matching and targeted amount weighting
  - `Baseline_improved_v5.ipynb`, `Baseline_improved_v4.ipynb`, etc.: Intermediate versions showing progressive improvements
  - `Generating_data.ipynb`: Notebook for generating synthetic fraud samples
  - Various model weight files (e.g., `baseline_improved_v7.pth`) capturing different stages of development

- **fraudEDA/**  
  Contains data exploration and preprocessing notebooks:
  - `Data_Preprocessing.ipynb`: Performs data cleaning, feature engineering, and transformations
  - `sparkovEDA.ipynb`: Exploratory analysis of the Sparkov dataset
  - `Data_Understanding_Sparkov.ipynb`: Detailed analysis of fraud patterns and distributions
  - `Sparkov_Analysis.ipynb`: Additional analysis of fraud characteristics

- **XGBOOST Model Training/**  
  Contains classifier implementation and evaluation notebooks:
  - `XGBoost_synthetic_controlled_&validation.ipynb`: Implementation of our controlled validation approach
  - `Baseline_XGBoost.py`: Baseline XGBoost model training
  - Various Python scripts implementing different XGBoost configurations
  - Visualization files showing classifier performance metrics

- **Results/**  
  Contains metrics, visualizations, and prediction files:
  - ROC and PR curve data (CSV files)
  - Model metrics in JSON format
  - Feature importance information
  - Prediction outputs and comparative visualizations

- **Documentation/**  
  Contains reference papers and research materials:
  - `FraudDiffuse_paper.pdf`: Original paper that inspired this project
  - `Related Work_RaghuRamSattanapalle.pdf`: Literature review on synthetic fraud generation approaches
  - `FraudFusion_Report.pdf`: Project report which outlines clear and well structured overview of the methodologies employed

## Model Evolution

Our enhanced FraudDiffuse model evolved through several versions, each addressing specific performance limitations:

- **Version 2:** Introduced range constraints for engineered features by computing observed min/max values in standardized space and adding penalty loss for values outside this range
- **Version 3:** Added feature-specific initialization for amount, implemented cyclical encoding for time features (hour, day, month, day of week), applied targeted loss weighting, and increased model capacity
- **Version 4:** Focused on stability with controlled distribution matching for amount feature, stability-preserving architecture changes, balanced loss weighting, and NaN prevention mechanisms
- **Version 5:** Enhanced distribution modeling to better capture the bimodal nature of the amount feature, improved age distribution modeling, and added feature-specific adjustments to the generation process
- **Version 7 (Final):** Implemented post-processing steps to enforce amount distribution matching, enhanced initialization specifically for the amount feature, applied more aggressive weighting for higher fraud amounts, and added distribution transformation matching using quantile-based CDF transformation. Introduced bimodal initialization strategy with a 90/10 split favoring high-value fraud samples and KDE-based peak detection.

## Enhanced Loss Functions

Our model incorporates several specialized loss components beyond the original FraudDiffuse paper:

- **Feature-Weighted L_norm:** MSE on predicted vs. true noise with increased weights for challenging features (e.g., transaction amount weighted at 1.8)
- **Probability-Based Loss:** Enforces predicted noise to follow a non-fraud prior using z-scores
- **Triplet Loss:** Ensures estimated clean samples are closer to true fraud samples and further from non-fraud instances
- **Engineered Range Loss:** Penalizes predictions for temporal features that fall outside observed bounds
- **Amount Distribution Loss:** Matches the bimodal transaction amount distribution by focusing on key percentiles (50th, 75th, 90th, 95th) with progressively higher weights (1.0, 3.0, 5.0, 8.0) for upper quantiles and a 4.0× weight for skewness matching

The final composite loss function combines these components with carefully tuned weights to balance different aspects of synthetic data quality.

## Key Results

Our experimental evaluation demonstrated significant improvements in fraud detection capability:

- **Synthetic Data Quality:** 
  - Version 7 successfully captured the bimodal distribution of fraud transaction amounts and preserved critical statistical properties across features
  - Achieved approximately 20-25% improvement in KS statistic for amount distribution compared to v4
  - Improved tail ratio accuracy (95th and 99th percentiles) bringing synthetic/real ratios closer to the ideal value of 1.0

- **Fraud Detection Performance:** 
  - The baseline XGBoost model achieved high precision (0.917) but missed approximately 17% of fraud cases
  - Models augmented with synthetic samples improved fraud capture by 5-6 percentage points (recall increased from 0.827 to 0.885)
  - Increasing synthetic data volume from 5,000 to 8,000 samples improved precision while maintaining enhanced recall

- **Operational Implications:** Our synthetic data approach effectively shifts the operating point toward higher sensitivity without significantly compromising the model's overall discrimination ability

## Evaluation Framework

We evaluated our enhanced FraudDiffuse model using a comprehensive dual assessment approach:

### Synthetic Data Quality Metrics
- **Distribution Metrics:** KS Statistic, Wasserstein Distance, Energy Distance
- **Tail Ratios:** Comparison of 95th & 99th percentiles between real and synthetic data
- **Visual Analysis:** QQ-plots and distribution histograms comparing synthetic to real data

### Controlled Validation Methodology
Our fraud detection evaluation employed a controlled validation approach:
- Maintained separate pure validation sets containing only real data
- Created synthetic validation sets combining real data with synthetic fraud samples
- Monitored both validation streams to ensure generalization to real fraud patterns
- Evaluated final performance on completely held-out test sets

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
   - Run the `Data_Preprocessing.ipynb` notebook in the `fraudEDA` folder to preprocess and generate required artifacts such as the standardized scaler, categorical vocabulary, and mapping.

3. **Model Training and Evaluation:**

   - For diffusion model training, explore the notebooks in the `Diffusion Models` folder. Start with earlier versions to understand the progression.
   - For generating synthetic samples, use the `Generating_data.ipynb` notebook with the latest model weights.
   - For XGBoost training and evaluation, use notebooks in the `XGBOOST Model Training` folder.

4. **Results Analysis:**

   - Analysis outputs, metrics and visualizations are stored in the `Results` folder.
   - View classifier metrics and performance comparisons to evaluate the effectiveness of synthetic data augmentation.

## Mathematical Formulation

The core of the synthetic fraud generation process is based on diffusion models. Key equations include:

- **Forward Diffusion Process:**

  $$x_t = \sqrt{\hat{\alpha}_t} \, x_0 + \sqrt{1 - \hat{\alpha}_t} \, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$

  where:
  - $$x_0$$ is the original (clean) input.
  - $$x_t$$ is the noisy input at timestep $$t$$.
  - $$\hat{\alpha}_t$$ is the cumulative product of $$\alpha$$ up to timestep $$t$$.

- **Enhanced Loss Function:**

  $$L_{\text{total}} = L_{\text{norm}} + w_1 \, L_{\text{prior}} + w_2 \, L_{\text{triplet}} + \lambda_{\text{eng}} \, L_{\text{eng}} + \lambda_{\text{amt}} \, L_{\text{amt}}$$

  where:
  - $$L_{\text{norm}}$$ is the MSE loss between the predicted and true noise.
  - $$L_{\text{prior}}$$ is a probability-based loss ensuring adherence to the non-fraud distribution.
  - $$L_{\text{triplet}}$$ is a contrastive loss enforcing separation between fraudulent and non-fraudulent samples.
  - $$L_{\text{eng}}$$ penalizes values of engineered features falling outside the observed range.
  - $$L_{\text{amt}}$$ is a specialized loss component targeting the bimodal amount distribution.
  - $$w_1$$, $$w_2$$, $$\lambda_{\text{eng}}$$, and $$\lambda_{\text{amt}}$$ are weighting parameters.

- **Distribution Transformation Function (v7):**

  $$y = F^{-1}_{\text{target}}(F_{\text{source}}(x))$$

  where:
  - $$F_{\text{source}}$$ is the CDF of the generated distribution
  - $$F^{-1}_{\text{target}}$$ is the inverse CDF (quantile function) of the target distribution
  - $$x$$ is the generated value
  - $$y$$ is the transformed value matching the target distribution

## Model Architecture Overview

The network architecture for the noise predictor consists of the following components:

- **Input Handling:**  
  - **Numeric Features:** Directly used after standard scaling (11 dimensions).
  - **Categorical Features:** Passed through embedding layers (8 categories with varying cardinality).
  - **Time Embedding:** Specialized cyclic encodings for temporal features (8 dimensions).
  - **Diffusion Timestep:** A normalized timestep is concatenated to the feature vector.

- **MLP Structure:**  
  - Four-layer feed-forward structure with hidden dimension 256
  - Gentle residual connections with scaling factor 0.1 between hidden layers
  - Layer normalization after each hidden layer for stability
  - ReLU activation functions
  - Dropout with rate 0.1 for regularization
  - Xavier initialization for weight stability

## Training Details

- **Hyperparameters:**  
  - Number of diffusion timesteps: 800  
  - Learning rate: 0.0003 with decay  
  - Batch size: 32
  - Number of epochs: 550 with early stopping

- **Training Stability Techniques:**  
  - Gradient clipping (max_norm=0.5)
  - Adaptive learning rate scheduling (ReduceLROnPlateau)
  - NaN detection and recovery
  - Weight decay (1e-5) for regularization

- **Generation Process:**
  - Bimodal initialization with 90/10 split favoring high fraud amounts
  - Adaptive noise reduction during generation (noise scale decreases as t approaches 0)
  - Post-processing with quantile-based distribution matching
  - Feature-specific constraints for engineered features

## Citation

> [FraudDiffuse: Diffusion-aided Synthetic Fraud Augmentation for Improved Fraud Detection](https://dl.acm.org/doi/pdf/10.1145/3677052.3698658)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback regarding FraudFusion, please contact:
- Raghu Ram Sattanapalle – [sattanapalle.r@northeastern.edu](mailto:sattanapalle.r@northeastern.edu)
- Anish Rao - [rao.anish@northeastern.edu](mailto:rao.anish@northeastern.edu)

---
