# GEMINI.md - Context & Instructions for SpO₂ Project (rPPG-Toolbox Edition)

## 🤖 AI Role & Objective
**Role:** You are an expert Biomedical AI Engineer specializing in Deep Learning for rPPG (Remote Photoplethysmography).
**Objective:** Develop a Python pipeline using the `rppg-toolbox` for data handling and `PhysNet` for modeling.
**Core Challenge:** Implement and compare two approaches:
1.  **Baseline:** Standard PhysNet (Data-driven).
2.  **Constrained:** PhysNet with biological constraints (Range & Smoothness).

---

## 🛠️ Project Architecture

### 1. Technology Stack
* **Core Framework:** `rppg-toolbox` (for Preprocessing & Data Loading).
* **Deep Learning:** PyTorch (v2.0+).
* **Model Architecture:** PhysNet (3D-CNN).
* **Data Format:** The toolbox outputs `.npy` chunks (Shape: `[Channels, Frames, Height, Width]`).

### 2. Dataset Structure (CRITICAL)
The code must be configured to read from this specific directory layout:
```text
project_root/
├── data/                       # Root Data Directory
│   ├── subject1/               # Subfolder per subject
│   │   ├── vid.avi             # Video file
│   │   └── ground_truth.txt    # Text file containing SpO2/BVP values
│   ├── subject2/
│   │   ├── ...
```


### 2. Pipeline Modules

#### A. `preprocess_config.yaml` (The "Eyes")
* **Goal:** Configuration file to run the rPPG-Toolbox preprocessing script.
* **Settings:**
    * Dataset: UBFC-rPPG.
    * Data Root Path: "/data"
    * Process: Face Crop $\rightarrow$ DiffNormalized $\rightarrow$ Chunking (180 frames).
    * Output: Saved `.npy` files in a cached directory.

#### B. `models_custom.py` (The "Brain")
Must define two model wrappers based on the standard PhysNet:
1.  **`PhysNetBaseline`:**
    * Imports standard `PhysNet` from `rppg-toolbox`.
    * Returns raw model output.
2.  **`PhysNetConstrained`:**
    * Inherits from `PhysNet`.
    * **Constraint 1 (Range):** Applies a custom activation on the final layer: $Output = 80 + 20 \times \text{Sigmoid}(x)$. (Forces 80-100% range).
    * **Constraint 2 (Smoothness):** Returns the temporal signal to allow the loss function to penalize jumps.

#### C. `loss_custom.py` (The Constraints)
Defines `PhysiologicalLoss` class:
* **Inputs:** Predictions ($Pred$), Ground Truth ($GT$).
* **Formula:** $L = L_{MSE} + \lambda_{smooth} L_{diff}$
* **$L_{MSE}$:** Standard Mean Squared Error.
* **$L_{diff}$ (Smoothness):** Calculate the first-order derivative (difference between frame $t$ and $t+1$) and minimize it to prevent jagged noise.

#### D. `train_hackathon.py` (The Execution)
A custom training loop that bypasses the complex Toolbox `main.py` to allow for our custom constraints.
* **Loader:** Uses `rppg_toolbox.dataset.UBFC` to load the `.npy` files.
* **Optimizer:** `AdamW` with Learning Rate `1e-4`.
* **Loop:** * Forward pass.
    * Calculate `PhysiologicalLoss`.
    * Backward pass.
    * Save best model based on MAE.

---

## 💻 CLI Generation Prompts
*Use these prompts with the Gemini CLI to generate the specific files.*

### Command 1: Generate Preprocessing Config
> "Using the context in GEMINI.md, write the `preprocess_config.yaml` file. It should be configured for the UBFC-rPPG dataset located at ./data. It must use 'DiffNormalized' data type, enable face cropping, and chunk the video into 180-frame segments."

### Command 2: Generate Custom Models
> "Using the context in GEMINI.md, write models_custom.py. It should import PhysNet from neural_methods.model.PhysNet. Create a PhysNetConstrained class that inherits from PhysNet. Overwrite the final layer to be a nn.Linear or nn.Conv1d that outputs a single channel (SpO2). In the forward pass, apply a scaled Sigmoid function 80 + 20 * sigmoid(x) to this output to strictly bound predictions between 80 and 100."

### Command 3: Generate Custom Loss
> "Using the context in GEMINI.md, write `loss_custom.py`. Create a `PhysiologicalLoss` class that calculates MSE and adds a temporal smoothness penalty (L1 norm of the difference between consecutive time steps)."

### Command 4: Generate Training Loop
> "Using the context in GEMINI.md, write train_hackathon.py. It should import the dataset loader from rppg-toolbox. Important: Configure the loader or the training loop to use SpO2 values (percentage) as the Ground Truth labels, NOT the pulse waveform. Run a training loop using PhysNetConstrained and PhysiologicalLoss. Include a validation step that prints MAE"

---

## 📊 Evaluation Logic
The `train_hackathon.py` script must output:
1.  **Epoch Loss:** (Training stability).
2.  **Val MAE:** (Accuracy).
3.  **Visualization:** (Optional) Save a plot of one batch's Prediction vs Ground Truth to see if the "Smoothness" constraint is working.