# 🍎 UROB HW3 - Fruit Image Analysis

This repository contains the code for the third homework assignment of the **UROB course at CTU**. The task involves training a neural network to:
- 🏷️ **Classify** images of fruits into 30 different categories
- 🎨 **Segment** fruits with pixel-level masks
- 🔍 **Learn** meaningful image embeddings

---

## 📁 Repository Structure

```
hw3_students/
├── confs/               # Configuration files for the project
├── data/                # Directory to store datasets (create after downloading)
├── model.py             # Neural network architecture definition
├── train.py             # Model training script
├── train_job.sbatch     # SLURM job script for cluster training
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

> 💡 **Note:** For cluster training details, see the [Course Page](https://urob-ctu.github.io/docs/)

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone git@github.com:urob-ctu/hw3-cnns.git
```

### 2️⃣ Set up your environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Download the dataset
Download the dataset and place it in the `data/` directory: [Dataset Link](https://drive.google.com/file/d/1cnD0vsuPSM-lkV4zqCxvXYihuvc6s7PF/view?usp=sharing)

### 4️⃣ Configure your experiment
Modify the configuration file in `confs/config.yml` as needed.

### 5️⃣ Train the model
- Define your model in `model.py`
- Fill the missing lines in `train.py` (look for ‼️‼️‼️‼️) [or write your own training script but then submit this training script as `train.py`]
- Run the training script:
```bash
python train.py
```

### 6️⃣ (Optional) Use SLURM cluster
If using a SLURM cluster, submit the job using:
```bash
sbatch train_job.sbatch
```

---

## ✅ Your Tasks

1. 🏗️ **Implement** the neural network architecture in `model.py`
   - must be named `MyModel` and accept `output_size` as an argument (output_size should be 30 for 30 fruit classes)
   - `forward` method must return three outputs:
     - class logits (shape: `[batch_size, 30]`)
     - segmentation mask (shape: `[batch_size, 1, 64, 64]`)
     - image embeddings (shape: `[batch_size, embedding_dim]`)
    - `get_embedding` method must return single output:
      - image embeddings (shape: `[batch_size, embedding_dim]`)
2. 🔄 **Complete** the training loop in `train.py` (look for ‼️‼️‼️‼️)
3. ⚙️ **Tune** hyperparameters in `confs/config.yml` for optimal performance
   - use tensorboard to monitor training progress: `tensorboard --logdir {path_to_logs}`

---

## 📊 Evaluation

### 🎯 Basic Evaluation (10 points)

This homework is worth **10 points**, distributed as follows:

| Task | Metric | Threshold | Points |
|------|--------|-----------|--------|
| **🍊 Fruit Classification** | Accuracy | 80% | 1 pt |
| | | 85% | 2 pts |
| **🎨 Segmentation Mask** | Mean IoU | 75% | 1 pt |
| | | 80% | 2 pts |
| | | 85% | 3 pts |
| **📈 Image Embeddings (ROC)** | AUC | 0.80 | 1 pt |
| | | 0.85 | 2 pts |
| **🎯 Image Embeddings (TPR)** | TPR @ 5% FPR | 0.75 | 1 pt |
| | | 0.80 | 2 pts |
| | | 0.85 | 3 pts |

### 🏆 Tournament Evaluation (up to +5 bonus points)

In addition to the basic evaluation, there will be a **tournament-style evaluation** where models are ranked based on their performance across all four tasks. 

**Scoring System:**
- Your final score is determined by the **sum of ranks** in each task
- **Lower total rank = Better score**
- In case of a tie, the model with the **earlier submission time** ranks higher

**Bonus Points:**
- 🥇 **Winner:** +5 points
- 🥈 **2nd place:** +4 points
- 🥉 **3rd place:** +3 points
- 🏅 **Top 10:** +2 points
- 🎖️ **Top 20:** +1 point

> **Maximum Score:** 15 points (10 from basic evaluation + 5 from tournament)

---


## 📦 Submission

Submit your code as a **zip file** containing:

- ✅ `model.py` - Your neural network architecture
- ✅ `train.py` - Your training script
- ✅ `weights.pth` - Trained model weights (state_dict)

> **Loading format:** Weights are loaded using:
> ```python
> model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
> ```

---

## 🚫 Important Policies

### 🤖 Pretrained Models Policy
❌ You are **not allowed** to use any pretrained models or transfer learning techniques.  
✅ All models **must be trained from scratch** using the provided dataset and your training code that you submit.

### 📝 Plagiarism Policy
✅ You are **strongly encouraged** to discuss ideas and approaches with your peers.  
⚠️ However, the code you submit **must be your own work**.  
❌ Copying code from others is **strictly prohibited** and will result in a **zero score** for the assignment.

---

<div align="center">
  
**Good luck! 🍀**

**In case of any questions, feel free to write me at hlavsja3@fel.cvut.cz and I will ask my LLM🔮**

</div>
