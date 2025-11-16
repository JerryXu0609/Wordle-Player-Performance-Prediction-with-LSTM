# Wordle Daily Player Statistics Dataset

This dataset contains daily aggregated Wordle player performance statistics for the year 2022, including guess distributions, success rates, difficulty indicators, and metadata about each puzzle. It is designed for time series forecasting, player performance modeling, and behavioral analysis using deep learning models such as LSTM, BiLSTM, or Transformer-based architectures.

The dataset has been preprocessed and normalized to ensure usability in machine learning workflows, especially for sequence modeling tasks that require sliding-window sampling.

---

## 📝 Dataset Summary

- **Range:** 2022-01-07 to 2022-12-31  
- **Total daily records:** 359  
- **Usable time-series sequences (7-day window):** 352  
- **Features:** guess distribution, mean attempts, success rate, hard-mode ratio, daily metadata  
- **Task Types:**  
  - Time series forecasting  
  - Regression  
  - Behavioral trend analysis  
  - Sequence modeling  

This dataset is appropriate for research involving Wordle difficulty prediction, guess behavior modeling, or understanding cognitive/gameplay trends.

---

## 📂 Dataset Structure

### **Data Example**
```json
{
  "Date": "2022-12-31",
  "Contest number": 560,
  "Word": "manly",
  "Number of reported results": 20380,
  "Number in hard mode": 1899,
  "1 try": 0.00,
  "2 tries": 0.02,
  "3 tries": 0.17,
  "4 tries": 0.37,
  "5 tries": 0.29,
  "6 tries": 0.12,
  "7+ tries (X)": 0.02,
  "mean_tries": 4.38,
  "success_rate": 0.97,
  "hard_mode_pct": 0.093,
  "year": 2022,
  "month": 12,
  "day": 31,
  "weekday": 5,
  "word_len": 5
}
```

------

## **📑 Data Fields**

| **Field**                  | **Type** | **Description**                          |
| -------------------------- | -------- | ---------------------------------------- |
| Date                       | string   | ISO-formatted date                       |
| Contest number             | int      | Wordle puzzle index                      |
| Word                       | string   | Daily solution word                      |
| Number of reported results | int      | Number of submitted game results         |
| Number in hard mode        | int      | Hard mode players                        |
| 1 try – 6 tries            | float    | Proportion of players solving in N tries |
| 7+ tries (X)               | float    | Failure rate                             |
| mean_tries                 | float    | Weighted average number of attempts      |
| success_rate               | float    | Percentage of successful players         |
| hard_mode_pct              | float    | Percentage of hard-mode players          |
| year, month, day, weekday  | int      | Date metadata                            |
| word_len                   | int      | Length of the solution word              |

------

## **🔧 Preprocessing Pipeline**

The following preprocessing steps were applied:

1. **Date normalization** (ISO-8601)
2. **Missing value processing**
3. **Feature engineering**:
   - mean_tries
   - success_rate
   - hard_mode_pct
   - weekday, word length
4. **Sliding-window sequence generation**
   - Input: 7 previous days
   - Target: next-day mean attempts
5. **Train/Validation/Test split**
   - Chronological split to avoid data leakage

These steps make the dataset suitable for LSTM/Transformer time series forecasting.

------

## **📁 Project Structure**

```
wordle-player-stats/
│
├── data.csv                   # Main dataset
├── preprocessed/              # Optional: sliding windows, scalers, etc.
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│
├── README.md                  # Dataset Card (this document)
└── license.txt                # License
```

------

## **🚀 Usage Example**

You can load the dataset directly with HuggingFace:

```
from datasets import load_dataset

dataset = load_dataset("your-username/wordle-player-stats")

print(dataset["train"][0])
```

### **Example: Build a sliding window time series**

```
import numpy as np

def create_window(seq, window=7):
    X, y = [], []
    for i in range(len(seq) - window):
        X.append(seq[i:i+window])
        y.append(seq[i+window])
    return np.array(X), np.array(y)

mean_tries = dataset["train"]["mean_tries"]
X, y = create_window(mean_tries)
```

------

## **📊 Benchmark Task: LSTM Forecasting**

This dataset supports:

- LSTM regression
- Multivariate time series forecasting
- Difficulty trend prediction
- Attention mechanism interpretability
- Data-driven analysis of puzzle difficulty

Example architecture:

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(32, input_shape=(7, 3)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
```

------

## **🔒 License**

Recommended license:

**CC BY-NC 4.0** — for non-commercial academic use.

Original Wordle data originates from *The New York Times*; redistribution should credit the data source.

------

## **📚 Citation**

If you use this dataset, please cite:

```
@dataset{wordle_stats_2022,
  title        = {Wordle Daily Player Statistics Dataset},
  author       = {Your Name},
  year         = {2024},
  url          = {https://huggingface.co/datasets/your-username/wordle-player-stats},
  note         = {Daily aggregated statistics of Wordle player performance in 2022.}
}
```

And acknowledge the original data:

```
The New York Times. (2022). Wordle Game Statistics Dataset.
```

------

## **❤️ Acknowledgements**

This dataset is built using publicly shared Wordle statistics.

We thank the Wordle community and The New York Times for making the game and its global participation possible.

------

