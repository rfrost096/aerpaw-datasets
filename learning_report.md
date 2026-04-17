# Learning Report: UAV RSRP Prediction Evolution

This report summarizes the transition from physics-based path loss models to deep learning and lightweight edge models for UAV-based 5G RSRP prediction.

## Experiment Configuration
- **Master Seed**: 42
- **Device**: cuda
- **Total Samples (Global)**: 9927
- **Test Samples (Global, Unseen)**: 1490

## 1. Physics Baseline: Line of Sight (FSPL)
- **Test RMSE**: 17.4918 dB
- **Inference Time**: 0.2394 ms/sample

## 2. Sample Scarcity & Convergence Speed
This section shows how much data is required for InceptionTime to reach a target accuracy of **5.0 dB RMSE**.

| Samples | Test RMSE | Epochs to Target | Train Time (s) |
|---------|-----------|------------------|----------------|
| 100 | 18.6126 | 25 | 45.42 |
| 500 | 11.8592 | 25 | 51.12 |
| 1000 | 9.4930 | 25 | 58.33 |
| 2000 | 5.0733 | 23 | 65.59 |
| 6948 | 4.9816 | 11 | 67.40 |

## 3. Lightweight Edge Model: ELM
ELM results on the same global sample constraints.

| Samples | Test RMSE | Fit Time (s) |
|---------|-----------|--------------|
| 100 | 7.7501 | 0.1314 |
| 500 | 4.1769 | 0.1755 |
| 1000 | 3.9045 | 0.5495 |
| 2000 | 3.7208 | 3.6117 |
| 6948 | 3.8771 | 2.6500 |

## 4. Real-Time In-Flight Scenario
**Target Flight**: `Dataset_24_PawPrints_5G_30m_Flight_2.csv`
This experiment simulates a UAV collecting data and adapting its model mid-flight. At each stage, the model is trained on the first X% of the flight and tested on the remaining (unseen) path.

| % Train | Samples | Inc RMSE | ELM RMSE | Inc Train (s) | ELM Fit (s) | Inc Inf (ms) | ELM Inf (ms) |
|---------|---------|----------|----------|---------------|-------------|--------------|--------------|
| 20% | 125 | 96.2048 | 37.9405 | 1.84 | 0.0404 | 0.4370 | 0.2827 |
| 40% | 251 | 97.5845 | 12.7268 | 3.70 | 0.0833 | 0.4547 | 0.2658 |
| 60% | 376 | 90.7486 | 5.8723 | 5.54 | 0.1193 | 0.4786 | 0.2882 |
| 80% | 502 | 23.9381 | 5.1637 | 7.41 | 0.1777 | 0.5381 | 0.2874 |

## 5. Visual Summary
![Real-Time Accuracy](plots/real_time_accuracy.png)
*Figure: Accuracy improvement as the UAV collects more data from the specific flight path.*

## Reproducibility
To recreate these exact results, run the following command:
```bash
python src/aerpaw_processing/paper/learning_experiment.py --seed 42 --flight Dataset_24_PawPrints_5G_30m_Flight_2.csv
```
