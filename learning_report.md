# Learning Report: UAV RSRP Prediction Evolution

This report summarizes the transition from physics-based path loss models to deep learning and lightweight edge models for UAV-based 5G RSRP prediction.

## Experiment Configuration
- **Master Seed**: 42
- **Device**: cuda
- **Total Samples (Global)**: 9927
- **Test Samples (Global, Unseen)**: 1490

## 1. Physics Baseline: Line of Sight (FSPL)
- **Test RMSE**: 17.4918 dB
- **Inference Time**: 0.2393 ms/sample

## 2. Sample Scarcity & Convergence Speed
This section shows how much data is required for InceptionTime to reach a target accuracy of **5.0 dB RMSE**.

| Samples | Test RMSE | Epochs to Target | Train Time (s) |
|---------|-----------|------------------|----------------|
| 100 | 5.6304 | 36 | 56.34 |
| 500 | 4.7801 | 40 | 69.36 |
| 1000 | 4.4286 | 34 | 68.01 |
| 2000 | 6.9432 | 21 | 56.07 |
| 6948 | 4.5771 | 11 | 57.65 |

**100-Epoch Limit Run (All Samples):**
- **Test RMSE**: 3.6677 dB
- **Epochs Run**: 100
- **Train Time**: 519.13 s

## 3. Lightweight Edge Model: ELM
ELM results on the same global sample constraints.

| Samples | Test RMSE | Fit Time (s) |
|---------|-----------|--------------|
| 100 | 7.7501 | 0.0726 |
| 500 | 4.1769 | 0.1188 |
| 1000 | 3.9045 | 0.4265 |
| 2000 | 3.7208 | 3.2270 |
| 6948 | 3.8771 | 1.9139 |

## 4. Real-Time In-Flight Scenario
**Target Flight**: `Dataset_24_PawPrints_5G_30m_Flight_2.csv`
This experiment simulates a UAV collecting data and adapting its model mid-flight. Results are shown for the specific target flight and as an average across all 7 available flights.

### Target Flight: `Dataset_24_PawPrints_5G_30m_Flight_2.csv`
| % Train | Inc RMSE | ELM RMSE | Inc Train (s) | ELM Fit (s) | Inc Inf (ms) | ELM Inf (ms) |
|---------|----------|----------|---------------|-------------|--------------|--------------|
| 20% | 93.9088 | 37.9405 | 1.74 | 0.0346 | 0.3423 | 0.2162 |
| 40% | 86.7686 | 12.7268 | 3.45 | 0.0539 | 0.2904 | 0.1783 |
| 60% | 90.8372 | 5.8723 | 5.16 | 0.0804 | 0.3017 | 0.1799 |
| 80% | 58.7961 | 5.1637 | 6.89 | 0.1165 | 0.3388 | 0.1854 |

### Average Across All Flights
| % Train | Inc RMSE | ELM RMSE | Inc Train (s) | ELM Fit (s) | Inc Inf (ms) | ELM Inf (ms) |
|---------|----------|----------|---------------|-------------|--------------|--------------|
| 20% | 90.6054 | 29.9600 | 4.08 | 0.0767 | 0.2918 | 0.1840 |
| 40% | 81.2879 | 11.0304 | 8.02 | 0.9903 | 0.2813 | 0.1804 |
| 60% | 63.8385 | 4.9897 | 12.02 | 1.0452 | 0.2999 | 0.1852 |
| 80% | 36.1372 | 4.7115 | 15.91 | 1.9529 | 0.3021 | 0.1891 |

## 5. Visual Summary
![Real-Time Accuracy](plots/real_time_accuracy.png)
*Figure: Accuracy improvement as the UAV collects more data. Solid lines represent the target flight, dashed lines represent the average across all flights.*

## Reproducibility
To recreate these exact results, run the following command:
```bash
python src/aerpaw_processing/paper/learning_experiment.py --seed 42 --flight Dataset_24_PawPrints_5G_30m_Flight_2.csv
```
