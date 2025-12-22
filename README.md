# BGFMOD Appendices



## Appendix.1 Real-IAD

Specifically, we adopt the industrial defect detection datasets provided by Real-IAD , which captures product samples using multi-angle industrial cameras under standardized lighting and workstation conditions, covering a wide range of defect areas and anomaly types (e.g., missing, pit, deformation, scratch, foreign object, etc.).

We selected six typical industrial components, including Audiojack, Button_battery, Pcb, Regulator, Terminalblock, and Transistor, totaling 30,210 images. For each sample, we treated all five fixed shooting perspectives (top view + four symmetrical 45° horizontal angles) of the same sample as different views. Following the standard setup , the anomaly ratio was controlled at 15%."

**The feature extraction process is detailed below:**

**Backbone Network:** We utilize a ResNet-50 network pre-trained on ImageNet for feature extraction.

**Layer Selection:** Considering the requirements of anomaly detection tasks, we extract intermediate features from the third residual layer of ResNet-50. This strategy allows us to preserve local structures and texture patterns while maintaining rich semantic information.

**Pooling & Dimension:** Subsequently, we apply Global Average Pooling (GAP) to the output of this layer, obtaining a 1024-dimensional feature vector for each view. All comparative methods in our experiments use these processed multi-view features as input.

**Note on Evaluation Scope:** It should be emphasized that our focus is on unsupervised sample-level detection rather than pixel-level anomalies. Therefore, we do not utilize pixel-level ground-truth annotations or metrics (e.g., P-PRO) to avoid confusion with localization tasks. The implementation of the detection method remains consistent with the synthetic outlier experiments.

#### Real-IAD Benchmark Results (AUC)

| DataSet        | MLRA   | CRMOD  | LDSR   | dPoE   | MFSR   | MQPR   | SRLSP         | MODGD         | RCPMOD        | MODGF  | Ours          |
| :------------- | :----- | :----- | :----- | :----- | :----- | :----- | :------------ | :------------ | :------------ | :----- | :------------ |
| Audiojack      | 0.5126 | 0.7656 | 0.8290 | 0.6685 | 0.5221 | 0.5041 | 0.8381        | <u>0.9459</u> | 0.9341        | 0.8175 | **0.9537**    |
| Button_battery | 0.5497 | 0.7456 | 0.7751 | 0.5830 | 0.4808 | 0.5121 | 0.8005        | 0.8277        | **0.9565**    | 0.8633 | <u>0.8976</u> |
| Pcb            | 0.6112 | 0.6940 | 0.7998 | 0.5913 | 0.4879 | 0.4964 | 0.7816        | <u>0.8165</u> | 0.8093        | 0.7691 | **0.8883**    |
| Regulator      | 0.4619 | 0.8482 | 0.8120 | 0.5831 | 0.4752 | 0.4662 | 0.8428        | 0.8636        | <u>0.8717</u> | 0.8513 | **0.9854**    |
| Terminalblock  | 0.5771 | 0.7785 | 0.8851 | 0.5493 | 0.5771 | 0.5441 | <u>0.9201</u> | 0.8545        | 0.9142        | 0.9056 | **0.9750**    |
| Transistor1    | 0.6184 | 0.8478 | 0.8654 | 0.4812 | 0.4879 | 0.4902 | 0.8774        | <u>0.9618</u> | 0.5572        | 0.8557 | **0.9722**    |
| **Average**    | 0.5552 | 0.7800 | 0.8277 | 0.5761 | 0.5052 | 0.5022 | 0.8434        | <u>0.8783</u> | 0.8405        | 0.8438 | **0.9454**    |



## Appendix.2 Runtime Experiment

#### Runtime Analysis (Seconds) on Standard CPU

| Dataset     | RCPMOD      | MODGF      | Ours       |
| :---------- | :---------- | :--------- | :--------- |
| h1          | 1053.23     | 326.47     | **29.98**  |
| h2          | 1049.87     | 345.14     | **31.75**  |
| h3          | 1075.65     | 363.63     | **33.14**  |
| h4          | 1090.08     | 413.30     | **29.35**  |
| h5          | 1195.02     | 376.52     | **26.98**  |
| h6          | 1177.38     | 434.83     | **27.51**  |
| c1          | 617.61      | 756.55     | **206.55** |
| c2          | 937.17      | 2013.64    | **198.17** |
| c3          | 1072.35     | 838.98     | **212.41** |
| c4          | 1146.56     | 825.66     | **223.47** |
| c5          | 1053.46     | 832.60     | **187.73** |
| c6          | 1025.19     | 773.83     | **183.90** |
| y1          | 9.09        | 26.34      | **4.35**   |
| y2          | 8.41        | 25.20      | **4.16**   |
| y3          | 8.79        | 25.20      | **4.21**   |
| y4          | 12.14       | 25.03      | **4.26**   |
| y5          | 9.79        | 24.94      | **4.48**   |
| y6          | 9.48        | 25.26      | **4.51**   |
| m1          | 3905.93     | 410.28     | **23.12**  |
| m2          | 4354.81     | 409.47     | **22.97**  |
| m3          | 3276.46     | 408.45     | **23.18**  |
| m4          | 3269.89     | 408.90     | **23.46**  |
| m5          | 3300.46     | 428.64     | **22.72**  |
| m6          | 3225.46     | 366.60     | **21.87**  |
| **Average** | **1411.85** | **453.56** | **69.76**  |

To evaluate the computational efficiency, we recorded the execution time of BGFMOD against the comparative methods across all datasets. The results are summarized in Table X. 

**Experimental Setup:** For the deep learning baseline (RCPMOD), considering the lack of standardized computational specifications in its original implementation, we performed the experiments on a machine equipped with an **NVIDIA RTX 3090 GPU** to prevent hardware bottlenecks. Other methods were executed under standard environment settings. 

**Results:** BGFMOD demonstrates significant algorithmic efficiency, with an average runtime of **69.76 seconds**. This represents a speedup of approximately **6.5x** over the graph-based MODGF (453.56s) and **20.2x** over the deep learning-based RCPMOD (1411.85s). The efficiency gains are attributed to our closed-form eigen-decomposition steps, which avoid the high computational cost of iterative gradient descent required by deep models.

## Appendix.3 Proofs of Convergence and  Converge Discussion

<img src="BGFMOD\fig\funciton.png" alt="image-20251217102039741" style="zoom:50%;" />

**Monotonicity:** This validates our *Alternating  Optimization* strategy. Since each iteration minimizes a quadratic sub-problem via an exact closed-form eigen-solution, the objective is mathematically guaranteed to be non-increasing.

**Efficiency:** The curve drops sharply in the first 5 iterations and converges to a stationary point within approximately 15 iterations. This high efficiency is attributed to the robust initialization using the consensus graph spectral basis, which places the optimization starting point close to the target manifold."
