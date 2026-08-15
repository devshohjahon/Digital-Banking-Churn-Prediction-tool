# Error Analysis - random_forest_balanced

## Held-out test set: 2000 customers

- False negatives (missed at-risk customers): 130
- False positives (flagged customers who stayed): 227

## False-negative pattern (predicted stay, actually left)
CreditScore         648.14
Age                  37.36
Balance           85081.81
NumOfProducts         1.33
IsActiveMember        0.45

## False-positive pattern (predicted leave, actually stayed)
CreditScore          644.92
Age                   45.61
Balance           104729.27
NumOfProducts          1.28
IsActiveMember         0.45

## Threshold note
The model outputs a churn probability, not just a 0/1 label. The default cutoff is 0.5. Since missing an at-risk customer is the costlier mistake, a lower cutoff (e.g. 0.35-0.4) trades some extra false positives for higher recall, and can be adjusted without retraining the model.
