# EDA Issue Log

Generated from an actual run against data/raw/Churn_Modelling.csv (10,000 rows).

- Rows: 10000, Columns: 14
- Duplicate rows: 0
- Duplicate CustomerId values: 0
- Missing values by column: none
- Target balance: Exited=0: 0.7963, Exited=1: 0.2037 (moderate class imbalance ~80/20)
- Geography values: ['France', 'Germany', 'Spain']
- Gender values: ['Female', 'Male']
- CreditScore range: 350-850
- Age range: 18-92
- Balance range: 0.00-250898.09 (36.2% of customers have a 0 balance)
- Churn rate by geography: {'France': 0.162, 'Germany': 0.324, 'Spain': 0.167}
- Churn rate by gender: {'Female': 0.251, 'Male': 0.165}

## Leakage check
RowNumber, CustomerId and Surname are unique per-row identifiers with no
predictive meaning about customer behavior; they are excluded from model
inputs (see src/churn_predictor/preprocessing.py). No other column in this
dataset represents information from *after* the observation point.
