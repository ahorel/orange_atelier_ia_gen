# Copilot Instructions - ML Price Inconsistency Workshop

## Project Overview

Educational workshop demonstrating XGBoost-based price inconsistency detection between EVOLLIS and HYBRIS order systems. The project progresses through 4 stages (ETAPE1-4) showing evolution from rule-based ML to probabilistic models with warm-start training.

**Core Business Logic**: Detect which system (EVOLLIS or HYBRIS) has pricing errors by comparing `total_price` against `quantity × unit_price`.

## Architecture & Data Flow

### Pipeline Stages (Sequential Learning Progression)

1. **ETAPE1**: Supervised ML with rule-based labels
   - Uses deterministic rules to label training data
   - Dual model: XGBClassifier (origin) + XGBRegressor (price correction)
   - Status detection via hard rules, ML learns from rule outputs

2. **ETAPE2**: Weak supervision (no hard origin rules)
   - Replaces certain rules with probabilistic predictions
   - Introduces `gap_evollis` and `gap_hybris` features
   - Predicts probabilities: `p_origin_evollis`, `p_origin_hybris`

3. **ETAPE3**: Augmented training data
   - Generates synthetic data via `ETAPE3_filesData_generator.py`
   - Calls `gen_hyb_prod.generate_files()` on init
   - Same model as ETAPE2 but with more training examples

4. **ETAPE4**: Warm-start training
   - Continues training from saved model (`xgb_origin_model.json`)
   - Uses `xgb_model=self.model_path` in XGBClassifier.fit()
   - Incremental learning without retraining from scratch

### Key Files & Naming Conventions

- **Input CSVs**: `{evollis,hybris}_products.csv` (base), `*_V2_01012026.csv` (augmented)
- **Output CSVs**: `ETAPE{N}_*.csv` (predictions for each stage)
- **Model Persistence**: `xgb_origin_model.json` (XGBoost native format)
- **Comments**: `commentaires/*.txt` (French explanations for each stage)

## Critical Code Patterns

### Feature Engineering (Standard Across Stages)

```python
self.FEATURES = [
    "quantity", "unit_price_EVOLLIS", "unit_price_HYBRIS",
    "total_price_EVOLLIS", "total_price_HYBRIS",
    "gap_evollis", "gap_hybris"  # Added in ETAPE2+
]

# Gap signals (weak indicators, not deterministic)
df["gap_evollis"] = abs(df["total_price_EVOLLIS"] - df["expected_price_evollis"])
df["gap_hybris"] = abs(df["total_price_HYBRIS"] - df["expected_price_hybris"])
```

### Training Data Selection (Only KO Cases)

```python
df_train = self.df[self.df["is_incoherent"] == 1]  # ETAPE2+
# OR
df_train = self.df[self.df["status"] == "KO"]      # ETAPE1
```

**Rationale**: ML only predicts when rules detect inconsistency. Never trains on OK cases.

### Warm-Start Pattern (ETAPE4)

```python
if os.path.exists(self.model_path):
    self.clf_origin.fit(X, y, xgb_model=self.model_path)
else:
    self.clf_origin.fit(X, y)
self.clf_origin.save_model(self.model_path)
```

### Data Merging Strategy

```python
self.df = self.df_evollis.merge(
    self.df_hybris,
    on=["order_id", "quantity", "product_name", "order_date"],
    suffixes=("_EVOLLIS", "_HYBRIS")
)
```

**Important**: Merge creates columns like `unit_price_EVOLLIS`, `total_price_HYBRIS`.

## Development Workflow

### Running Individual Stages

```powershell
python ETAPE1_modelML_supervise.py       # Baseline supervised
python ETAPE2_modelML_noRuleBased.py     # Probabilistic
python ETAPE3_modelML_augmentedTrainingFile.py  # + synthetic data
python ETAPE4_modelML_warmStart.py       # Warm-start training
```

Each script:
- Reads `{evollis,hybris}_products*.csv`
- Outputs `ETAPE{N}_*.csv` with predictions
- Prints preview of key columns (order_id, product_name, predictions)

### Data Generation (ETAPE3)

```python
import ETAPE3_filesData_generator as gen_hyb_prod
gen_hyb_prod.generate_files("evollis")  # Creates ETAPE3_evollis_products.csv
gen_hyb_prod.generate_files("hybris")   # Creates ETAPE3_hybris_products.csv
```

**Note**: Generator has incomplete/broken functions (`create_evollis_products_csv` logic is partial). Focus on `generate_files()` as entry point.

## CSV Format Conventions

**Separator**: `;` (semicolon)  
**Date Format**: `YYYY-MM-DD`  
**Core Columns**:
```
order_id;quantity;product_name;order_date;unit_price;total_price
```

**Output Columns** (vary by stage):
- ETAPE1: `status`, `error_origin`, `predicted_origin_label`, `corrected_price`
- ETAPE2-4: `is_incoherent`, `p_origin_evollis`, `p_origin_hybris`, `predicted_origin`

## XGBoost Hyperparameters

```python
# Classification (origin detection)
XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    objective="multi:softprob",  # Changed from softmax in ETAPE2+
    num_class=2,
    eval_metric="mlogloss",
    random_state=42
)

# Regression (price correction - ETAPE1 only)
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)
```

## Common Issues & Solutions

1. **CSV Encoding**: Use `sep=";"` always, avoid assuming comma delimiters
2. **Model Loading**: Check `os.path.exists(model_path)` before warm-start
3. **Feature Count Mismatch**: Ensure FEATURES list matches training/prediction data
4. **Data Generator**: `ETAPE3_filesData_generator.py` has incomplete functions - verify before use
5. **Missing Data**: ETAPE3 generator intentionally creates None values (10% probability)

## Testing Strategy

No formal test suite. Validation via:
- Visual inspection of output CSV predictions
- Comparing predicted vs expected origin labels
- Checking probability distributions (`p_origin_*` columns)
- French commentary files in `commentaires/` explain expected behaviors

## Language & Comments

**Code**: Python with French variable names and comments  
**Documentation**: Mix of French (README, commentaires) and English (code structure)  
**Key Terms**:
- `incohérence` = inconsistency
- `prix unitaire` = unit price
- `entrainement` = training
- `à chaud` = warm-start

## Extension Points

When adding new stages:
1. Follow `ETAPE{N}_*` naming convention
2. Inherit feature engineering from prior stages
3. Document changes in `commentaires/commentaires_*.txt`
4. Update CSV output with new prediction columns
5. Preserve existing FEATURES for model compatibility
