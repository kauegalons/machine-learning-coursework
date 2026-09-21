# Machine Learning Coursework

Two supervised and unsupervised learning exercises in Python, originally handed in
as the two halves of a single Intelligent Systems assignment: customer
segmentation with K-Means and wine quality classification with Random Forest.

Each folder holds a notebook together with the dataset it consumes, so the
notebooks run without any external storage.

## Contents

| Folder | Task | Algorithm | Dataset |
| --- | --- | --- | --- |
| `clustering/` | customer segmentation | K-Means | Wholesale customers, 440 rows |
| `classification/` | wine quality prediction | Random Forest | WineQT, 1143 rows |

## Requirements

- Python 3
- jupyter
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- imbalanced-learn, used for SMOTE in the classification notebook

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install jupyter pandas numpy scipy scikit-learn matplotlib imbalanced-learn
```

## Running

Open either notebook from inside its own folder, since datasets are referenced by
relative path.

```bash
cd clustering
jupyter notebook Cluster_Training.ipynb
```

```bash
cd classification
jupyter notebook Classification_Training.ipynb
```

---

## clustering — Customer segmentation with K-Means

The Wholesale customers dataset describes annual spending across six product
categories, plus two categorical attributes.

| Attribute | Type |
| --- | --- |
| `Channel` | categorical, sales channel |
| `Region` | categorical, region |
| `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen` | numeric, annual spending |

### Pipeline

1. `Channel` and `Region` are one-hot encoded into `channel_1`, `channel_2`,
   `region_1`, `region_2` and `region_3`.
2. The six numeric columns are scaled with `MinMaxScaler`. The fitted scaler is
   persisted as `normalizador1.pkl` so that new observations can be transformed
   the same way later.
3. Scaled numeric columns and the dummy columns are concatenated into the final
   eleven-feature training matrix.

### Choosing the number of clusters

Rather than reading the elbow off the chart by eye, the notebook determines it
numerically. It fits K-Means for every `k` from 1 to 100, recording the distortion
as the mean minimum Euclidean distance from each point to its centroid. It then
draws the straight line between the first and last points of the distortion curve
and computes the perpendicular distance from every point to that line. The `k`
with the largest distance is the elbow, and that value is used for the final
model.

The distortion curve is saved as `elbow_distorcao.png`.

### Final model and interpretation

The definitive `KMeans` is fitted with `random_state=42` and persisted as
`Wholesale_Customer_cluster_2024.pkl`.

The last part of the notebook exists to make the result readable. It takes an
observation, predicts its cluster, retrieves that cluster's centroid, and then
reverses both transformations: `inverse_transform` restores the numeric columns to
their original spending scale, and an `undummify` helper collapses the one-hot
columns back into single `Channel` and `Region` values. The output is a centroid
expressed in the same units as the source data instead of normalised values.

---

## classification — Wine quality with Random Forest

The WineQT dataset holds eleven physicochemical measurements per wine sample, a
`quality` score used as the label, and an `Id` column that is discarded.

| Attribute group | Columns |
| --- | --- |
| Acidity | `fixed acidity`, `volatile acidity`, `citric acid`, `pH` |
| Composition | `residual sugar`, `chlorides`, `density`, `sulphates`, `alcohol` |
| Sulphur | `free sulfur dioxide`, `total sulfur dioxide` |
| Label | `quality` |

### Pipeline

1. `quality` and `Id` are dropped from the feature matrix, and `quality` becomes
   the target.
2. Quality scores are heavily imbalanced in this dataset, so **SMOTE** synthesises
   minority-class samples to balance the classes before training.
3. A `RandomForestClassifier` is trained on a 70/30 split of the balanced data.

### Evaluation

Two evaluations are run. A single hold-out split gives a provisional accuracy, and
then a ten-fold cross-validation measures macro precision and macro recall, which
is the more trustworthy figure because it does not depend on one lucky split.

| Metric | Value recorded in the notebook |
| --- | --- |
| Hold-out accuracy, 30% test | 0.8644 |
| Macro precision, 10-fold CV | 0.8054 |
| Macro recall, 10-fold CV | 0.8145 |

The gap between the hold-out accuracy and the cross-validated figures is the
expected outcome: a single split flatters the model, and averaging over ten folds
brings it back to a realistic level.

A confusion matrix is rendered with `ConfusionMatrixDisplay` to show which quality
scores get confused with which.

### Hyperparameter search

A randomized search explores the number of trees, maximum depth, maximum features,
minimum samples per split and per leaf, and bootstrap sampling. The best
combination found was:

```python
{'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt',
 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
```

Trained models are persisted as `wine_quality_forest.pkl`,
`wine_quality_forest_class.pkl` and `wine_quality_forest_cross_class.pkl`.

---

## Notes on reproducing

The notebooks were originally written in Google Colab and read their data from a
mounted Google Drive. They now read from the dataset sitting next to them, and the
Drive mount has been removed, so they run in any local Jupyter installation.

Pickled models, the elbow figure and Jupyter checkpoints are generated by running
the notebooks and are excluded through `.gitignore`, so only source notebooks and
datasets are versioned.

Two details worth knowing before running the classification notebook on an
up-to-date environment. The hyperparameter grid includes `max_features = 'auto'`,
which was removed from `RandomForestClassifier` in scikit-learn 1.3 and now raises
an error; `'sqrt'` is the replacement, and it is already the value the search
selected. And an early cell fits a `MinMaxScaler` over the full dataframe whose
result is never used downstream, which is harmless here because tree ensembles do
not require feature scaling.

## License

Academic work, shared for reference.
