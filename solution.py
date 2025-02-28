''' 
1st  problem  solution ''' 


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DataHandling:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train_test_splitting(self, test_size=0.2):
        x_sampled, y_sampled = self.imbalance_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_sampled, y_sampled, test_size=test_size, random_state=42
        )

    def imbalance_dataset(self, technique='adasyn', n_neighbors=5):
        from imblearn.over_sampling import ADASYN, SMOTE

        if technique.lower() == 'adasyn':
            sampling = ADASYN(n_neighbors=n_neighbors, random_state=42)
        elif technique.lower() == 'smote':
            sampling = SMOTE(k_neighbors=n_neighbors, random_state=42)
        else:
            raise ValueError("Invalid technique. Use 'adasyn' or 'smote'.")

        return sampling.fit_resample(self.x, self.y)

    def compute_class_weight(self, technique='using_library'):
        if technique.lower() == 'manually':
            class_weight_0 = len(self.y) / (2 * sum(self.y == 1))
            class_weight_1 = len(self.y) / (2 * sum(self.y == 0))
        elif technique.lower() == 'using_library':
            class_weights = compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
            class_weight_0, class_weight_1 = class_weights
        else:
            raise ValueError("Invalid technique. Use 'manually' or 'using_library'.")

        return class_weight_0, class_weight_1

    def best_threshold(self, model):
        y_predict_proba = model.predict_proba(self.x_test)[:, 1]
        thresholds = np.linspace(0.01, 0.90, 1000)

        scores = [f1_score(self.y_test, (y_predict_proba >= t).astype(int)) for t in thresholds]
        return thresholds[np.argmax(scores)]

    def make_classification_model(self):
        self.train_test_splitting()
        _0_class, _1_class = self.compute_class_weight()

        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, n_jobs=-1, class_weight={0: _0_class, 1: _1_class}
        )
        model.fit(self.x_train, self.y_train)

        best_thresh = self.best_threshold(model)
        y_predict = (model.predict_proba(self.x_test)[:, 1] >= best_thresh).astype(int)

        return f1_score(self.y_test, y_predict)

# Testing the implementation
from sklearn.datasets import make_classification

x, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_redundant=5,
    n_clusters_per_class=2, weights=[0.99, 0.01], flip_y=0.02, random_state=42
)

model_accuracy = DataHandling(x, y).make_classification_model()
print("Model F1 Score:", model_accuracy)

''' 2nd problem solution ''' 
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

x, y = make_classification(n_samples=10000, n_features=20, n_informative=10, 
                           n_redundant=5, weights=[0.995, 0.005], flip_y=0.02, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
x_train_resampled, y_train_resampled = smote_tomek.fit_resample(x_train, y_train)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
 RandomForestClassifier(n_estimators=300, class_weight=class_weight_dict, max_depth=5, random_state=42)
rf_model.fit(x_train_resampled, y_train_resampled)


base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ('xgb', XGBClassifier(scale_pos_weight=class_weights[1], use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(class_weight="balanced"))
]

meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5)
stacking_model.fit(x_train_resampled, y_train_resampled)

 = stacking_model.predict_proba(x_test)[:, 1]  # Get probability scores
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print("Precision-Recall AUC:", pr_auc)


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss=focal_loss(), metrics=['AUC'])
nn_model.fit(x_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_data=(x_test, y_test))

