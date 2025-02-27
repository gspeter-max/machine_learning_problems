''' 
1 problem  solution ''' 


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
 
