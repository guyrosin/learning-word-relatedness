import logging
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

import utils

classifiers = {
    'logistic_regression': LogisticRegression(),
    'svm': SVC(class_weight='balanced', probability=True),
    'linear_svm': SVC(kernel='linear', C=0.025, class_weight='balanced', probability=True),
    'svm_rbf': SVC(gamma=2, C=1.0, probability=True),
    'mlp': MLPClassifier(alpha=0.01, solver='adam'),
    'rf': RandomForestClassifier(class_weight='balanced', n_estimators=1000, n_jobs=-1)
}


class Classifier:
    def __init__(self, X, y, classifier_type='svm'):
        self.X = X
        self.y = y
        self.classifier_type = classifier_type
        self.classifier = classifiers[classifier_type]
        self.colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        self.line_width = 2
        self.folds_num = 10
        # cross validation (stratified, to preserve the percentage of samples for each class)
        self.cv = StratifiedKFold(n_splits=self.folds_num, shuffle=True)

    def evaluate(self):
        logging.info('Evaluating {} classifier'.format(self.classifier_type))
        accuracy_sum = 0
        recall_sum = 0
        precision_sum = 0
        f1_sum = 0
        roc_auc_sum = 0
        for fold_index, ((train_index, test_index), color) in enumerate(zip(self.cv.split(self.X, self.y), self.colors), start=1):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.train(X_train, y_train)

            y_pred = list(self.classifier.predict(X_test))  # predicted classes
            probs = list(self.classifier.predict_proba(X_test))  # probabilities for the true class
            y_prob = np.array(probs)[:, 1]

            # Compute various metrics
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_sum += accuracy
            recall = recall_score(y_test, y_pred)
            recall_sum += recall
            precision = precision_score(y_test, y_pred)
            precision_sum += precision
            f1 = f1_score(y_test, y_pred)
            f1_sum += f1
            logging.info("Accuracy: %.2f", accuracy)
            logging.info("Recall: %.2f", recall)
            logging.info("Precision: %.2f", precision)
            logging.info("F-measure: %.2f", f1)

            # Compute ROC curve and AUC
            if len(y_test) == len(y_prob):
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_auc_sum += roc_auc
                logging.info("AUC: %.2f", roc_auc)
                plt.plot(fpr, tpr, lw=self.line_width, color=color,
                         label='ROC fold %d (area = %.2f)' % (fold_index, roc_auc))

        logging.info("Avg. accuracy: %.2f", accuracy_sum / self.folds_num)
        logging.info("Avg. recall: %.2f", recall_sum / self.folds_num)
        logging.info("Avg. precision: %.2f", precision_sum / self.folds_num)
        logging.info("Avg. F-measure: %.2f", f1_sum / self.folds_num)
        if roc_auc_sum > 0:
            logging.info("Avg. AUC: %.2f", roc_auc_sum / self.folds_num)
            utils.show_roc_graph(show_legend=False)

    def cross_validate(self):
        for (train_index, test_index), color in zip(self.cv.split(self.X, self.y), self.colors):
            # remove unknowns from the training data
            train_index = [idx for idx in train_index if self.y[idx] >= 0]
            X_train = self.X[train_index]
            y_train = self.y[train_index]
            classifier = SVC(class_weight='balanced', probability=True)
            model = classifier.fit(X_train, y_train)
            yield model, test_index

    def train(self, X_train=None, y_train=None):
        if X_train is None and y_train is None:
            X_train = self.X
            y_train = self.y
        self.classifier.fit(X_train, y_train)
        return self.classifier
