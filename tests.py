import pytest
from sklearn import svm, ensemble
from sklearn.neural_network import MLPClassifier

from project_classes import Dataset, Correlation, pd


class TestDataset:
    first_frame = pd.read_csv('train_dataset.csv')
    second_frame = pd.read_csv('train_salaries.csv')
    salary = pd.merge(first_frame, second_frame, on='jobId')
    salary = salary.iloc[:1000, :]

    def test_create_dataset(self):
        dataset = Dataset(self.salary)
        assert isinstance(dataset.dataset_pd, pd.DataFrame)
        assert isinstance(dataset, Dataset)


class TestCorrelation:
    first_frame = pd.read_csv('train_dataset.csv')
    second_frame = pd.read_csv('train_salaries.csv')
    salary = pd.merge(first_frame, second_frame, on='jobId')
    salary = salary.iloc[:1000, :]

    correlation = Correlation(salary)

    def test_create_correlation(self):
        assert self.correlation.accuracy_compare is None
        assert isinstance(self.correlation, Correlation)

    def test_train_correlation(self):
        svc = self.correlation.train(svm.SVC())
        ens = self.correlation.train(ensemble.RandomForestClassifier())
        mlpc = self.correlation.train(MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000))

        assert svc is not None
        assert len(svc) == 4
        assert ens is not None
        assert len(ens) == 4
        assert mlpc is not None
        assert len(mlpc) == 4
