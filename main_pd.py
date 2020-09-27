from models.KNN_pd import KNN
import preprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Import data
df = preprocessing.import_voter()

# Extract tuning set
tuning_set = df.sample(frac=0.1, random_state=2)
df = df.drop(tuning_set.index)

# Set up stratified 5-fold cross-validation; only necessary for classificaton
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
training_sets, test_sets = [], []
for fold, (train, test) in enumerate(skf.split(X=np.zeros(len(df)), y=df.iloc[:, -1:])):
    training_sets.append(df.iloc[train])
    test_sets.append(df.iloc[test])

# Train; run 5 experiments in total
trained_models = []
for training_set in training_sets:
    my_knn = KNN(num_neighbors=3, data=training_set, attribute_names=list(df.columns))
    my_knn.train()
    # my_knn.train(edited=True)
    # my_knn.condense_data()
    my_knn.report_training_stats()
    trained_models.append(my_knn)

# Tune k for 5 trained models
for model in trained_models:
    parameters = [_ for _ in range(1, 11)]
    optimal_k = model.tune_k(tuning_set, parameters=parameters)
    model.report_k_tuning_stats()
    print("Optimal k: ", optimal_k)

# Test
for model, test_set in zip(trained_models, test_sets):
    model.test(test_set)
    model.report_testing_stats()
