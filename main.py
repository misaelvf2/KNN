from models.KNN import KNN

my_data = [
    [5, 7, 8, 'Blue'],
    [2, 4, 1, 'Red'],
    [3, 4, 1, 'Red'],
    [8, 1, 3, 'Red'],
    [9, 8, 9, 'Yellow'],
    [2, 1, 4, 'Yellow'],
    [3, 1, 8, 'Red'],
    [4, 5, 6, 'Blue'],
    [6, 1, 9, 'Blue'],
    [4, 7, 3, 'Yellow'],
]

my_query_point = [4, 3, 2]

my_knn = KNN(1, my_data, 3)
print(my_knn.classify(my_query_point))
