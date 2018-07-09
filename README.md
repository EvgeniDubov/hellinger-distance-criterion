# Hellinger Dinstance Criterion
Hellinger Distance criterion for Random Forest and Decision Tree classifiers sklearn implementation 

# Build 
You will need a cython "header" file (.pxd) from sklearn.

In case you've installed sklearn from source code package, you've already got it.

In case you've installed sklearn using pip install sklearn then you need to get it.

- Got to https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/tree
- Download **_criterion.pxd** and place it your local sklearn installation folder
  - Linux usually at /usr/local/lib/python3.5/dist-packages/sklearn/tree
  - Windows usually at C:\Users\\[user name]\AppData\Local\Continuum\Anaconda3\Lib\site-packages\sklearn\tree\

```
python setup.py build_ext --inplace
```

# Example
```
>>> import numpy as np
>>> from hellinger_distance_criterion import HellingerDistanceCriterion
>>> from sklearn.ensemble import RandomForestClassifier
>>> hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
>>> clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
>>> clf.fit(X_train, y_train)
>>> print('hellinger distance score: ', clf.score(X_test, y_test))
