# hellinger-random-forest
Random Forest model using Hellinger Distance as split criterion

# build 
You will need a cython header file (.pxd) from sklearn source code.
In case you've installed sklearn from source code package, you've already got it.
In case you've installed sklearn using pip install sklearn then you need to get it.
- Got to https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/tree
- Download **_criterion.pxd** and place it your local sklearn installation folder
  - Linux usually at /usr/local/lib/python3.5/dist-packages/sklearn/tree
  - Windows usually at C:\Users\\[user name]\AppData\Local\Continuum\Anaconda3\Lib\site-packages\sklearn\tree\

python setup.py build_ext --inplace
