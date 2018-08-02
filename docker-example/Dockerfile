FROM python:3.5
RUN pip install Cython scikit-learn numpy scipy &&\
wget https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/tree/_criterion.pxd -P /usr/local/lib/python3.5/site-packages/sklearn/tree/ &&\
wget https://raw.githubusercontent.com/EvgeniDubov/hellinger-distance-criterion/master/hellinger_distance_criterion.pyx &&\
wget https://raw.githubusercontent.com/EvgeniDubov/hellinger-distance-criterion/master/setup.py
RUN python3 setup.py build_ext --inplace
RUN pip install pandas
COPY ./random_forest_train.py .
CMD python random_forest_train.py
