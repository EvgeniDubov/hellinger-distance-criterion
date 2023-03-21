# Author: Evgeni Dubov <evgeni.dubov@gmail.com>
#
# License: MIT

from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t

import numpy as np
cdef double INFINITY = np.inf

from libc.math cimport sqrt, pow
from libc.math cimport abs


cdef class HellingerDistanceCriterion(ClassificationCriterion):    
    
    cdef double proxy_impurity_improvement(self) noexcept nogil:
        cdef double impurity_left
        cdef double impurity_right
        
        self.children_impurity(&impurity_left, &impurity_right)
       
        return impurity_right + impurity_left
    
    cdef double impurity_improvement(self, double impurity_parent, double impurity_left, double impurity_right ) noexcept nogil:
        
        return impurity_right + impurity_left
    
    cdef double node_impurity(self) noexcept nogil:
        cdef double hellinger = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                hellinger += 1.0

        return hellinger / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) noexcept nogil:
        cdef double hellinger_left = 0.0
        cdef double hellinger_right = 0.0
        cdef double count_k1 = 0.0
        cdef double count_k2 = 0.0

        cdef SIZE_t k
        cdef SIZE_t c

        # stop splitting in case reached pure node with 0 samples of second class
        if self.sum_left[k,1] + self.sum_right[k,1] == 0:
            impurity_left[0] = -INFINITY
            impurity_right[0] = -INFINITY
            return
        
        for k in range(self.n_outputs):
            if(self.sum_left[k,0] + self.sum_right[k,0] > 0):
                count_k1 = sqrt(self.sum_left[k,0] / (self.sum_left[k,0] + self.sum_right[k,0]))
            if(self.sum_left[k,1] + self.sum_right[k,1] > 0):
                count_k2 = sqrt(self.sum_left[k,1] / (self.sum_left[k,1] + self.sum_right[k,1]))

            hellinger_left += pow((count_k1  - count_k2),2)
            
            if(self.sum_left[k,0] + self.sum_right[k,0] > 0):    
                count_k1 = sqrt(self.sum_right[k,0] / (self.sum_left[k,0] + self.sum_right[k,0]))
            if(self.sum_left[k,1] + self.sum_right[k,1] > 0):
                count_k2 = sqrt(self.sum_right[k,1] / (self.sum_left[k,1] + self.sum_right[k,1]))

            hellinger_right += pow((count_k1  - count_k2),2)
        
        impurity_left[0]  = hellinger_left  / self.n_outputs
        impurity_right[0] = hellinger_right / self.n_outputs
