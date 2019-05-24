from mrjob.job import MRJob
import numpy as np
import re
import sys
from datetime import datetime

class MRGet_accuracy(MRJob):

    def mapper(self,_,line):
        # remove the "" and split based on comma
        line = line.replace('"', '').split('\t')
        data = re.findall('[+-]?\d+\.\d+',line[1])
        n_data = len(data)/2
        data_array = np.array(data).reshape(n_data,2).astype(float)
        sum_delta_squared = np.sum(np.square(data_array[:,0]-data_array[:,1]))
        yield 'rmse',(n_data,sum_delta_squared)

    def combiner(self,key,sum_delta_squared):
        squared_sums = list(sum_delta_squared)
        sum_of_delta_squared = 0
        sum_n_data = 0
        for i in range(0, len(squared_sums)):
            sum_n_data += squared_sums[i][0]
            sum_of_delta_squared += squared_sums[i][1]
            
        yield key, (sum_n_data, sum_of_delta_squared)


    def reducer(self,key,sum_delta_squared):
        squared_sums = list(sum_delta_squared)
        sum_of_delta_squared = 0
        sum_n_data = 0
        for i in range(0, len(squared_sums)):
            sum_n_data += squared_sums[i][0]
            sum_of_delta_squared += squared_sums[i][1]
            
         # calculate the average from all predictions
        rmse = np.sqrt(sum_of_delta_squared/sum_n_data)

        yield key, rmse


if __name__ == '__main__':
    start_time=datetime.now()
    MRGet_accuracy.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    sys.stderr.write(str(elapsed_time))