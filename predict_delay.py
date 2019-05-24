from mrjob.job import MRJob
from mrjob.step import MRStep
import random
import csv
import numpy as np
import sys
from datetime import datetime

class DecisionTree:
    def __init__(self, training_data, field_types):
        self.training_data = np.array(training_data)
        self.field_types = field_types # N=numerical, C=categorical

    def mean_squared_error(self, records):
        if records.shape[0] == 0:
            return 0
        targets = records[:, -1].astype('float')
        value = np.mean(targets)
        mse = np.mean(np.square(targets - value))
        return mse

    def find_best_attr(self, records):
        result = {'attr': None, 'split_cond': None, 'splits': None}
        min_mse = -1
        for i in np.arange(0, records.shape[1] - 1):
            split_cond = None
            splits = {}
            if self.field_types[i] == 'N':
                # numerical attribute
                left_split = list()
                right_split = list()
                split_cond = np.mean(records[:, i].astype('float')) # mean value of attribute i
                for record in records:
                    if record[i] < split_cond:
                        left_split.append(record)
                    else:
                        right_split.append(record)
                splits['left'] = np.array(left_split)
                splits['right'] = np.array(right_split)
            else:
                # categorical attribute
                split_cond = list(set(records[:, i]))
                splits = {cond:list() for cond in split_cond}
                for record in records:
                    splits[record[i]].append(record)
                splits = {k: np.array(v) for k,v in splits.items()}

            # calculate MSE of splits
            error = 0
            for cond in splits:
                split = splits[cond]
                error += (split.shape[0]/records.shape[0])*self.mean_squared_error(split)
            
            if min_mse == -1 or error < min_mse:
                result['attr'] = i # index of chosen attribute for splitting
                result['split_cond'] = split_cond
                result['splits'] = splits
                min_mse = error
        
        return result

    def split(self, node):
        # split the records in a node, apply recursively to child nodes
        splits = node['splits']
        min_record = 5 # if the number of record in a split < min_record, we make a leaf node
        for i in splits:
            split = splits[i]
            if split.shape[0] <= min_record:
                node[i] = np.mean(split[:, -1].astype('float')) # make a leaf node
            else:
                node[i] = self.find_best_attr(split) # make an internal node
                self.split(node[i]) # split recursively on the child node

    def build_model(self):
        root_node = self.find_best_attr(self.training_data)
        self.split(root_node)
        return root_node
        
    def apply_model(self, node, record):
        if self.field_types[node['attr']] == 'N':
            # numerical node
            if record[node['attr']] < node['split_cond']:
                if isinstance(node['left'], dict):
                    return self.apply_model(node['left'], record)
                else:
                    return node['left'] # leaf node
            else:
                if isinstance(node['right'], dict):
                    return self.apply_model(node['right'], record)
                else:
                    return node['right'] # leaf node
        else:
            # categorical node
            cat = record[node['attr']]
            if cat not in node['split_cond'] and len(node['split_cond']) > 0:
                # not equal to any categorical values, set to the first category as default
                cat = node['split_cond'][0]
            
            if isinstance(node[cat], dict):
                return self.apply_model(node[cat], record)
            else:
                return node[cat] # leaf node

    def predict(self, model, test_data):
        predictions = []
        for record in test_data:
            pred_val = self.apply_model(model, record)
            predictions.append([record[-1], pred_val])
        return predictions



class MRPredictDelay(MRJob):
    field_types = ['C', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'N', 'N', 'N']

    def configure_options(self):
        super(MRPredictDelay, self).configure_options()
        self.add_passthrough_option('--maxSplitNumber', type='int', default=10)
        self.add_passthrough_option('--sampleNumber', type='int', default=100)
        self.add_passthrough_option('--sampleSize', type='int', default=4000)
        self.add_passthrough_option('--predict', default='departure', choices=['departure', 'arrival'])
        self.add_file_option('--testFile')

    def steps(self):
        return [
            MRStep(mapper = self.mapper_prepare_data,
                    reducer = self.reducer_prepare_data),
            MRStep(mapper_init = self.mapper_init_predict,
                    mapper = self.mapper_predict,
                    combiner = self.combiner_predict,
                    reducer = self.reducer_predict)
        ]

    def mapper_prepare_data(self,_,line):
        key = random.randint(0, self.options.maxSplitNumber)
        line = line.replace('"', '').split('\t')
        record = []
        if len(line) > 1:
            record = line[1].split(',')
            if self.options.predict == 'departure':
                # remove column of arr_delay
                del record[-1]
            else:
                # remove column of dep_delay
                del record[-2]
        yield key, record

    def reducer_prepare_data(self, key, values):
        values_list = np.array(list(values))
        l = values_list.shape[0]
        if l < self.options.sampleSize:
            yield key, list(values)
        else:
            # sampling without replacement
            for i in range(0, self.options.sampleNumber):
                idx = np.random.choice(l, size=self.options.sampleSize, replace=False)
                yield "{}_{}".format(key, i), values_list[idx, :].tolist()
    
    def mapper_init_predict(self):
        self.test_set = []
        with open(self.options.testFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if len(row) > 1:
                    record = row[1].split(',')
                    for i in [1, 8, 9, 10, 11, 12, 13]:
                        record[i] = float(record[i]) # convert numerical values into float
            
                    if self.options.predict == 'departure':
                        # remove column of arr_delay
                        del record[-1]
                    else:
                        # remove column of dep_delay
                        del record[-2]
                    
                    self.test_set.append(record)


    def mapper_predict(self, key, trainingset):
        # create a decision tree
        DT = DecisionTree(training_data = trainingset, field_types = self.field_types)
        model = DT.build_model()
        predictions = DT.predict(model, self.test_set)
        # As combiner is not guaranteed to be executed, change format that can be handle 
        # in reducer in case this will go directly to reducer
        yield 'predictions',(1,predictions)

    def combiner_predict(self, key, predictions):
        predictions = list(predictions)
        combined_prediction = []
        predictions_number = len(predictions)
        for i in range(0, predictions_number):
            if i == 0:
                combined_prediction = predictions[i][1]
            else:
                combined_prediction = np.add(combined_prediction, predictions[i][1])
        yield key, (predictions_number, combined_prediction.tolist())

    def reducer_predict(self, key, predictions):
        predictions = list(predictions)
        final_prediction = []
        predictions_number = 0
        for i in range(0, len(predictions)):
            predictions_number += predictions[i][0]
            if i == 0:
                final_prediction = predictions[i][1]
            else:
                final_prediction = np.add(final_prediction, predictions[i][1])
        
        # calculate the average from all predictions
        final_prediction = final_prediction/predictions_number

        yield key, final_prediction.tolist()

if __name__ == '__main__':
    start_time=datetime.now()
    MRPredictDelay.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    sys.stderr.write(str(elapsed_time))
    
