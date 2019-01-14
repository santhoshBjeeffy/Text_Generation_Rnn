'''
Created on Nov 13, 2018

@author: santhosh
'''
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

class DownloadDataSet(object):

    def __init__(self):
        '''
        Constructor
        '''
        
    def downloadFile(self):
        file_path = tf.keras.utils.get_file('C:\Users\santhob\.spyder-py3\TextClass\shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt');
        return file_path;
    
class ReadDataSet(object):
    filePath = None
    def __init__(self, filePath):
        self.filePath = filePath
        
    def readData(self):
        return open(self.filePath).read()
        
    def getUniqueChars(self):
        data = self.readData()
        uniqueChars = sorted(set(data));
        return uniqueChars
    
    def getUniqueCharToIndex(self):
        uniqueChars = self.getUniqueChars()
        char2idx = {u:i for i, u in enumerate(uniqueChars)}
        return char2idx
    
    def getIndexToChar(self):
        return np.array(self.getUniqueChars())
    
    def getTextAsInt(self):
        dataSet = self.readData()
        char2idx = self.getUniqueCharToIndex()
        text_as_int = np.array([char2idx[c] for c in dataSet])
        return text_as_int 
    
class TrainAndTargetData(object):
    sequenceLength = None
    dataSetPath = None
    BUFFER_SIZE = None
    BATCH_SIZE = None
    
    def __init__(self, sequenceLength, dataSetPath, BUFFER_SIZE, BATCH_SIZE):
        self.sequenceLength = sequenceLength
        self.dataSetPath = dataSetPath
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        
    def split_input_target(self, chunk):
        '''
            From chunk input is except last char
            From chunk target is from 2nd char to last char
        '''
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
        
    def getTrainAndTargetDataSet(self):
        '''
            In chunks the entire data is divided to batches of size sequenceLength
        '''
        chunks = tf.data.Dataset.from_tensor_slices(ReadDataSet(self.dataSetPath).getTextAsInt()).batch(self.sequenceLength + 1, drop_remainder=True)       
        dataset = chunks.map(self.split_input_target)
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

        return dataset
    
        
        