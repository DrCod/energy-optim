import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        dataframe['dt'] = pd.to_datetime(dataframe['dt'])
        dataframe = dataframe.set_index('dt')

        #dataframe= dataframe.drop(columns=['dt'])
        i_split = int(len(dataframe.get(cols).resample('h').mean()) * split)
        self.data_train = dataframe.get(cols).resample('h').mean().values[:i_split]
        self.data_test  = dataframe.get(cols).resample('h').mean().values[i_split:]
        np.savetxt("Test_Data.txt",self.data_train, delimiter=',')
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.scaler     =  MinMaxScaler(feature_range =(0,1))
        self.len_train_windows = None

    def get_test_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        print(self.len_test)
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False)
        x = data_windows[:,:-1]
        y = data_windows[:,-1,[0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with in the range of 0 and 1'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = self.scaler.fit_transform(window)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data).astype(float)