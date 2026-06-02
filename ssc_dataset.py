import h5py
import os
import torch
import numpy as np

device = 'cuda'

def getData(dataset):
    dataset = dataset
    root_path = '/data/hyx/ViT-dend/data/extract'
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)

class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=int)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                device)
            y_batch = torch.tensor(self.labels_[batch_index], device=device)
            self.counter += 1
            return X_batch.to(device=device), y_batch.to(device=device)

        else:
            raise StopIteration
        

if __name__=="__main__":
    T = 250
    max_time = 1.4
    in_dim = 700
    (x_train, y_train), (x_test, y_test) = getData('SSC')
    train_loader = SpikeIterator(x_train, y_train, 8, T, in_dim, max_time, shuffle=True)
    test_loader = SpikeIterator(x_test, y_test, 8, T, in_dim, max_time, shuffle=False)

    x,y = next(iter(train_loader))
    print(x.shape, y.shape)