import socket
import h5py, os, random, glob
import numpy as np

class DataSetLoader:
    def __init__(self, path="./data", num_files=None):
        """
        Constructor for DataLoader class.

        Args:
        - path (str): path to directory containing data files (default: "./data")
        """
        self.path = path
        self.num_files = num_files
        self.data = {}

    @staticmethod
    def _hdf5_read(path="./data.h5"):
        """
        Read a dataset in HDF5 format.

        Args:
        - path (str): path to HDF5 file (default: "./data.h5")

        Returns:
        - data_dict (dict): dictionary with keys and values from the HDF5 file
        """
        with h5py.File(path, "r") as f:
            data_entry = {}
            for key in f.keys():
                data_entry[key] = f[key]['values'][:]
        return data_entry

    def load_dataset(self):
        """
        Load and combine data from HDF5 files in the directory specified by `self.path`.

        Returns:
        - data_dict (dict): dictionary containing combined data from all HDF5 files
        """
        # Get a list of all HDF5 files in the directory specified by `self.path`.
        file_paths = glob.glob(os.path.join(self.path, "*.h5"))

        # Shuffle the list of file paths to randomize the order in which the files are read.
        np.random.shuffle(file_paths)
        file_paths = file_paths[:self.num_files]

        # Calculate the required amount of Frequency and amplitude columns
        column_value = []
        for file_path in file_paths:
            data_entry = self._hdf5_read(file_path)
            # Combine data_entry with existing data dictionary.
            for key, value in data_entry.items():
                if key == 'amplitude_array' or key == 'frequency_array':
                    column_value.append(int(value.shape[0]))

        # Read each file and add its contents to the `self.data` dictionary.
        for file_path in file_paths:
            data_entry = self._hdf5_read(file_path)
            print('file_path:', file_path)

            # Combine data_entry with existing data dictionary.
            for key, value in data_entry.items():
                if key == 'amplitude_array' or key == 'frequency_array':
                    pad_width = [(0, 0)] * value.ndim  # initialize padding to zero for all dimensions
                    pad_width[-1] = (0, max(column_value) - value.shape[-1])  # pad along last dimension
                    value = np.pad(value, pad_width, mode='constant')
                if key not in self.data:
                    self.data[key] = value[np.newaxis, :]  # add new key and convert to row vector
                else:
                    self.data[key] = np.vstack((self.data[key], value))  # stack arrays vertically
            try:
                print(np.shape(self.data[key]))
            except:
                pass
        return self.data

    @staticmethod
    def load_hammerstein_coefficients(path, remove_linear_term=False, distorted_mode='False'):
        ''' Load Hammerstein coeficients'''
        with h5py.File(path, 'r') as f:
            coeff = f['nonlinear_coeff'][:]
        # print('org:', coeff)
        coeff_row1 = coeff[0, :].reshape(1, -1)
        coeff_rest = coeff[1:]
        if distorted_mode == 'False':
            coeff_rest = -coeff_rest

        for ii in range(0, len(coeff_rest[0])):
            vslide = coeff_rest[:, ii]
            if ii == 0:
                result = vslide
            else:
                result = (np.concatenate([result, vslide]))

        if remove_linear_term == False:
            result = np.concatenate([coeff_row1[0], result], axis=0)
            coeff = np.concatenate([coeff_row1, coeff_rest], axis=0)
        # print('coeff_row1:', np.shape(coeff_row1), coeff_row1)
        # print('coeff_rest:', np.shape(coeff_rest), coeff_rest)
        # print('result:', np.shape(result), result)

        # coeff2 = np.concatenate([coeff_row1, coeff_rest], axis=0)

        result = result.reshape(-1, 1)
        return result,coeff

class DataSetWriter:
    def __init__(self, dataset_features, train_path="./train/", test_path="./test/", train_pct=0.8):
        """
        A class for writing datasets in hdf5 format.

        Args:
            dataset_features (dict): Dictionary of dataset features, including amount_bits, fs, total_term
            train_path (string): Path to write training data to. Default: "./train/"
            test_path (string): Path to write test data to. Default: "./test/"
            train_pct (float): Percentage of data to use for training. Default: 0.8
        """
        self.dataset_features = dataset_features
        self.train_path = train_path
        self.test_path = test_path
        self.train_pct = train_pct
        self.coeff_path = './'

    @staticmethod
    def _hdf5_write_groups(data, path="./data.h5"):
        """
        Write a dataset in hdf5 format

        Args:
            data (dict): Dictionary of data to store, where keys are group names and values are numpy arrays
            path (string): Path to write the data to. Default: "./data.h5"
        """
        with h5py.File(path, "w") as f:
            for key in data.keys():
                f.create_group(key)
                f[key].create_dataset('values', data=data[key])
        return

    def DiscWriter(self, V, X, nonlinnear_coeff=None):
        """
        Write a dataset in hdf5 format to specified training and test directories,
        based on input distorted and pure signals, frequency array and amplitude array.

        Args:
            V (numpy array): Distorted signals
            X (numpy array): Pure signals

        Returns:
            None
        """
        # Determine the number of samples to use for training
        num_train = int(self.train_pct * len(V))

        # Shuffle the indices to randomly split the data
        indices = list(range(len(V)))
        random.shuffle(indices)
        train_indices = indices[:num_train]

        # Loop through the indices, write each sample to appropriate directory
        for i in indices:
            if i in train_indices:
                folder = self.train_path
                name_set = '_train'
            else:
                folder = self.test_path
                name_set = '_test'
            # Create a dictionary of data for this sample
            data_generator = {
                'distorted_signal': V[i].astype('float64'),
                'pure_signal': X[i].astype('float64'),
            }

            # Generate a filename for this sample
            filename = 'signal_' + str(1+i) + str(name_set)+'.h5'

            # Set the file path based on the sample's directory and filename
            filepath = os.path.join(folder, filename)

            try:
                # Write the data to the hdf5 file
                self._hdf5_write_groups(data_generator, filepath)
            except Exception as e:
                print(f"Error writing data using function ._hdf5_write_groups")

        if nonlinnear_coeff is not None:
            path_parts = self.train_path.split('/')
            path_parts = path_parts[:-1]
            coeff_main_path = '/'.join(path_parts) + '/nonlinear_coeff'
            # h5path = coeff_main_path + '.h5'
            # coeff_filepath = '/'.join(path_parts) + '/nonlinear_coeff.h5'
            with h5py.File(coeff_main_path+'.h5', mode='w') as fh5:
                fh5.create_dataset(name='nonlinear_coeff', data=nonlinnear_coeff)

            with open(coeff_main_path+'.txt', 'w') as ftxt:
                # Save Matlab format
                ftxt.write('[')
                # Assuming nonlinnear_coeff is a numpy array or similar
                for i, row in enumerate(nonlinnear_coeff):
                    # Convert each row to a string, elements separated by spaces
                    row_str = ' '.join(map(str, row))
                    # For MATLAB, end rows with a semicolon except for the last one
                    if i < len(nonlinnear_coeff) - 1:
                        ftxt.write(row_str + '; ')
                    else:
                        ftxt.write(row_str)
                # Close the MATLAB matrix format
                ftxt.write('];\n')


class PathManager:
    def __init__(self):
        self.main_path, self.host = self.check_path_by_host()

    @staticmethod
    def make_path(root_path, local_path, current_path, state=None):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if state == 'data':
            test_path = os.path.join(root_path, local_path, current_path, 'test')
            train_path = os.path.join(root_path, local_path, current_path, 'train')

            if not os.path.exists(test_path):
                os.makedirs(test_path)
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            return train_path, test_path
        elif state == 'simulations':
            simulations_path = os.path.join(root_path, local_path, current_path)
            print('simulations_path:', simulations_path)
            if not os.path.exists(simulations_path):
                os.makedirs(simulations_path)
            return simulations_path

    @staticmethod
    def load_path(root_path, local_path, folder_prefix, state=None):
        # print('root_path', root_path)
        # print('root_path', local_path)
        # print('root_path', current_path)
        if state == 'data':
            matching_folders = [folder for folder in os.listdir(os.path.join(root_path, local_path)) if folder.startswith(folder_prefix)]
            print('matching_folders:', matching_folders)
            test_path = os.path.join(root_path, 'datasets', str(matching_folders[0]), 'test')
            train_path = os.path.join(root_path, 'datasets', str(matching_folders[0]), 'train')
            return train_path, test_path
        elif state == 'simulations':
            matching_folders = [folder for folder in os.listdir(os.path.join(root_path, local_path)) if folder.startswith(folder_prefix)]
            simulation_path = os.path.join(root_path, 'trained_model', str(matching_folders[0]))
            return simulation_path
    def check_path_by_host(self):
        main_path = os.path.join('./')
        host = socket.gethostname()
        return main_path, host



