import h5py
import numpy as np

def print_hdf5_contents(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")
        if obj.size < 10:  # Print small datasets entirely
            print(f"  Data: {obj[:]}")
        else:  # Print first and last 3 elements for larger datasets
            print(f"  First 3 elements: {obj[:3]}")
            print(f"  Last 3 elements: {obj[-3:]}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

def read_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Contents of {file_path}:")
            f.visititems(print_hdf5_contents)
    except IOError:
        print(f"Error: Unable to open file {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    file_path = input("Enter the path to your HDF5 file: ")
    read_hdf5_file(file_path)
