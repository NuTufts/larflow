import h5py
import os

def print_dataset_sizes(filename):
    """
    Opens an HDF5 file and prints the storage size of each dataset in MB.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    with h5py.File(filename, 'r') as f:
        print(f"--- Analyzing file: {filename} ---")
        f.visititems(visitor_func)

def visitor_func(name, obj):
    """
    A visitor function for h5py.File.visititems() that checks if the object
    is a dataset and prints its size.
    """
    if isinstance(obj, h5py.Dataset):
        # Get the size in bytes using the low-level API
        size_bytes = obj.id.get_storage_size()
        # Convert bytes to megabytes (MB)
        size_mb = size_bytes / (1024 * 1024)
        # Print the dataset name and its size
        print(f"Dataset: {name}")
        print(f"  Size on disk: {size_mb:.4f} MB")
        # Optional: Print the uncompressed shape and dtype
        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

# --- Example Usage ---
# Replace 'your_file.h5' with the path to your HDF5 file
# If you don't have a file, the script will print an error message.
# You can create a sample file for testing if needed.
# Example:
# with h5py.File('sample.h5', 'w') as f:
#     f.create_dataset('small_data', data=range(1000))
#     f.create_dataset('large_data', shape=(1000, 1000), dtype='f4', compression='gzip')

print_dataset_sizes('out_test.h5')
