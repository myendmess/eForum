import numpy as np
import random
from scipy.stats import norm

def show_array(arr):
    print(arr)
    print(type(arr))
    print(f"----")

# Creating NumPy arrays
arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
show_array(arr1)

arr2 = np.array([42])  # Create a 1D array
show_array(arr2)

arr3 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 4, 6, 8, 10, 12, 14, 16, 19]])
show_array(arr3)
print(arr3[0, 1])  # Access element at row 0, column 1

arr4 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
show_array(arr4)
print(arr4[0, 1, 2])  # Access element at row 0, column 1, inner array index 2

# Working with different data types
arr5 = np.array([1, 2, 3, 4, 5])
show_array(arr5)

arr6 = np.array(["dog", "cat", "giraffe", "velociraptor"])
show_array(arr6)

arr7 = np.array([1, 2, 3, 4])  # These are still integers
show_array(arr7)

# Separate arrays for better data type management
arr8 = np.array(["a", 2])  # Consider separate arrays for strings and numbers
show_array(arr8)

arr9 = np.array([1.2, 2.3, 3.1, 4.5])
show_array(arr9)

arr10 = arr9.astype('i')  # Conversion to integers might cause data loss
show_array(arr10)

# Slicing examples
print(arr1[3:])
print(arr1[-3:-1])
print(arr1[::2])

show_array(arr3)
print(arr3[0:3, 2:4])

# Get array dimensions
print(arr4.shape)

# Generate random numbers (using NumPy's random module)
random_numbers = np.random.rand(6)  # Array of 6 random floats between 0 and 1
print(random_numbers)

# Generate random integers (using NumPy's random module)
correct_random_ints = np.random.randint(100, size=(3, 5))
print(correct_random_ints)

# Generate random choices from a list (using NumPy's random module)
ch = [22, 55, 77, 99, 12, 5, 98]
correct_random_choices = np.random.choice(ch, size=(3, 5))
print(correct_random_choices)

# Generate random samples from a normal distribution with mean 100 and standard deviation 15
normal_samples = norm.rvs(loc=100, scale=15, size=1000)  # Use norm.rvs from scipy.stats
print(normal_samples[:100])  # Print only the first 10 samples for brevity