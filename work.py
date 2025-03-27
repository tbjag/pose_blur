import scipy.io

# Load the .mat file
mat_file_path = "/media/Data_2/person-search/dataset/annotation/Images.mat"
mat_data = scipy.io.loadmat(mat_file_path)

# Print available keys
print("Keys in Images.mat:", mat_data.keys())

# Check the structure of the "Images" key
if "Img" in mat_data:
    images_info = mat_data["Img"]
    print("\nType of 'Images':", type(images_info))
    print("Shape of 'Images':", images_info.shape)

    # Print the first element's structure
    print("\nFirst element structure:")
    print(images_info[0])  # Print full structure for debugging

    # Check if it's a structured array
    if isinstance(images_info[0], np.ndarray):
        print("\nData fields in first element:")
        print(images_info[0].dtype)
else:
    print("\n'Images' key not found in .mat file! Available keys:", mat_data.keys())
