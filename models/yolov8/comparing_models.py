from difflib import unified_diff

# Read the contents of the two files
with open('myyolo.txt', 'r') as file1:
    myyolo_content = file1.readlines()

with open('test_yolo.txt', 'r') as file2:
    yolo_content = file2.readlines()

# Find the differences
diff = unified_diff(myyolo_content, yolo_content, fromfile='myyolo.txt', tofile='test_yolo.txt')

# Print the differences
for line in diff:
    print(line, end='')