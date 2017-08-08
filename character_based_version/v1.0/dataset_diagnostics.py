import os
import sys

min_size = sys.maxsize
max_size = 0
sum_size = 0
num_files = 0
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'
smaller = 0
equal = 0

for subdir, dirs, files in os.walk(ROOTDIR):
    print(subdir)
    name = str(subdir).split('/')[-1]
    n_files = len(files)
    completed = 0
    smaller = 0
    for f in files:
        num_files += 1
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)
        raw_text = open(SOURCE).read()
        size = len(raw_text)
        if size < 900000:
            smaller += 1
        if size == 932088:
            equal += 1
        sum_size += size
        if size < min_size:
            min_size = size
        if size > max_size:
            max_size = size
        completed += 1
        print('Progress: {}/{}'.format(completed, n_files))

print('Number of files:', num_files)
print('Minimum size of an article is {}'.format(min_size))
print('Maximum size of an article is {}'.format(max_size))
print('Average size of article is {}'.format(int(sum_size/num_files)))
print('Number of files that are smaller than average is', smaller)
print('Number of files that are equal in size as average file is', equal)
print('Number of files that are smaller or equal than average is', smaller + equal)

# with open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/character_based_version/v1.0/shortest.txt', 'w+') as f_save:
#     f_save.write(str(min_size))
