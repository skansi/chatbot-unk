import os
from conversion import conversion

rootdir ='/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'

l = []

for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    name = str(subdir).split('/')[-1]
    l.append(str(name))
    for f in files:
        print('Working on file:', str(name) + '/' + str(f))
        SOURCE = str(subdir) + '/' + str(f)
        conversion(SOURCE)

print(sorted(l))
print('Done.')
