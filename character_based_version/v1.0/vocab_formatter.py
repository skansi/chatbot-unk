import os

ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump/A/AB'

with open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/character_based_version/v1.1/vocab2', 'r') as v:
    VOCAB = eval(v.read())

for subdir, dirs, files in os.walk(ROOTDIR):
    print(subdir)
    name = str(subdir).split('/')[-1]
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)
        with open(SOURCE, 'r') as d:
            data = d.read()
            data = data.split()

            for k in range(len(data)):
                data[k] = ''.join([i for i in data[k] if i in VOCAB])

            data = ' '.join(data)
        with open(SOURCE, 'w+') as s:
            s.write(data)
