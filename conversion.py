from random import randint

with open('/home/prometej/Workspaces/PythonWorkspace/project_chatbot/sample.txt', 'r+') as myfile:
    data = myfile.read().replace('\n', '')

list = data.split(' ')

for i in list:
    if i[len(i)-1] == ',':
        i = i[:len(i)-1]

for i in range(200):
    a = randint(0, 943)
    list[a] += '$#'

f = " ".join(list)
file = open('/home/prometej/Workspaces/PythonWorkspace/project_chatbot/sample_changed.txt', 'w+')
file.write(f)
