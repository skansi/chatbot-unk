FILE_1 = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/dataset.txt'
FILE_2 = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Investment_banking.txt'

with open(FILE_1, 'r') as file_1:
    data_1 = file_1.read()

with open(FILE_2, 'r') as file_2:
    data_2 = file_2.read()

merged_document = data_1 + ' ' + data_2

dest = open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/dataset.txt', 'w')
dest.write(merged_document)
