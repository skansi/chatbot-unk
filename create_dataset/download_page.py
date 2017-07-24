import wikipedia

p = wikipedia.page('Finance')

content = p.content

file = open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Finance.txt', 'w+')
file.write(content)
