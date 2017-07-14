import re

SOURCE = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Finance.txt'

# all titles
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('=+\s([a-zA-Z\x7f-\xff]+\s)*=+')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# all square brackets and everything inside them
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\[[a-zA-Z\x7f-\xff]*[0-9]*\]')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# all life span sequences
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\([0-9]{4}.?[0-9]+\)')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# all years
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('[0-9]+')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# all brackets and everything inside them
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\((.*?)\)')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# all commas
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile(',')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# replace all abbreviations
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\s[A-Z][a-z]\.')
data = p.sub('', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# replace all abbreviations -> capital letters
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\s[A-Z][A-Z]\.')
data = p.sub(' ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# check if after any dot there is no space, than add one
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\.')
data = p.sub('. ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# replace all dots with END_OF_SENTENCE symbol
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\.\s')
data = p.sub('$# ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# replace two spaces with one
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\s\s')
data = p.sub(' ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# replace new_line with single space
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('\n')
data = p.sub(' ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)

# with open(SOURCE, 'r+') as myfile:
#     data = myfile.read().replace('\n', '')
#
# words = data.split(' ')
