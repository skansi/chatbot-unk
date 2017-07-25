import re

SOURCE = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Finance.txt'

# remove all titles
with open(SOURCE, 'r+') as myfile:
    data_whole = myfile.read()

p = re.compile('=+\s([a-zA-Z\x7f-\xff]+\s)*=+')
data = p.sub('', data_whole)

# remove all square brackets and everything inside them
p = re.compile('\[[a-zA-Z\x7f-\xff]*[0-9]*\]')
data = p.sub('', data_whole)

# remove all life span sequences
p = re.compile('\([0-9]{4}.?[0-9]+\)')
data = p.sub('', data_whole)

# remove all years
p = re.compile('[0-9]+')
data = p.sub('', data_whole)

# remove all brackets and everything inside them
p = re.compile('\((.*?)\)')
data = p.sub('', data_whole)

# remove all commas
p = re.compile(',')
data = p.sub('', data_whole)

# remove replace all abbreviations
p = re.compile('\s[A-Z][a-z]\.')
data = p.sub('', data_whole)

# remove replace all abbreviations -> capital letters
p = re.compile('\s[A-Z][A-Z]\.')
data = p.sub(' ', data_whole)

# remove check if after any dot there is no space, than add one
p = re.compile('\.')
data = p.sub('. ', data_whole)

# remove replace all dots with END_OF_SENTENCE symbol
p = re.compile('\.\s')
data = p.sub('$# ', data_whole)

# remove replace two spaces with one
p = re.compile('\s\s')
data = p.sub(' ', data_whole)

# replace new_line with single space
p = re.compile('\n')
data = p.sub(' ', data_whole)

file = open(SOURCE, 'w+')
file.write(data)
