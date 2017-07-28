import re

def conversion(SOURCE):

    # SOURCE = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/enwiki-20170720-pages-articles-multistream.xml'
    #
    # DEST = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump.txt'

    # remove all titles
    with open(SOURCE, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('=+\s([a-zA-Z\x7f-\xff]+\s)*=+')
    data = p.sub('', data_whole)

    # remove all lines that start with <
    p = re.compile('\<.*?\n')
    data = p.sub('', data)

    # remove all lines that start with *
    p = re.compile('\*.*')
    data = p.sub('', data)

    # remove all square brackets and everything inside them
    p = re.compile('\[[a-zA-Z\x7f-\xff]*[0-9]*\]')
    data = p.sub('', data)

    # remove all life span sequences
    p = re.compile('\([0-9]{4}.?[0-9]+\)')
    data = p.sub('', data)

    # remove all years
    p = re.compile('[0-9]+')
    data = p.sub('', data)

    # remove all brackets and everything inside them
    p = re.compile('\((.*?)\)')
    data = p.sub('', data)

    # remove all brackets
    p = re.compile('\(')
    data = p.sub('', data)

    # remove all brackets
    p = re.compile('\)')
    data = p.sub('', data)

    # remove all commas
    p = re.compile(',')
    data = p.sub('', data)

    # remove all commas
    p = re.compile(':')
    data = p.sub('', data)

    # remove all commas
    p = re.compile(';')
    data = p.sub('', data)

    # remove all one letter abbreviations
    p = re.compile('\s[A-z]\.')
    data = p.sub('', data)

    # remove all abbreviations
    p = re.compile('\s[A-z][A-z]\.')
    data = p.sub('', data)

    # remove all " characters
    p = re.compile('\"')
    r = re.compile('\'')
    data = p.sub('', data)
    data = r.sub('', data)

    # remove ... sequences
    p = re.compile('\.{3}')
    data = p.sub('', data)

    # check if after any dot there is no space, than add one
    p = re.compile('\.')
    data = p.sub('. ', data)

    # replace all dots with END_OF_SENTENCE symbol
    p = re.compile('\.\s')
    data = p.sub(' $# ', data)

    # replace two spaces with one
    p = re.compile('\s\s')
    data = p.sub(' ', data)

    # # add space before END_OF_SENTENCE symbol
    # p = re.compile('\$\#')
    # data = p.sub(' $#', data)

    # replace new_line with single space
    p = re.compile('\n')
    data = p.sub(' ', data)

    # replace two spaces with one
    p = re.compile('\s\s')
    data = p.sub(' ', data)

    file = open(SOURCE, 'w+')
    file.write(data)

    return
