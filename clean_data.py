# Question format:
# {"id":"ACTAAP_2007_5_27","question":{"stem":"A new substance is found. The substance is extremely lightweight. Most remarkably, it will expand and fill each of the containers below. Which type of matter is this new substance? {img:images/States2017/AR/AR_5thGr_2007_3.png}","choices":[{"text":"gas","label":"A"},{"text":"solid","label":"B"},{"text":"liquid","label":"C"},{"text":"colloid","label":"D"}]},"answerKey":"A"}

import re, sys

SOURCE_LIST = ['/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-DMC-Train.jsonl', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-DMC-Dev.jsonl', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-DMC-Test.jsonl', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-NDMC-Dev.jsonl', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-NDMC-Train.jsonl', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/ElementarySchool/Elementary-NDMC-Test.jsonl']

TEMP = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/adapted.jsonl'

SAVE_FILE = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/Q&A/final.txt'

for SOURCE in SOURCE_LIST:
    print(SOURCE)
    # remove all picutre annotations
    with open(SOURCE, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('\{img:.*?\}')
    data = p.sub('', data_whole)

    file = open(TEMP, 'w+')
    file.write(data)

    # remove all 'Use ... below to answer the question' sentences
    with open(TEMP, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('Use th.*? answer .*?question.*?\.')
    data = p.sub('', data_whole)

    file = open(TEMP, 'w+')
    file.write(data)

    # remove all 'Then answer question...' sentences
    with open(TEMP, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('Then answer question.*?\.')
    data = p.sub('', data_whole)

    file = open(TEMP, 'w+')
    file.write(data)

    # remove replace two spaces with one
    with open(TEMP, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('\s\s')
    data = p.sub(' ', data_whole)

    file = open(TEMP, 'w+')
    file.write(data)

    def get_answer_text(line, a_letter):

        answers = re.findall('\[.*?\]', line)
        answers_list = re.findall('\{.*?\}', answers[0])

        i = ord(a_letter) - 65

        temp = answers_list[i]
        temp = temp.split('\"')
        ans = temp[3]

        return ans

    with open(TEMP, 'r+') as myfile:
        d = myfile.readlines()

    with open(SAVE_FILE, 'a+') as dest:

        index = 0

        while index < len(d):
            data = d[index]
            question = data.split('\"')[9]
            answer_letter = data.split('\"')[-2]

            answer = get_answer_text(data, answer_letter)

            q_and_a = "Q: " + str(question) + " ---> A: " + str(answer).capitalize() + ".\n"

            dest.write(q_and_a)

            index += 1

            # if index >= len(d):
    dest.close()

    # remove replace two spaces with one
    with open(SAVE_FILE, 'r+') as myfile:
        data_whole = myfile.read()

    p = re.compile('\s\s')
    data = p.sub(' ', data_whole)

    r = re.compile('\.\.')
    data = r.sub('.', data_whole)

    file = open(SAVE_FILE, 'w+')
    file.write(data)

print('Done. Exiting...')
sys.exit()
