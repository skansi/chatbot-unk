import os

ROOTDIR = '/home/novak_luka93/wikidump'
char_list = set()
with open('/home/novak_luka93/chatbot-unk/character_based_version/v1.0/vocab', 'r') as v:
    VOCAB = eval(v.read())

for subdir, dirs, files in os.walk(ROOTDIR):
    print(subdir)
    name = str(subdir).split('/')[-1]
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)
        with open(SOURCE, 'r') as d:
            data = d.read()
            data = ''.join([i for i in data if i in VOCAB])

        with open(SOURCE, 'w+') as s:
            s.write(data)
