<p align="center">
  <img src="https://valohai.com/static/img/support-logos/keras-text.svg" heigth=100 width=405> <img src="https://camo.githubusercontent.com/ee91ac3c9f5ad840ebf70b54284498fe0e6ddb92/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f7472616e73702e706e67" height=110 width="140">  <img src="https://jdrch.files.wordpress.com/2013/04/python_logo_and_wordmark-svg.png" height=100 width="338">
</p>

# chatbot-unk
Chatbot for QnA in an unknown language. There are several versions planned, ranging from SRN, LSTM, MemNN and Q-learning. Team members: dr. sc. Sandro Skansi (team lead), dr. sc. Branimir Dropuljić, Luka Novak, Antonio Šajatović

Chatbot is implemented in Python3 using [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend.

# Prequisites installation steps:
  - [Install Tensorflow and NVIDIA prerequisites](http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html)
  - [Install Keras](https://keras.io/): <code>sudo pip3 install keras</code>

# Usage:
  ## ***Important***: <code>python --version</code> ==> Python 3.5.2

  - Clone this repo
  - Download [Wikidump](https://dumps.wikimedia.org/backup-index.html) and extract it with:
      
      <code>bzip2 -dk <.your_wikidump.bz2></code>
      
  - Clone [WikiExtractor](https://github.com/attardi/wikiextractor) and run it like this:
      
      <code>python wikiextractor/WikiExtractor.py .o <output_folder> --processes <number_of_processes_to_use> <your_wikidump.xml></code>
  
  - Go to [create_dataset/format_files.py](https://github.com/skansi/chatbot-unk/blob/master/create_dataset/format_files.py) and run:
  
      <code>python format_files.py</code>
  
  - Go to [character_based_version/v1.0/vocab_formatter.py](https://github.com/skansi/chatbot-unk/blob/master/character_based_version/v1.0/vocab_formatter.py) and run:
   
      <code>python vocab_formatter.py</code>

 
