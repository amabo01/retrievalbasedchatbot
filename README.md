# Retrieval Based Chatbot
A retriveal based chatbot is a chatbot that takes information from the words a user inputs into the chat and produces a related response in an effort to help the user. This chatbot was customized to potentially be used on a podiatric office's website. I chose a podiatric office because as a former pre-med student, I spent 4 years working at the same podiatric office. Many of the frequently asked questions could have been answered online with the help of a chatbot opposed to calling the office frequently.

This program was written in Python.

## How to Run the Code
This code is run from the command line. To run the program, you must first run python train_chatbot.py and then run chatbotgui.py to run the gui.

## Libraries used
The front end and back end were in two separate python files. For the back end file, the following libraries were used: NLTK, JSON, pickle, PANDAS, NumPy, MatPlotLib, and ScikitLearn. NLTK, known as Natural Language Toolkit, is used to tokenize and lemmatize the words. JSON is used to import the JSON file in which the tag phrases are located. ScikitLearn is used to vectorize the words. ScikitLearn is also assisting in creating the neural network. Without the neural network, there would be no way for the system to retrieve the responses for the users. In the neural network, the user’s input is taken and compared to the key phrases stored. When a match is found, a response is generated. This response will be generated based on the input. This response will be relevant to the user’s input and will hopefully lead the user closer to their answer. For the front end, NLTK, NumPy, and Tkinter. Tkinter is used for the GUI, graphical user interface. This is what displays the chatbot. 

## What Needs to be Installed to Run the Program
In order to run this program, you must have the libraries previously listed installed. Without those libraries, you would not be able to run the program. And Python must also be installed.

## The Data
The data being analyzed is the words input by the user. NLTK tokenizes the words and ScikitLearn analyzes how many times that word was used and if that word is related to any of the words in the JSON file.
