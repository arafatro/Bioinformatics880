# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm!!')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from bnlp import BengaliWord2Vec

bwv = BengaliWord2Vec()
model_path = "model/bengali_word2vec.model"
word = 'গ্রাম'
vector = bwv.generate_word_vector(model_path, word)
print(vector.shape)
print(vector)
