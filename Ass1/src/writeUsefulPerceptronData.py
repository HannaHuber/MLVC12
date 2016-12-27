# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:04:23 2016

@author: Lena

The given file perceptrondata.csv contains blanks at random places.
This file writes a new file perceptrondataUseful.csx, where the data of one line
is seperated by comma (,) and the different lines are written in a new line 
(seperated by \n)
"""

def load_file(file_name):
    '''
    file_name (string): the name of the file containing 
    the data    
    
    Returns: a list containing the data of one line as a string, entries separated
    by a comma
    '''
    print('Loading file...')
    # inFile: file
    in_file = open(file_name, 'r')
    # line: string
    lines = in_file.readlines()
    output = []
    for line in lines:
        # word_list: list of strings
        word_list = line.split()
        output.append(','.join(map(str, word_list))) 
    in_file.close()
    print('Finished loading of file...')
    return output
    
def write_file(file_name, data):
    '''
    file_name (string): the name of the new file to store the data

    data(list strings): writes the data separated by \n 
    '''
    print('Write file...')
    # inFile: file
    out_file = open(file_name, 'w')    
    out_file.write('\n'.join(data))
    print('Finished writing of file...')
    out_file.close()
       
data = load_file('perceptrondata.csv')
write_file('perceptrondataUseful.csv',data)