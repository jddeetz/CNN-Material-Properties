'''The following script queries the Materials Project (MP) database, cleans 
the data retrieved from the database constructs 3D images of each crystal 
structure, and fits a convolutional neural network to the dataset.'''

#Import libraries
import numpy as np
import pandas as pd
import pickle
from os import listdir

#Import custom modules
import RetrieveData

#Specify the private key to be used to query the database
#This is just a string, and you can replace the Private_Key module below with
#a string
import Private_Key
key=Private_Key.Private_Key()

#Make a list of the elements in the materials to be queried for
elements=["Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","F","Cl","Br","I",
          "O","S","Se","Te"]

#Make a list of the properties to be queried for    
properties=["material_id","structure","unit_cell_formula","elasticity",
            "density","band_gap","pretty_formula","spacegroup","warnings"]

#Query the DB if PKL file does not exist, otherwise load it.
if 'entries.pkl' in listdir('.'):
    entries=pickle.load(open("entries.pkl",'rb'))
else:
    #Retrieve data from the Materials Project (MP) database
    entries = RetrieveData.RetrieveData(elements,properties,key)
    pickle.dump(entries, open("entries.pkl",'wb'))

