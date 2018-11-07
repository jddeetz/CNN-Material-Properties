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
import Get_Properties

#Specify the private key to be used to query the database
#This is just a string, and you can replace the Private_Key module below with
#a string for your own personal key
import Private_Key #Feel free to delete this
key=Private_Key.Private_Key() #Feel replace with your own key as a string here.

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
    
#For each entry, perform several operations:
#Gather quantities we wish to predict, such as density and band gap energy
#Make a 3D image of the material, based on each atoms coordinates and elements
    
properties=pd.DataFrame([],columns=["material_id","density","band_gap"])

for entry in entries:
    
    #If this material has information on both its density and band gap energy
    if entry["density"] and entry["band_gap"]:
        
        #Get the material_id, density and band gap to put into a pd dataframe
        material_id,density,band_gap = Get_Properties.Get_Properties(entry)
        
        #Add these to a pandas dataframe
        properties.loc[len(properties)] = {"material_id":material_id,"density":density,"band_gap":band_gap}
