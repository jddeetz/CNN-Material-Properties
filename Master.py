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
import Coordinates2Image as c2i
import CNNOptimization as cnn_opt
import SplitData as sd

#Import keras libraries
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D
from keras.layers import AveragePooling3D, MaxPooling3D, Dropout, GlobalMaxPooling3D, GlobalAveragePooling3D
from keras.models import Model
from keras import regularizers

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
    
properties=pd.DataFrame([],columns=["material_id","density"])

#Make an empty list of images of materials
images=[]

for entry in entries:
    
    #Filter out materials that do not have both elements from the list above
    found=0
    for key,value in entry["unit_cell_formula"].items():
        if key not in elements: found=1
    if found==1: continue
        
    #If this material has information on its density
    if entry["density"]:
        
        #Get the material_id, density and band gap to put into a pd dataframe
        material_id,density = Get_Properties.Get_Properties(entry)
        
        #Add these to a pandas dataframe
        properties.loc[len(properties)] = {"material_id":material_id,"density":density}
        
        #Convert the coordinates and elements of a material into a 3D "picture"
        #In practice, this will be stored as a list of 3D numpy arrays where 
        #each of the 3D are used for a dimension of the material coordinates 
        
        #Each of the 3D arrays is an image, where a 0 corresponds to empty space,
        #and a different non-zero value corresponds to which element occupies that space
        
        #Define the resolution of the image, for example if side_length=8
        #the resulting image will be 8x8x8
        side_length=8
        
        #Generate image of the material
        image=c2i.Coordinates2Image(entry,elements,side_length)
        
        #Record the image
        images.append(image)

#Split the dataset into a training set and test set
X_train,X_test,y_train,y_test = sd.SplitData(images,properties,split=0.3)

#Initiate and compile the CNN model
cnn = cnn_opt.CNNModel(X_train.shape[1:])
cnn.compile('adam', loss='mean_squared_error', metrics=['mae'])

#Train the cnn model
cnn.fit(X_train, y_train, epochs=100, batch_size=50)
   
#Predict the densities of materials given the optimized CNN 
y_predict = cnn.predict(X_test)

#Summarize the root mean square difference (RMSD)
RMSD=np.mean((y_predict-y_test)**2)**0.5
print("The RMSD in density prediction is ",RMSD)