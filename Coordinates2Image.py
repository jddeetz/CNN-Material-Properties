#This function converts a materials entry into a 3D "image"
#White pixels (zero's) correspond to
import numpy as np

def Coordinates2Image(entry,elements,side_length):
    #Construct a numpy array consisting of zeroes with dimensions side_length^3 
    image=np.zeros([side_length,side_length,side_length],dtype=int)
    
    for atom in range(len(entry["structure"])):
        
        #Get the coordinates of this atom
        coords=entry["structure"][atom].frac_coords #This is a numpy array
        
        #Get the element string of this atom
        element=entry["structure"][atom].species_string #This is a string
        
        #Calculate the color of this atom in the image, based off its element number
        color=int((elements.index(element)+1)*256/len(elements)//1)-1
        
        #Find the location of the pixel in the image, based off of its coordinates
        x=int((coords[0]*side_length-0.000001)//1)
        y=int((coords[1]*side_length-0.000001)//1)
        z=int((coords[2]*side_length-0.000001)//1)
        
        if x >= side_length:
            x=side_length-1
        elif y >= side_length:
            y=side_length-1
        elif z >= side_length: 
            z=side_length-1
        
        #Assign the pixel color to the atoms location
        image[x,y,z]=color
        
    return image
        