#Retrieve data from the Materials Project (MP) website https://materialsproject.org

#If you do not have the pymatgen library installed:
#In Anaconda: conda install -c matsci pymatgen
#Otherwise: pip install pymatgen

def RetrieveData(elements,properties,private_key):
    #Import the MPRester API library
    from pymatgen import MPRester

    #Specify private key for MP database
    mpr = MPRester(private_key)

    #Query MP-DB for all materials ids of binary alloys of transition metals. Returns a list of dictionaries.
    entries = mpr.query({"elements":{"$in":elements}, "nelements":2}, properties)

    return entries


