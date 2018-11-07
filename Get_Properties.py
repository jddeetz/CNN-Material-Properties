#Get certain properties from a materials entry and return them

def Get_Properties(entry):
    material_id=entry["material_id"]
    density=entry["density"]
    band_gap=entry["band_gap"]
    
    return material_id,density,band_gap

