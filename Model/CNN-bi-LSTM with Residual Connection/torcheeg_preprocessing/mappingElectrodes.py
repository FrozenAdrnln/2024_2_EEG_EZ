####################################################################################################
# mappingElectrodes.py: Mapping electrode to each row with dictionary
####################################################################################################

def mapElectrodes():
    res = {}
    f = open('electrode_positions.txt', 'r')
    num = 0
    for line in f.readlines():
        res[line.split(" ")[0]] = num
        num += 1

    return res



    