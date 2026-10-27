from eppy.modeleditor import IDF

iddfile = "/Applications/EnergyPlus-8-8-0/Energy+.idd"
IDF.setiddname(iddfile)

idfname = "/Users/bhabalaj/Repositories/COML/pyEp/pyEp/pyEp/example_buildings/LargeOffice/LargeOfficeFUN.idf"
epwfile = "/Users/bhabalaj/Repositories/COML/pyEp/pyEp/SPtMasterTable_587017_2012_amy.epw"

idf = IDF(idfname, epwfile)
idf.run()
