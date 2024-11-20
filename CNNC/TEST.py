import utils

resultspath = "./CNNC_data1_GRN/"
resultfile = 'CNNC-GRN.csv'

utils.results_summary(resultspath, resultfile)

resultspath = "./CNNC_data2_TRN/"
resultfile = 'CNNC-TRN.csv'

utils.results_summary(resultspath, resultfile)
