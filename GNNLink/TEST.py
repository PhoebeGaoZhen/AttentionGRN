import utils, os
# resultspath = "GNNLink_data2_TRN/"
# if not os.path.isdir(resultspath):
#     os.makedirs(resultspath)
# resultfile = 'GNNlink_TRN_data2.csv'
# utils.results_summary(resultspath, resultfile)

resultspath = "GNNLink_data1_GRN/"
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)
resultfile = 'GNNlink-GRN-data1.csv'
utils.results_summary(resultspath, resultfile)