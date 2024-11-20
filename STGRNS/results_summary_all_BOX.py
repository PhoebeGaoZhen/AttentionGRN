import utils_data

save_folder = 'STGRNS_data2_TRN\\'
modelname = 'STGRNS'
iteration = 5
evaluation = '3cv'
utils_data.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils_data.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)

save_folder = 'STGRNS_data1_GRN\\'
modelname = 'STGRNS'
iteration = 5
evaluation = '311'
utils_data.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils_data.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)
