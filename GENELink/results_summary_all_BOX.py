import utils

save_folder = 'GENELink_data2_TRN\\'
modelname = 'GENELink'
iteration = 5
evaluation = '3cv'
utils.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)

save_folder = 'GENELink_data1_GRN\\'
modelname = 'GENELink'
iteration = 5
evaluation = '311'
utils.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)
