import utils_gz

# save_folder = 'STGRNS_data2_TRN\\'
# modelname = 'STGRNS'
# iteration = 5
# evaluation = '3cv'
# utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
# utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)

save_folder = 'AttentionGRN_data1_GRN\\'
modelname = 'AttentionGRN'
iteration = 5
evaluation = '311'
utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)
