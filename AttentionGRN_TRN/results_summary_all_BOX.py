import utils_gz

save_folder = 'AttentionGRN_data2_TRN\\'
modelname = 'AttentionGRN-TFgene'
iteration = 5
evaluation = '3cv'
utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils_gz.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)

