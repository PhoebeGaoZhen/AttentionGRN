import utils_Train_Test_Split

save_folder = 'DeepFGRN_data2_TRN\\'
modelname = 'DeepFGRN'
iteration = 5
evaluation = '3cv'
utils_Train_Test_Split.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
utils_Train_Test_Split.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)

# save_folder = 'DeepFGRN_data1_GRN\\'
# modelname = 'DeepFGRN'
# iteration = 5
# evaluation = '311'
# utils_Train_Test_Split.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=500)
# utils_Train_Test_Split.results_summary(save_folder, modelname, iteration, evaluation,Rank_num=1000)
