clc
clear variables
close all

path = ['Executions/Dataset1/',num2str(datetime().Year),'_',num2str(datetime().Month),'_',num2str(datetime().Day),'_',num2str(datetime().Hour),'_',num2str(datetime().Minute),'_',num2str(datetime().Second,2)];
mkdir(path);

path_data_file = [path,'/Dataset1Data.xls'];

%% Dataset Anallysis

dataToUse = detectImportOptions('Fish.csv');

dataToUse.VariableNames(1) = {'Species'};
dataToUse.VariableNames(2) = {'Weight'};
dataToUse.VariableNames(3) = {'VerticalLength'};
dataToUse.VariableNames(4) = {'DiagonalLength'};
dataToUse.VariableNames(5) = {'CrossLength'};
dataToUse.VariableNames(6) = {'Height'};
dataToUse.VariableNames(7) = {'Width'};

dataToUse = setvartype(dataToUse,'Species','string'); 
dataToUse = setvartype(dataToUse,{'Species'},'categorical');
dataToUse = setvartype(dataToUse,{'Weight','VerticalLength','DiagonalLength','CrossLength','Height','Width'},'double');

dataToUse = setvaropts(dataToUse,{'Species'},'FillValue','Miscellaneous');

dataToUse_ds = readtable('Fish.csv',dataToUse);

close all

dataToUse_ds_normalized = dataToUse_ds;
dataToUse_ds_normalized.Weight = (dataToUse_ds_normalized.Weight - min(dataToUse_ds_normalized.Weight))/(max(dataToUse_ds_normalized.Weight) - min(dataToUse_ds_normalized.Weight));

dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Bream',1)) = 0;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Roach',1)) = 1;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Whitefish',1)) = 2;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Parkki',1)) = 3;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Perch',1)) = 4;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Pike',1)) = 5;
dataToUse_ds_normalized.SpeciesCat(ismember(dataToUse_ds_normalized.Species == 'Smelt',1)) = 6;

dataToUse_ds_normalized.VerticalLength = filloutliers(dataToUse_ds_normalized.VerticalLength,"linear");
dataToUse_ds_normalized.DiagonalLength = filloutliers(dataToUse_ds_normalized.DiagonalLength,"linear");
dataToUse_ds_normalized.Weight = filloutliers(dataToUse_ds_normalized.Weight,"linear");

features = [dataToUse_ds_normalized.VerticalLength, dataToUse_ds_normalized.DiagonalLength];
target = dataToUse_ds_normalized.Weight;

corr_vehicle = corrcoef([target,features])*100;
labels_heatmap = {'Weight', 'Vertical Length', 'Diagonal Length'};


figure
heatmap(labels_heatmap,labels_heatmap,corr_vehicle);
title('Correlation between features and target in %');
saveas(gcf,[path,'/heatmap_corr_features_target.png']);

close all


%% Fitting Neural Network

% Fitting Neural Network
% Training data is 70%
% Validation data the next 15%
% Test data the next 15% (unless dataset is already divided)


% To appropriate test the algorithms, ANN with a single hidden layer will be created with random
% initial weights and different structures:
% - 4, 7, 10, 12, 15 and 20 neurons in the hidden layer;
% - 10 ANN of each type;
% - Initial weights must be in the range [-2,4/I, 2.4/I], where I is the number of inputs;
% - Initial and final weights must be saved and delivered for each network and each problem;


features = transpose(features);
target = transpose(target);

target = (target - min(target))/(max(target) - min(target));

num_neurons_hidden_layer = [4,7,10,12,15,20];
num_nets = 10;
num_alg = 2;
num_inputs = size(features,1);
range_weights = 2.4/num_inputs;

train_alg = ["trainscg";"trainrp"];

nets = cell(2,10);
training = cell(2,10);


for algIndex = 1:num_alg
    for numNetIndex = 1:num_nets
        path_net = [path,'/algNum-',num2str(algIndex),'_netNum-',num2str(numNetIndex)];
        mkdir(path_net);
        
        sheet_name = ['Alg_ANN_num-',num2str(algIndex),'_',num2str(numNetIndex)];
        
        hiddenLayerSizeIndex = randi(size(num_neurons_hidden_layer,2));
        nets{algIndex,numNetIndex} = fitnet(num_neurons_hidden_layer(hiddenLayerSizeIndex),train_alg(algIndex,:));
        
        nets{algIndex,numNetIndex}.inputs{1}.processFcns={'mapstd'};
        nets{algIndex,numNetIndex}.outputs{2}.processFcns={'mapstd'};
        
        nets{algIndex,numNetIndex}.trainParam.goal = 0;
        nets{algIndex,numNetIndex}.trainParam.mu=1.0000e-003;
        nets{algIndex,numNetIndex}.trainParam.mu_inc=10;
        nets{algIndex,numNetIndex}.trainParam.mu_dec=0.1;
        nets{algIndex,numNetIndex}.trainParam.epochs =5000;
        nets{algIndex,numNetIndex}.trainParam.max_fail=5000;
        
        nets{algIndex,numNetIndex}.initFcn = 'initlay';
        nets{algIndex,numNetIndex}.layers{1}.initFcn = 'initwb';
        nets{algIndex,numNetIndex}.layers{2}.initFcn = 'initwb';
        
        nets{algIndex,numNetIndex}.inputWeights{1,1}.initFcn = 'rands';
        nets{algIndex,numNetIndex}.layerWeights{2,1}.initFcn = 'rands';
        
        
        nets{algIndex,numNetIndex}.divideParam.trainRatio = 0.70;
        nets{algIndex,numNetIndex}.divideParam.valRatio = 0.15;
        nets{algIndex,numNetIndex}.divideParam.testRatio = 0.15;
        
        
        nets{algIndex,numNetIndex} = configure(nets{algIndex,numNetIndex},features,target);
        
        IW = range_weights * rands(num_neurons_hidden_layer(hiddenLayerSizeIndex),num_inputs);
        
        LW = range_weights * rands(nets{algIndex,numNetIndex}.numOutputs,num_neurons_hidden_layer(hiddenLayerSizeIndex));
        
        nets{algIndex,numNetIndex}.IW{1,1} = IW;
        nets{algIndex,numNetIndex}.LW{2,1} = LW;
        
        
        [nets{algIndex,numNetIndex},training{algIndex,numNetIndex}] = train(nets{algIndex,numNetIndex},features,target);
        
        
        predicted_targets = nets{algIndex,numNetIndex}(features);
        prediction_target_error = gsubtract(target,predicted_targets);
        r_coefficient_matrix = corrcoef(target,predicted_targets);
        r_coefficient = r_coefficient_matrix(1,2);
        performance = perform(nets{algIndex,numNetIndex},target,predicted_targets);
        
        
        %Save Data
        
        table_column_Index = 'A';
        
        writetable(table(num_neurons_hidden_layer(hiddenLayerSizeIndex),'VariableNames',{'Neurons_Hidden_Layer'}),path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
        table_column_Index = table_column_Index + 1;
        
        
        for numIWIndex = 1:size(IW,2)
            save_initialIW_temp = table(IW(:,numIWIndex));
            save_initialIW_temp.Properties.VariableNames = {['Initial_IW',num2str(numIWIndex)]};
            
            writetable(save_initialIW_temp,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
            table_column_Index = table_column_Index + 1;
        end
        
        for numLWIndex = 1:size(transpose(LW),2)
            save_initialLW_temp = table(transpose(LW));
            save_initialLW_temp = save_initialLW_temp(:,numLWIndex);
            save_initialLW_temp.Properties.VariableNames = {['Initial_LW',num2str(numLWIndex)]};
            
            writetable(save_initialLW_temp,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
            table_column_Index = table_column_Index + 1;
        end
        
        for numIWIndex = 1:size(nets{algIndex,numNetIndex}.IW{1,1},2)
            save_endIW_temp = table(nets{algIndex,numNetIndex}.IW{1,1}(:,numIWIndex));
            save_endIW_temp.Properties.VariableNames = {['End_IW',num2str(numIWIndex)]};
            
            writetable(save_endIW_temp,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
            table_column_Index = table_column_Index + 1;
        end
        
        for numLWIndex = 1:size(transpose(nets{algIndex,numNetIndex}.LW{2,1}),2)
            save_endLW_temp = table(transpose(nets{algIndex,numNetIndex}.LW{2,1}));
            save_endLW_temp = save_endLW_temp(:,numLWIndex);
            save_endLW_temp.Properties.VariableNames = {['End_LW',num2str(numLWIndex)]};
            
            writetable(save_endLW_temp,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
            table_column_Index = table_column_Index + 1;
        end
        
        save_r_coefficient = table(r_coefficient,'VariableNames',{'R_Coefficient'});
        save_performance = table(performance,'VariableNames',{'Performance_OR_MSE'});
        
        writetable(save_performance,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
        table_column_Index = table_column_Index + 1;
        
        writetable(save_r_coefficient,path_data_file,'Sheet',sheet_name,'Range',[char(table_column_Index),'1']);
        table_column_Index = table_column_Index + 1;
        
        
        figure, plotperform(training{algIndex,numNetIndex})
        saveas(gcf,[path_net,'/plotperform.png']);
        
        figure, plottrainstate(training{algIndex,numNetIndex})
        saveas(gcf,[path_net,'/plottrainstate.png']);
        
        figure, ploterrhist(prediction_target_error)
        saveas(gcf,[path_net,'/ploterrhist.png']);
        
        figure, plotregression(target,predicted_targets)
        saveas(gcf,[path_net,'/plotregression.png']);
        
        close all
    end
end



clear variables
close all
