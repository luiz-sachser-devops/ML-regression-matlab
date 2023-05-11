clc
clear variables
close all

path = ['Executions/Dataset3/',num2str(datetime().Year),'_',num2str(datetime().Month),'_',num2str(datetime().Day),'_',num2str(datetime().Hour),'_',num2str(datetime().Minute),'_',num2str(datetime().Second,2)];
mkdir(path);

path_data_file = [path,'/Dataset3Data.xls'];

%% Dataset Anallysis

dataToUse_opts = detectImportOptions('Bitcoin.csv');
getvaropts(dataToUse_opts,{'Date','Close'});
dataToUse_opts.SelectedVariableNames = {'Date','Close'};

dataToUse_ds = readtable('Bitcoin.csv',dataToUse_opts);
dataToUse_ds.Properties.VariableNames = {'Date','Close'};
dataToUse_ds.Date.Format = 'yyyy-MM-dd';
dataToUse_ds = rmmissing(dataToUse_ds);


close all

% Normalize data
dataToUse_ds.CloseNormalized = (dataToUse_ds.Close - min(dataToUse_ds.Close))/(max(dataToUse_ds.Close) - min(dataToUse_ds.Close));
pUnit = dataToUse_ds.Date;
features = datenum(dataToUse_ds.Date);
features = num2cell(features.');
targets = dataToUse_ds.Close;
targets = num2cell(targets.');


figure, plot(dataToUse_ds.Date,dataToUse_ds.CloseNormalized), title('Scatter Close Value Normalized')
saveas(gcf,[path,'/scatter_Scatter Close Value Normalized.png']);

hold on;

close all
clear variable pUnit


%% Time Series Neural Network

% Time Series Neural Network
% Training data is 70%
% Validation data the next 15%
% Test data the next 15% (unless dataset is already divided)

% To appropriate test the algorithms, ANN with a single hidden layer will be created with random
% initial weights and different structures:
% - 4, 7, 10, 12, 15 and 20 neurons in the hidden layer;
% - 10 ANN of each type;

inputDelays = 2;
feedbackDelays = 2;
num_neurons_hidden_layer = [4,7,10,12,15,20];
num_nets = 10;
num_alg = 2;

train_alg = ["trainscg"; "trainrp"];

nets = cell(2,10);
training = cell(2,10);

for algIndex = 1:num_alg 
    for numNetIndex = 1:num_nets 
        path_net = [path,'/algNum-',num2str(algIndex),'_netNum-',num2str(numNetIndex)];
        mkdir(path_net);
        
        sheet_name = ['Alg_ANN_num-',num2str(algIndex),'_',num2str(numNetIndex)];
        
        hiddenLayerSizeIndex = randi(size(num_neurons_hidden_layer,2));
        nets{algIndex,numNetIndex} = narxnet(inputDelays,feedbackDelays,num_neurons_hidden_layer(hiddenLayerSizeIndex),'open',train_alg(algIndex,:));
        
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
        nets{algIndex,numNetIndex}.inputWeights{1,2}.initFcn = 'rands';
        nets{algIndex,numNetIndex}.layerWeights{2,1}.initFcn = 'rands';
       
        [x,xi,ai,targets_to_predict] = preparets(nets{algIndex,numNetIndex},features,{},targets);
        
        nets{algIndex,numNetIndex}.divideParam.trainRatio = 0.70;
        nets{algIndex,numNetIndex}.divideParam.valRatio = 0.15;
        nets{algIndex,numNetIndex}.divideParam.testRatio = 0.15;
        
        nets{algIndex,numNetIndex} = configure(nets{algIndex,numNetIndex},x,targets_to_predict);
        
        num_inputs = nets{algIndex,numNetIndex}.numInputs;
        range_weights = 2.4/num_inputs;
        
        IW = range_weights * rands(num_neurons_hidden_layer(hiddenLayerSizeIndex),num_inputs);
        
        LW = range_weights * rands(nets{algIndex,numNetIndex}.numOutputs,num_neurons_hidden_layer(hiddenLayerSizeIndex));
        
        nets{algIndex,numNetIndex}.IW{1,1} = IW(:,1);
        nets{algIndex,numNetIndex}.IW{1,2} = IW(:,2);
        nets{algIndex,numNetIndex}.LW{2,1} = LW;
        
        % Train the Network
        [nets{algIndex,numNetIndex},training{algIndex,numNetIndex}] = train(nets{algIndex,numNetIndex},x,targets_to_predict,xi,ai);
        
        
        % Test the Network
        predicted_targets = nets{algIndex,numNetIndex}(x,xi,ai);
        
        r_coefficient_matrix = corrcoef(cell2mat(targets_to_predict),cell2mat(predicted_targets));
        r_coefficient = r_coefficient_matrix(1,2);
        
        prediction_target_error = gsubtract(targets_to_predict,predicted_targets);
        performance = perform(nets{algIndex,numNetIndex},targets_to_predict,predicted_targets);
        
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
        
        for numIWIndex = 1:size(nets{algIndex,numNetIndex}.IW{1,2},2)
            save_endIW_temp = table(nets{algIndex,numNetIndex}.IW{1,2}(:,numIWIndex));
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
        
        figure, plotregression(targets_to_predict,predicted_targets)
        saveas(gcf,[path_net,'/plotregression.png']);
        
        figure, plotresponse(targets_to_predict,predicted_targets)
        saveas(gcf,[path_net,'/plotresponse.png']);
        
        figure, ploterrcorr(prediction_target_error)
        saveas(gcf,[path_net,'/ploterrcorr.png']);
        
        figure, plotinerrcorr(x,prediction_target_error)
        saveas(gcf,[path_net,'/plotinerrcorr.png']);
        
        close all
    end
end


clear variables
close all