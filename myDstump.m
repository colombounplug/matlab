function [split_and_error, labels] = myDstump(file, fold)

    %Call the function to get the 10 folds
    tenFoldCell = GetTenFold(file, fold);
    
    %storing split and error rate for each fold
    split_and_error = zeros(length(tenFoldCell),2);
        
    %Iterate over training folds
    for n=1:length(tenFoldCell)        
        %Acessing training selt of each fold
        currentCellTrain = tenFoldCell{n,1};
        
        %Test data
        current_test_set = tenFoldCell{n,2};
        
        %Get train labels from the currnt fold
        curr_Y = currentCellTrain(:,size(currentCellTrain,2));
        
        %place holder for tree
        stump_holder = {1:size(currentCellTrain,1)};
        
        %feature to split
        %results_cell = cell(10,2);
        feature_to_split_cell = cell(size(currentCellTrain,2)-1,4);
        
        %iterate over each feature to find the best split,
        %-1 is last column is label column
        for feature_idx=1:(size(currentCellTrain,2) - 1)
            
            %get current feature
            curr_X = currentCellTrain(:,feature_idx);
            
            %identify the unique values
            unique_values_in_feature = unique(curr_X);
            
            H = get_entropy(curr_Y); %This is actually H(X) in slides
            %temp entropy holder
   
            %Storage for feature element's class
            element_class = zeros(size(unique_values_in_feature,1),2);
            
            %conditional probability H(X|y)
            H_cond = zeros(size(unique_values_in_feature,1),1);
            
            for aUnique=1:size(unique_values_in_feature,1)
                %M(M(:,1)==1,:)
                %fit=curr_feature == aUnique;
                %match = train_label(fit);
                %count = 0
%                 for i=1:length(curr_X)
%                     if curr_X(i) == aUnique
%                         display(curr_X(i))
%                         count = count + 1;
%                     end
%                 end
                %display(count)
                match = curr_X(:,1)==aUnique;
                mat = curr_Y(match);
                majority_class = mode(mat);
                element_class(aUnique,1) = unique_values_in_feature(aUnique);
                element_class(aUnique,2) = majority_class;
                H_cond(aUnique,1) = (length(mat)/size((curr_X),1)) * get_entropy(mat);
            end
        
            %Getting the information gain
            IG = H - sum(H_cond);
            
            %Storing the IG of features
            feature_to_split_cell{feature_idx, 1} = feature_idx;
            feature_to_split_cell{feature_idx, 2} = max(IG);
            feature_to_split_cell{feature_idx, 3} = unique_values_in_feature;
            feature_to_split_cell{feature_idx, 4} = element_class;
        end
        feature_to_split_cell;
        %set feature to split zero for every fold
        feature_to_split = 0;
        
        %getting the max IG of the fold
        max_IG_of_fold = max([feature_to_split_cell{:,2:2}]);
        
        %vector to store values in the best feature
        values_of_best_feature = zeros(size(15,1));
        
        %Iterating over cell to get get the index and the values under best
        %splited feature.
        for i=1:length(feature_to_split_cell)
            if (max_IG_of_fold == feature_to_split_cell{i,2});
                feature_to_split = i;
                values_of_best_feature = feature_to_split_cell{i,4};
                labels = values_of_best_feature;
            end
        end
        split_feature = feature_to_split;
        %values_of_best_feature
        %display(element_class)
        %get all the rows and iterate over the rows
        temp_error = 0;
        for j=1:size(current_test_set(:,1))
            %getting the test label
            test_label = current_test_set(j,size(current_test_set,2));
            % take the feature that we trained to be best split
            value_of_test_feature = current_test_set(j,split_feature);
            for k=1:length(element_class)
                if (value_of_test_feature == values_of_best_feature(k,1))
                    if (values_of_best_feature(k,2) ~= test_label)
                        temp_error = temp_error + 1;
                    end
                end
            end
        end
        error_rate = temp_error/size(current_test_set,1);
        test_error = error_rate;
        split_and_error(n,1) = split_feature;
        split_and_error(n,2) = test_error;
    end
    fprintf('10 fold: Feature to split and error rate')
end

%next steps
%Remove the feature that