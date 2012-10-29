function [trainErr, testErr] = myDtree(fileName, level, fold)

    %Call the function to get the 10 folds
    tenFoldCell = GetTenFold(fileName, fold);
    
    %iterate over folds
    for foldNum = 1:length(tenFoldCell)
        %now the engineering begins, we need to recurse for each fold for
        %given levels
        tree = zeros(1,100);
        
        %Accessing training set of each fold
        current_Cell_Train = tenFoldCell{foldNum,1};
        
        %Accessing test set of each fold
        current_Cell_Test = tenFoldCell{foldNum,2};
        
        %Get train labels from the current fold
        curr_Y_train = current_Cell_Train(:,size(current_Cell_Train,2));
        
        %Get test labels for the current fold
        curr_Y_test = current_Cell_Train(:,size(current_Cell_Test, 2));
        
        inds = {1:size(current_Cell_Train,1)};
        
        [branches, best_feature] = get_best_feature(current_Cell_Train, curr_Y_train)
        level1_cell = cell(size(branches,1),3);
        for i = 1:length(branches)
           mat1 = current_Cell_Train(current_Cell_Train(:,best_feature)== branches(i,1),:); 
           level1_cell{i,1} = mat1;
        end
        %Iterate over level one data to get best split for the next level
        for i = 1:size(level1_cell,1)
            [branches1, best_feat2] = get_best_feature(level1_cell{i,1}, curr_Y_train)
            level1_cell{i,2} = branches1;
            level1_cell{i,3} = best_feat2;
        end
        
        level2_cell = cell(20,4);
        
        for i = 1:size(level1_cell,1)
            branch_array = level1_cell{i,2}
            for a = 1:length(branch_array)
                mat2 = level1_cell{i,1}(level1_cell{i,1}(:,level1_cell{i,3}) == branch_array(a,1))
                %level2_cell{a,1} = mat2
                t = branch_array(a,1)
                level2_cell{t,1}  = mat2
            end
        end
                
        %check if the split should happen or not
        for i = 1:size(level1_cell,1)
            for a = 1:length(level1_cell{i,2})
                class1 = sum(level1_cell{i,2} == 1)
                class_1 = sum(level1_cell{i,2} == -1)
                if (class1(:,2) ~= 0 || class_1(:,2) ~= 0)
                    if (class1 ~= 1 & class_1 ~= 1)
                        [branches2, best_feat3] = get_best_feature(level1_cell{i,1}, curr_Y_train)
                        level2_cell{i,1} = i
                        level2_cell{i,2} = branches2
                        level2_cell{i,3} = best_feat3
                    else
                        level2_cell{i,1} = i
                    end
                end
            end
        end
        
        level3_cell = cell(10,3);
        %set up is complete so call the split function to do the recursive
        %spilit
        [branches, split] = split_node(current_Cell_Train, curr_Y_train, current_Cell_Test, curr_Y_test, 2, inds, 1);
        
        
    end
end

function [indeces_of_node, best_split] = split_node(X_train, Y_train, X_test, Y_test, level, inds, node) %indeces_of_node, p, labels, node)
    
    %cell to save split information
    feature_to_split_cell = cell(size(X_train,2)-1,4);
    
    %iterate over features
    for feature_idx=1:(size(X_train,2) - 1)
        %get current feature
        curr_X_feature = X_train(:,feature_idx);

        %identify the unique values
        unique_values_in_feature = unique(curr_X_feature);

        H = get_entropy(Y_train); %This is actually H(X) in slides
        %temp entropy holder

        %Storage for feature element's class
        element_class = zeros(size(unique_values_in_feature,1),2);

        %conditional probability H(X|y)
        H_cond = zeros(size(unique_values_in_feature,1),1); 
        
        for aUnique=1:size(unique_values_in_feature,1)
            match = curr_X_feature(:,1)==unique_values_in_feature(aUnique);
            mat = Y_train(match);
            majority_class = mode(mat);
            element_class(aUnique,1) = unique_values_in_feature(aUnique);
            element_class(aUnique,2) = majority_class;
            H_cond(aUnique,1) = (length(mat)/size((curr_X_feature),1)) * get_entropy(mat);
        end
        
        %Getting the information gain
        IG = H - sum(H_cond);
        
        %Storing the IG of features
        feature_to_split_cell{feature_idx, 1} = feature_idx;
        feature_to_split_cell{feature_idx, 2} = max(IG);
        feature_to_split_cell{feature_idx, 3} = unique_values_in_feature;
        feature_to_split_cell{feature_idx, 4} = element_class;
    end
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
        end
    end
    display(feature_to_split)
    display(values_of_best_feature(:,1)')
    
    curr_X_feature = X_train(:,feature_to_split);
    
    best_split = feature_to_split
    indeces_of_node = unique(curr_X_feature)
    
    %testing
    for k = 1 : length(values_of_best_feature)
        % Condition to stop the recursion, if clases are pure then we are
        % done splitting, if both classes have save number of attributes
        % then we are done splitting.
        if (sum(values_of_best_feature(:,2) == -1) ~= sum(values_of_best_feature(:,2) == 1))
            if((sum(values_of_best_feature(:,2) == -1) ~= 0) || (sum(values_of_best_feature(:,2) == 1) ~= 0))
                mat1 = X_train(X_train(:,5)== values_of_best_feature(k),:);
                [indeces_of_node, best_split] = split_node(mat1, Y_train, 1, 2, 3, inds, node);
            end
        end
    end
end

function [tree] = store_tree()
    tree = cell(100,2)
    
end