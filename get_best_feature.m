function [branches, best_feature] = get_best_feature(X_train, Y_train)
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
    best_feature= feature_to_split;
    branches = values_of_best_feature;
end