%Function that accept data file as a name and the number of folds
%For the cross fold
function [results_cell] = GetTenFold(dataFile, x)

%loading the data file
dataMatrix = load(dataFile);

%removing the 11th feature
dataMatrix.data(:,11) = [];

%combine the data and labels as one matrix
X = [dataMatrix.data dataMatrix.labels];

%geting the length of the of matrix
dataRowNumber = length(dataMatrix.data);

%shuffle the matrix while keeping rows intact 
shuffledMatrix = X(randperm(size(X,1)),:);

crossValidationFolds = x;
%Assinging number of rows per fold
numberOfRowsPerFold = round(dataRowNumber / crossValidationFolds);

%Assigning 10X2 cell to hold each fold as training and test data
results_cell = cell(10,2);
    %starting from the first row and segment it based on folds
    i = 1;
    for startOfRow = 1:numberOfRowsPerFold:dataRowNumber
        %Find the end point of the group
        group_end = round(i*dataRowNumber/crossValidationFolds);
        if (group_end <= dataRowNumber)
            testRows = startOfRow:group_end;

            if (startOfRow == 1)
                trainRows = (max(testRows)+1:dataRowNumber);
            else
                trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
                %i = i + 1;
            end
            %%adding to a the cell
            results_cell{i,1} = shuffledMatrix(trainRows ,:);
            results_cell{i,2} = shuffledMatrix(testRows ,:);
            i = i + 1;
    end
end