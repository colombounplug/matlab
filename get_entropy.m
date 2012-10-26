%Given a vector the function will calculate the entropy
%Since we know each vector is a feature we can pass 
%a vector as a parameter to calculate entropy
%Theory of entropy
%entropy = -sum((dBelongToClass/TotalPoints) .* log2(dBelongToClass/TotalPoints))
function entropy = get_entropy(label_vector)
    frequency_table = tabulate(label_vector);
    
    % Remove zero-entries
    frequency_table = frequency_table(frequency_table(:,3)~=0,:);
    
    prob = frequency_table(:,3) / 100;
    % Get entropy
    entropy = -sum((prob+1e-100) .* log2(prob + 1e-100));
    %entropy = -sum(prob2 .* log2(prob2));
end
