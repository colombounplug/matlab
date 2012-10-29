function conEntrop = get_conditional_entropy(feature)
    %Calculate entropy for label given the feature that we are looking
    conEntrop = 0;
    
    %building the frequency table for given feature
    freq_table = tabulate(feature);
    % Remove zero-entries
    freq_table = freq_table(freq_table(:,3)~=0,:);    
    for i = 1:size(freq_table,1)
        %Entropy for a label (1,-1) considering feature as current value.
        H = get_entropy((feature == freq_table(i,1)));
        %H = ent(Y(X == tab(i,1)));
        
        %Now we calculate probability
        prob = freq_table(i,3)/100;
        
        %finally we are doing the summation
        conEntrop = conEntrop + prob * H;
    end
    
end
