function per_error = Per_Err(err)
    error = 0;
    for i= 1:size(err,1)
        for n= 1:size(err,2)
            if(err(i,n) > 0) error = error + err(i,n);
            end
        end
    end
    per_error = error / size(err,1);
end