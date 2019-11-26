A = importdata('ann-train.data');
input = A(:,1:21); % I give 21 features into input matrix from ann-train.data
B = A(:,22);       % I give real output data into B matrix from ann-train.data
for i= 1:size(B,1) % There are 3 different classes in B matrix ,so I seperate them to 3772x3 matrix. 
    if(B(i)==1)
        correct_Output(i,1) = 1;
    elseif(B(i)==2)
        correct_Output(i,2) = 1;
    else
        correct_Output(i,3) = 1;
    end 
end

% I did 3 layer neural network. The first layer has 21 input, second one is
% hidden layer ,so I made 5 different hidden value, The last layer is our 
% output layer ,so 3 output represent for 3 different classes.

Wbias1  = 2*rand(1,5);      % Then, Wbias1 means first layer bias weights matrix (matrix size is 1x5 )
Wbias2  = 2*rand(1,3);      % Wbias2 means second layer bias weights matrix (matrix size is 1x3 )
Weight1 = 2*rand(21,5);     % Weight1 means first layer weight matrix (matrix size is 21x5 )
Weight2 = 2*rand(5,3);      % Weight2 means second layer weight matrix (matrix size is 21x5 )

Learning = 0.1;             % Learning is Learning Rate
delta1(1:21,1:5) = 0;       % delta1 means first layer delta term
delta2(1:5,1:3) = 0;        % delta2 means second layer delta term
deltaBias1(1,1:5) = 0;      % I made deltaBias term because our weights and
deltaBias2(1,1:3) = 0;      % Wbiases made different matrixes
biasMatrix(1:size(A,1),1) = 1; %biasMatrix matrix contains A matrix size ones.

% I want to ensure are our Weights changing or not ,so I made Before_update_weight that 
% means first version of weight datas.
Before_update_weight1 = Weight1; 
Before_update_weight2 = Weight2;
Before_update_wbias2 = Wbias2;
Before_update_wbias1 = Wbias1;

% This for loop does forward prop and backward prob 100 times and updates
% weigths.
for epoch = 1:100 
    Weighted_sum1 = input * Weight1;

    for i = 1:size(Weight1,2)
        for n = 1:size(input,1)
        hidden_value(n,i) = Sigmoid(Weighted_sum1(n,i)+Wbias1(1,i));
        end
    end

    Weighted_sum2 = hidden_value * Weight2;
    
    for i = 1:size(Weight2,2)
        for n = 1:size(input,1)
        output(n,i) = Sigmoid(Weighted_sum2(n,i)+Wbias2(1,i));
        end
    end
    %error3 means last layer (3rd layer) error.(error = hq - y)
    error3 = output - correct_Output;
    error2 = (Weight2 * error3.').' .* (hidden_value.*(1-hidden_value));   ;%?2 = (?2)T ?3 . *(a2 . * (1 - a2))
   
    delta2 = delta2 + hidden_value.' * error3;
    D2 = (1/size(input,1)) * delta2 + Learning * Weight2;
    
    deltaBias2 = deltaBias2 + biasMatrix.' * error3;
    Dbias2 = (1/size(input,1))*deltaBias2;
       
     
    
    delta1 = delta1 + input.' * error2;
    D1 = (1/size(input,1)) * delta1 + Learning * Weight1;
    
    deltaBias1 = deltaBias1 + biasMatrix.' * error2;
    Dbias1 = (1/size(input,1))*deltaBias1;
    
    Weight2 = Weight2 - D2;
    %Wbias2 = Wbias2 - Dbias2;
    %Wbias1 = Wbias1 - Dbias1;
    Weight1 = Weight1 - D1;
    
end
Error = Per_Err(error3);
% This trained weight updates this 'Trained_Network.mat'
save('Trained_Network.mat');