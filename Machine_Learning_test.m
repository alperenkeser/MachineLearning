clear
A = importdata('ann-test.data');
load('Trained_Network.mat','Weight1','Weight2','Wbias1','Wbias2');
input = A(:,1:21);
B = A(:,22);
for i= 1:size(B,1)
    if(B(i)==1)
        correct_Output(i,1) = 1;
    elseif(B(i)==2)
        correct_Output(i,2) = 1;
    else
        correct_Output(i,3) = 1;
    end 
end

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

error3 = output - correct_Output;