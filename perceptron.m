close all; 
%% Generate Training Set
dataCount = 10;

trainSetA = 10*rand(dataCount,2) + 10;
trainSetB = -10*rand(dataCount,2) + 10;

plot(trainSetB(:,1),trainSetB(:,2),'x');
hold on;
plot(trainSetA(:,1),trainSetA(:,2),'o');

setA = ones(dataCount,1);
setB = -ones(dataCount,1);
trainSetA = [trainSetA,setA]';
trainSetB = [trainSetB,setB]';
trainSet = [trainSetA,trainSetB];

%% Perceptron Algorithm
eta = 0.98;

w = [0;0];
b = 0;
k = 0;
R = sqrt(800);
mistake = 0;

while (mistake == 0)
    mistake = 1;
    
    for i = 1:20
        if (trainSet(3,i)*(dot(w,trainSet(1:2,i)) + b) <= 0)
            w = w + eta*trainSet(3,i)*trainSet(1:2,i);
            b = b + eta*trainSet(3,i)*R^2;
            k = k + 1;
            mistake = 0;
        end
    end
    
end

x = -10:0.1:20;
plot(x,(-w(1)/w(2))*x - (b/w(2)));
plot(0,0,'or');
