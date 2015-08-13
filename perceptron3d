close all; 
clear;
hold on;
view(3);
grid on;

%% Options
trainingDataSource = 1;
drawPlane = 0;

%% Generate training data from Iris file
if trainingDataSource == 0
    trainSet = importdata('iris.txt');
    trainSet = [trainSet(:,1:3),trainSet(:,5)];
    trainSet = trainSet';
    hold on
    view(3)
    grid on
    for j = 1:length(trainSet)
        if trainSet(end,j) == 1
            scatter3(trainSet(1,j),trainSet(2,j),trainSet(3,j),'or');
        elseif trainSet(end,j) == -1
            scatter3(trainSet(1,j),trainSet(2,j),trainSet(3,j),'xg');
        end
    end
end

%% Generating Training Set from seagull
if trainingDataSource == 1
    seagull = double(imread('seagull.jpg'));
    trainSetA = [reshape(seagull(10:60,10:60,1),[1,51^2]);reshape(seagull(10:60,10:60,2),[1,51^2]);reshape(seagull(10:60,10:60,3),[1,51^2]);ones(1,51^2)];
    trainSetB = [reshape(seagull(30:40,80:100,1),[1,11*21]);reshape(seagull(30:40,80:100,2),[1,11*21]);reshape(seagull(30:40,80:100,3),[1,11*21]);-1*(ones(1,11*21))];
    trainSet = [trainSetA(:,1:1000),trainSetB];
    for j = 1:length(trainSet)
        if trainSet(end,j) == 1
            scatter3(trainSet(1,j),trainSet(2,j),trainSet(3,j),'or');
        elseif trainSet(end,j) == -1
            scatter3(trainSet(1,j),trainSet(2,j),trainSet(3,j),'xg');
        end
    end
end

%% Perceptron Algorithm
eta = 0.1;

w = zeros(size(trainSet(1:end-1,1)));
b = 0;
k = 0;
R = max(trainSet(1,:).^2 + trainSet(2,:).^2 + trainSet(3,:).^2);
mistake = 0;

while (mistake == 0)
    mistake = 1;
    
    for i = 1:length(trainSet)
        if (trainSet(end,i)*(dot(w,trainSet(1:end-1,i)) + b) <= 0)
            w = w + eta*trainSet(end,i)*trainSet(1:end-1,i);
            b = b + eta*trainSet(end,i)*R;
            k = k + 1;
            mistake = 0;
        end
    end
    
end
%% Draw hyperplane
if drawPlane == 1
    shading flat
    x = 0:.1:200;
    [X, Y] = meshgrid(x);
    Z = (-b - w(1)*X - w(2)*Y)/w(3);
    surf(X,Y,Z)
end

%% Display Segmented Image
figure;
segmented = zeros(size(seagull,1),length(seagull));
for i = 1:size(seagull,1)
    for j = 1:length(seagull)
        if (dot(w,squeeze(seagull(i,j,1:3))) + b <= 0)
            segmented(i,j) = 0;
        else 
            segmented(i,j) = 1;
        end
    end
end
imagesc(segmented);
