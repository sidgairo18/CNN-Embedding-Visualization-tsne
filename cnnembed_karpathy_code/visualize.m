fileID = fopen('fc7_features_vgg16.txt', 'r');
formatSpec = "";
f = "%f";
dims = 4096;
sizeA = [dims Inf];

for i=1:1:dims
    formatSpec = formatSpec + " " + f;
end

features = fscanf(fileID, formatSpec, sizeA);
features = features';

fprintf('Features Loaded \n');

Y = tsne(features,'Algorithm','barneshut','NumPCAComponents',50);

fprintf('t-sne finished running, result in Y \n');

%Running Karpathy's code from here.

x = Y;
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

%% load validation image filenames
fs = textread('image_names_vgg16.txt', '%s');
N = length(fs);

%% create an embedding image
fprintf('Running embedding 1 \n');

S = 2000; % size of full embedding image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every single image

S = 2000; % size of full embedding image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every single image

Ntake = 10000;
for i=1:Ntake
    
    if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = imread(fs{i});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

%imshow(G);

imwrite(G, 'cnn_embed_2k.jpg', 'jpg');

%% do a guaranteed quade grid layout by taking nearest neighbor
fprintf('Running embedding 2 \n');

S = 2000; % size of final image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every image thumbnail

xnum = S/s;
ynum = S/s;
used = false(N, 1);

qq=length(1:s:S);
abes = zeros(qq*2,2);
i=1;
for a=1:s:S
    for b=1:s:S
        abes(i,:) = [a,b];
        i=i+1;
    end
end
%abes = abes(randperm(size(abes,1)),:); % randperm

for i=1:size(abes,1)
    a = abes(i,1);
    b = abes(i,2);
    %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
    %yf = ((b-1)/S - 0.5)/2 + 0.5;
    xf = (a-1)/S;
    yf = (b-1)/S;
    dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
    I = imread(fs{di});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);

    G(a:a+s-1, b:b+s-1, :) = I;

    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end

%imshow(G);

imwrite(G, 'cnn_embed_full_2k.jpg', 'jpg');
