% author = biringaChi
% email = biringachidera@gmail.com

% This file contains code for performing image construction 
% using Singular Value Decomposition (SVD)

% Read image
image = imread('flower.bmp');
figure(1) 
imshow(image)
title('Figure 1: Original Image');
 
% Grayscale conversion
grayscale = rgb2gray(image);
figure(2)
imshow(grayscale)
title('Figure 2: Grayscale Image');

% Double data format conversion
grayscale_double = double(grayscale);

% SVD implementaion
[u, s, v] = svd(grayscale_double);

% Singular values rank
singular_values = diag(s);
figure(3)
plot(singular_values)
title('Figure 3: Rank of all the singular values');
xlabel('K values');
ylabel('Singular values');


% Top 10 singular values 
top_10 = singular_values(1:10, 1);
figure(4)
plot(top_10)
title('Figure 4: Rank of top 10 singular values');
xlabel('K values');
ylabel('Singular values');


% K = 10
mat_top_10 = diag((top_10)');
zeros_top_10 = zeros(300);
zeros_top_10(1:10, 1:10) = mat_top_10;
img_top_10 = (u * zeros_top_10) * (v');
figure(5)
imshow(img_top_10, [0 255])
title('Figure 5: K = 10');

% K = 50
top_50 = singular_values(1:50, 1);
mat_top_50 = diag((top_50)');
zeros_top_50 = zeros(300);
zeros_top_50(1:50,1:50) = mat_top_50;
img_top_50 = (u * zeros_top_50) * (v');
figure(6)
imshow(img_top_50, [0 255])
title('Figure 6: K = 50');

% K = 100
top_100 = singular_values(1:100, 1);
mat_top_100 = diag((top_100)');
zeros_top_100 = zeros(300);
zeros_top_100(1:100,1:100) = mat_top_100;
img_top_100 = (u * zeros_top_100) * (v');
figure(7)
imshow(img_top_100, [0 255])
title('Figure 7: K = 100');

% K = 200
top_200 = singular_values(1:200, 1);
mat_top_200 = diag((top_200)');
zeros_top_200 = zeros(300);
zeros_top_200(1:200, 1:200) = mat_top_200;
img_top_200 = (u * zeros_top_200) * (v');
figure(8)
imshow(img_top_200, [0 255])
title('Figure 8: K = 200');







