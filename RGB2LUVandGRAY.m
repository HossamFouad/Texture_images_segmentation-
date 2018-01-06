RGB = imread('A.jpg');

% Converting from RGB to Gray level
Gray = rgb2gray(RGB);

% Converting from RGB to XYZ
XYZ_struct = makecform('srgb2xyz');
XYZ = applycform(RGB,XYZ_struct);

% Converting from XYZ to LUV
LUV_struct = makecform('xyz2uvl');
LUV = applycform(XYZ,LUV_struct);

figure
imshow(RGB);
title('RGB');
figure
imshow(Gray);
title('Gray');
figure
imshow(XYZ);
title('XYZ');
figure
imshow(LUV);
title('LUV');


% Seperating L U V
U = LUV(:,:,1);
V = LUV(:,:,2);
L = LUV(:,:,3);
figure
subplot(2,2,1), imshow(L) , title('L');
subplot(2,2,2), imshow(U) , title('U');
subplot(2,2,3), imshow(V) , title('V');




