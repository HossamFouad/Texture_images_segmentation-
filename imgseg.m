img=imread('cat.jpg');
img=im2double(img);
LUV=zeros(size(img));
h = ones(5,5)/25;
MELH=zeros(size(img,1),size(img,2));
MEHL=zeros(size(img,1),size(img,2));
LUV = RGB2ULV(img);
img_gray=rgb2gray(img);
MELH_HL=wavlet_decomposition(img_gray) ;
Enl=0.00001;
err=0.0001;
Entropy_ns=0;
alpha=0.5;
X1=[];
%%%%%%%%%TL%%%%%%%%%%
while 1
[Entropy_ns, TL,IL] = ns(LUV(:,:,1),Enl);
fprintf('Enl= %f , Entropy_ns= %f , error= %f\n', Enl,Entropy_ns,(Entropy_ns-Enl)/Enl);
if ((Entropy_ns-Enl)/Enl)< err
    break
else
    Enl=Entropy_ns ;
end
end
disp('---------------------');
X=zeros(size(img,1),size(img,2));
TL_mean = imfilter(TL,h,'same');
X(IL<alpha)=TL(IL<alpha);
X(IL>=alpha)=TL_mean(IL>=alpha); %required
imshow(X)
X1(:,:,1)=X;
%%%%%%%%%TU%%%%%%%%%%
Enl=0.00001;
while 1
[Entropy_ns, TU,IU] = ns(LUV(:,:,2),Enl);
fprintf('Enl= %f , Entropy_ns= %f , error= %f\n', Enl,Entropy_ns,(Entropy_ns-Enl)/Enl);
if ((Entropy_ns-Enl)/Enl)< err
    break
else
    Enl=Entropy_ns ;
end
end
disp('---------------------');
X=zeros(size(img,1),size(img,2));
TU_mean = imfilter(TU,h,'same');
X(IU<alpha)=TU(IU<alpha);
X(IU>=alpha)=TU_mean(IU>=alpha); %required
imshow(X)
X1(:,:,2)=X;
%%%%%%%%%%TV%%%%%%%%%
Enl=0.00001;
while 1
[Entropy_ns, TV,IV] = ns(LUV(:,:,3),Enl);
fprintf('Enl= %f , Entropy_ns= %f , error= %f\n', Enl,Entropy_ns,(Entropy_ns-Enl)/Enl);
if ((Entropy_ns-Enl)/Enl)< err
    break
else
    Enl=Entropy_ns ;
end
end
disp('---------------------');
X=zeros(size(img,1),size(img,2));
TV_mean = imfilter(TV,h,'same');
X(IV<alpha)=TV(IV<alpha);
X(IV>=alpha)=TV_mean(IV>=alpha); %required

imshow(X)
X1(:,:,3)=X;
%%%%%%%%%TLH%%%%%%%%%%
 Enl=0.00001;
while 1
[Entropy_ns, TLH,ILH] = ns(MELH_HL(:,:,1),Enl);
fprintf('Enl= %f , Entropy_ns= %f , error= %f\n', Enl,Entropy_ns,(Entropy_ns-Enl)/Enl);
if ((Entropy_ns-Enl)/Enl)< err
    break
else
    Enl=Entropy_ns ;
end
end
disp('---------------------');
X=zeros(size(img,1),size(img,2));
TLH_mean = imfilter(TLH,h,'same');
X(ILH<alpha)=TLH(ILH<alpha);
X(ILH>=alpha)=TLH_mean(ILH>=alpha); %required

imshow(X)
X1(:,:,4)=X;
ans=X;
%%%%%%%%%THL%%%%%%%%%%
Enl=0.00001;
while 1
[Entropy_ns, THL,IHL] = ns(MELH_HL(:,:,2),Enl);
fprintf('Enl= %f , Entropy_ns= %f , error= %f\n', Enl,Entropy_ns,(Entropy_ns-Enl)/Enl);
if ((Entropy_ns-Enl)/Enl)< err
    break
else
    Enl=Entropy_ns ;
end
end
disp('---------------------');
X=zeros(size(img,1),size(img,2));
THL_mean = imfilter(THL,h,'same');
X(IHL<alpha)=THL(IHL<alpha);
X(IHL>=alpha)=THL_mean(IHL>=alpha); %required
imshow(X)
X1(:,:,5)=X;
%%%%%%%%%%%%%%%%%%%
nrows = size(X1,1);
ncols = size(X1,2);
X1 = reshape(X1,nrows*ncols,5);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(X1,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
imshow(pixel_labels,[]), title('image labeled by cluster index');


%% get Entropy of input image
function [Entropy_ns, T,I]= ns(G,Enl)
% Calculate local mean value of the image
h = ones(5,5)/25;
g_mean = imfilter(G,h,'same');
% figure;
% subplot(1,2,1),imagesc(G),title('Original');
% subplot(1,2,2),imagesc(g_mean),title('Mean');
% get g_mean max and min value
g_min=min(min(im2double(g_mean)));
g_max=max(max(im2double(g_mean)));
[rows,cols] = size(G);

% En=entropy(G);
% Enmax=-log2(1/(rows*cols));
% Enmin=0;
% AlphaMax=0.1;
% AlphaMin=0.01;
% Alpha=AlphaMin+((AlphaMax-AlphaMin)*(En-Enmin)/(Enmax-Enmin));
% Beta=1-Alpha;

g_min = g_min.*ones(rows,cols);
g_max = g_max.*ones(rows,cols);
%figure,imshow(gmax),title('maxg');
%figure,imshow(gmin),title('ming');
% Absolute Value
o=abs(G-g_mean);
%figure,imshow(o),title('dif');
% get absolute value max and min
mino=min(min(im2double(o)));
maxo=max(max(im2double(o)));
o_min = mino.*ones(rows,cols);
o_max = maxo.*ones(rows,cols);
%figure,imshow(omax),title('maxo');
%figure,imshow(omin),title('mino');
% get True , Flase and Indetermincy
T=(im2double(g_mean)-g_min)./(g_max-g_min);
I=(im2double(o)-o_min)./(o_max-o_min);
F=1-T;

% figure,
% subplot(1,3,1),imshow(T),title('T');
% subplot(1,3,2),imshow(I),title('I');
% subplot(1,3,3),imshow(F),title('F');
% En=entropy(T)+entropy(I)+entropy(F);
% get Entorpy
Enmax=-log2(1/(rows*cols));
Enmin=0;
AlphaMax=0.1;
AlphaMin=0.01;
Alpha=AlphaMin+((AlphaMax-AlphaMin)*(Enl-Enmin)/(Enmax-Enmin));
Beta=1-Alpha;
Entropy_ns=entropy(I);
fprintf('Alpha= %f , Beta = %f\n', Alpha,Beta);
% T_alpha mean
meanT = imfilter(T,h,'same');
AlphaT=ones(rows,cols);
AlphaT(I<Alpha)=T(I<Alpha);
AlphaT(I>=Alpha)=meanT(I>=Alpha); %required

% F_alpha mean
meanF = imfilter(F,h,'same');
AlphaF=ones(rows,cols);
AlphaF(I<Alpha)=F(I<Alpha);
AlphaF(I>=Alpha)=meanF(I>=Alpha); % required

AlphameanT = imfilter(AlphaT,h,'same');
meanI=abs(AlphaT-AlphameanT);
minmeanI=min(min(im2double(meanI)));
maxmeanI=max(max(im2double(meanI)));
meanImin = minmeanI.*ones(rows,cols);
meanImax = maxmeanI.*ones(rows,cols);
AlphaI=(im2double(meanI)-meanImin)./(meanImax-meanImin); %requierd

% figure,
% subplot(1,3,1),imshow(AlphaT),title('Alpha mean T');
% subplot(1,3,2),imshow(AlphaI),title('Alpha mean I');
% subplot(1,3,3),imshow(AlphaF),title('Alpha mean F');


EnhT=ones(rows,cols);
EnhT(AlphaT<=0.5) = 2*EnhT(AlphaT<=0.5).^2;
EnhT(AlphaT>0.5) = 1-2*(1-EnhT(AlphaT>0.5)).^2;

EnhF=ones(rows,cols);
EnhF(AlphaF<=0.5) = 2*EnhF(AlphaF<=0.5).^2;
EnhF(AlphaF>0.5) = 1-2*(1-EnhF(AlphaF>0.5)).^2;

% figure,
% subplot(1,2,1),imshow(EnhT),title('Enhanced T');
% subplot(1,2,2),imshow(EnhF),title('Enhanced F');

BetaEnhT=ones(rows,cols);
BetaEnhT(AlphaI<Beta)=AlphaT(AlphaI<Beta);
BetaEnhT(AlphaI>=Beta)=EnhT(AlphaI>=Beta); %required

BetaEnhF=ones(rows,cols);
BetaEnhF(AlphaI<Beta)=AlphaF(AlphaI<Beta);
BetaEnhF(AlphaI>=Beta)=EnhF(AlphaI>=Beta); %required

BetaT = imfilter(BetaEnhT,h);
BetaI=abs(BetaEnhT-BetaT);
minbetaI=min(min(im2double(BetaI)));
maxbetaI=max(max(im2double(BetaI)));
betaImin = minbetaI.*ones(rows,cols);
betaImax = maxbetaI.*ones(rows,cols);
BetaEnhI=(im2double(BetaI)-betaImin)./(betaImax-betaImin); %required
% figure,
% subplot(1,3,1),imshow(BetaEnhT),title('Beta Enhanced T');
% subplot(1,3,2),imagesc(BetaEnhI),title('Beta Enhanced I');
% subplot(1,3,3),imshow(BetaEnhF),title('Beta Enhanced F');

% thres = graythresh(T);
% err=1;
% while err>0.0001,
%     mu1 = mean(T(T<=thres));
%     mu2 = mean(T(T>thres));
%     thres2 = (mu1+mu2)/2;
%     err = thres2-thres;
%     thres = thres2;
% end
% tt = thres;
% BinT = im2bw(T,thres);
% thres = graythresh(F);
% err=1;
% while err>0.0001,
%     mu1 = mean(F(F<=thres));
%     mu2 = mean(F(F>thres));
%     thres2 = (mu1+mu2)/2;
%     err = thres2-thres;
%     thres = thres2;
% end
% tf = thres;
% BinF = im2bw(F,thres);
% 
% [Gx, Gy] = gradient(im2double(A));
% eg = sqrt(Gx.*Gx+Gy.*Gy);
% eg=  eg/max(max(eg));
% sd = colfilt(im2double(A),[7 7],'sliding',@std);
% sd = sd/max(max(sd));
% 
% Homogen = 1 - sd.*eg;
% Indet = 1-Homogen;
% 
% lambda = 0.01;
% IO = zeros(size(T));
% IO((T>=tt)&(Indet<lambda)) = 1;
% IE = zeros(size(T));
% IE(((T<tt)|(F<tf))&(Indet>=lambda))=1;
% IB = zeros(size(F));
% IB((F>=tf)&(Indet<lambda))=1;
% 
% BinImage = ones(size(T));
% BinImage((IO==1)|(IB==1)|(IE==0))=0;
% 
% figure,imshow(BinImage),title('Binary Image based on T,I,F');
% 
% BinImage = 1-BinImage;
% cc = bwconncomp(BinImage);
% 
% numr = cc.NumObjects;
% 
% [RA, GA, BA] = deal(Orig);
% SegR = Orig;
% cumlab = zeros(size(T));
% for i=1:numr,
%     
%     labels=zeros(size(T));
%     labels(cc.PixelIdxList{i})=1;
%     labels=bwperim(imerode(labels,strel('disk',3)));
%     if sum(sum(labels))/numel(labels)>0.01,
%         cumlab = cumlab+labels;
%     end
%     
% end
% RA(cumlab>0)=1.0;
% GA(cumlab>0)=0.0;
% BA(cumlab>0)=0.0;
% colorImage = cat(3,RA,GA,BA);
% figure,imshow(colorImage),title('Segmentation Result');
% SegR(cumlab>0)=1.0;
% imwrite(SegR,'segmented.jpg','jpg');
% figure,imshow(SegR),title('Segmentation Result');
% 

%Entns1=entropy(T)+entropy(I)+entropy(F);
%Entns2=entropy(AlphaT)+entropy(AlphaI)+entropy(AlphaF);


end




%% Function convert grayscale image to wavelet subband HL and LH
function MELH_HL = wavlet_decomposition(X) 
%[Lo_D,Hi_D,Lo_R,Hi_R] = biorfilt(DF,RF) computes four filters associated 
%with the biorthogonal wavelet specified by decomposition filter DF 
%and reconstruction filter RF. These filters are
[Rf,Df] = biorwavf('bior2.2');
%It is well known in the subband filtering community that if the same FIR 
%filters are used for reconstruction and decomposition, then symmetry and 
%exact reconstruction are incompatible (except with the Haar wavelet). 
%Therefore, with biorthogonal filters, two wavelets are introduced instead 
%of just one
[LoD,HiD,LoR,HiR] = biorfilt(Df,Rf);
% plot
% figure;
% subplot(211); stem(LoD);
% title('Dec. low-pass filter bior2.2');
% subplot(212); stem(HiD);
% title('Dec. high-pass filter bior2.2');
%read image

HA = conv2(X,LoD(:)','same');
HD = conv2(X,HiD(:)','same');
V_LH1 = conv2(HA',HiD(:)','same');
V_LH1 =V_LH1';
H_HL1 = conv2(HD',LoD(:)','same');
H_HL1 = H_HL1';



%wcodemat rescales an input matrix to a specified range for display. 
%If the specified range is the full range of the current colormap
V1img = wcodemat(V_LH1,255,'mat',1);
H1img = wcodemat(H_HL1,255,'mat',1);
% figure;
% imagesc(V1img);
% title('Vertical(LH subband) detail Coef. of Level 1');
% figure;
% imagesc(H1img);
% title('Horizontal(HL subband) detail Coef. of Level 1');

filter_E=1/25*ones(5,5);
MELH=conv2(V_LH1,filter_E,'same');
MEHL=conv2(H_HL1,filter_E,'same');
% figure;
% imagesc(MELH);
% title('MELH');
% figure;
% imagesc(MEHL);
% title('MEHL');
MELH_HL(:,:,1)=MELH;
MELH_HL(:,:,2)=MEHL;
end


%% Convert color Image to LUV space
function LUV = RGB2ULV(RGB)
clc;
% Converting from RGB to Gray level
%Gray = rgb2gray(RGB);

% Converting from RGB to XYZ
XYZ_struct = makecform('srgb2xyz');
XYZ = applycform(RGB,XYZ_struct);

% Converting from XYZ to LUV
LUV_struct = makecform('xyz2uvl');
LUV = applycform(XYZ,LUV_struct);

% figure
% imshow(RGB);
% title('RGB');
% figure
% imshow(Gray);
% title('Gray');
% figure
% imshow(XYZ);
% title('XYZ');
% figure
% imshow(LUV);
% title('LUV');


% Seperating L U V

% figure
% subplot(2,2,1), imagesc(LUV(:,:,1)) , title('L');
% subplot(2,2,2), imagesc(LUV(:,:,2)) , title('U');
% subplot(2,2,3), imagesc(LUV(:,:,3)) , title('V');
end

function []=kmean(I,K)
%% K-means Segmentation (option: K Number of Segments)
% Alireza Asvadi
% http://www.a-asvadi.ir
% 2012
% Questions regarding the code may be directed to alireza.asvadi@gmail.com

%% Load Image
I = im2double(I);                    % Load Image
F = reshape(I,size(I,1)*size(I,2),5);                 % Color Features
%% K-means                                            % Cluster Numbers
CENTS = F( ceil(rand(K,1)*size(F,1)) ,:);             % Cluster Centers
DAL   = zeros(size(F,1),K+2);                         % Distances and Labels
KMI   = 10;                                           % K-means Iteration
for n = 1:KMI
   for i = 1:size(F,1)
      for j = 1:K  
        DAL(i,j) = norm(F(i,:) - CENTS(j,:));      
      end
      [Distance, CN] = min(DAL(i,1:K));               % 1:K are Distance from Cluster Centers 1:K 
      DAL(i,K+1) = CN;                                % K+1 is Cluster Label
      DAL(i,K+2) = Distance;                          % K+2 is Minimum Distance
   end
   for i = 1:K
      A = (DAL(:,K+1) == i);                          % Cluster K Points
      CENTS(i,:) = mean(F(A,:));                      % New Cluster Centers
      if sum(isnan(CENTS(:))) ~= 0                    % If CENTS(i,:) Is Nan Then Replace It With Random Point
         NC = find(isnan(CENTS(:,1)) == 1);           % Find Nan Centers
         for Ind = 1:size(NC,1)
         CENTS(NC(Ind),:) = F(randi(size(F,1)),:);
         end
      end
   end
end

X = zeros(size(F));
for i = 1:K
idx = find(DAL(:,K+1) == i);
X(idx,:) = repmat(CENTS(i,:),size(idx,1),1); 
end
T = reshape(X,size(I,1),size(I,2),5);
%% Show
figure()
subplot(121); imshow(I); title('original')
subplot(122); imshow(T); title('segmented')
disp('number of segments ='); disp(K)
end