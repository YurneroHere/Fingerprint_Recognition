clc;
clear all; 
close all;

cd Database


DF=[]

for i=1:2

    
str=int2str(i)

str=strcat(str,'.jpg');

I=imread(str);

% I=imresize(I,[64 64],'bilinear');
[N N]=size(I)

I=im2double(I);



[m nn c]=size(I)

if c==3
    b=rgb2gray(I);
else
    b=I;
end

% % normalise
nor=(b-mean2(b))./(std2(b));

% segment

t=graythresh(nor);

seg=im2bw(nor,t);

sigma=0.1;
psi=0.2;
gamma=0.1;
n1=4;
lambda=[5 7 9 11];
n2=4;
theta=[30 35 60 120];


for i=1:n1
    l=lambda(i);
    
    for j=1:n2
        t=theta(j);
        g1=gabor_fn(sigma,t,l,psi,gamma);
    
        GT=conv2(b,double(g1),'same');
        
    end
end 


bi=im2bw(GT);


% % mopholgy

st=strel('disk',2)

clos=imclose(bi,st);

BW2 = bwmorph(clos,'remove');



%Read Input Image
binary_image=BW2;

%Small region is taken to show output clear
binary_image = binary_image(120:400,20:250);


%Thinning
thin_image=~bwmorph(binary_image,'thin',Inf);


%Minutiae extraction
s=size(thin_image);
N=3;%window size
n=(N-1)/2;
r=s(1)+2*n;
c=s(2)+2*n;
double temp(r,c);   
temp=zeros(r,c);bifurcation=zeros(r,c);ridge=zeros(r,c);
temp((n+1):(end-n),(n+1):(end-n))=thin_image(:,:);
outImg=zeros(r,c,3);%For Display
outImg(:,:,1) = temp .* 255;
outImg(:,:,2) = temp .* 255;
outImg(:,:,3) = temp .* 255;
for x=(n+1+10):(s(1)+n-10)
    for y=(n+1+10):(s(2)+n-10)
        e=1;
        for k=x-n:x+n
            f=1;
            for l=y-n:y+n
                mat(e,f)=temp(k,l);
                f=f+1;
            end
            e=e+1;
        end;
         if(mat(2,2)==0)
            ridge(x,y)=sum(sum(~mat));
            bifurcation(x,y)=sum(sum(~mat));
         end
    end;
end;


[ridge_x ridge_y]=find(ridge==2);
len=length(ridge_x);
%For Display
for i=1:len
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),2:3)=0;
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),2:3)=0;
    outImg((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    outImg((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),1)=255;
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),1)=255;
    outImg((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
    outImg((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
end

%FINDING
[bifurcation_x bifurcation_y]=find(bifurcation==4);
len=length(bifurcation_x);
%For Display
for i=1:len
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),1:2)=0;
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),1:2)=0;
    outImg((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    outImg((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),3)=255;
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),3)=255;
    outImg((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
    outImg((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
end



% % % minimum distance graph
core=outImg(m/2,nn/2);

% find next 
ver=0

for i=1:size(outImg,1)
    
    for j=1:size(outImg,2)
      
        if outImg(i,j)>min(core)
        
          ver=ver+0;
        else
            
          ver=ver+1;
        end
        
    end
    
end



DF=[DF ;ver]
end

cd ..

inp=input('ENTER IMAGE : ')
I=imread(inp);

% I=imresize(I,[64 64],'bilinear');
[N N]=size(I)
imshow(I)
I=im2double(I);



[m nn c]=size(I)

if c==3
    b=rgb2gray(I);
else
    b=I;
end

figure,imshow(b)

% % normalise
nor=(b-mean2(b))./(std2(b));

% segment

t=graythresh(nor);

seg=im2bw(nor,t);

sigma=0.1;
psi=0.2;
gamma=0.1;
n1=4;
lambda=[5 7 9 11];
n2=4;
theta=[30 35 60 120];


for i=1:n1
    l=lambda(i);
    
    for j=1:n2
        t=theta(j);
        g1=gabor_fn(sigma,t,l,psi,gamma);
    
        GT=conv2(b,double(g1),'same');
        
    end
end 
figure,imshow(GT)

bi=im2bw(GT);

figure,imshow(bi)

% % mopholgy

st=strel('disk',2)

clos=imclose(bi,st);

BW2 = bwmorph(clos,'remove');

figure,imshow(BW2)

%Read Input Image
binary_image=BW2;

%Small region is taken to show output clear
binary_image = binary_image(120:400,20:250);
figure;imshow(binary_image);title('Input image');

%Thinning
thin_image=~bwmorph(binary_image,'thin',Inf);
figure;imshow(thin_image);title('Thinned Image');

%Minutiae extraction
s=size(thin_image);
N=3;%window size
n=(N-1)/2;
r=s(1)+2*n;
c=s(2)+2*n;
double temp(r,c);   
temp=zeros(r,c);bifurcation=zeros(r,c);ridge=zeros(r,c);
temp((n+1):(end-n),(n+1):(end-n))=thin_image(:,:);
outImg=zeros(r,c,3);%For Display
outImg(:,:,1) = temp .* 255;
outImg(:,:,2) = temp .* 255;
outImg(:,:,3) = temp .* 255;
for x=(n+1+10):(s(1)+n-10)
    for y=(n+1+10):(s(2)+n-10)
        e=1;
        for k=x-n:x+n
            f=1;
            for l=y-n:y+n
                mat(e,f)=temp(k,l);
                f=f+1;
            end
            e=e+1;
        end;
         if(mat(2,2)==0)
            ridge(x,y)=sum(sum(~mat));
            bifurcation(x,y)=sum(sum(~mat));
         end
    end;
end;


[ridge_x ridge_y]=find(ridge==2);
len=length(ridge_x);
%For Display
for i=1:len
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),2:3)=0;
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),2:3)=0;
    outImg((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    outImg((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),1)=255;
    outImg((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),1)=255;
    outImg((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
    outImg((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
end

%FINDING
[bifurcation_x bifurcation_y]=find(bifurcation==4);
len=length(bifurcation_x);
%For Display
for i=1:len
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),1:2)=0;
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),1:2)=0;
    outImg((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    outImg((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),3)=255;
    outImg((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),3)=255;
    outImg((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
    outImg((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
end
figure;imshow(outImg);title('Minutiae');


% % % minimum distance graph
core=outImg(m/2,nn/2);

% find next 
ver=0

for i=1:size(outImg,1)
    
    for j=1:size(outImg,2)
      
        if outImg(i,j)>min(core)
        
          ver=ver+0;
        else
            
          ver=ver+1;
        end
        
    end
    
end

figure,
subplot(2,2,1)
imshow(I)
title('input')

subplot(2,2,2)
imshow(GT)
title('gabor')

subplot(2,2,3)
imshow(BW2)
title('thinning ')

subplot(2,2,4)
imshow(outImg)
title('minitiuae')






QF=ver;

if QF<3e4
    msgbox('RECOGNIZED')
else
    msgbox('NOT RECOGNIZED')
end






