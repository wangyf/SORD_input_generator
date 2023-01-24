clear all

L = [60e3,20e3,40e3];
dx = 50;
dy = 50;
nn = floor(L/dx)+1;


%% compute normal and shear stress
stress1 = zeros(nn(1)-1,nn(2)-1); %cell
stress2 = zeros(nn(1)-1,nn(2)-1); %cell

dep = ((1:nn(2)-1)-0.5)*dy;

rho = 2670;
rhow = 1e3;
g=9.8;
ratio = g*(rho-rhow)/1e6;
%ratio = 16.2/1e3;
rsn = 0.40;
cohesion = 3;
sratio = 80*rsn/((80-cohesion)/ratio);
% for i = 1:nn(1)-1
%     stress1(i,:)=dep*ratio-cohesion;
%     stress2(i,:)=-dep*ratio*1/3;
% end
%% linear shear and normal stress
% off=0;
% for i = 1:nn(2)-1
%     dp = dep(i);
%     sign = min(dp*ratio+cohesion,80);
%     sigs = min((dp*sratio),80*rsn);
%     if(abs(sign-80)<1e-3)
%         off=off+1;
%         if(off==1) dps = dp;end
%     end
% %     if(dp>15e3)
% %         sigs = max(rsn*80-(dp-15e3)*16.2/1e3,0.4*80);
% %     end
%     stress1(:,i)=-sign*ones(nn(1)-1,1);
%     stress2(:,i)=sigs*ones(nn(1)-1,1);
% end

%% linear ratio of shear and normal stress
off=0;
for i = 1:nn(2)-1
    dp = dep(i);
    sign = min(dp*ratio+cohesion,80);

    if(abs(sign-80)<1e-3)
        off=off+1;
        if(off==1) dps = dp;end
    end
    stress1(:,i)=-sign*ones(nn(1)-1,1);
end

a1 = 0.32;%ratio of shear and normal stress at top layer
a2 = rsn;
snr = a2*ones(nn(2)-1,1);

for i = 1:nn(2)-1
    dp = dep(i);

    if(dp<dps)
        snr(i) = a1+(dp-dep(1))*(a2-a1)/(dps-dep(1));
    end
    stress2(:,i)=-stress1(:,i)*snr(i);
end

%%

stress1 = stress1*1e6;
stress2 = stress2*1e6;

figure(1)
x = 0:dx:L(1);
y = 0:dx:L(2);
clf
subplot(211)
pcolor(-stress1');
%pcolor(x(1:end-1)+0.5*dx,y(1:end-1)+0.5*dx,stress1');
axis equal ij
shading flat
colorbar
% xlim([min(x),max(x)]);
% ylim([min(y),max(y)]);

subplot(212)
pcolor(stress2');
%pcolor(x(1:end-1)+0.5*dx,y(1:end-1)+0.5*dx,stress1');
axis equal ij
shading flat
colorbar
% xlim([min(x),max(x)]);
% ylim([min(y),max(y)]);


figure(2)
clf
subplot(121)
plot(-stress1(1,:),dep,'linewidth',3);
hold on
plot(stress2(1,:),dep,'linewidth',3);
grid on
axis ij
legend('Normal Stress','Shear Stress');
subplot(122)
plot(stress2(1,:)./-stress1(1,:),dep,'linewidth',3);
grid on
axis ij

fname='stress';
h=gcf;
set(h,'PaperPositionMode','auto');         
set(h,'renderer','Painters')
% set(h,'renderer','openGL')
print(h,'-r600','-dpdf','-bestfit',fname); 

fid = fopen(['databin/normal1.bin'],'wb');
fwrite(fid,stress1,'single','b'); %to mira
%fwrite(fid,stress1,'single'); %to bluewater
fclose(fid);
fid = fopen(['databin/normal2.bin'],'wb');
fwrite(fid,stress1,'single','b'); %to mira
% fwrite(fid,stress1,'single'); %to bluewater
fclose(fid);
fid = fopen(['databin/normal3.bin'],'wb');
fwrite(fid,stress1,'single','b'); %to mira
% fwrite(fid,stress1,'single'); %to bluewater
fclose(fid);

fid = fopen(['databin/shear.bin'],'wb');
fwrite(fid,stress2,'single','b'); %to mira
% fwrite(fid,stress2,'single'); %to bluewater
fclose(fid);

%% friction property
%------------ af ----------------
x0 = L(1)/2;
y0 = 4e3;
thickness = 5e3;
lx1 = 25e3;
lx2 = 25e3;
ly1 = 10e3;%large enough
ly2 = 11e3;
[bx,by]=bmatrix(nn(1),nn(2),dx,dy,x0,y0,thickness,lx1,lx2,ly1,ly2);

bxx = bx.*ones(1,nn(2));
byy = ones(nn(1),1).*by;

iy1 = 1;
sthickness = dps;
iy2 = round((sthickness)/dy)+iy1;

ix1 = round((x0-lx1-thickness/2)/dx)+1;
ix2 = round((x0+lx2+thickness/2)/dx)+1;

%%
ashallow = 0.015;
acore = 0.01;
aout = 0.03;
a = acore*ones(nn(1),nn(2));
da  = aout-acore;

dv = (acore-ashallow)/da;%percent increase from zero
byy = ones(nn(1),1).*by;
byy=shallow(byy,ix1,ix2,iy1,iy2,dv);

bxy = bxx.*byy;

Nr = 4;
Nc = 1;

afs = a+da*(ones(nn(1),nn(2))-bxy);
%smooth;
afs = smooth2a(afs,Nr,Nc);

fid = fopen(['databin/af.bin'],'wb');
fwrite(fid,afs,'single','b'); %to mira
% fwrite(fid,afs,'single'); %to bluewater
fclose(fid);

%------------- vw -------------
vwshallow = 0.1;
vwcore = 0.1; %0.1(or 0.075) pulse like; 0.05 crack like
vwout = 2;
vw = vwcore*ones(nn(1),nn(2));
dvw  = vwout-vwcore;

dv = (vwcore-vwshallow)/dvw;%percent increase from zero
byy = ones(nn(1),1).*by;
byy=shallow(byy,ix1,ix2,iy1,iy2,dv);

bxy = bxx.*byy;

vwfs = vw+dvw*(ones(nn(1),nn(2))-bxy);
%smooth;
vwfs = smooth2a(vwfs,Nr,Nc);

fid = fopen(['databin/vwf.bin'],'wb');
fwrite(fid,vwfs,'single','b'); %to mira
% fwrite(fid,vwfs,'single'); %to bluewater
fclose(fid);

% % %------- f0 -------
% f0f = 10*ones(nn(1),nn(2));
% f0 = 0.7;
% 
% 
% ix1=floor(xbounds/dx)+1;
% ix2=nn(1)-ix1+1;
% %iy1=floor(bounds/dx)+1;
% % iy1 = 1;
% % iy2=floor(sbounds/dx/2)+1;
% % llf(ix1:ix2,iy1:iy2) = 0.7;
% 
% iy1 = 1;
% % iy1 = iy2+1;
% iy2=nn(2)-floor(ybounds/dx)+1+1;
% f0f(ix1:ix2,iy1:iy2) = f0;
% %gaussian smooth
% f0fs = smooth2a(f0f,Nr,Nc);
% 
% %move upwards or downwards the desired pattern
% % yoff = floor(1e3/dx);
% % for i = 1:3*Nc+1
% %     vwfs(:,i) = vwfs(:,i+yoff);
% % end
% 
% fid = fopen(['databin/f0f.bin'],'wb');
% fwrite(fid,f0fs,'single','b'); %to mira
% %fwrite(fid,stress1,'single'); %to bluewater
% fclose(fid);


% % %------- fw -------
% fwf = 10*ones(nn(1),nn(2));
% fw = 0.2;
% 
% 
% ix1=floor(xbounds/dx)+1;
% ix2=nn(1)-ix1+1;
% %iy1=floor(bounds/dx)+1;
% % iy1 = 1;
% % iy2=floor(sbounds/dx/2)+1;
% % llf(ix1:ix2,iy1:iy2) = 0.7;
% 
% iy1 = 1;
% % iy1 = iy2+1;
% iy2=nn(2)-floor(ybounds/dx)+1+1;
% fwf(ix1:ix2,iy1:iy2) = fw;
% %gaussian smooth
% fwfs = smooth2a(fwf,Nr,Nc);
% 
% %move upwards or downwards the desired pattern
% % yoff = floor(1e3/dx);
% % for i = 1:3*Nc+1
% %     vwfs(:,i) = vwfs(:,i+yoff);
% % end
% 
% fid = fopen(['databin/fwf.bin'],'wb');
% fwrite(fid,fwfs,'single','b'); %to mira
% %fwrite(fid,stress1,'single'); %to bluewater
% fclose(fid);
%%
figure(3)
x = 0:dx:L(1);
y = 0:dx:L(2);
clf
subplot(211)
pcolor(x,y,afs');
%pcolor(x(1:end-1)+0.5*dx,y(1:end-1)+0.5*dx,stress1');
axis equal ij
shading flat
colorbar
xlim([min(x),max(x)]);
ylim([min(y),max(y)]);

subplot(212)
pcolor(x,y,vwfs');
%pcolor(x(1:end-1)+0.5*dx,y(1:end-1)+0.5*dx,stress1');
axis equal ij
shading flat
colorbar
xlim([min(x),max(x)]);
ylim([min(y),max(y)]);

figure(4)
clf
subplot(121)
plot(afs(floor(nn(1)/2),:),y,'linewidth',3);
hold on
plot(0.014*ones(size(y)),y,'linewidth',3);
grid on
axis ij
legend('a','b');

subplot(122)
plot(vwfs(floor(nn(1)/2),:),y,'linewidth',3);
grid on
axis ij
legend('vw');

fname='friction';
h=gcf;
set(h,'PaperPositionMode','auto');         
set(h,'renderer','Painters')
% set(h,'renderer','openGL')
print(h,'-r600','-dpdf','-bestfit',fname); 

% subplot(313)
% pcolor(x,y,llfs');
% %pcolor(x(1:end-1)+0.5*dx,y(1:end-1)+0.5*dx,stress1');
% axis equal ij
% shading flat
% colorbar
% xlim([min(x),max(x)]);
% ylim([min(y),max(y)]);


function outv = shallow(inv,ix1,ix2,iy1,iy2,dv)
    outv = inv;
    for j = iy1:iy2
        outv(ix1:ix2,j) = outv(ix1:ix2,j)+dv-(j-iy1)*dv/(iy2-iy1);
    end
end




function [bx,by] = bmatrix(nx,ny,dx,dy,x0,y0,thickness,lx1,lx2,ly1,ly2)
    
    x = (0:nx-1)*dx;
    y = (0:ny-1)*dx;
    
    ix0 = floor(x0/dx)+1;
    iy0 = floor(y0/dx)+1;
    
    bx = zeros(nx,1);
    by = zeros(1,ny);
    
    ix1 = 1;
    ix2 = round((x0-lx1-thickness)/dx)+1;
    ix3 = round((x0-lx1)/dx)+1;
    ix4 = round((x0+lx2)/dx)+1;
    ix5 = round((x0+lx2+thickness)/dx)+1;
    ix6 = nx;
% bmatrix along x
    for i = 1:nx
        xi=x(i);
        if(i>=1 && i<ix2)
            bx(i) = 0;
        elseif(i>=ix2 && i<ix3)
            dist = x0-lx1-xi;
            
            if(abs(dist-thickness)<1e-3)
                bx(i)=0;
            elseif(abs(dist)<1e-3)
                bx(i)=1;
            else
                bx(i) = 1/2*(1+tanh...
                    (thickness/(dist-thickness)+thickness/(dist)));
            end
        elseif(i>=ix3 && i<ix4)
            bx(i) = 1;
        elseif(i>=ix4 && i<ix5)
            dist = xi-x0-lx2;
            
            if(abs(dist-thickness)<1e-3)
                bx(i)=0;
            elseif(abs(dist)<1e-3)
                bx(i)=1;
            else
                bx(i) = 1/2*(1+tanh...
                    (thickness/(dist-thickness)+thickness/(dist)));
            end
        elseif(i>=ix5 && i<=ix6)
            bx(i) = 0;   
        end
    end
    
% bmatrix along y
    iy1 = 1;
    iy2 = round((y0-ly1-thickness)/dy)+1;
    iy3 = round((y0-ly1)/dy)+1;
    iy4 = round((y0+ly2)/dy)+1;
    iy5 = round((y0+ly2+thickness)/dy)+1;
    iy6 = ny;

    for i = 1:ny
        yi=y(i);
        if(i>=1 && i<iy2)
            by(i) = 0;
        elseif(i>=iy2 && i<iy3)
            dist = y0-ly1-yi;
            
            if(abs(dist-thickness)<1e-3)
                by(i)=0;
            elseif(abs(dist)<1e-3)
                by(i)=1;
            else
                by(i) = 1/2*(1+tanh...
                    (thickness/(dist-thickness)+thickness/(dist)));
            end
        elseif(i>=iy3 && i<iy4)
            by(i) = 1;
        elseif(i>=iy4 && i<iy5)
            dist = yi-y0-ly2;
            
            if(abs(dist-thickness)<1e-3)
                by(i)=0;
            elseif(abs(dist)<1e-3)
                by(i)=1;
            else
                by(i) = 1/2*(1+tanh...
                    (thickness/(dist-thickness)+thickness/(dist)));
            end
        elseif(i>=iy5 && i<=iy6)
            by(i) = 0;   
        end
    end
    
    return
end

