clear all

nx = 3081;
ny = 351;
nz = 1715;

fid=fopen('ts.bin','rb');
data = fread(fid,'single');
ts = reshape(data,nx-1,ny-1);

fid=fopen('tn.bin','rb');
data = fread(fid,'single');
tn = reshape(data,nx-1,ny-1);

fid=fopen('ts2.bin','rb');
data = fread(fid,'single');
ts2 = reshape(data,nx-1,ny-1);

fid=fopen('tn2.bin','rb');
data = fread(fid,'single');
tn2 = reshape(data,nx-1,ny-1);

figure(1)
subplot(211)
pcolor(ts2'./ts')
shading flat
colorbar
axis ij equal

subplot(212)
pcolor(tn2'./tn')
shading flat
colorbar
axis ij equal