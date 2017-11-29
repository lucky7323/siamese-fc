% -------------------------------------------------------------------------------------------------
function [s_x, z_features, targetPosition, targetSize, classId, avgChans] = tracker_z(targetDet, startImgFile, p,net_z, zFeatId)

    index = find(targetDet(:,1));
    numDet = length(index);
    targetPosition=zeros(numDet,2);
    targetSize=zeros(numDet,2);
    classId = zeros(numDet,1);
    for i=1:numDet
        tdReal= targetDet(index(i),:);     
        l=tdReal(1);t=tdReal(2);w=tdReal(3);h=tdReal(4);
        cx=l+w/2; cy=t+h/2;
    
        targetPosition(i,:) = [cy cx]; % centre of the bounding box
        targetSize(i,:) = [h w];
        classId(i,:) = tdReal(5);
    end
    
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
    % These are the default hyper-params for SiamFC-3S
    % The ones for SiamFC (5 scales) are in params-5s.txt
    im = imresize(startImgFile{1},p.scale);
    im = gpuArray(im);
 %   im = imresize(im,bboxes[600,800]);
    % if grayscale repeat one channel to match filters size
	if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    % Init visualization
    
    % get avg for padding
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

    wc_z = targetSize(:,2) + p.contextAmount*sum(targetSize,2); %numDet
    hc_z = targetSize(:,1) + p.contextAmount*sum(targetSize,2); %numDet 
    s_z = sqrt(wc_z.*hc_z);  %numDet
    scale_z = p.exemplarSize ./ s_z;  %numDet
    % initialize the exemplar
    
    z_crop = zeros(p.exemplarSize,p.exemplarSize,3,numDet,'single');
    z_crop = gpuArray(z_crop);
    for k=1:numDet
        [z_crop(:,:,:,k), ~] = get_subwindow_tracking(im, targetPosition(k,:), [p.exemplarSize p.exemplarSize], [round(s_z(k,:)) round(s_z(k,:))], avgChans);
    end
%    if p.subMean
%        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
%    end
    d_search = (p.instanceSize - p.exemplarSize)/2;
    pad = d_search./scale_z; %numDet
    s_x = s_z + 2*pad; %numDet
    % arbitrary scale saturation

    % evaluate the offline-trained network for exemplar z features
    for k=1:numDet
        net_z.eval({'exemplar', z_crop(:,:,:,k)});
        z_features_temp = net_z.vars(zFeatId).value;
        z_features(:,:,:,:,k) = repmat(z_features_temp, [1 1 1 p.numScale]);    %numDet
    end
    
end
