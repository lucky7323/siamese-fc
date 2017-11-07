% -------------------------------------------------------------------------------------------------
function [bboxes, targetPosition, targetSize, s_x] = tracker(targetPosition, targetSize, imgFile, p, net_x, scoreId, s_x, z_features, avgChans)       
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
% These are the default hyper-params for SiamFC-3S
% The ones for SiamFC (5 scales) are in params-5s.txt

    numDet = size(targetPosition,1);  %numDet
    bboxes = zeros(numDet, 4);

    min_s_x = 0.2*s_x; %numDet
    max_s_x = 5*s_x; %numDet

    switch p.windowing
        case 'cosine'
            window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
        case 'uniform'
            window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    end
    % make the window sum 1
    window = window / sum(window(:));
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));

    % load new frame on GPU
    %im = gpuArray(single(imgFiles{i}));
    im = imresize(imgFile{1},p.scale);
    im = gpuArray(im);  
    % if grayscale repeat one channel to match filters size
    if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    scaledInstance = s_x * scales;  % numDet*numScale
    scaledTarget(:,:,1) = targetSize(:,1) * scales;  % numDet*numScale
    scaledTarget(:,:,2) = targetSize(:,2) * scales;  % numDet*numScale

    % extract scaled crops for search region x at previous target position
    % x_crops = zeros(p.intanceSize,p.intanceSize,3, p.numScale, numDet,'single');
    % x_crops = gpuArray(x_crops);

    for k=1:numDet
        x_crops = make_scale_pyramid(im, targetPosition(k,:), scaledInstance(k,:), p.instanceSize, avgChans, p.stats, p);

        % evaluate the offline-trained network for exemplar x features

        [newTargetPosition, newScale] = tracker_eval(net_x, round(s_x(k,:)), scoreId, z_features(:,:,:,:,k), x_crops, targetPosition(k,:), window, p);
        targetPosition(k,:) = gather(newTargetPosition);
        % scale damping and saturation
        s_x(k,:) = max(min_s_x(k,:), min(max_s_x(k,:), (1-p.scaleLR)*s_x(k,:) + p.scaleLR*scaledInstance(k,newScale)));
        targetSize(k,:) = (1-p.scaleLR)*targetSize(k,:) + p.scaleLR*[scaledTarget(k,newScale,1) scaledTarget(k,newScale,2)];
    end

    rectPosition = [[targetPosition(:,2),targetPosition(:,1)] - [targetSize(:,2),targetSize(:,1)] / 2, [targetSize(:,2) , targetSize(:,1)]];       
    bboxes(:, :)=rectPosition;

    % at the first frame output position and size passed as input (ground truth)         
    save tracker
end
