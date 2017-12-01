% -------------------------------------------------------------------------------------------------
function [bboxes, targetPosition, net_x] = tracker(net_x, targetPosition, targetSize, imgFile, p, scoreId, z_features, avgChans, s_x)       
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
% These are the default hyper-params for SiamFC-3S
% The ones for SiamFC (5 scales) are in params-5s.txt
    
    numDet = size(targetPosition,1);  %numDet

    window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    %switch p.windowing
    %    case 'cosine'
    %        window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    %    case 'uniform'
    %        window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    %end
    % make the window sum 1
    window = window / sum(window(:));

    % load new frame on GPU
    im = imresize(imgFile{1},p.scale);
    im = gpuArray(im);  
    % if grayscale repeat one channel to match filters size
    
    %if(size(im, 3)==1)
    %    im = repmat(im, [1 1 3]);
    %end
    
    for k=1:numDet
        x_crops = get_subwindow_tracking(im, targetPosition(k,:), [p.instanceSize p.instanceSize]...
                ,[round(s_x(k)) round(s_x(k))],avgChans);
      
        % evaluate the offline-trained network for exemplar x features
        [newTargetPosition] = tracker_eval(net_x, round(s_x(k,:)), scoreId, z_features(:,:,:,:,k), x_crops, targetPosition(k,:), window, p);
        targetPosition(k,:) = gather(newTargetPosition);
        % scale damping and saturation
    end
    %rectPosition;
    bboxes = [[targetPosition(:,2),targetPosition(:,1)] - [targetSize(:,2),targetSize(:,1)] / 2, [targetSize(:,2) , targetSize(:,1)]];
    
    % at the first frame output position and size passed as input (ground truth)         
end
