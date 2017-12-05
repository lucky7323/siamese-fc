function []= Copy_of_runAll_vot15(vd, num_idx, opts, model, rpn_net, fast_rcnn_net)
    close all;
    legends = {'airplane';'antelope';'bear';'bicycle';'bird';'bus';'car';'cattle';'dog';'cat';'elephant';'fox'; ...
        'giant_panda';'hamster';'horse';'lion';'lizard';'monkey';'motorcycle';'rabbit';'red_panda';'sheep';'snake'; ...
        'squirrel';'tiger';'train';'turtle';'watercraft';'whale';'zebra'};
%VISUAL    colors_candidate = colormap('hsv');
%VISUAL    colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/30)):end, :);
%VISUAL    colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
%VISUAL    colors = colors_candidate;

    restart = 1;
%    startup;
    %active_caffe_mex(1,'caffe_faster_rcnn');

    video_path = ['/home/eunho/LARGE_DATASET2/val3/' vd '/'];

    %load all jpg files in the folder
    img_files = dir([video_path '*.JPEG']);
    assert(~isempty(img_files), 'No image files to load.')
    img_files = sort({img_files.name});

    %eliminate frame 0 if it exists, since frames should only start at 1
    img_files(strcmp('00000000.JPEG', img_files)) = [];
    img_files = strcat(video_path, img_files);

    % read all frames at once
    imgs = vl_imreadjpeg(img_files,'numThreads', 12);

    % scale
    fix_width = 800;
    imsz = size(imgs{1});
    scale = fix_width / imsz(2);

    %%=================================================================
%    [opts, model, rpn_net, fast_rcnn_net] = faster_init();
%    [zFeatId,scoreId,p,net_z,net_x]=siamese_init(scale);

    %%=================================================================

    numImg=length(imgs);    
    ii=1;
    elapse = zeros(numImg, 1);
    elapse2 = zeros(numImg, 1);

    %tic;
    submission = [];
    while ii<=numImg
%       if mod(ii,3)==1
%            figure(ii);
            tic;
            % bbDet(i,:) = (xmin, ymin, xmax, ymax) 
            targetDet=script_faster_rcnn_demo(img_files{ii}, opts, model, rpn_net, fast_rcnn_net);
            elapse(ii,1) = toc;
            drawnow;

            if size(targetDet)~=0
                bbDet = [];
                tmp = targetDet(:,1:4) / scale;
                bbDet(:,3) = tmp(:,1);
                bbDet(:,4) = tmp(:,2);
                bbDet(:,5) = tmp(:,1) + tmp(:,3);
                bbDet(:,6) = tmp(:,2) + tmp(:,4);
                bbDet(:,1:2) = targetDet(:,5:6);
                submission = [submission; ones(size(bbDet,1),1)*(ii + num_idx - numImg) bbDet];
            end
            
%            tic;
%            if size(targetDet)~=0
%                [s_x, z_features, pos, target_sz, classId, avgChans, net_z]=tracker_z(net_z, targetDet, imgs(ii), p, zFeatId);
%            end
%            elapse2(ii,1)=toc;

            ii=ii+1;
%        else

%            tic;
%            if size(targetDet)~=0
%                [bboxes, pos, net_x] = tracker(net_x, pos, target_sz, imgs(ii), p, scoreId, z_features, avgChans, s_x);
%            end
%            elapse(ii,1) = toc;

%            if size(targetDet)~=0
%                bbDet2 = [];
%                tmp = bboxes / scale;
%                bbDet2(:,3) = tmp(:,1);
%                bbDet2(:,4) = tmp(:,2);
%                bbDet2(:,5) = tmp(:,1) + tmp(:,3);
%                bbDet2(:,6) = tmp(:,2) + tmp(:,4);
%                bbDet2(:,1:2) = targetDet(:,5:6);
%                submission = [submission; ones(size(bbDet2,1),1)*(ii + num_idx - numImg) bbDet2];
%                end
            
            %% Visualization
%             img = imread(img_files{ii});
%             img = imresize(img, scale);               
%     
%             figure(ii)
%             image(img); 
%             axis image;
%             axis off;
%             set(gcf, 'Color', 'white');
%     
%             for j=1:size(targetDet,1)
%                 rectangle('Position',bboxes(j,:),'LineWidth', 2,'EdgeColor',colors{classId(j,1)});
%                 label = sprintf('%s : %.3f', legends{classId(j,1)}, targetDet(j,6));
%                 text(double(bboxes(j,1))+2, double(bboxes(j,2)), label, 'BackgroundColor', 'w');
%             end
%             drawnow     
% 
%             ii=ii+1;   
%        end
    end
    fid = fopen('/home/eunho/vision_project/OD_fromVideo/Vsubmission.txt', 'a');
    fprintf(fid, '%d %d %.4f %.3f %.3f %.3f %.3f\n', submission');
    fclose(fid);
%    caffe.reset_all(); 
%   clear mex;
    sprintf('%s',vd)
end
