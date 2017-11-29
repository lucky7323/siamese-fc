legends = {'airplane';'antelope';'bear';'bicycle';'bird';'bus';'car';'cattle';'dog';'cat';'elephant';'fox'; ...
    'giant_panda';'hamster';'horse';'lion';'lizard';'monkey';'motorcycle';'rabbit';'red_panda';'sheep';'snake'; ...
    'squirrel';'tiger';'train';'turtle';'watercraft';'whale';'zebra'};
colors_candidate = colormap('hsv');
colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/30)):end, :);
colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
colors = colors_candidate;

restart=1;      
startup;
%active_caffe_mex(1,'caffe_faster_rcnn');

video_path = '/home/eunho/vision_project/OD_fromVideo/siamese-fc/demo-sequences/test_1030/imgs/';

%load all jpg files in the folder
img_files = dir([video_path '*.JPEG']);
assert(~isempty(img_files), 'No image files to load.')
img_files = sort({img_files.name});

%eliminate frame 0 if it exists, since frames should only start at 1
img_files(strcmp('00000000.jpg', img_files)) = [];
img_files = strcat(video_path, img_files);

% read all frames at once
imgs = vl_imreadjpeg(img_files,'numThreads', 12);

% scale
fix_width = 800;
imsz = size(imgs{1});
scale = fix_width / imsz(2);

%%=================================================================
[opts, model, rpn_net, fast_rcnn_net] = faster_init();
[zFeatId,scoreId,p,net_z,net_x]=siamese_init(scale);

%%=================================================================

numImg=length(imgs);    
ii=1;
elapse = zeros(numImg, 1);
elapse2 = zeros(numImg, 1);

%tic;
while ii<=numImg
    if mod(ii,3)==1
        figure(ii);
        tic;
        targetDet=script_faster_rcnn_demo(img_files{ii}, opts, model, rpn_net, fast_rcnn_net);
        elapse(ii,1) = toc;
        drawnow;
        
        ii=ii+1;
        
        tic;
        [s_xz, z_features, pos, target_sz, classId, avgChans]=tracker_z(targetDet, imgs(ii-1), p, net_z, zFeatId);
        elapse2(ii,1)=toc;
        
        s_x = s_xz;
 
    else
        
        tic;
        [bboxes, pos, target_sz, s_x] = tracker(net_x, pos, target_sz, imgs(ii), p, scoreId, s_xz, z_features, avgChans, s_x);
        elapse(ii,1) = toc;
 
        %% Visualization
        img = imread(img_files{ii});
        img = imresize(img, scale);               

        figure(ii)
        image(img); 
        axis image;
        axis off;
        set(gcf, 'Color', 'white');
        %imshow(img)
        for j=1:size(targetDet,1)
            rectangle('Position',bboxes(j,:),'LineWidth', 2,'EdgeColor',colors{classId(j,1)});
            label = sprintf('%s', legends{classId(j,1)});
            text(double(bboxes(j,1))+2, double(bboxes(j,2)), label, 'BackgroundColor', 'w');
        end
        drawnow     

        ii=ii+1;   
    end
end
%total_time = toc;
caffe.reset_all(); 
clear mex;