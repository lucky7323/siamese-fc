legends = {'aeroplane';'bicycle';'bird';'boat';'bottle';'bus';'car';'cat';'chair';'cow';'diningtable';'dog';'horse';'motorbike';'person';'pottedplant';'sheep';'sofa';'train';'tvmonitor'};
   colors_candidate = colormap('hsv');
            colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/20)):end, :);
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


%%
numImg=length(imgs);    
ii=1;
while ii<=numImg
    if mod(ii,3)==1
%        load /home/eunho/vision_project/OD_fromVideo/faster_rcnn/a.mat
%        targetDet = a;
%        targetDet=faster_rcnn(imgs(i)); %(a.mat)
         figure(ii);
         targetDet=script_faster_rcnn_demo(img_files{ii},restart);
         ii=ii+1;
         drawnow;
    else
        [pos, target_sz, classId]=load_video_info(targetDet);
        bboxes = tracker(pos, target_sz, imgs(ii:ii+1));
        
 
       %% Visualization
           for i=1:2
                iImg=ii+i-1;
                img = imread(img_files{iImg});
                img = imresize(img, [450, 800]);               
     
                figure(iImg)
                imshow(img)
                for j=1:size(targetDet,1)
                    rectangle('Position',bboxes(i,j,:),'LineWidth', 2,'EdgeColor',colors{classId(j,1)});
                    label = sprintf('%s', legends{classId(j,1)});
                    text(double(bboxes(i,j,1))+2, double(bboxes(i,j,2)), label, 'BackgroundColor', 'w');
                end
           drawnow     
           end
           
        ii=ii+2;   
    end
end
%    seq={
%     'test_1030'
%   };
%    for s=1:numel(seq)   
%        bb = run_tracker(seq{s}, visualization);
%    end
    