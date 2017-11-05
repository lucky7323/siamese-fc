video_path = '/home/eunho/vision_project/OD_fromVideo/siamese-fc/demo-sequences/test_1030/imgs/';

%load all jpg files in the folder
img_files = dir([video_path '*.jpg']);
assert(~isempty(img_files), 'No image files to load.')
img_files = sort({img_files.name});

%eliminate frame 0 if it exists, since frames should only start at 1
img_files(strcmp('00000000.jpg', img_files)) = [];
img_files = strcat(video_path, img_files);

% read all frames at once
imgs = vl_imreadjpeg(img_files,'numThreads', 12);

%For Siamese-CNN
startup;

%%
numImg=length(imgs);    
ii=1;
while ii<=numImg
    if mod(ii,3)==1
        ii=ii+1;
%        load /home/eunho/vision_project/OD_fromVideo/faster_rcnn/a.mat
%        targetDet = a;
%        targetDet=faster_rcnn(imgs(i)); %(a.mat)
         targetDet=script_faster_rcnn_demo(imgs(ii));
    else
        [pos, target_sz, classId]=load_video_info(targetDet);
        tracker(pos, target_sz, classId, imgs(ii:ii+1));
        ii=ii+2;
    end
end

%    seq={
%     'test_1030'
%   };
%    for s=1:numel(seq)   
%        bb = run_tracker(seq{s}, visualization);
%    end
    