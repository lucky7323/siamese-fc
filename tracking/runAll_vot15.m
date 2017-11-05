%load all jpg files in the folder
	img_files = dir([video_path '*.jpg']);
	assert(~isempty(img_files), 'No image files to load.')
	img_files = sort({img_files.name});

	%eliminate frame 0 if it exists, since frames should only start at 1
	img_files(strcmp('00000000.jpg', img_files)) = [];
    img_files = strcat(video_path, img_files);
    % read all frames at once
    imgs = vl_imreadjpeg(img_files,'numThreads', 12);
    
numImg=length(imgs);    
for i=1:numImg
    if mod(i,3)==1
        faster_rcnn(imgs(i));
    else
        setting
        tracker
%frame Num%3==1
faster
else
    setting
    siamese
end



startup;
%% Parameters that should have no effect on the result.
params.video = 'test_1030';
params.visualization = 1;
params.gpus = 1;
%% Call the main tracking function
tracker(params);   
    

%    seq={
%     'test_1030'
%   };
%    for s=1:numel(seq)   
%        bb = run_tracker(seq{s}, visualization);
%    end
    