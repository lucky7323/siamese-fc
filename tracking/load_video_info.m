% -------------------------------------------------------------------------------------------------
function [imgs, pos, target_sz] = load_video_info(base_path, video)
%LOAD_VOT_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx4, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
% -------------------------------------------------------------------------------------------------
	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/imgs/'];

	%load ground truth from text file
%	ground_truth = csvread([base_path '/' video '/' 'groundtruth.txt']);
    ground_truth = csvread([base_path video '/' 'groundtruth.txt']);

	region = ground_truth(1, :);
	[cx, cy, w, h] = get_axis_aligned_BB(region);
    
    load /home/eunho/vision_project/OD_fromVideo/faster_rcnn/a.mat
    
 %  index = find(~cellfun(@isempty,a));
    index = find(a(:,1));
    numDet = length(index);
    pos=zeros(numDet,2);
    target_sz=zeros(numDet,2);
    for i=1:numDet
        aa= a(index(i),:);     
        l=aa(1);t=aa(2);w=aa(3);h=aa(4);
        cx=l+w/2; cy=t+h/2;
    
        pos(i,:) = [cy cx]; % centre of the bounding box
        target_sz(i,:) = [h w];
    end
    
	%load all jpg files in the folder
	img_files = dir([video_path '*.jpg']);
	assert(~isempty(img_files), 'No image files to load.')
	img_files = sort({img_files.name});

	%eliminate frame 0 if it exists, since frames should only start at 1
	img_files(strcmp('00000000.jpg', img_files)) = [];
    img_files = strcat(video_path, img_files);
    % read all frames at once
    imgs = vl_imreadjpeg(img_files,'numThreads', 12);
    
end
