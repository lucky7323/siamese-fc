vid_path = '/home/eunho/Downloads/ILSVRC2015/Data/VID/val/';
val_videos = dir(sprintf('%s/*val*',vid_path));

num_val_files = 0;
for i = 1:numel(val_videos)
    if val_videos(i).isdir
        val_files = dir(sprintf('%s/%s/*.JPEG',vid_path,val_videos(i).name));
        num_val_files = num_val_files + numel(val_files);
    end
    runAll_vot15(val_videos(i).name, num_val_files)
end