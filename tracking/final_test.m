vid_path = '/home/eunho/LARGE_DATASET2/test/';
val_videos = dir(sprintf('%s/*val*',vid_path));

num_val_files = 0;
for i = 1:numel(val_videos)
    if val_videos(i).isdir
        val_files = dir(sprintf('%s/%s/*.JPEG',vid_path,val_videos(i).name));
        num_val_files = num_val_files + numel(val_files);
    end
    runAll_vot15(val_videos(i).name, num_val_files)
    if i==ceil(numel(val_videos) / 4)
        sprintf('======== 25%% complete ========= \n')
        pause(2.0);
        copyfile('submission.txt', 'submission_25_cache.txt');
        pause(1.0);
    elseif i==ceil(numel(val_videos) / 2)
        sprintf('======== 50%% complete ========= \n')
        pause(2.0);
        copyfile('submission.txt', 'submission_50_cache.txt');
        pause(1.0);
    elseif i==ceil(3 * numel(val_videos) / 4)
        sprintf('======== 75%% complete ========= \n')
        pause(2.0);
        copyfile('submission.txt', 'submission_75_cache.txt');
        pause(1.0);
    end
end