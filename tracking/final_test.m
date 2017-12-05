vid_path = '/home/eunho/LARGE_DATASET2/val6/';
test_videos = dir(sprintf('%s/*val*',vid_path));

%71884, 51369, 63973, 77032, 50918
num_val_files = 0;
for i = 1:numel(test_videos)
    if test_videos(i).isdir
        val_files = dir(sprintf('%s/%s/*.JPEG',vid_path,test_videos(i).name));
        num_val_files = num_val_files + numel(val_files);
    end
    runAll_vot15(test_videos(i).name, num_val_files)
%     if i==ceil(numel(test_videos) / 4)
%         sprintf('======== 25%% complete ========= \n')
%         pause(2.0);
%         copyfile('submission.txt', 'submission5_25_cache.txt');
%         pause(1.0);
%     elseif i==ceil(numel(test_videos) / 2)
%         sprintf('======== 50%% complete ========= \n')
%         pause(2.0);
%         copyfile('submission.txt', 'submission5_50_cache.txt');
%         pause(1.0);
%     elseif i==ceil(3 * numel(test_videos) / 4)
%         sprintf('======== 75%% complete ========= \n')
%         pause(2.0);
%         copyfile('submission.txt', 'submission5_75_cache.txt');
%         pause(1.0);
%     end
end