vid_path = '/home/eunho/LARGE_DATASET2/test5/';
test_videos = dir(sprintf('%s/*test*',vid_path));
startup;
[opts, model, rpn_net, fast_rcnn_net] = faster_init();
%71884, 51369, 63973, 77032, 50918
num_val_files = 264258;
for i = 1:numel(test_videos)
    if test_videos(i).isdir
        val_files = dir(sprintf('%s/%s/*.JPEG',vid_path,test_videos(i).name));
        num_val_files = num_val_files + numel(val_files);
    end
    Copy_of_runAll_vot15(test_videos(i).name, num_val_files, opts, model, rpn_net, fast_rcnn_net)
%     if i==ceil(numel(test_videos) / 4)
%         sprintf('======== 25%% complete ========= \n')
%         pause(1.0);
%         copyfile('Vsubmission.txt', 'Vsubmission3_25_cache.txt');
%     elseif i==ceil(numel(test_videos) / 2)
%         sprintf('======== 50%% complete ========= \n')
%         pause(1.0);
%         copyfile('Vsubmission.txt', 'Vsubmission3_50_cache.txt');
%     elseif i==ceil(3 * numel(test_videos) / 4)
%         sprintf('======== 75%% complete ========= \n')
%         pause(1.0);
%         copyfile('Vsubmission.txt', 'Vsubmission3_75_cache.txt');
%     end
end
caffe.reset_all(); 
clear mex;