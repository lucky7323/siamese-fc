% -------------------------------------------------------------------------------------------------
function [zFeatId,scoreId,p,net_z,net_x] = siamese_init(scale)
    
    p.scale = scale;
    p.numScale = 1;
    p.scaleStep = 1.0375;
    p.scalePenalty = 0.9745;
    p.scaleLR = 0.59; % damping factor for scale update
    p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
    p.windowing = 'cosine'; % to penalize large displacements
    p.wInfluence = 0.176; % windowing influence (in convex sum)
    p.net = '2016-08-17.net.mat';
%    p.net = '2017-11-13.net.mat';
    %% execution, visualization, benchmark
    p.video = 'ILSVRC2015_val_00036010';
    p.visualization = false;
    p.gpus = 1;
    p.bbox_output = false;
    p.fout = -1;
    %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;%17;
    p.totalStride = 8;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
    %% SiamFC prefix and ids
    p.prefix_z = 'a_'; % used to identify the layers of the exemplar
    p.prefix_x = 'b_'; % used to identify the layers of the instance
    p.prefix_join = 'xcorr';
    p.prefix_adj = 'adjust';
    p.id_feat_z = 'a_feat';
    p.id_score = 'score';
    % Overwrite default parameters with varargin
%   p = vl_argparse(p, varargin);
% -------------------------------------------------------------------------------------------------

    % Get environment-specific default paths.
    p = env_paths_tracking(p);
    % Load ImageNet Video statistics
    if exist(p.stats_path,'file')
        stats = load(p.stats_path);
    else
%        warning('No stats found at %s', p.stats_path);
        stats = [];
    end
    
    % Load two copies of the pre-trained network
    net_z = load_pretrained([p.net_base_path p.net], p.gpus);
    net_x = load_pretrained([p.net_base_path p.net], p.gpus);
    
    % Divide the net in 2
    % exemplar branch (used only once per video) computes features for the target
    remove_layers_from_prefix(net_z, p.prefix_x);
    remove_layers_from_prefix(net_z, p.prefix_join);
    remove_layers_from_prefix(net_z, p.prefix_adj);
    % instance branch computes features for search region x and cross-correlates with z features
    remove_layers_from_prefix(net_x, p.prefix_z);
    zFeatId = net_z.getVarIndex(p.id_feat_z);
    scoreId = net_x.getVarIndex(p.id_score);
    % get the first frame of the video
    
    p.stats= stats;
end
