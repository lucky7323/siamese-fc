    visualization = false;
    gpus = 1;

    seq={
     'test_1030'

    };

    for s=1:numel(seq)   
        bb = run_tracker(seq{s}, visualization);
    end
    

