vlist = {'/ILSVRC2015_val_00005002/', '/ILSVRC2015_val_00078001/'};
for i=1:size(vlist,2)
    runAll_vot15(vlist{i})
    pause(0.5);
end