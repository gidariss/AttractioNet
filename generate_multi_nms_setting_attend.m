function set = generate_multi_nms_setting_attend()
% for fcn
% model.stage1_rpn.nms.nms_iou_thrs   = [0.90, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50];
% model.stage1_rpn.nms.max_per_image  = [3000, 2000, 1000, 800,  400,  200,  100,  50];

nms_range{1} = [0.9 : -0.05 : 0.5];
nms_range{2} = [0.9 : -0.1 : 0.5];

max_im = [5000, 4000, 3000, 2000, 1000];
decay_factor = [0.9, 0.75, 0.6, 0.5, 0.3];

cnt = 0;
for i = 1:length(max_im)
%     if i == 3
%         keyboard;
%     end
    for j = 1:length(decay_factor)
        for k = 1:length(nms_range)
            cnt = cnt + 1;
            set(cnt).nms_iou_thrs = nms_range{k};
            set(cnt).max_per_image = generate_vec(max_im(i), ...
                decay_factor(j), length(nms_range{k}));
        end
    end
end

    function vec = generate_vec(max_im, decay, total_length)
        vec = zeros(total_length, 1);
        for kk = 1:total_length
            if kk == 1, vec(kk) = max_im;
            else vec(kk) = floor(decay*vec(kk-1)); end
        end
    end
end