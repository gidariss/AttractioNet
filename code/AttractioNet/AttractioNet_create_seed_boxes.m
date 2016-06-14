function seed_boxes = AttractioNet_create_seed_boxes( im, num_seed_boxes)
%AttractioNet_create_seed_boxes: it generates equaly distributed boxes in
%the image
% 
% This file is part of the code that implements the following paper:
% Title      : "Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% code       : https://github.com/gidariss/AttractioNet
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
%
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

im_wh  = [size(im, 2), size(im, 1)];

%************************** CREATE SEED BOX SIZES *************************
% List of seed box sizes:
List_of_seed_box_widths   = [16; 32; 16; 32; 50; 25; 50; 72; 36; 72; 96; 48; 96; 128;  64; 128; 192;  96; 192; 256; 128; 256; 384; 192; 384];
List_of_seed_box_heights  = [16; 16; 32; 32; 25; 50; 50; 36; 72; 72; 48; 96; 96;  64; 128; 128;  96; 192; 192; 128; 256; 256; 192; 384; 384];
List_of_seed_box_wh_sizes = [List_of_seed_box_widths, List_of_seed_box_heights];
original_num_seed_box_sizes = size(List_of_seed_box_wh_sizes, 1);

% we filter seed box sizes to fit inside the image
inside_img = List_of_seed_box_wh_sizes(:,1) < im_wh(1) & List_of_seed_box_wh_sizes(:,2) < im_wh(2);
List_of_seed_box_wh_sizes = List_of_seed_box_wh_sizes(inside_img,:);

num_seed_boxes_sizes = size(List_of_seed_box_wh_sizes, 1);
assert(num_seed_boxes > num_seed_boxes_sizes);

seed_boxes = {};
if num_seed_boxes_sizes ~= original_num_seed_box_sizes, 
    % we add one candidate that covers the whole image size
   seed_boxes{1} = [0, 0, im_wh];
end
%**************************************************************************

%************* FIND THE PROPER STRIDE FOR EACH SEED BOX SIZE **************
% Figure out the number of seed boxes per seed box size
ratio_w = floor(max(List_of_seed_box_wh_sizes(:,1)) ./ List_of_seed_box_wh_sizes(:,1)); 
ratio_h = floor(max(List_of_seed_box_wh_sizes(:,2)) ./ List_of_seed_box_wh_sizes(:,2));
ratio   = max(1,min(24,ratio_w)).* max(1,min(24,ratio_h));
num_seed_boxes_per_size = ratio * floor(num_seed_boxes / sum(ratio)); % number of seed boxes per seed box size

stride_per_size         = zeros(2, num_seed_boxes_sizes); % stride per seed box size
total_placed_seed_boxes = size(seed_boxes, 1); % total placed seed boxes so far
for i = 1:num_seed_boxes_sizes % iterate over the possible seed box sizes
    
    seed_box_size_this = List_of_seed_box_wh_sizes(i, :);
    % assert that the seed box size fits inside the image
    assert((im_wh(1) - seed_box_size_this(1)) > 0); 
    assert((im_wh(2) - seed_box_size_this(2)) > 0);
    
    % starting stride for this seed box size: the starting stride is 8 
    % times smaller than the width/height (but no smaller than 4 pixels)
    stride_w = max(4,seed_box_size_this(1)/8); 
    stride_h = max(4,seed_box_size_this(2)/8);
    assert(stride_w > 0);
    assert(stride_h > 0);
    
    % find the best stride for this seed box size such that the number of
    % seed boxes (placements) with this size to not exceed the number: num_seed_boxes_per_size(i)
    num_seed_boxes_placed = comp_num_seed_box_placements_with_this_size_and_stride(im_wh, seed_box_size_this, stride_w, stride_h);
    while(num_seed_boxes_placed > num_seed_boxes_per_size(i))
        % enlarge the stride (by two pixels) in order the total number of
        % placed seed boxes for this seed box size to be less or equal than
        % the allowed box placements for this seed box size (=num_seed_boxes_per_size(i))
        stride_w = stride_w + 2; % larger stride, less placements
        stride_h = stride_h + 2; % larger stride, less placements
        num_seed_boxes_placed = comp_num_seed_box_placements_with_this_size_and_stride(im_wh, seed_box_size_this, stride_w, stride_h);
    end
    assert(num_seed_boxes_placed <= num_seed_boxes_per_size(i));

    % keep record of the total number of seed boxes placed
    total_placed_seed_boxes = total_placed_seed_boxes + num_seed_boxes_placed;
    assert(total_placed_seed_boxes <= num_seed_boxes);

    % store the stride for this seed box size
    stride_per_size(1, i) = stride_w; 
    stride_per_size(2, i) = stride_h;
end    
assert(total_placed_seed_boxes <= num_seed_boxes);

% reamining number of seed boxes till we reach the asked number of seed
% boxes (num_seed_boxes)
remaining_num_seed_boxes = num_seed_boxes - total_placed_seed_boxes;
assert(remaining_num_seed_boxes >= 0);

% perform three runs during which the stride will be reduced till the asked
% number of seed boxes is reached
for k = 1:3     
    for i = num_seed_boxes_sizes:-1:1 % start from the bigest box size
        seed_box_size_this = List_of_seed_box_wh_sizes(i, :); % seed box size
        stride_wh = stride_per_size(:,i); % stride for this seed box size
        % number of seed box placements for this seed box size
        num_seed_boxes_placed = comp_num_seed_box_placements_with_this_size_and_stride(...
            im_wh, seed_box_size_this, stride_wh(1),stride_wh(2));
        for alpha = 0.75:0.05:1.0 % alpha is the shrinkage factor of the stride for this seed box size

            new_stride_wh = stride_wh * alpha; % shrink the stride
            % compute new number of seed box placement with the new stride
            new_num_seed_boxes_placed = comp_num_seed_box_placements_with_this_size_and_stride(...
                im_wh, seed_box_size_this, new_stride_wh(1),new_stride_wh(2));

            delta_seed_boxes = new_num_seed_boxes_placed - num_seed_boxes_placed;
            if remaining_num_seed_boxes > delta_seed_boxes,
                stride_per_size(:,i) = new_stride_wh;
                total_placed_seed_boxes = total_placed_seed_boxes + delta_seed_boxes;
                remaining_num_seed_boxes = remaining_num_seed_boxes - delta_seed_boxes;
                break;
            end
        end
    end
end    
%**************************************************************************

%****** GENERATE SEED BOXES FOR THE GIVEN SEED BOX SIZES AND STRIDES ******
for i = 1:num_seed_boxes_sizes,
    seed_box_size_this = List_of_seed_box_wh_sizes(i, :);
    stride_wh = stride_per_size(:,i);
    [x_dots, y_dots] = create_xy_dots_wh(im_wh, seed_box_size_this, stride_wh(1),stride_wh(2));
    [xx, yy] = meshgrid(x_dots, y_dots);
    top_left_xy = [xx(:) yy(:)];
    seed_boxes{end+1} = [top_left_xy, top_left_xy(:,1)+seed_box_size_this(1),top_left_xy(:,2)+seed_box_size_this(2)];
end
seed_boxes = cell2mat(seed_boxes(:));
assert(size(seed_boxes, 1) == total_placed_seed_boxes);    
assert(size(seed_boxes, 1) <= num_seed_boxes);
%**************************************************************************

end

function [x_dots, y_dots]= create_xy_dots_wh(im_wh, seed_box_size_this, stride_w, stride_h)
x1 = 1;
y1 = 1;
x2 = im_wh(1) - seed_box_size_this(1);
y2 = im_wh(2) - seed_box_size_this(2);
assert(x2 > 0);
assert(y2 > 0);
x_dots = x1 + (mod(x2 - x1, stride_w) / 2): stride_w : x2;
y_dots = y1 + (mod(y2 - y1, stride_h) / 2): stride_h : y2;
end

function [ret] = comp_num_seed_box_placements_with_this_size_and_stride(im_wh, seed_box_size_this, stride_w, stride_h)
[x_dots, y_dots] = create_xy_dots_wh(im_wh, seed_box_size_this, stride_w, stride_h);
num_seed_boxes_placed = size(x_dots, 2) * size(y_dots, 2);
ret = num_seed_boxes_placed;
end