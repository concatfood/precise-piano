# Precise Piano

A simple pipeline for processing marker positions recorded by an OptiTrack system with the goal of hand joint tracking.

This repository contains the implementation of the paper 'Precise Hand Pose Data Collection for Piano Players' submitted
to the CHI 2021 workshop Human Augmentation for Skill Acquisition and Skill Transfer (HAA 2021).

The dataset we recorded will be made public soon.

## Usage

### Requirements

The requirements include the following Python packages:
1. numpy
2. matplotlib
3. tqdm

The code in this repository is very basic which is why the steps listed below need to be verified one after another.

1. `markers_by_time.py` can be used to find a good seed frame for the matching step later in the process. A good seed
   is identified by the correct number of tracked markers (21) as well as other good frames in its temporal vicinity.
2. `show_order.py` is used to highlight the order in which OptiTrack has tracked the markers in the seed frame. This
order is expected to differ from the order depicted in the image below. For verification, the order can be noted in the 
   file `show_order_test.py` in lines 43-63. As an example: If the first marker you see is marker 8 in the image below,
   you replace the corresponding line with `frame_seed_ordered[8] = frame_seed[0]`. If the second marker is 15, you
   replace the next corresponding line with `frame_seed_ordered[15] = frame_seed[1]` and so on.    
3. Copy the same lines of code into lines 45-65 in `match.py` to enforce the same labeling of all markers in all frames.
4. Lastly, run `map.py` to map the markers into the centers of the hand joints.

At the top of the file, there is a variable `optitrack_file` as well as `f_seed` which denotes the seed frame number.
Note that this code only works with OptiTrack's .csv marker files in the correct format. We use millimeters as global
coordinate unit.

![marker labels](labels.png)

In the future, a more stable and fully automated process might be made public.

In case there are any questions, feel free to contact us immediately.
